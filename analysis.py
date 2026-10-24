"""Retrospective analysis of deposited bulk RNA-seq TPM, not a clinical model."""

from pathlib import Path
import csv, gzip, hashlib, itertools, json, platform, re, urllib.request
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from Bio import Medline

ROOT = Path(__file__).parent
SOURCES = {
    "GSE107011_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE107nnn/GSE107011/matrix/GSE107011_series_matrix.txt.gz",
    "GSE107011_TPM.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE107nnn/GSE107011/suppl/GSE107011_Processed_data_TPM.txt.gz",
    "paper.medline": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=30726743&rettype=medline&retmode=text",
}


def acquire(root=ROOT):
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "data/sources.json").read_text())
    for name, url in SOURCES.items():
        p = raw / name
        if not p.exists():
            tmp = p.with_suffix(".download")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != manifest[name]["sha256"]:
            raise ValueError(
                f"Source changed: {name}; review rather than silently accept"
            )
    return raw


def metadata(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("!Sample_"):
                rows.append(next(csv.reader([line], delimiter="\t")))

    def one(key):
        found = [r[1:] for r in rows if r[0] == key]
        if len(found) != 1:
            raise ValueError(f"Ambiguous metadata field {key}")
        return found[0]

    titles = one("!Sample_title")
    gsms = one("!Sample_geo_accession")
    source = one("!Sample_source_name_ch1")
    if len(set(titles)) != len(titles) or len(set(gsms)) != len(gsms):
        raise ValueError("Duplicate sample identity")
    chars = [r[1:] for r in rows if r[0] == "!Sample_characteristics_ch1"]
    cell = [r for r in chars if all(x.startswith("cell type: ") for x in r)]
    if len(cell) != 1:
        raise ValueError("Missing cell type")
    types = [x.split(": ", 1)[1] for x in cell[0]]
    if types != source:
        raise ValueError("Cell type metadata sources disagree")
    out = pd.DataFrame(
        {
            "gsm": gsms,
            "title": titles,
            "sample_id": [re.sub(r"_rep\d+$", "", t) for t in titles],
            "donor": [t.split("_", 1)[0] for t in titles],
            "cell_type": types,
        }
    )
    if out.sample_id.duplicated().any() or (out.donor == "").any():
        raise ValueError("Ambiguous normalized identity")
    return out


def validate_expression(x, meta):
    if not x.index.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate expression identifier")
    if set(x.index) != set(meta.sample_id):
        raise ValueError("Expression/metadata sample sets differ")
    values = x.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("TPM must be finite and nonnegative")
    if x.shape[1] < 500:
        raise ValueError("Unexpectedly truncated expression matrix")
    return x.loc[meta.sample_id]


def bh(p):
    p = np.asarray(p, float)
    if (
        p.ndim != 1
        or len(p) == 0
        or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[
        ::-1
    ]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def exact_paired_p(differences):
    d = np.asarray(differences, float)
    if (
        d.ndim != 2
        or not 2 <= d.shape[0] <= 16
        or d.shape[1] == 0
        or not np.isfinite(d).all()
    ):
        raise ValueError(
            "Need complete finite differences: 2-16 donors and at least one gene"
        )
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    observed = np.abs(d.mean(axis=0))
    null = np.abs(signs @ d / len(d))
    return (null >= observed - 1e-12).mean(axis=0)


def evaluate(x, y, groups, splitter, grouped):
    pred = np.empty(len(y), dtype=object)
    baseline = np.empty(len(y), dtype=object)
    records = []
    folds = []
    splits = splitter.split(x, y, groups) if grouped else splitter.split(x, y)
    for fold, (tr, te) in enumerate(splits):
        overlap = set(groups[tr]) & set(groups[te])
        if grouped and overlap:
            raise ValueError("Donor leakage")
        pipe = make_pipeline(
            VarianceThreshold(0.01),
            SelectKBest(f_classif, k=300),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2500, solver="lbfgs"),
        )
        pipe.fit(x[tr], y[tr])
        pred[te] = pipe.predict(x[te])
        dummy = DummyClassifier(strategy="prior").fit(x[tr], y[tr])
        baseline[te] = dummy.predict(x[te])
        records.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "overlapping_donors": len(overlap),
                "accuracy": accuracy_score(y[te], pred[te]),
                "macro_f1": f1_score(y[te], pred[te], average="macro", zero_division=0),
            }
        )
        folds.extend(
            {
                "sample_index": int(i),
                "fold": fold,
                "split": "test",
                "donor": str(groups[i]),
            }
            for i in te
        )
    return pred, baseline, records, folds


def selection_outside_folds_accuracy(x, y, groups, k=300):
    """Deliberately wrong comparison: choose the k features on ALL samples, then hold donors out. Leaks labels into selection."""
    chosen = make_pipeline(VarianceThreshold(0.01), SelectKBest(f_classif, k=k)).fit(x, y).transform(x)
    pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneGroupOut().split(chosen, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2500, solver="lbfgs"))
        pred[te] = model.fit(chosen[tr], y[tr]).predict(chosen[te])
    return float(accuracy_score(y, pred))


def permutation_control(x, y, groups, seeds=range(20)):
    """Negative control: shuffle the cell-type labels (class counts kept) and repeat the donor-held-out evaluation.

    The in-fold pipeline should fall to chance. Selecting features on all samples first should not, because the shuffled
    labels have already shaped the features that are kept."""
    rows = []
    for seed in seeds:
        shuffled = np.random.default_rng(20261022 + seed).permutation(y)
        pred, _, _, _ = evaluate(x, shuffled, groups, LeaveOneGroupOut(), True)
        rows.append(
            {
                "seed": int(seed),
                "in_fold_accuracy": float(accuracy_score(shuffled, pred)),
                "selection_outside_folds_accuracy": selection_outside_folds_accuracy(x, shuffled, groups),
            }
        )
    return pd.DataFrame(rows)


def run(output=ROOT / "artifacts/run"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    raw = acquire()
    meta = metadata(raw / "GSE107011_series_matrix.txt.gz")
    expr = pd.read_csv(raw / "GSE107011_TPM.txt.gz", sep="\t", index_col=0).T
    expr = validate_expression(expr, meta)
    meta.to_csv(output / "sample_manifest.csv", index=False)
    # Exclusion uses biological metadata only, never performance or expression labels.
    counts = meta.loc[meta.cell_type != "PBMCs"].groupby("cell_type").donor.nunique()
    eligible = counts[counts >= 3].index
    keep = meta.cell_type.isin(eligible)
    m = meta.loc[keep].reset_index(drop=True)
    x = np.log2(expr.loc[m.sample_id].to_numpy() + 1)
    y = m.cell_type.to_numpy()
    groups = m.donor.to_numpy()
    qc = pd.DataFrame(
        {
            "sample_id": meta.sample_id,
            "sum_TPM": expr.sum(axis=1).to_numpy(),
            "detected_genes": (expr > 0).sum(axis=1).to_numpy(),
            "included": keep.to_numpy(),
            "cell_type": meta.cell_type,
        }
    )
    qc.to_csv(output / "quality_control.csv", index=False)
    pred, dummy, folds, assign = evaluate(x, y, groups, LeaveOneGroupOut(), True)
    p_random, _, random_folds, _ = evaluate(
        x, y, groups, StratifiedKFold(3, shuffle=True, random_state=20261022), False
    )
    control = permutation_control(x, y, groups)
    control.to_csv(output / "permutation_control.csv", index=False)
    chance = 1 / len(set(y))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=control.melt(id_vars="seed", var_name="pipeline", value_name="accuracy").replace(
            {"in_fold_accuracy": "Selection inside each training fold", "selection_outside_folds_accuracy": "Selection on all samples first"}
        ),
        x="pipeline", y="accuracy", ax=ax, size=7, color="#2b8cbe",
    )
    ax.axhline(chance, color="black", ls="--", label=f"Chance for {len(set(y))} balanced classes")
    ax.set(ylim=(0, max(0.5, float(control.max(numeric_only=True).max()) + 0.05)), xlabel="", ylabel="Donor-held-out accuracy, shuffled labels",
           title="Negative control: 20 label shuffles")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "permutation_control.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(folds).to_csv(output / "donor_holdout_scores.csv", index=False)
    pd.DataFrame(assign).to_csv(output / "fold_assignments.csv", index=False)
    pd.DataFrame(
        {
            "sample_id": m.sample_id,
            "donor": groups,
            "truth": y,
            "donor_holdout_prediction": pred,
            "sample_split_prediction": p_random,
            "baseline": dummy,
        }
    ).to_csv(output / "predictions.csv", index=False)
    # Descriptive PCA may use all included samples; it is not a validation feature pipeline.
    variance = x.var(axis=0)
    indices = np.argsort(variance)[-500:]
    pc = PCA(2, svd_solver="full").fit_transform(x[:, indices])
    lineage = m.cell_type.map(
        lambda t: (
            "B lineage"
            if ("B cells" in t or t == "Plasmablasts")
            else (
                "CD8 T"
                if "CD8" in t
                else (
                    "CD4 T"
                    if any(
                        z in t
                        for z in [
                            "CD4",
                            "Th1",
                            "Th2",
                            "Th17",
                            "T regulatory",
                            "Follicular",
                        ]
                    )
                    else (
                        "Monocyte"
                        if "monocyte" in t.lower()
                        else "Dendritic" if "dendritic" in t else "Other"
                    )
                )
            )
        )
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=pc[:, 0], y=pc[:, 1], hue=lineage, style=groups, s=65, ax=ax)
    ax.set(
        title="Exploratory PCA of sorted bulk immune populations",
        xlabel="PC1 (log2 TPM, top 500 variable genes)",
        ylabel="PC2",
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "pca.png", dpi=160)
    plt.close(fig)
    labels = sorted(set(y))
    fig, ax = plt.subplots(figsize=(13, 12))
    sns.heatmap(
        confusion_matrix(y, pred, labels=labels),
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set(
        xlabel="Predicted cell type",
        ylabel="Depositor cell type",
        title="Out-of-donor confusion matrix",
    )
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    # All four donors have both NK and naive B samples; exact paired sign-flips.
    a = m[m.cell_type == "Natural killer cells"].set_index("donor")
    b = m[m.cell_type == "Naive B cells"].set_index("donor")
    donors = sorted(set(a.index) & set(b.index))
    if len(donors) != 4:
        raise ValueError("Predefined NK/B paired design changed")
    nk = np.log2(expr.loc[a.loc[donors, "sample_id"]].to_numpy() + 1)
    bc = np.log2(expr.loc[b.loc[donors, "sample_id"]].to_numpy() + 1)
    d = nk - bc
    ps = exact_paired_p(d)
    q = bh(ps)
    contrast = pd.DataFrame(
        {
            "gene_id": expr.columns,
            "mean_paired_log2_TPM_plus1_difference": d.mean(axis=0),
            "exact_sign_flip_p": ps,
            "BH_q": q,
        }
    )
    contrast.to_csv(output / "paired_NK_vs_naive_B.csv", index=False)
    top = np.argsort(np.abs(d.mean(axis=0)))[-12:]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        d[:, top].T,
        xticklabels=donors,
        yticklabels=expr.columns[top],
        center=0,
        cmap="vlag",
        ax=ax,
    )
    ax.set(
        title="NK minus naive B: descriptive paired log2(TPM + 1)",
        xlabel="Donor",
        ylabel="Ensembl gene ID",
    )
    fig.tight_layout()
    fig.savefig(output / "paired_contrast.png", dpi=160)
    plt.close(fig)
    with (raw / "paper.medline").open() as f:
        papers = list(Medline.parse(f))
    if len(papers) != 1 or papers[0].get("PMID") != "30726743":
        raise ValueError("Wrong publication record")
    citation = {k: papers[0].get(k) for k in ["PMID", "TI", "AU", "JT", "DP", "LID"]}
    (output / "citation.json").write_text(json.dumps(citation, indent=2))
    result = {
        "dataset": "GSE107011",
        "source_samples": len(meta),
        "source_genes": expr.shape[1],
        "source_donors": meta.donor.nunique(),
        "included_samples": len(m),
        "included_donors": m.donor.nunique(),
        "included_cell_types": len(set(y)),
        "excluded_samples": meta.loc[~keep, ["sample_id", "cell_type"]].to_dict(
            "records"
        ),
        "donor_holdout_accuracy": accuracy_score(y, pred),
        "donor_holdout_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "dummy_accuracy": accuracy_score(y, dummy),
        "sample_split_accuracy": accuracy_score(y, p_random),
        "permutation_control": {
            "shuffles": len(control),
            "chance_accuracy": chance,
            "in_fold_mean_accuracy": float(control.in_fold_accuracy.mean()),
            "in_fold_max_accuracy": float(control.in_fold_accuracy.max()),
            "selection_outside_folds_mean_accuracy": float(control.selection_outside_folds_accuracy.mean()),
            "selection_outside_folds_max_accuracy": float(control.selection_outside_folds_accuracy.max()),
            "reading": "Shuffled labels carry no signal. Accuracy near chance is the expected result of a leak-free pipeline; accuracy well above chance after selecting features on all samples shows the leak the in-fold design avoids.",
        },
        "grouped_fold_scores": folds,
        "sample_split_fold_scores": random_folds,
        "paired_contrast_donors": len(donors),
        "minimum_attainable_two_sided_exact_p": 2 / 2 ** len(donors),
        "BH_q_below_005": int((q < 0.05).sum()),
        "scope": "Deposited bulk TPM, healthy sorted cells, four development donors. Not single-cell RNA-seq, not raw-read QC, no independent external cohort or clinical validation.",
        "source_sha256": {
            name: hashlib.sha256((raw / name).read_bytes()).hexdigest()
            for name in SOURCES
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
