from pathlib import Path
import sys, subprocess, json, os, hashlib, platform

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ.setdefault("MPLCONFIGDIR", str(root / "artifacts/mpl"))
for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[k] = "1"
(root / "artifacts").mkdir(exist_ok=True)
sys.path.insert(0, str(root))
from analysis import acquire

acquire()
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "--junitxml=artifacts/pytest.xml"],
    check=True,
)
subprocess.run([sys.executable, "analysis.py"], check=True)
from nbclient import NotebookClient
import nbformat

nb = nbformat.read(root / "notebooks/analysis.ipynb", as_version=4)
NotebookClient(
    nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root)}}
).execute()
nbformat.validate(nb)
# Preserve warning text while making environment paths portable.
for cell in nb.cells:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            output["text"] = (
                output["text"]
                .replace(sys.prefix, "<python-env>")
                .replace(str(root), "<repo>")
            )
nbformat.write(nb, root / "notebooks/analysis.executed.ipynb")
import xml.etree.ElementTree as ET

suite = ET.parse(root / "artifacts/pytest.xml").getroot().find("testsuite")
receipt = {
    "pytest": {
        k: int(suite.attrib[k]) for k in ["tests", "failures", "errors", "skipped"]
    },
    "requirements_sha256": hashlib.sha256(
        (root / "requirements.lock").read_bytes()
    ).hexdigest(),
    "python": platform.python_version(),
    "notebook": "executed without cell errors",
    "summary_sha256": hashlib.sha256(
        (root / "artifacts/run/summary.json").read_bytes()
    ).hexdigest(),
    "analysis_sha256": hashlib.sha256((root / "analysis.py").read_bytes()).hexdigest(),
}
(root / "artifacts/validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
