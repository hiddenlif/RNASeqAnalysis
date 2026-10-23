FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
WORKDIR /work
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
RUN useradd --uid 10001 --create-home scientist && chown scientist:scientist /work
COPY --chown=scientist:scientist . .
ENV MPLCONFIGDIR=/tmp/mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
USER scientist
CMD ["python", "scripts/validate.py"]
