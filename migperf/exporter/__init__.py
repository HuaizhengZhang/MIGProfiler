import os
from pathlib import Path
DCGM_URL = "http://dcgm_exporter:9400/metrics"
DATA_DIR = Path(os.getcwd()).parent.parent / Path("data/results/A30/")
METRICS_DIR = Path(os.getcwd()).parent.parent / Path("mig_perf/metrics/metrics.txt")