import os
from datetime import datetime
from pathlib import Path
import shutil

def create_run_folder(model_type: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = Path(f"models/{model_type}/runs/run_{ts}")
    base.mkdir(parents=True, exist_ok=True)
    (base / "latest").mkdir(exist_ok=True)
    return base

def save_config_copy(config_path, dest_folder):
    shutil.copy(config_path, dest_folder / "config_used.yaml")
