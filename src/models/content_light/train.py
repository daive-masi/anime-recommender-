# src/models/content_light/train.py
# src/models/content_light/train.py
import argparse
import logging
import pandas as pd
from pathlib import Path
from .model import ContentRecommenderLight


# Configuration des logs
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "content_light.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("content_light")

def train(args):
    logger.info("Starting training process")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    model = ContentRecommenderLight(args.config).fit(df)
    model.save(args.output_dir)
    logger.info("Training completed successfully")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))  # Ajoute le chemin racine au sys.path

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed/anime_master_clean.csv", help="Chemin vers le dataset")
    p.add_argument("--config", default="src/models/content_light/config.yaml", help="Chemin vers le fichier de config")
    p.add_argument("--output_dir", default="models/content_light/", help="Dossier de sortie pour les modèles")
    args = p.parse_args()
    train(args)
