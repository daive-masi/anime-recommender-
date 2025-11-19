import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Utils
from src.utils.run_manager import create_run_folder, save_config_copy
from src.utils.logger import get_logger


def build_tfidf(df, cfg, logger):
    logger.info("Building TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        max_features=cfg["tfidf"]["max_features"],
        min_df=cfg["tfidf"]["min_df"],
        ngram_range=tuple(cfg["tfidf"]["ngram_range"])
    )

    X = vectorizer.fit_transform(df["synopsis_clean"])
    logger.info(f"TF-IDF shape: {X.shape}")

    return X, vectorizer


def build_genres(df, logger):
    logger.info("Building one-hot genres matrix...")
    mlb = df["genres_list"].str.join("|").str.get_dummies()
    logger.info(f"Genres shape: {mlb.shape}")
    return mlb


def build_numeric(df, cfg, logger):
    logger.info("Building numeric features...")
    numeric_cols = cfg["numeric_features"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols])

    logger.info(f"Numeric shape: {X.shape}")
    return X, scaler


def concat_features(parts, logger):
    from scipy.sparse import hstack

    logger.info("Concatenating all feature matrices...")
    X = hstack(parts).tocsr()
    logger.info(f"Final feature matrix shape: {X.shape}")
    return X


def train_knn(X, cfg, logger):
    logger.info("Training KNN model...")

    model = NearestNeighbors(
        n_neighbors=cfg["knn"]["n_neighbors"],
        metric=cfg["knn"]["metric"],
        algorithm="auto"
    )
    model.fit(X)

    logger.info("KNN training done.")
    return model


def train(cfg, config_path):
    """
    MAIN TRAIN FUNCTION FOR LIGHT MODEL
    """

    # ---------------------------------------------------------
    # 1) Prepare run folder + logging
    # ---------------------------------------------------------
    run_folder = create_run_folder("light")
    save_config_copy(config_path, run_folder)
    log_path = run_folder / "training.log"
    logger = get_logger(log_path)

    logger.info("===== START TRAINING LIGHT MODEL =====")

    # ---------------------------------------------------------
    # 2) Load data
    # ---------------------------------------------------------
    df_path = Path(cfg["dataset"])
    logger.info(f"Loading dataset: {df_path}")

    df = pd.read_csv(df_path)
    logger.info(f"Dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 3) Build features
    # ---------------------------------------------------------
    X_parts = []
    meta_objects = {}

    # TF-IDF
    X_tfidf, vectorizer = build_tfidf(df, cfg, logger)
    X_parts.append(X_tfidf)
    meta_objects["tfidf_vectorizer"] = vectorizer

    # Genres one-hot
    X_genres = build_genres(df, logger)
    X_parts.append(X_genres)

    # Numeric
    X_num, scaler = build_numeric(df, cfg, logger)
    X_parts.append(X_num)
    meta_objects["numeric_scaler"] = scaler

    # Concatenate
    X = concat_features(X_parts, logger)

    # ---------------------------------------------------------
    # 4) Train model
    # ---------------------------------------------------------
    knn_model = train_knn(X, cfg, logger)

    # ---------------------------------------------------------
    # 5) Save outputs
    # ---------------------------------------------------------
    out_folder = run_folder / "latest"
    out_folder.mkdir(exist_ok=True)

    logger.info("Saving model and assets...")

    joblib.dump(knn_model, out_folder / "knn_model.joblib")
    joblib.dump(meta_objects, out_folder / "meta.pkl")

    # save sparse matrix efficiently
    from scipy import sparse
    sparse.save_npz(out_folder / "features.npz", X)

    # Save metadata CSV (titles, mal_id…)
    df[["mal_id", "title"]].to_csv(out_folder / "metadata.csv", index=False)

    logger.info("All artifacts saved successfully.")
    logger.info("===== TRAINING LIGHT MODEL FINISHED =====")

    # Return summary
    return {
        "run_folder": str(run_folder),
        "num_items": len(df),
        "feature_dim": X.shape[1]
    }
