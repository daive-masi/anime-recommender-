# src/config.py
import pandas as pd
import numpy as np

# Paths
RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"
MODELS_PATH = "../models/"

# Random seed
RANDOM_STATE = 42

# Test set for evaluation
TEST_QUERIES_COUNT = 100

# Feature Engineering
TFIDF_MAX_FEATURES = 5000
EMBEDDING_DIM = 100