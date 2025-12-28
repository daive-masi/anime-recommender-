# Anime Recommender System - AI Coding Agent Instructions

## Project Overview
A machine learning comparative analysis project implementing multiple recommendation algorithms (content-based, collaborative filtering, supervised ML, deep learning) for personalized anime recommendations. Python 3.13.1, Jupyter-based exploration with scriptable training pipelines.

## Architecture & Data Flows

### Two-Track Model Architecture
The project has **dual parallel implementations** with different computational approaches:

- **Light Model** (`src/models/content_light/`): Fast, memory-efficient
  - TF-IDF vectorization (20k features, bigrams)
  - Optional UMAP dimensionality reduction (50 components)
  - KNN search (30 neighbors, cosine distance)
  - Config: `config/content_light.yaml`

- **Premium Model** (`src/models/content_premium/`): Advanced with ANN
  - TF-IDF vectorization (50k features)
  - SVD or UMAP dimensionality reduction
  - FAISS-based approximate nearest neighbor search
  - Optional GPU acceleration (CUDA)
  - Config: `config/content_premium.yaml`

### Data Pipeline
```
data/raw/ (MyAnimeList dataset)
  ↓
data/processed/ (anime_master_clean.csv - cleaned & merged)
  ↓
Feature Engineering → TF-IDF Vectorization → Dimensionality Reduction
  ↓
src/models/{content_light|content_premium}/ (trained artifacts)
  ↓
runs/{model_type}_{timestamp}/ (timestamped run outputs)
```

## Configuration System
- **YAML-based**: All model configs in `config/` with preset variants in `config/presets/`
- **Loader**: `src/utils/config_loader.py` - simple `load_config(path)` function
- **Run Manager**: `src/utils/run_manager.py` - creates timestamped run folders, saves config copies
- **Pattern**: `python train_{light|premium}.py --config config/{filename}.yaml [--gpu] [--debug]`

## Key Conventions

### Training Entry Points
- **Light**: `train_light.py` → `src/models/content_light/train.py`
- **Premium**: `train_premium.py` → `src/models/content_premium/train.py`
- Both use `--config` CLI argument (YAML path required)
- Premium accepts `--gpu` and `--debug` flags

### Notebook Workflow Order
1. `notebooks/prepared_dataset/01_clean_master.ipynb` - data cleaning
2. `notebooks/prepared_dataset/02_feature_engineering.ipynb` - feature extraction
3. `notebooks/04_train_content_light.ipynb` - light model training
4. `notebooks/05_train_content_premium.ipynb` - premium model training
5. `notebooks/06_train_random_forest.ipynb`, `07_train_svd_collaborative.ipynb` - alternative models

### File Organization
- **Source code**: `src/` - all reusable Python modules
- **Models**: `src/models/{type}/train.py` defines the training logic (receives config dict)
- **Utils**: `src/utils/` - config_loader, text_cleaning, logger, feature_scaling, genre_processing
- **Artifacts**: Trained models saved to `runs/{type}_{timestamp}/artifacts/` (joblib/pickle format)

### Path Handling
- **Relative paths**: All paths in YAML configs are relative to project root
- **Critical**: Import system expects `src/` to be in Python path (handled by setup.py + editable install)
- **Example**: Dataset loaded from `data/processed/anime_master_clean.csv` (relative to root)

## Development Workflows

### Training a Model
```bash
# Light model with default preset
python train_light.py --config config/presets/light_fast.yaml

# Premium model with GPU support
python train_premium.py --config config/content_premium.yaml --gpu

# Custom config
python train_light.py --config config/content_light.yaml
```

### Running Analysis
```bash
# All models in sequence
bash scripts/train_all.sh

# Evaluation across all models
bash scripts/eval_all.sh
```

### Environment Setup
```bash
pip install -r requirements.txt
python scripts/setup_data.py  # Downloads & prepares data
```

## Critical Dependencies & Integration Points

### External Libraries
- **ML**: scikit-learn (TF-IDF, KNN), FAISS (ANN search), XGBoost, TensorFlow
- **Dimensionality**: UMAP, scikit-learn SVD
- **Data**: pandas, numpy
- **Serialization**: joblib (preferred for sklearn models)
- **Config**: PyYAML

### Cross-Component Communication
- Config YAML → loaded by `config_loader.py` → passed as dict to `train()` function
- Trained artifacts → saved with `run_manager.py` → loaded in evaluation notebooks
- Feature processing → shared utilities in `src/utils/` (text_cleaning, genre_processing, feature_scaling)

## Project-Specific Patterns

### Pattern: Config-Driven Training
Each train function signature: `train(config: dict, config_path: str, **kwargs)`
- Config dict has `data`, `model` (with nested `tfidf`, `umap`/`svd`, `knn`/`faiss`), `output` sections
- `config_path` used by `run_manager` to copy YAML to run folder for reproducibility
- Avoid hardcoding; always read from config dict

### Pattern: Timestamped Run Organization
```python
from src.utils.run_manager import create_run_folder, save_config_copy
run_folder = create_run_folder("content_light")
save_config_copy("config/content_light.yaml", run_folder)
# Artifacts saved to: run_folder / "artifacts" / {model_name}.joblib
```

### Pattern: Model Artifact Naming
Joblib files follow pattern: `{component}_{model_type}.joblib`
- Example: `tfidf_vectorizer.joblib`, `knn_model.joblib`, `umap_model.joblib`
- Avoid custom naming; follow this convention for reproducibility

## Common Issues & Solutions

**Issue**: Import errors with `src` modules
- **Fix**: Ensure `setup.py` installs package in editable mode: `pip install -e .`
- Verify `src/__init__.py` exists (can be empty)

**Issue**: Config file not found
- **Fix**: Provide absolute path or verify path is relative to project root
- YAML paths in configs must also be relative to project root

**Issue**: UMAP/FAISS not available
- **Fix**: Check `config/presets/` for alternative configs
- `premium_no_umap.yaml` uses SVD instead; `light_fast.yaml` skips dimensionality reduction

**Issue**: GPU not found for FAISS
- **Fix**: Ensure CUDA-compatible FAISS installed
- Falls back to CPU; --gpu flag optional

## Testing & Validation
- No formal test suite; validation done in notebooks with synthetic test queries
- Light model trained on 100 test queries (see `src/config.py: TEST_QUERIES_COUNT`)
- Evaluation metrics in `src/evaluation/metrics.py`
- Comparative analysis notebooks in `notebooks/analyse_comparative/`
