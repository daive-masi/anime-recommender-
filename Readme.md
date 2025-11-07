# 🎌 Anime Recommendation System

*A comprehensive machine learning project comparing multiple recommendation approaches for anime, built with Python 3.13.1 and VSCode.*

---

## 📊 Project Overview

This project implements and compares various recommendation algorithms to provide personalized anime recommendations. The goal is to evaluate the performance of different approaches and identify the most effective method for this specific use case.

### **Implemented Approaches:**
- **Content-based filtering** (TF-IDF, metadata similarity)
- **Supervised ML models** (Random Forest, XGBoost)
- **Deep Learning approaches** (LSTM, CNN, multimodal)
- **Hybrid models** (combining multiple approaches)

---

## 🛠️ Technical Environment

| **Category**       | **Details**                          |
|--------------------|--------------------------------------|
| **Python Version** | 3.13.1                               |
| **IDE**            | Visual Studio Code (with Jupyter)    |
| **ML Frameworks**  | Scikit-learn, TensorFlow              |
| **Visualization**  | Matplotlib, Seaborn, Plotly           |
| **Data Handling**  | Pandas, NumPy                         |

---

## 📁 Project Structure

```bash
anime-recommender/
├── notebooks/          # Jupyter notebooks (step-by-step analysis)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_exploration.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── ...             # Model training notebooks
├── src/                # Python source code
├── apps/               # Streamlit demo app (optional)
├── data/               # Dataset files
│   ├── raw/            # ⚠️ Download required (see below)
│   └── external/       # Data source documentation
├── scripts/            # Utility scripts
└── reports/            # Analysis reports & figures
```

---

## 🚀 Quick Start

### **1. Clone & Setup**
```bash
git clone https://github.com/daive-masi/anime-recommender-.git
cd anime-recommender-
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Download Data**
```bash
python scripts/setup_data.py
```
**Manually download the dataset from:**
[Kaggle - MyAnimeList Dataset](https://www.kaggle.com/datasets/your-dataset-link)
Extract the files to `data/raw/`.

### **4. Run Analysis**
Open the project in **VSCode** and execute the notebooks in the following order:
1. `01_data_collection.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_data_exploration.ipynb`
4. `04_feature_engineering.ipynb`
5. ... (model training notebooks)

---

## 📚 Dataset

### **Source:**
[MyAnimeList Dataset on Kaggle](https://www.kaggle.com/datasets/your-dataset-link)

### **Files Used:**
| **File**            | **Description**                          |
|---------------------|------------------------------------------|
| `details.csv`       | Anime metadata (title, genres, synopsis) |
| `stats.csv`         | Ratings and popularity statistics        |
| `recommendations.csv`| Anime-to-anime recommendations           |

---

## 🎯 Project Goals

1. **Data Collection and Cleaning**
   - Gather and preprocess anime data.
   - Handle missing values and outliers.

2. **Exploratory Data Analysis**
   - Understand data distribution and relationships.
   - Visualize key features and patterns.

3. **Feature Engineering**
   - Extract meaningful features from text and metadata.
   - Encode categorical variables.

4. **Baseline Model Implementation**
   - Implement content-based and collaborative filtering models.

5. **Advanced Model Comparison**
   - Compare ML (Random Forest, XGBoost) vs. DL (LSTM, CNN) approaches.

6. **Model Evaluation and Analysis**
   - Evaluate models using metrics like RMSE, precision, and recall.
   - Analyze strengths and weaknesses of each approach.

7. **Streamlit Demo Application (Optional)**
   - Build an interactive demo for real-time recommendations.

---

## 🧪 Experimental Approach

| **Model Type**      | **Algorithms**               | **Use Case**                     |
|---------------------|------------------------------|----------------------------------|
| **Content-based**   | TF-IDF + Cosine Similarity   | Baseline recommendation          |
| **Supervised ML**   | Random Forest, XGBoost       | Regression/Classification tasks  |
| **Deep Learning**   | LSTM (text), CNN (images)    | Semantic similarity              |
| **Hybrid**          | Feature combination          | Optimal performance              |

---

## 🤝 Contributing

We welcome contributions! Here’s how you can help:

1. **Fork the project**
2. **Create your feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

---

## 📜 License

This project is for educational purposes as part of a Master's degree in Machine Learning.

