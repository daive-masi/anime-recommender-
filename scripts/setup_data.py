# scripts/setup_data.py
import os
import zipfile
from pathlib import Path

def setup_directories():
    """Create directory structure"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models',
        'reports/figures',
        'apps'  # Pour l'app Streamlit plus tard
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created")

def manual_download_instructions():
    """Instructions pour téléchargement manuel"""
    print("\n📋 MANUAL DOWNLOAD REQUIRED:")
    print("1. Visit: https://www.kaggle.com/datasets/neelagiriaditya/anime-dataset-jan-1917-to-oct-2025")
    print("2. Click 'Download' button")
    print("3. Extract zip to 'data/raw/'")
    print("4. Required files: details.csv, stats.csv, recommendations.csv")
    
    # Vérification des fichiers
    required_files = ['details.csv', 'stats.csv', 'recommendations.csv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(f"data/raw/{file}"):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing: {missing_files}")
    else:
        print("✅ All files found!")

if __name__ == "__main__":
    print("🚀 Anime Recommender - Setup")
    setup_directories()
    manual_download_instructions()