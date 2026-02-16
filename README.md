# 🧬 Protein Function Classifier

A machine learning application that predicts enzyme functional class (EC number) from protein amino acid sequences.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🎯 Project Overview

This project uses machine learning to classify proteins into one of **7 major enzyme classes** based on their amino acid sequence:

| EC Class | Name | Function |
|----------|------|----------|
| EC 1 | Oxidoreductases | Catalyze oxidation-reduction reactions |
| EC 2 | Transferases | Transfer functional groups |
| EC 3 | Hydrolases | Catalyze hydrolysis reactions |
| EC 4 | Lyases | Break bonds without hydrolysis |
| EC 5 | Isomerases | Catalyze structural rearrangements |
| EC 6 | Ligases | Join molecules using ATP |
| EC 7 | Translocases | Move molecules across membranes |

## 📊 Results

- **Best Model:** XGBoost Classifier
- **Test Accuracy:** 60.2%
- **F1 Score (Macro):** 60.3%

*Note: Random baseline for 7 classes would be ~14%, so our model performs significantly better than chance.*

## 🔬 Features Extracted

The model uses **437 features** derived from protein sequences:

- **Amino Acid Composition (20):** Frequency of each amino acid
- **Dipeptide Composition (400):** Frequency of amino acid pairs
- **Physicochemical Properties (10):** Hydrophobicity, molecular weight, charge, polarity
- **Secondary Structure Propensity (4):** Helix and sheet forming tendencies
- **Sequence Complexity (3):** Entropy and repetitiveness measures

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Manju-Selvakumaran/protein-function-classifier.git
cd protein-function-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Train the Model (Optional)

```bash
# Download data from UniProt
python src/data_loader.py

# Run training notebook
jupyter notebook notebooks/03_model_training.ipynb
```

## 📁 Project Structure

```
protein-function-classifier/
│
├── data/
│   ├── raw/                    # Raw UniProt data
│   └── processed/              # Processed features
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_loader.py          # UniProt data fetching
│   ├── features.py             # Feature extraction
│   └── model.py                # Model training utilities
│
├── models/
│   ├── best_model.pkl          # Trained XGBoost model
│   ├── scaler.pkl              # Feature scaler
│   └── label_encoder.pkl       # Label encoder
│
├── figures/                    # Visualizations
├── app.py                      # Streamlit web application
├── requirements.txt
└── README.md
```

## 📈 Model Comparison

| Model | Train Acc | Test Acc | F1 Macro |
|-------|-----------|----------|----------|
| Logistic Regression | 70.8% | 47.6% | 47.0% |
| Random Forest | 100% | 55.7% | 55.9% |
| **XGBoost** | **95.2%** | **60.2%** | **60.3%** |

## 🖼️ Screenshots

### Web Interface
*Paste a protein sequence and get instant predictions*

### Confusion Matrix
*Model performance across all enzyme classes*

### Feature Importance
*Top features for classification*

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **Streamlit** - Web application
- **matplotlib/seaborn** - Visualization
- **UniProt REST API** - Protein data

## 📚 Dataset

- **Source:** UniProt (Swiss-Prot reviewed entries)
- **Size:** ~5,300 protein sequences
- **Classes:** 7 EC classes (balanced)
- **Sequence length:** 50-2000 amino acids


## 📄 License

MIT License - feel free to use this project for learning and development.

## 👤 Author

**Manju Selvakumaran**
- GitHub: https://github.com/Manju-Selvakumaran
- LinkedIn: https://www.linkedin.com/in/binf-manju-selvakumaran/
  
**Acknowledgments**:
- UniProt — For providing the protein sequence database and REST API used to collect training data
- scikit-learn — For machine learning tools and model evaluation utilities
- XGBoost — For the gradient boosting implementation
- Streamlit — For the easy-to-use web application framework
- Claude AI (Anthropic) — For analysis assistance

---

**Last Updated**: February 2026

---

*Built as a portfolio project demonstrating machine learning applications in bioinformatics.*
