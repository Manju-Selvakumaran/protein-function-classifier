"""
Model training and evaluation module for protein function classification.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# EC class names for display
EC_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases"
}


def load_data(features_path='data/processed/X_features.npy',
              labels_path='data/processed/y_labels.npy'):
    """
    Load preprocessed features and labels.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    """
    X = np.load(features_path)
    y = np.load(labels_path)
    print(f"Loaded data: X={X.shape}, y={y.shape}")
    return X, y


def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Split and optionally scale the data.
    Also encodes labels to start from 0 for XGBoost compatibility.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
    scale : bool
        Whether to standardize features
    
    Returns
    -------
    dict
        Dictionary containing train/test splits, scaler, and label encoder
    """
    # Encode labels to start from 0 (required for XGBoost)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Original labels: {np.unique(y)}")
    print(f"Encoded labels: {np.unique(y_encoded)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Features standardized")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': label_encoder
    }


def get_models():
    """
    Get dictionary of models to train.
    
    Returns
    -------
    dict
        Dictionary of model name -> model object
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric='mlogloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
    }
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Train and evaluate a single model.
    
    Parameters
    ----------
    model : estimator
        Scikit-learn compatible model
    X_train, X_test : np.ndarray
        Feature matrices
    y_train, y_test : np.ndarray
        Labels (encoded, starting from 0)
    model_name : str
        Name for display
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    print(f"\nTraining {model_name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'f1_macro': f1_score(y_test, y_pred_test, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred_test, average='weighted'),
        'precision_macro': precision_score(y_test, y_pred_test, average='macro'),
        'recall_macro': recall_score(y_test, y_pred_test, average='macro'),
        'y_pred': y_pred_test,
        'model': model
    }
    
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"  F1 (macro):     {results['f1_macro']:.4f}")
    
    return results


def cross_validate_model(model, X, y, cv=5, model_name="Model"):
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : estimator
        Scikit-learn compatible model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    cv : int
        Number of folds
    model_name : str
        Name for display
    
    Returns
    -------
    dict
        Cross-validation results
    """
    print(f"\nCross-validating {model_name} ({cv}-fold)...")
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Accuracy scores
    acc_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    # F1 scores
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    
    results = {
        'model_name': model_name,
        'cv_accuracy_mean': acc_scores.mean(),
        'cv_accuracy_std': acc_scores.std(),
        'cv_f1_mean': f1_scores.mean(),
        'cv_f1_std': f1_scores.std(),
        'cv_scores': acc_scores
    }
    
    print(f"  CV Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std']*2:.4f})")
    print(f"  CV F1 Macro: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std']*2:.4f})")
    
    return results


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models.
    
    Returns
    -------
    list
        List of result dictionaries for each model
    """
    models = get_models()
    all_results = []
    
    
    print("TRAINING AND EVALUATING MODELS")
    
    
    for name, model in models.items():
        results = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        all_results.append(results)
    
    return all_results


def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from a tree-based model.
    
    Parameters
    ----------
    model : estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to return
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def save_model(model, scaler, label_encoder, 
               model_path='models/best_model.pkl',
               scaler_path='models/scaler.pkl',
               encoder_path='models/label_encoder.pkl'):
    """
    Save trained model, scaler, and label encoder to disk.
    
    Parameters
    ----------
    model : estimator
        Trained model
    scaler : StandardScaler
        Fitted scaler
    label_encoder : LabelEncoder
        Fitted label encoder
    model_path : str
        Path to save model
    scaler_path : str
        Path to save scaler
    encoder_path : str
        Path to save label encoder
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f" Saved model to {model_path}")
    
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
        print(f" Saved scaler to {scaler_path}")
    
    if label_encoder is not None:
        joblib.dump(label_encoder, encoder_path)
        print(f" Saved label encoder to {encoder_path}")


def load_model(model_path='models/best_model.pkl',
               scaler_path='models/scaler.pkl',
               encoder_path='models/label_encoder.pkl'):
    """
    Load trained model, scaler, and label encoder from disk.
    
    Returns
    -------
    tuple
        (model, scaler, label_encoder)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    label_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    
    print(f" Loaded model from {model_path}")
    
    return model, scaler, label_encoder


def print_classification_report(y_true, y_pred, label_encoder=None):
    """
    Print detailed classification report with EC class names.
    """
    print("DETAILED CLASSIFICATION REPORT")
    
    # Get unique classes (these are encoded 0-6)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Map encoded labels back to original EC numbers
    if label_encoder is not None:
        original_classes = label_encoder.inverse_transform(classes)
        target_names = [f"EC {c} ({EC_NAMES.get(c, 'Unknown')})" for c in original_classes]
    else:
        # If no encoder, assume encoded labels are 0-6 mapping to EC 1-7
        target_names = [f"EC {c+1} ({EC_NAMES.get(c+1, 'Unknown')})" for c in classes]
    
    print(classification_report(y_true, y_pred, target_names=target_names))


def compare_models(results_list):
    """
    Create comparison DataFrame of all models.
    
    Parameters
    ----------
    results_list : list
        List of result dictionaries
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train Acc': f"{r['train_accuracy']:.4f}",
            'Test Acc': f"{r['test_accuracy']:.4f}",
            'F1 Macro': f"{r['f1_macro']:.4f}",
            'F1 Weighted': f"{r['f1_weighted']:.4f}",
            'Precision': f"{r['precision_macro']:.4f}",
            'Recall': f"{r['recall_macro']:.4f}"
        }
        for r in results_list
    ])
    
    return comparison


# Main execution for testing
if __name__ == "__main__":
    print("Testing model module")
    
    # Load data
    X, y = load_data()
    
    # Prepare data
    data = prepare_data(X, y)
    
    # Train a single model as test
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    results = evaluate_model(
        rf,
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        "Random Forest (Test)"
    )
    
    print("\n Model module working!")