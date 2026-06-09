"""
Model Training module for the Student Study Coach.
This module defines the ModelTrainer class, which handles hyperparameter tuning,
model selection, and artifact persistence.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# Machine learning tools for cross-validation and hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, GridSearchCV
# Model algorithms: Linear (LogisticRegression) and Ensemble (XGBoost)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# Unsupervised learning for student clustering
from sklearn.cluster import KMeans
# Evaluation metrics to measure model performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from src.config import MODELS_DIR

class ModelTrainer:
    """
    Handles the training, hyperparameter tuning, and evaluation of student
    performance prediction models.
    """
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        """
        Initializes the trainer with a directory for saving artifacts.
        """
        self.models_dir = models_dir
        self.best_model: Any = None      # Stores the winning model after GridSearchCV
        self.best_name: str = ""         # Name of the winning model
        self.kmeans: Optional[KMeans] = None
        self.metrics: Dict[str, float] = {} # Final performance metrics

    def train_and_evaluate(
        self,
        X_train_sm: np.ndarray,
        y_train_sm: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> None:
        """
        Performs grid search across multiple model candidates using cross-validation.
        Selects the best performing model based on F1-weighted score.
        
        Args:
            X_train_sm: SMOTE-balanced training features.
            y_train_sm: SMOTE-balanced training targets.
            X_test: Original test features.
            y_test: Original test targets.
        """
        # Define the Cross-Validation strategy
        # StratifiedKFold ensures each fold has the same proportion of Pass/Fail classes as the original data
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Define model candidates and their hyperparameter grids for tuning
        candidates = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=2000
                ),
                "params": {"C": [0.1, 1.0, 10.0]}, # C controls regularization strength
            },
            # XGBoost (eXtreme Gradient Boosting): A powerful ensemble model
            "XGBoost": {
                "model": XGBClassifier(
                    eval_metric="logloss", random_state=42, n_jobs=-1
                ),
                "params": {
                    "n_estimators": [50, 100], # Number of trees
                    "max_depth": [3, 6],       # Depth of trees
                    "learning_rate": [0.05, 0.1], # Step size shrinkage
                },
            },
        }

        print(
            "\n{:<20} {:>10} {:>12} {:>10} {:>12}".format(
                "Model", "Accuracy", "Precision", "Recall", "F1 (weighted)"
            )
        )
        print("-" * 66)

        best_f1 = -1.0
        best_params: Dict[str, Any] = {}

        for name, config in candidates.items():
            print(f"\nRunning GridSearchCV for {name}...")
            
            # GridSearchCV (Grid Search Cross-Validation): Automates hyperparameter tuning.
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="f1_weighted", # We optimize for F1 score to balance precision and recall
                cv=cv_strategy,       # Use our StratifiedKFold strategy
                n_jobs=-1,            # Use all available CPU cores
                verbose=1,
            )
            
            # Train the grid search on SMOTE-balanced training data
            grid_search.fit(X_train_sm, y_train_sm)
            best_estimator = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")

            # Evaluate the best version of this model on the test set
            y_pred = best_estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            print(
                "{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>12.4f}".format(
                    name, acc, prec, rec, f1
                )
            )

            # Update self.best_model if this candidate performed better than previous ones
            if f1 > best_f1:
                best_f1, self.best_name, self.best_model, best_params = (
                    f1,
                    name,
                    best_estimator,
                    grid_search.best_params_,
                )

        print(f"\n🏆 Best overall model: {self.best_name}  (F1 = {best_f1:.4f})")
        print(f"Winning hyperparameters: {best_params}")

        # Final Evaluation: Print detailed metrics for the best model
        y_pred_final = self.best_model.predict(X_test)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_final)),
            "precision": float(precision_score(y_test, y_pred_final, average="weighted")),
            "recall": float(recall_score(y_test, y_pred_final, average="weighted")),
            "f1": float(f1_score(y_test, y_pred_final, average="weighted")),
        }

        print(f"\n── Final Metrics ({self.best_name}) ──────────────────")
        for k, v in self.metrics.items():
            print(f"  {k.capitalize():<10}: {v:.4f}")

        # Detailed breakdown per class (Pass/Fail)
        print("\n── Classification Report ──────────────────────────────")
        print(classification_report(y_test, y_pred_final, target_names=["Fail", "Pass"]))

        # Visualizing errors: False Positives vs False Negatives
        cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1])
        print("── Confusion Matrix ───────────────────────────────────")
        print(f"  TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}")
        print(f"  FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}")

    def train_kmeans(
        self, df: pd.DataFrame, feature_cols: List[str], scaler: Any
    ) -> None:
        """
        Trains a KMeans clustering model to segment students based on performance.
        
        Args:
            df: Cleaned dataframe.
            feature_cols: List of features to use for clustering.
            scaler: Trained scaler to normalize features before clustering.
        """
        # Train KMeans clustering to group students into 3 performance tiers
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        self.kmeans.fit(scaler.transform(df[feature_cols]))

    def save_artifacts(
        self, scaler: Any, feature_names: np.ndarray, feature_cols: List[str]
    ) -> None:
        """
        Saves all trained models, scalers, and metadata to the models directory.
        
        Args:
            scaler: Trained scaler object.
            feature_names: Array of feature names.
            feature_cols: List of feature column names.
        """
        # Save all trained objects and metadata to files
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(self.models_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.kmeans, os.path.join(self.models_dir, "kmeans.pkl"))
        joblib.dump(
            feature_names.tolist(),
            os.path.join(self.models_dir, "feature_names.pkl"),
        )
        joblib.dump(
            feature_cols, os.path.join(self.models_dir, "input_feature_cols.pkl")
        )
        joblib.dump(
            {"best_model_name": self.best_name, **self.metrics},
            os.path.join(self.models_dir, "meta.pkl"),
        )
        print("\n✅ All artifacts saved to models/")

