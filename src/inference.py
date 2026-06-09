"""
Inference module for the Student Study Coach.
This module provides the StudentPredictor class, which handles loading
trained ML models and performing predictions on new student data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.config import MODELS_DIR, FEATURES

class StudentPredictor:
    """
    Handles ML model inference and recommendation generation.
    Loads saved model artifacts (classifier, scaler, clusterer) and provides
    methods to process raw inputs and return predictions.
    """
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        """
        Initializes the predictor by defining paths to model artifacts.
        """
        self.models_dir = models_dir
        self.feature_cols = FEATURES
        self.model_files = {
            "model":  os.path.join(self.models_dir, "model.pkl"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "kmeans": os.path.join(self.models_dir, "kmeans.pkl"),
            "meta":   os.path.join(self.models_dir, "meta.pkl"),
        }
        # Load all necessary model artifacts (model, scaler, clusters, metadata)
        self.bundle = self._load_pretrained_model()

    def _load_pretrained_model(self) -> Optional[Dict[str, Any]]:
        """
        Loads the pickled model files from the models directory.
        Performs a consistency check on the scaler features.
        
        Returns:
            Dict containing the model, scaler, kmeans, and meta objects, or None if missing.
        """
        # Check if all required files exist before trying to load
        if not all(os.path.exists(v) for v in self.model_files.values()):
            return None
            
        try:
            bundle = {
                "model":  joblib.load(self.model_files["model"]),
                "scaler": joblib.load(self.model_files["scaler"]),
                "kmeans": joblib.load(self.model_files["kmeans"]),
                "meta":   joblib.load(self.model_files["meta"]),
            }

            # Consistency check: Ensure the scaler matches the number of features defined in config
            scaler = bundle["scaler"]
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(self.feature_cols):
                raise RuntimeError(
                    f"Loaded scaler expects {scaler.n_features_in_} features but config defines "
                    f"{len(self.feature_cols)}. Re-run training to synchronize."
                )
            return bundle
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

    def predict_bundle(
        self, X_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes raw input data through the full prediction pipeline.
        Pipeline: Numeric Conversion -> Clipping -> Scaling -> Prediction -> Clustering.
        
        Args:
            X_df: DataFrame containing the raw student features.
            
        Returns:
            Tuple of (probabilities, class_predictions, cluster_labels).
        """
        if not self.bundle:
            raise RuntimeError("Models not loaded. Ensure artifacts exist in models/ directory.")
        
        # 1. Feature Selection & Data Cleaning
        Xc = X_df[self.feature_cols].copy()
        
        for col in self.feature_cols:
            # Force numeric and fill NaNs with 0 to prevent pipeline crashes
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0)
            
        # 2. Value Clipping
        # Ensure input values don't exceed the min/max seen during training
        scaler = self.bundle["scaler"]
        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            for i, col in enumerate(self.feature_cols):
                Xc[col] = Xc[col].clip(scaler.data_min_[i], scaler.data_max_[i])
        
        # 3. Feature Scaling
        X_s = scaler.transform(Xc)
        
        # 4. Success Prediction
        # Get probability of 'Pass' (class 1)
        probas = self.bundle["model"].predict_proba(X_s)[:, 1]
        
        # 5. Hard Classification
        # 0.5 threshold for Pass/Fail
        preds = (probas >= 0.5).astype(int)
        
        # 6. Performance Clustering
        # Assign student to a performance group/tier (e.g., High-Flyer, At-Risk)
        clusters = self.bundle["kmeans"].predict(X_s)
        
        return probas, preds, clusters

    def get_student_recommendations(self, row: pd.Series) -> List[str]:
        """
        Analyzes a student's metrics to generate rule-based actionable tips.
        This provides immediate feedback before the user starts the chat session.
        
        Args:
            row: A pandas Series containing student features.
            
        Returns:
            List of strings containing personalized tips.
        """
        recs = []
        
        # Rule-based logic for lifestyle and academic factors
        if float(row.get("daily_study_hours", 5)) < 2:
            recs.append("📚 Increase daily study to at least 2-3 hours.")
        if float(row.get("attendance_rate", 1.0)) < 0.75:
            recs.append("🏫 Aim for 75%+ attendance to reduce knowledge gaps.")
        if float(row.get("math_score", 100)) < 50:
            recs.append("🔢 Focus on math — practice problem sets daily.")
        if float(row.get("reading_score", 100)) < 50:
            recs.append("📖 Improve reading — summarize one article daily.")
        if float(row.get("writing_score", 100)) < 50:
            recs.append("✍️ Work on writing structure and grammar daily.")
        if float(row.get("stress_level", 5)) > 7:
            recs.append("🧘 High stress detected — try the Pomodoro technique.")
        if float(row.get("sleep_hours", 8)) < 6:
            recs.append("😴 Aim for 7-8 hours of sleep for better retention.")
        if float(row.get("motivation_score", 50)) < 30:
            recs.append("🎯 Set small SMART goals to rebuild motivation daily.")
        
        # Default encouragement if no critical areas identified
        if not recs:
            recs.append("🌟 Great habits! Keep consistency and add spaced revision.")
        return recs

    def get_meta(self) -> Dict[str, Any]:
        """Returns metadata about the trained model (e.g., training date, metrics)."""
        return self.bundle["meta"] if self.bundle else {}

    def is_ready(self) -> bool:
        """Checks if the models were successfully loaded and are ready for inference."""
        return self.bundle is not None

