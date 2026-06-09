"""
Main entry point for the Student Study Coach application.
This script handles:
1. Environment variable loading (API keys).
2. Streamlit page configuration.
3. Automatic model training if artifacts are missing (e.g., on first deploy).
4. Multi-page navigation setup.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Ensure API keys are available in os.environ for other modules
# On Streamlit Cloud, keys are fetched from secrets if not in environment
for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    if key not in os.environ and key in st.secrets:
        os.environ[key] = st.secrets[key]

# Configure global Streamlit settings
st.set_page_config(page_title="Student Study Coach", layout="wide", page_icon="🎓")

# ── Auto-train if models are missing (e.g. first deploy on Streamlit Cloud) ──
from src.config import MODELS_DIR

# Check for required model artifacts
required = ["model.pkl", "scaler.pkl", "kmeans.pkl", "meta.pkl"]
models_missing = not all((MODELS_DIR / f).exists() for f in required)

if models_missing:
    with st.spinner("Setting up models for the first time — this takes about 30 seconds..."):
        from src.data_processor import DataProcessor
        from src.model_trainer import ModelTrainer

        # Initialize processor and trainer
        processor = DataProcessor()
        trainer = ModelTrainer()

        # Run the full training pipeline
        df = processor.load_data()
        df = processor.fill_nulls(df)
        df_clean = processor.remove_outliers(df)
        X_scaled, y, df_clean = processor.preprocess(df_clean)
        X_train_sm, X_test, y_train_sm, y_test = processor.split_and_balance(X_scaled, y)
        trainer.train_and_evaluate(X_train_sm, y_train_sm, X_test, y_test)
        trainer.train_kmeans(df_clean, processor.feature_cols, processor.scaler)
        trainer.save_artifacts(processor.scaler, processor.feature_names, processor.feature_cols)

    st.success("Models ready!")
    st.rerun()

# Define navigation pages
dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="🎓", default=True)
chat = st.Page("pages/chat_interface.py", title="Study Coach", icon="🤖")

# Initialize navigation
pg = st.navigation([dashboard, chat])
pg.run()

