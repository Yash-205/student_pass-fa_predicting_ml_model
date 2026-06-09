"""
Data Processing module for the Student Study Coach.
This module defines the DataProcessor class, which handles data loading,
cleaning, feature scaling, and class balancing using SMOTE.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, FEATURES

class DataProcessor:
    """
    Handles the end-to-end data pipeline from raw CSV to balanced training sets.
    """
    def __init__(self, data_path: str = str(DATA_PATH)) -> None:
        """Initializes the processor with data path and feature definitions."""
        self.data_path = data_path
        self.feature_cols = FEATURES
        self.scaler = MinMaxScaler()
        self.feature_names: np.ndarray = np.array(FEATURES)

    def load_data(self) -> pd.DataFrame:
        """Loads the raw dataset from CSV."""
        df = pd.read_csv(self.data_path)
        print(f"Loaded: {df.shape[0]:,} rows | Columns: {df.columns.tolist()}")
        return df

    def fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in feature columns with their respective medians."""
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        print(f"Null values filled with column medians.")
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes outliers using the Interquartile Range (IQR) method."""
        for col in self.feature_cols:
            if col in df.columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        print(f"After IQR outlier removal: {df.shape[0]:,} rows")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Processes features and targets.
        Steps: Target encoding, 1% quantile clipping, and MinMax scaling.
        """
        df = df.copy()
        # Encode Pass/Fail as 1/0
        df["result"] = df["pass_fail"].apply(
            lambda x: 1 if str(x).strip().lower() == "pass" else 0
        )
        vc = df["result"].value_counts()
        print(
            f"Class distribution → Pass: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)  "
            f"Fail: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)"
        )

        X_raw = df[self.feature_cols].copy()
        y = df["result"].copy()

        # Clip extreme values to the 1st and 99th percentiles
        for col in self.feature_cols:
            X_raw[col] = X_raw[col].clip(
                X_raw[col].quantile(0.01), X_raw[col].quantile(0.99)
            )

        X_scaled = self.scaler.fit_transform(X_raw)
        return X_scaled, y, df

    def split_and_balance(
        self, X: np.ndarray, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Splits data into train/test sets and applies SMOTE to the training set
        to handle class imbalance.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

        # SMOTE (Synthetic Minority Over-sampling Technique)
        # Generates synthetic samples for the minority class (Fail) to balance the dataset.
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        vc_sm = pd.Series(y_train_sm).value_counts()
        print(f"After SMOTE → Pass: {vc_sm.get(1,0):,}  Fail: {vc_sm.get(0,0):,}")
        return X_train_sm, X_test, y_train_sm, y_test

