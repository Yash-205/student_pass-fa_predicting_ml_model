from dotenv import load_dotenv
load_dotenv()

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


def main() -> None:
    print("🚀 Starting training pipeline...")

    processor = DataProcessor()
    trainer = ModelTrainer()

    print("\n[1] Loading data...")
    df = processor.load_data()

    print("\n[2] Filling null values...")
    df = processor.fill_nulls(df)

    print("\n[3] Removing outliers...")
    df_clean = processor.remove_outliers(df)

    print("\n[4] Preprocessing features and target...")
    X_scaled, y, df_clean = processor.preprocess(df_clean)

    print("\n[5] Splitting and balancing with SMOTE...")
    X_train_sm, X_test, y_train_sm, y_test = processor.split_and_balance(X_scaled, y)

    print("\n[6] Training and evaluating models...")
    trainer.train_and_evaluate(X_train_sm, y_train_sm, X_test, y_test)

    print("\n[7] Training KMeans clustering...")
    trainer.train_kmeans(df_clean, processor.feature_cols, processor.scaler)

    print("\n[8] Saving artifacts...")
    trainer.save_artifacts(
        processor.scaler, processor.feature_names, processor.feature_cols
    )


if __name__ == "__main__":
    main()
