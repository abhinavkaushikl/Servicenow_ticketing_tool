import pandas as pd
from features.Missing_null_pipeline import DataProcessor
from features.handletimeseriesdata import TimeSeriesProcessor
from features.Encodingfeatures import FeatureEncoder
from features.leakageandsmote import SMOTEHandler,LeakyFeatureRemover
from models.Modeltraining_sla_breach import ModelTrainer
from evaluation.model_evaluation.evaluation import evaluate_and_save_best_model

def run_null_handling(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Running Null Handling...")
    processor = DataProcessor(df)
    processor.remove_high_null_columns()
    processor.handle_missing_values()

    total_null_values = processor.df_cleaned.isnull().sum().sum()
    print(" Cleaned DataFrame Preview:")
    print(processor.df_cleaned.head())
    print(f" Total remaining null values: {total_null_values}")

    return processor.df_cleaned


def run_time_series_processing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⏱ Running Time Series Feature Engineering...")
    print(f" Input DataFrame shape: {df.shape}")
    
    processor = TimeSeriesProcessor(df, reference_date_col='created_date')
    df_processed = processor.process()

    print(f" Time Series Feature Engineering Complete. Shape: {df_processed.shape}")
    return df_processed


def run_feature_encoding(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning Feature Encoding...")
    print(f"DataFrame shape before encoding: {df.shape}")
    
    encoder = FeatureEncoder(df, target_col='SLA Breach', verbose=True)
    df_encoded, encoding_strategy_df, encoders = encoder.encode()

    print(f"Encoding complete. Shape: {df_encoded.shape}")
    return df_encoded


def run_leak_removal_and_smote(df: pd.DataFrame):
    print("\n Running Leaky Feature Removal + SMOTE...")
    
    remover = LeakyFeatureRemover(target_col='SLA Breach', verbose=True)
    X, y = remover.fit_transform(df)

    print(f" After leak removal -> Features: {X.shape}, Target: {y.shape}")

    smoter = SMOTEHandler(verbose=True)
    X_resampled, y_resampled = smoter.apply(X, y)

    print(f" After SMOTE -> Balanced Features: {X_resampled.shape}, Target: {y_resampled.shape}")
    return X_resampled, y_resampled


def full_feature_pipeline():
    print("\n Starting In-Memory Feature Engineering Pipeline...")

    # Step 1: Load raw data from config
    df = pd.read_csv("data/raw/itsm_sla_tickets_dataset_extended.csv")

    # Step 2: Null handling
    df_clean = run_null_handling(df)

    # Step 3: Time series feature engineering
    df_time_features = run_time_series_processing(df_clean)

    # Step 4: Encoding
    df_encoded = run_feature_encoding(df_time_features)
    
    df_encoded.to_csv("data/processed/encoded_data.csv", index=False)
    
    df_encoded = pd.read_csv("data/processed/encoded_data.csv")
    

    # Step 5: Leak removal + SMOTE
    X_final, y_final = run_leak_removal_and_smote(df_encoded)

    print("\n Feature Pipeline Completed Successfully!")
    return X_final, y_final

































if __name__ == "__main__":
    try:
        X, y = full_feature_pipeline()
    except Exception as e:
        print(f"Feature pipeline failed: {e}")
