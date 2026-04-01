import mlflow
from src.ingest import fetch_s3_data
from src.preprocess import split_data, preprocess_pipeline
from src.train import tune_and_log_model  # Updated import
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Pipeline Started--")


def main():
    # --------- DATA INGESTION
    BUCKET = "staywise-data-bucket/airbnb/raw_data"
    FILE_KEY = "AB_NYC_2019.csv"
    
    print("Starting Pipeline...")
    df_raw = fetch_s3_data(BUCKET, FILE_KEY)

    # --------- PREPROCESSING
    train_raw, val_raw, test_raw = split_data(df_raw)
    train, val, test = preprocess_pipeline(train_raw, val_raw, test_raw)

    # --------- MLFLOW EXPERIMENT SETUP
    mlflow.set_experiment("NYC_Airbnb_Price_Prediction")

    # --------- PARAMETER GRIDS FOR HYPERPARAMETER TUNING
    
    # 1. Ridge Grid
    ridge_grid = {
        'alpha': [0.01, 0.1, 1.0, 10, 100, 1000]
    }

    # 2. Random Forest Grid
    rf_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }

    # 3. XGBoost Grid
    xgb_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'max_depth': [3, 5, 7, 10],
        'n_estimators': [100, 200, 500, 1000],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'objective': ['reg:squarederror']  # Fixed for all runs
    }

    # --------- MODEL TRAINING WITH HYPERPARAMETER TUNING
    
    print("\n" + "="*60)
    print("MODEL 1: RIDGE REGRESSION (Baseline)")
    best_ridge = tune_and_log_model(
        model_name="Ridge_Model",
        model_obj=Ridge(random_state=42),
        train_df=train,
        val_df=val,
        param_grid=ridge_grid
    )

    print("\n" + "="*60)
    print("MODEL 2: RANDOM FOREST")
    best_rf = tune_and_log_model(
        model_name="Random_Forest_Model",
        model_obj=RandomForestRegressor(random_state=42, n_jobs=-1),
        train_df=train,
        val_df=val,
        param_grid=rf_grid
    )

    print("\n" + "="*60)
    print("MODEL 3: XGBOOST")
    best_xgb = tune_and_log_model(
        model_name="XGBoost_Model",
        model_obj=XGBRegressor(random_state=42),
        train_df=train,
        val_df=val,
        param_grid=xgb_grid
    )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
  
    print("\nRun 'mlflow ui' to view results in the MLflow UI")

if __name__ == "__main__":
    main()