import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error
)
import time

def tune_and_log_model(model_name, model_obj, train_df, val_df, param_grid, n_iter=10):
    """
    Performs hyperparameter tuning using RandomizedSearchCV and logs results to MLflow.
    """
    X_train = train_df.drop(columns=['price_log'])
    y_train = train_df['price_log']
    X_val = val_df.drop(columns=['price_log'])
    y_val = val_df['price_log']

    # Log the Dataset Context
    dataset = mlflow.data.from_pandas(train_df, name="NYC_Airbnb_Train")

    # Start Parent Run for this model
    with mlflow.start_run(run_name=f"{model_name}_Tuning"):
        mlflow.log_input(dataset, context="training")
        
        # Log hyperparameter search configuration
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("model_type", model_name)
        
        print(f"\n{'='*60}")
        print(f"Starting RandomizedSearchCV for {model_name}")
        print(f"Testing {n_iter} random parameter combinations...")
        
        start_time = time.time()
        
        # Configure RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model_obj,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=1,
            return_train_score=True
        )
        
        # Perform hyperparameter search
        search.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Get best model
        best_model = search.best_estimator_
        
        # Make predictions on validation set
        val_preds = best_model.predict(X_val)
        train_preds = best_model.predict(X_train)
        
        # Calculate metrics on validation set
        val_metrics = {
            "val_RMSE": np.sqrt(mean_squared_error(y_val, val_preds)),
            "val_MAE": mean_absolute_error(y_val, val_preds),
            "val_R2": r2_score(y_val, val_preds),
            "val_MAPE": mean_absolute_percentage_error(y_val, val_preds),
            "val_MedAE": median_absolute_error(y_val, val_preds)
        }
        
        # Calculate metrics on training set (to check overfitting)
        train_metrics = {
            "train_RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
            "train_MAE": mean_absolute_error(y_train, train_preds),
            "train_R2": r2_score(y_train, train_preds),
            "train_MAPE": mean_absolute_percentage_error(y_train, train_preds)
        }
        
        # Log all metrics
        mlflow.log_metrics({**val_metrics, **train_metrics})
        
        # Log best parameters
        mlflow.log_params(search.best_params_)
        
        # Log cross-validation score
        mlflow.log_metric("cv_best_score", -search.best_score_)  # Negative because scoring is neg_mse
        
        # Calculate and log overfitting metric
        overfitting_gap = train_metrics['train_R2'] - val_metrics['val_R2']
        mlflow.log_metric("overfitting_gap_R2", overfitting_gap)
        
        # Log feature importances if available
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save as artifact
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            # Log top 5 features as params
            for idx, row in feature_importance.head(5).iterrows():
                mlflow.log_param(f"top_feature_{idx+1}", row['feature'])
        
        # Register the best model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=f"NYC_Airbnb_{model_name}"
        )
        
        # Log all CV results as a table artifact
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv("cv_results.csv", index=False)
        mlflow.log_artifact("cv_results.csv")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"{model_name} Tuning Complete!")

        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Best Cross-Validation RMSE: {-search.best_score_:.4f}")
        print(f"\nBest Parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {val_metrics['val_RMSE']:.4f}")
        print(f"  MAE:  {val_metrics['val_MAE']:.4f}")
        print(f"  R²:   {val_metrics['val_R2']:.4f}")
        print(f"  MAPE: {val_metrics['val_MAPE']:.4f}")
        print(f"\nTrain vs Val R² (Overfitting Check):")
        print(f"  Train R²: {train_metrics['train_R2']:.4f}")
        print(f"  Val R²:   {val_metrics['val_R2']:.4f}")
        print(f"  Gap:      {overfitting_gap:.4f}")

        
        return best_model


def evaluate_on_test(model, test_df, model_name):
    """
    Evaluate the final model on test set and log results.
    """
    X_test = test_df.drop(columns=['price_log'])
    y_test = test_df['price_log']
    
    with mlflow.start_run(run_name=f"{model_name}_Test_Evaluation"):
        # Make predictions
        test_preds = model.predict(X_test)
        
        # Calculate test metrics
        test_metrics = {
            "test_RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
            "test_MAE": mean_absolute_error(y_test, test_preds),
            "test_R2": r2_score(y_test, test_preds),
            "test_MAPE": mean_absolute_percentage_error(y_test, test_preds),
            "test_MedAE": median_absolute_error(y_test, test_preds)
        }
        
        mlflow.log_metrics(test_metrics)
        mlflow.log_param("model_name", model_name)
        
        print(f"\n{model_name} Test Set Results:")
        print(f"  RMSE: {test_metrics['test_RMSE']:.4f}")
        print(f"  MAE:  {test_metrics['test_MAE']:.4f}")
        print(f"  R²:   {test_metrics['test_R2']:.4f}")
        print(f"  MAPE: {test_metrics['test_MAPE']:.4f}\n")