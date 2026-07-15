"""
End-to-end pipeline function for the Student Exam Pass Prediction project.
Chains data loading, cleaning, outlier treatment, and multi-model
training/evaluation, driven by config.yaml.
"""

import os

import joblib

from src.data.load_data import load_data
from src.features.preprocessing import (
    clean_data,
    encode_categorical_columns,
    get_column_types,
    treat_outliers,
)
from src.models.train_models import get_models, split_data, train_and_evaluate
from src.utils.config import load_config


def run_pipeline(config: dict = None) -> dict:
    """
    Run the full student exam pass prediction pipeline: load, clean,
    encode, treat outliers (excluding the target), split, train, and
    compare all candidate models.

    Args:
        config (dict, optional): Parsed config. Loads config/config.yaml
                                  if not provided.

    Returns:
        dict: {
            "results": DataFrame of model_name -> [accuracy, precision, recall, f1],
            "best_model_name": str,
            "best_model": fitted estimator,
        }
    """
    config = config or load_config()
    target_col = config["features"]["target_col"]
    feature_cols = config["features"]["columns"]

    # 1. Load & clean
    data = load_data(config["data"]["raw_path"])
    data = clean_data(data)
    data = encode_categorical_columns(data)

    # 2. Outlier treatment — exclude the target column (binary label, not
    #    a continuous measurement)
    _, numerical = get_column_types(data)
    cols_to_treat = [col for col in numerical if col != target_col]
    data = treat_outliers(data, cols_to_treat)

    # 3. Save processed data
    processed_path = config["data"]["processed_path"]
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    data.to_csv(processed_path, index=False)

    # 4. Split
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # 5. Train & compare models
    models = get_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    best_model_name = results.index[0]
    best_model = models[best_model_name]

    # 6. Save the best model
    model_path = config["output"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }
