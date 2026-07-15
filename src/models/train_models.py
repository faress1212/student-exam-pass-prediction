"""
Model training and evaluation utilities for the Student Exam Pass
Prediction project.

Compares multiple classifiers to see which fits this dataset best, rather
than committing to a single model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> dict:
    """
    Build the dictionary of candidate classification models, including
    both models from the original notebook (Logistic Regression, Naive
    Bayes) plus additional common classifiers for comparison.

    Args:
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Mapping of model name -> unfitted scikit-learn estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
    }


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split features/target into stratified train/test sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Fit each model and compute accuracy, precision, recall, and F1 score
    on the test set.

    Args:
        models (dict): Mapping of model name -> unfitted estimator.
        X_train, X_test, y_train, y_test: Train/test split data.

    Returns:
        pd.DataFrame: Rows = models, columns = [accuracy, precision, recall, f1],
                      sorted by accuracy descending.
    """
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

    return pd.DataFrame(results).T.sort_values("accuracy", ascending=False)


def predict_new_student(model, study_hours: float, previous_exam_score: float) -> int:
    """
    Predict pass/fail for a single new student.

    Args:
        model: A fitted scikit-learn classifier expecting columns
               ['Study Hours', 'Previous Exam Score'].
        study_hours (float): Hours studied.
        previous_exam_score (float): Score on a previous exam.

    Returns:
        int: Predicted class (0 = Fail, 1 = Pass).
    """
    new_data = pd.DataFrame({"Study Hours": [study_hours], "Previous Exam Score": [previous_exam_score]})
    return model.predict(new_data)[0]
