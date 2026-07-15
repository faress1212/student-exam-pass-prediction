"""
Data loading utilities for the Student Exam Pass Prediction project.
"""

import os

import pandas as pd

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "student_exam_data.csv")


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the student exam dataset from a CSV file.

    Args:
        path (str): Path to student_exam_data.csv. Defaults to
                    data/raw/student_exam_data.csv relative to the project root.

    Returns:
        pd.DataFrame: Raw student exam dataframe.
    """
    return pd.read_csv(path)
