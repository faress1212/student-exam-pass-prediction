"""
Preprocessing utilities for the Student Exam Pass Prediction project:
cleaning, categorical encoding, and outlier treatment.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe: drop duplicates and fill missing numeric
    values with the column median.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.drop_duplicates().copy()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode any object-dtype (categorical) columns in place.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with categorical columns label-encoded.
    """
    df = df.copy()
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = encoder.fit_transform(df[col])
    return df


def get_column_types(df: pd.DataFrame) -> tuple:
    """
    Split column names into categorical (object dtype) and numeric groups.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        tuple: (categorical_columns list, numeric_columns list)
    """
    categorical = [col for col in df.columns if df[col].dtype == "object"]
    numerical = [col for col in df.columns if df[col].dtype != "object"]
    return categorical, numerical


def treat_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Clip outliers in the given numeric columns to their IQR-based bounds.

    Note: the target column (e.g. 'Pass/Fail') should NOT be passed here —
    it's a binary class label, not a continuous measurement, so IQR-based
    clipping doesn't make sense and would distort the labels themselves.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): Numeric feature column names to treat (target
                         column excluded by the caller).

    Returns:
        pd.DataFrame: Dataframe with outliers clipped in the given columns.
    """
    df = df.copy()

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)

    return df
