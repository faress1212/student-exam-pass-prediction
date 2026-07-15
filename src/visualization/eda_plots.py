"""
Exploratory Data Analysis utilities for the Student Exam Pass Prediction
project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_target_distribution(y: pd.Series, save_path: str = None) -> None:
    """
    Plot a histogram of the target variable's distribution.

    Args:
        y (pd.Series): Target column (e.g. Pass/Fail).
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure()
    plt.hist(y, bins=20)
    plt.title("Distribution of Target Variable")
    plt.xlabel("Pass/Fail")
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a correlation heatmap for all numeric columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        save_path (str, optional): If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame, numerical_cols: list, save_path: str = None) -> None:
    """
    Plot histograms for the given numeric columns, each with a mean line,
    arranged in a grid.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (list): Numeric column names to plot.
        save_path (str, optional): If provided, save the figure to this path.
    """
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    axvline_kwargs = {"color": "red", "linestyle": "--", "linewidth": 2}

    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[col], bins=50, color="skyblue", edgecolor="black")
        plt.axvline(df[col].mean(), **axvline_kwargs, label=f"Mean: {df[col].mean():.1f}")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
