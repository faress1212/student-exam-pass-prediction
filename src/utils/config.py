"""
Small utility for loading the project's YAML config file.
"""

import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load the project YAML config.

    Args:
        path (str): Path to config.yaml. Defaults to config/config.yaml
                    relative to the project root.

    Returns:
        dict: Parsed configuration.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
