"""
Entry point for the Student Exam Pass Prediction project.

Run with:
    python main.py

Uses config/config.yaml for all paths and hyperparameters.
"""

from src.pipeline import run_pipeline
from src.models.train_models import predict_new_student
from src.utils.config import load_config


def main():
    config = load_config()
    output = run_pipeline(config)

    print("Model comparison (sorted by accuracy):")
    print(output["results"].round(4))

    best_model_name = output["best_model_name"]
    best_model = output["best_model"]
    print(f"\nBest model: {best_model_name}")

    example = config["example_prediction"]
    prediction = predict_new_student(
        best_model,
        study_hours=example["study_hours"],
        previous_exam_score=example["previous_exam_score"],
    )
    print(
        f"Example prediction ({example['study_hours']} study hours, "
        f"{example['previous_exam_score']} previous score): "
        f"{'Pass' if prediction == 1 else 'Fail'}"
    )


if __name__ == "__main__":
    main()
