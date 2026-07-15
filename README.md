# Student Exam Pass Prediction 🎓

A classification project predicting whether a student passes or fails an exam, based on study hours and a previous exam score, comparing multiple classification models.

## Project Overview

- Data cleaning (duplicates, missing values)
- IQR-based outlier treatment — applied only to feature columns, never the target
- Exploratory Data Analysis (target distribution, correlation heatmap, feature distributions)
- **Comparing 6 classification models** to find the best fit
- A worked example: predicting pass/fail for a new student

## Project Structure

```
student-exam-pass-prediction/
├── config/
│   └── config.yaml            # Paths, feature columns, split settings
├── data/
│   ├── raw/
│   │   └── student_exam_data.csv   # Included — small, self-contained dataset
│   └── processed/               # Cleaned data saved here after running the pipeline
├── notebooks/
│   └── student_exam_pass_prediction.ipynb   # Full walkthrough notebook
├── src/
│   ├── data/
│   │   └── load_data.py        # Data loading
│   ├── features/
│   │   └── preprocessing.py    # Cleaning, encoding, outlier treatment
│   ├── models/
│   │   └── train_models.py     # Model training & evaluation (multi-model comparison)
│   ├── visualization/
│   │   └── eda_plots.py        # Visualization / EDA functions
│   ├── utils/
│   │   └── config.py           # Config loader
│   └── pipeline.py             # Chains all steps together
├── models/                     # Saved best model (.pkl)
├── reports/
│   └── figures/                # Saved plots
├── main.py                     # Runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Dataset

A small (500-row), self-contained dataset with 3 columns:

| Column               | Description                          |
|----------------------|----------------------------------------|
| Study Hours          | Hours the student studied              |
| Previous Exam Score  | Score on a previous exam               |
| Pass/Fail            | Target: 1 = Pass, 0 = Fail              |

## A Note on Outlier Treatment

The target column (`Pass/Fail`) must be **excluded** from IQR-based outlier clipping. It's a binary class label, not a continuous measurement — clipping it doesn't make statistical sense and risks corrupting the labels themselves. This project explicitly excludes the target from the outlier treatment step.

## Models Compared

| Model               | Notes                                    |
|----------------------|--------------------------------------------|
| Logistic Regression | Linear baseline (from the original analysis) |
| Naive Bayes          | Probabilistic classifier (from the original analysis) |
| Decision Tree        | Non-linear, interpretable classifier       |
| Random Forest        | Ensemble of decision trees                 |
| KNN                   | Instance-based classifier                  |
| SVM                   | Support Vector Machine classifier          |

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/faress1212/student-exam-pass-prediction.git
cd student-exam-pass-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

Or explore the full analysis interactively:
```bash
jupyter notebook notebooks/student_exam_pass_prediction.ipynb
```

## Results

Tree-based models (Random Forest, Decision Tree) tend to achieve the highest accuracy on this dataset, suggesting the pass/fail boundary isn't purely linear. Comparing multiple models confirms this isn't an artifact of any single algorithm's assumptions.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## License

This project is open source and available under the [MIT License](LICENSE).
