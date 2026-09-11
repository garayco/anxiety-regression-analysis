# ML Assignment 1: Anxiety Level Predictor

This project develops a machine learning pipeline to estimate anxiety levels from variables such as stress, sleep, physical activity, caffeine intake, therapy sessions, and other lifestyle and health-related indicators.

The work includes exploratory data analysis (EDA), preprocessing, feature engineering, model comparison, and a small Streamlit web app for real-time prediction.

## Objective

Build a predictive model that estimates a person's anxiety level based on their personal characteristics and daily habits, with emphasis on:

- linear regression as a baseline model
- model evaluation using quantitative metrics
- exploratory analysis of the dataset
- simple deployment for inference via a web interface

## Dataset

The dataset used comes from Kaggle and corresponds to a social/anxiety-related dataset with demographic and lifestyle features.

Original source:
- `natezhang123/social-anxiety-dataset`

Relevant variables in the dataset:
- `Stress Level (1-10)`
- `Sleep Hours`
- `Caffeine Intake (mg/day)`
- `Physical Activity (hrs/week)`
- `Therapy Sessions (per month)`
- `Heart Rate (bpm)`
- `Breathing Rate (breaths/min)`
- `Anxiety Level (1-10)`
- additional categorical variables such as gender, smoking, family history, medication, dizziness, occupation, etc.

## Project Structure

```text
ML_Assignment_1/
├── app.py                      # Streamlit app for inference
├── dataset_download.py         # Kaggle dataset downloader
├── EDA_utils.py                # Utility functions for visualization and model evaluation
├── ML_EDA_Linear_regression.ipynb
│                              # Main notebook with EDA and model training
├── pyproject.toml             # Project dependencies
├── README.md                  # Project documentation
├── dataset/                   # Downloaded dataset folder
├── outputs/                   # Saved models and generated charts
└── .venv/                     # Local virtual environment (if created)
```

## Technologies Used

- Python 3.12
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit

## Requirements

Before running the project, ensure you have:

- Python 3.12
- Poetry
- Kaggle API credentials configured in `~/.kaggle/kaggle.json` if you want to download the dataset automatically

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ML_Assignment_1
```

2. Install dependencies with Poetry:

```bash
poetry install
```

3. If you prefer a manual virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install matplotlib numpy seaborn scipy scikit-learn jupyter plotly streamlit
```

## Dataset Download

The script `dataset_download.py` downloads the dataset directly from Kaggle.

```bash
python - <<'PY'
from dataset_download import dataset_download

dataset_download("natezhang123/social-anxiety-dataset")
PY
```

> Note: Kaggle authentication must be configured correctly on your machine for this step to work.

## Running the Notebook

Open and execute the notebook:

```bash
jupyter notebook ML_EDA_Linear_regression.ipynb
```

Inside it, the project performs:
- dataset loading and validation
- duplicate and null-value handling
- univariate and categorical exploratory analysis
- feature engineering and scaling of the target variable
- model training and comparison
- saving the final trained model

## Running the App

To launch the prediction interface:

```bash
poetry run streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The app allows the user to modify values such as:
- stress level
- sleep hours
- daily caffeine intake
- weekly physical activity
- therapy sessions per month

and returns a prediction for the anxiety level.

## Model

The trained model is saved in:

```text
outputs/models/ols_anxiety_model.pkl
```

This file is loaded by `app.py` to perform inference from user-entered values.

## Evaluation Metrics

The analysis uses the following metrics:

- RMSE
- MAE
- R²
- MAPE

These metrics help assess how well the model explains the variability in anxiety levels and how close the predictions are to real values.

## Model Comparison Summary

The notebook compares several models, including:

- OLS Linear Regression
- Lasso
- Ridge
- SVR
- KNN

The results show that linear models perform acceptably and remain interpretable, but nonlinear models such as scaled SVR and KNN achieve better predictive performance.

The best-performing models were the scaled versions of:
- SVR (RBF)
- KNN

These models reached the best values in RMSE, MAE, and R², highlighting the importance of feature scaling for algorithms sensitive to input magnitude.

## Notes

- This project is an academic machine learning exercise focused on regression and EDA.
- The current model is used as a practical baseline for understanding the problem and deploying a first functional solution.
- Future improvements may include feature selection refinement, cross-validation, or more advanced nonlinear models.

## Author

- Felipe Garaycochea Lozada

## License

This project is distributed for academic and learning purposes. If you reuse the content, please credit the original dataset and project source.

