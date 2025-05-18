from pandas.core.dtypes.api import is_numeric_dtype, is_string_dtype
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

def get_distinct_colors(n, cmap_name="tab20"):
    cmap = cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]


def get_columns_type(data):
    num_cols = [c for c in data.columns if is_numeric_dtype(data[c])]
    cat_cols = [c for c in data.columns if is_string_dtype(data[c])]
    return num_cols, cat_cols


def delete_duplicated_data(data):
    print("Duplicate Records (before):", data[data.duplicated()].shape[0])
    data.drop_duplicates(inplace=True)
    print("Duplicate Records (after):", data[data.duplicated()].shape[0])


def plot_numeric_distributions(data, num_cols):
    if num_cols:

        cols_length = len(num_cols)

        fig, ax = plt.subplots(2, cols_length, figsize=(6 * cols_length, 10))

        if cols_length == 1:
            ax = np.expand_dims(ax, axis=1)

        colors = get_distinct_colors(cols_length)

        for i, c in enumerate(num_cols):
            color = colors[i]

            # Histogram
            sns.histplot(data[c], kde=True, ax=ax[0, i], color=color)

            # Si es entera y tiene pocos valores únicos, usar xticks enteros
            # if pd.api.types.is_integer_dtype(data[c]) and data[c].nunique() <= 20:
            #     min_val = int(data[c].min())
            #     max_val = int(data[c].max())
            #     ax[0, i].set_xticks(range(min_val, max_val + 1, 2))

            ax[0, i].set_title(c)

            # Boxplot
            sns.boxplot(x=data[c], ax=ax[1, i], color=color)
            ax[1, i].set_title(c)
        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(data, cat_cols):
    if cat_cols:
        fig, ax = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 5))

        if len(cat_cols) == 1:
            ax = [ax]

        colors = get_distinct_colors(len(cat_cols))

        for i, c in enumerate(cat_cols):
            sns.countplot(x=data[c], ax=ax[i], color=colors[i])
            ax[i].set_title(c)
            ax[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


def plot_pairplots_by_category(data, cat_cols, height=2, aspect=1.5):
    for cat in cat_cols:
        print(f"Generating pairplot with hue = '{cat}'")
        sns.pairplot(data, hue=cat, height=height, aspect=aspect)
        


def plot_distribution(data, title="Predicciones", xlabel="Valor predicho"):

    fig, axs = plt.subplots(nrows=2, figsize=(8, 7))

    sns.histplot(data, kde=True, bins=30, edgecolor="black", ax=axs[0])
    axs[0].set(
        title=f"Histograma de {title.lower()}",
        xlabel=xlabel,
        ylabel="Frecuencia",
    )

    sns.boxplot(x=data, ax=axs[1])
    axs[1].set(title=f"Boxplot de {title.lower()}")

    plt.tight_layout()
    plt.show()

def model_evaluation(model, x, y, model_name):
    y_pred = model.predict(x)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    
    print(f"\n{model_name}:")
    print(
        "-------------\nRMSE = {:.4f}\nR2 = {:.4f}\nMAE = {:.4f}\nMAPE = {:.4f}".format(
            rmse, r2, mae, mape
        )
    )
    
    return y_pred, {"RMSE": rmse, "R2": r2, "MAE": mae, "MAPE": mape}
