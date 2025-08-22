# Stress Level Analysis

This project analyzes and predicts stress levels using a dataset of various personal and environmental factors. It includes a detailed data analysis in a Jupyter Notebook and an interactive web-based dashboard for visualizing the data and making predictions.

## Dataset

The project uses the `StressLevelDataset.csv` file, which contains various features related to stress levels of individuals. The dataset includes information such as anxiety level, self-esteem, mental health history, depression, and other lifestyle factors.

## Features

*   **In-depth Data Analysis:** The `Analysisi.ipynb` notebook provides a comprehensive analysis of the dataset, including data cleaning, exploratory data analysis (EDA), feature engineering, and model training.
*   **Interactive Dashboard:** The `dashboard.py` file launches a Streamlit web application that allows users to visualize the data, explore feature correlations, and even upload their own data to get stress level predictions.
*   **Machine Learning Model:** The project uses a LightGBM model to predict stress levels, and the trained model is saved in `stress_lgbm.pkl`.
*   **SHAP integration:** The dashboard includes SHAP (SHapley Additive exPlanations) waterfall plots to explain individual predictions, providing insights into why the model makes a certain decision.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd Stresslevel_analysis
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Analysis

To explore the data analysis process, open and run the `Analysisi.ipynb` notebook using Jupyter:

```bash
jupyter notebook Analysisi.ipynb
```

### Interactive Dashboard

To launch the interactive dashboard, run the following command in your terminal:

```bash
streamlit run dashboard.py
```

This will open a new tab in your web browser with the dashboard.

## File Descriptions

*   `Analysisi.ipynb`: Jupyter Notebook containing the data analysis, visualization, and model training.
*   `dashboard.py`: Streamlit application for the interactive dashboard.
*   `StressLevelDataset.csv`: The dataset used for this project.
*   `stress_lgbm.pkl`: The trained LightGBM model.
*   `requirements.txt`: A list of the Python libraries required to run the project.
*   `README.md`: This file.

## Libraries Used

*   **Data Manipulation and Analysis:** pandas, numpy
*   **Data Visualization:** matplotlib, seaborn, plotly
*   **Machine Learning:** scikit-learn, xgboost, lightgbm, optuna, shap
*   **Web Dashboard:** streamlit, streamlit-extras, streamlit-toggle
*   **Model Persistence:** joblib
