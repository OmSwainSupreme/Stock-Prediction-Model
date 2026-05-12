# Stock Prediction Model

This project is an end-to-end machine learning pipeline designed to predict the next day's closing price of a stock using historical data and technical indicators.

## Project Structure

*   **`data/`**: Stores raw historical stock data (CSV format) fetched via `yfinance` to prevent redundant API calls.
*   **`notebooks/`**: Contains Jupyter Notebooks used for exploration, analysis, and model training.
    *   `01_EDA_and_Modeling.ipynb`: The main notebook containing the full pipeline (Data Fetching -> EDA -> Feature Engineering -> Modeling).
*   **`models/`**: Intended for saving serialized versions of trained machine learning models (e.g., using `joblib` or `pickle`) for future use.
*   **`images/`**: A directory to store generated plots and visualizations.
*   **`Prediction Model.py`**: A placeholder script intended for future development of a standalone command-line application.

## Features Implemented in Notebook

1.  **Data Acquisition**: Automatically fetches historical stock data (Open, High, Low, Close, Volume) using the `yfinance` library.
2.  **Exploratory Data Analysis (EDA)**: Includes static (`matplotlib`/`seaborn`) and interactive (`plotly`) visualizations of price history and trading volume.
3.  **Feature Engineering**: Calculates key technical indicators to serve as features for the machine learning model:
    *   Simple Moving Averages (SMA: 20-day, 50-day)
    *   Exponential Moving Average (EMA: 20-day)
    *   Relative Strength Index (RSI: 14-day)
    *   Daily Returns and Rolling Volatility
4.  **Modeling & Evaluation**:
    *   Utilizes a `RandomForestRegressor` from `scikit-learn`.
    *   Employs chronological train-test splitting to respect the time-series nature of the data.
    *   Evaluates performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
    *   Visualizes predicted vs. actual prices.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:OmSwainSupreme/Stock-Prediction-Model.git
    cd Stock-Prediction-Model
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn yfinance notebook seaborn plotly
    ```

## Usage

1.  Activate your virtual environment.
2.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
3.  Open `notebooks/01_EDA_and_Modeling.ipynb`.
4.  Run the cells sequentially to fetch data, engineer features, and train the model. You can change the `ticker` variable in the second code cell to analyze different stocks (e.g., 'MSFT', 'TSLA').
