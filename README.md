# Stock Market Trend Predictor

This project is an end-to-end machine learning solution designed to predict short-term stock market trends. Instead of forecasting exact prices, it uses a **Random Forest Classifier** to predict whether a stock's closing price will move **UP** or **DOWN** on the following trading day.

## Project Structure

*   **`dashboard/`**: A professional Streamlit-based web application for interactive analysis and real-time predictions.
    *   `app.py`: Main UI layout and dashboard logic.
    *   `utils.py`: Backend engine for data processing and model inference.
*   **`data/`**: Stores raw historical stock data (CSV format) fetched via `yfinance`.
*   **`notebooks/`**: Contains Jupyter Notebooks used for exploration, analysis, and model training.
    *   `01_EDA_and_Modeling.ipynb`: The main notebook containing the full pipeline with academic documentation.
*   **`models/`**: Intended for saving serialized versions of trained machine learning models.
*   **`images/`**: A directory to store generated plots and visualizations.

## Features

1.  **Dynamic Data Acquisition**: Fetches real-time historical data for any ticker (AAPL, TSLA, RELIANCE.NS, etc.).
2.  **Advanced Feature Engineering**: Calculates technical indicators like SMA (10, 50), EMA (20), RSI (14), and Volatility.
3.  **Machine Learning Classification**: Uses a Random Forest Classifier to identify directional trends.
4.  **Interactive Visualizations**: High-quality Candlestick charts, RSI plots, and Volume analysis using Plotly.
5.  **Model Explainability**: Includes **Feature Importance** analysis to show which indicators drive the model's decisions.
6.  **Professional Dashboard**: A clean, "Bloomberg-Lite" interface for end-user interaction.

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
    pip install pandas numpy matplotlib scikit-learn yfinance notebook seaborn plotly streamlit
    ```

## Usage

### 1. Interactive Dashboard (Streamlit)
The most professional way to interact with the project:
1. Activate your virtual environment.
2. Run the application:
   ```bash
   streamlit run dashboard/app.py
   ```

### 2. Jupyter Notebook
For research and detailed analysis:
1. Activate your virtual environment.
2. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
3. Open `notebooks/01_EDA_and_Modeling.ipynb`.

## Disclaimer
This project is for **educational purposes only** (Semester 2 College Project) and is not financial advice.
