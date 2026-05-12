import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

def run_model(ticker='AAPL'):
    # 1. Fetch Data
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start='2015-01-01')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Feature Engineering
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Target: 1 if next day close is higher than current close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI', 'Daily_Return', 'Volatility']
    X = df[features]
    y = df['Target']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Model Training
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluation
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print(f"Precision Score: {precision:.2%}")
    print(f"Recall Score: {recall:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # 6. Prediction for Next Day
    latest_data = X.tail(1)
    prediction = model.predict(latest_data)[0]
    confidence = model.predict_proba(latest_data)[0]
    
    print("\n--- NEXT DAY PREDICTION ---")
    print(f"Predicted Trend: {'UP 📈' if prediction == 1 else 'DOWN 📉'}")
    print(f"Confidence Score: {confidence[prediction]:.2%}")

if __name__ == "__main__":
    run_model()
