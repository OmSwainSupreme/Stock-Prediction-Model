import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def fetch_data(ticker, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def add_technical_indicators(df):
    """Calculate technical indicators for feature engineering."""
    # Moving Averages
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Returns and Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Drop rows with NaN values created by rolling windows
    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    """Train the model on historical data and predict the next day trend."""
    # Define features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI', 'Daily_Return', 'Volatility']
    
    # Target: 1 if next day close is higher than current close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Use data up to the second to last row for training/testing
    # The last row's target is NaN because we don't know "tomorrow" yet
    model_df = df.dropna().copy()
    
    X = model_df[features]
    y = model_df['Target']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train Random Forest
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Get latest data for prediction (the very last row in our original df)
    latest_features = df[features].iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    confidence = model.predict_proba(latest_features)[0]
    
    # Calculate accuracy on test set
    accuracy = model.score(X_test, y_test)
    
    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    return {
        'model': model,
        'prediction': prediction,
        'confidence': confidence,
        'accuracy': accuracy,
        'feature_importance': importance,
        'test_size': len(X_test)
    }

def get_insights(df):
    """Generate simple text insights based on current data."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    insights = []
    
    # Trend insight
    if latest['Close'] > latest['MA50']:
        insights.append("Stock is currently trading above its 50-day Moving Average, indicating a potential long-term bullish trend.")
    else:
        insights.append("Stock is trading below its 50-day Moving Average, suggesting long-term bearish momentum.")
        
    # RSI insight
    if latest['RSI'] > 70:
        insights.append("RSI is above 70, suggesting the stock might be overbought.")
    elif latest['RSI'] < 30:
        insights.append("RSI is below 30, suggesting the stock might be oversold.")
        
    # Volatility insight
    avg_vol = df['Volatility'].mean()
    if latest['Volatility'] > avg_vol * 1.2:
        insights.append("Recent volatility is higher than average, indicating increased market uncertainty.")
        
    return insights
