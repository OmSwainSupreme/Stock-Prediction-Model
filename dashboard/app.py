import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils import fetch_data, add_technical_indicators, train_and_predict, get_insights

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR SECTION ---
st.sidebar.header("🔍 Search & Settings")

ticker_list = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "NVDA", "GOOGL"]
ticker = st.sidebar.selectbox("Select Stock Ticker", ticker_list)

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5) # Default 5 years

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, end_date),
    max_value=end_date
)

predict_btn = st.sidebar.button("Run Prediction Engine")

st.sidebar.divider()
st.sidebar.markdown("""
### 🎓 Project Info
**Title:** Stock Trend Prediction  
**Level:** Semester 2 Project  
**Model:** Random Forest Classifier  
**Features:** Technical Indicators
""")

st.sidebar.info("This project is for educational purposes only and is not financial advice.")

# --- MAIN DASHBOARD CONTENT ---
st.title("📈 Stock Market Trend Analytics")
st.markdown(f"Exploring trends and predictions for **{ticker}**")

# Load and process data
with st.spinner("Fetching market data..."):
    raw_data = fetch_data(ticker, date_range[0], date_range[1])
    data = add_technical_indicators(raw_data.copy())

if data.empty:
    st.error("Not enough data found for this period. Please try a longer date range.")
else:
    # --- TOP SECTION: KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_diff = current_price - prev_price
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_diff:.2f}")
    
    # Placeholder for model results (will be updated if button clicked)
    with col2:
        st.metric("Predicted Trend", "---")
    with col3:
        st.metric("Model Accuracy", "---")
    with col4:
        st.metric("Confidence", "---")

    st.divider()

    # --- MIDDLE SECTION: CHARTS ---
    tab1, tab2 = st.tabs(["📊 Price & Trends", "📈 Technical Analysis"])

    with tab1:
        # Candlestick with MAs
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Market Data"
        ))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MA10'], line=dict(color='orange', width=1), name="MA 10"))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='blue', width=1), name="MA 50"))
        
        fig_price.update_layout(
            title=f"{ticker} Candlestick Chart with Moving Averages",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume Chart
        fig_vol = px.bar(data, x=data.index, y='Volume', title="Trading Volume Over Time")
        fig_vol.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        # RSI and Volatility
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_rsi = px.line(data, x=data.index, y='RSI', title="Relative Strength Index (RSI)")
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col_t2:
            fig_volat = px.line(data, x=data.index, y='Volatility', title="Rolling Volatility (20-Day)")
            st.plotly_chart(fig_volat, use_container_width=True)

    # --- PREDICTION LOGIC ---
    if predict_btn:
        with st.spinner("Training Model..."):
            results = train_and_predict(data)
            
            # Update KPI cards using streamlit containers (redrawing the metrics)
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            trend = "UP 📈" if results['prediction'] == 1 else "DOWN 📉"
            color = "green" if results['prediction'] == 1 else "red"
            
            with res_col1:
                st.markdown(f"### Next Day Trend: <span style='color:{color}'>{trend}</span>", unsafe_allow_html=True)
                confidence_score = results['confidence'][1] if results['prediction'] == 1 else results['confidence'][0]
                st.write(f"**Confidence:** {confidence_score:.2%}")
                
            with res_col2:
                st.write(f"**Model Accuracy:** {results['accuracy']:.2%}")
                st.write(f"**Trained on:** {len(data) - results['test_size']} samples")
                st.write(f"**Tested on:** {results['test_size']} samples")
                
            with res_col3:
                signal = "BUY" if results['prediction'] == 1 and confidence_score > 0.55 else "SELL" if results['prediction'] == 0 and confidence_score > 0.55 else "NEUTRAL"
                st.markdown(f"### Trade Signal: **{signal}**")

            # Analytical Visuals
            st.divider()
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.write("#### 📊 Feature Importance")
                fig_imp = px.bar(
                    results['feature_importance'], 
                    orientation='h', 
                    labels={'value': 'Importance Score', 'index': 'Feature'},
                    color_discrete_sequence=['teal']
                )
                fig_imp.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
                st.caption("This graph shows which indicators influenced the model's decision the most.")

            with col_a2:
                st.write("#### 💡 Automated Insights")
                insights = get_insights(data)
                for insight in insights:
                    st.write(f"- {insight}")
                
                # Add a prediction insight
                if results['prediction'] == 1:
                    st.success(f"The model identifies bullish patterns with {confidence_score:.1%} confidence.")
                else:
                    st.error(f"The model identifies bearish patterns with {confidence_score:.1%} confidence.")

    # --- FOOTER ---
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: gray; font-size: 14px;">
        Made for College Semester 2 Project | Built with Streamlit, Plotly & Scikit-Learn<br>
        <b>Disclaimer:</b> This application is for educational demonstration purposes only. Financial markets are risky.
    </div>
    """, unsafe_allow_html=True)

    # Recent Data Table
    with st.expander("📄 View Recent Raw Data"):
        st.dataframe(data.tail(10), use_container_width=True)
