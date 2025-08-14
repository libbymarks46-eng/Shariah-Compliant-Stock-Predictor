import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from models.predictor import StockPredictor
from utils.shariah_stocks import get_shariah_compliant_stocks
from utils.data_fetcher import fetch_stock_data, get_technical_indicators

# Page configuration
st.set_page_config(
    page_title="Shariah Compliant Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 Shariah Compliant Stock Price Predictor")
st.markdown("""
This application provides stock price predictions for Shariah compliant securities using machine learning models.
Select a stock from our curated list of Shariah compliant companies to view predictions and analysis.
""")

# Disclaimer
with st.expander("⚠️ Important Disclaimer"):
    st.warning("""
    **Investment Risk Warning**: This application is for educational and informational purposes only. 
    Stock price predictions are based on historical data and statistical models, which may not accurately 
    predict future performance. Past performance does not guarantee future results. Always consult with 
    qualified financial advisors before making investment decisions. Trading in stocks involves substantial 
    risk of loss and is not suitable for all investors.
    """)

# Sidebar for stock selection and parameters
st.sidebar.header("Stock Selection & Parameters")

# Load Shariah compliant stocks
shariah_stocks = get_shariah_compliant_stocks()
stock_symbols = list(shariah_stocks.keys())
stock_names = [f"{symbol} - {info['name']}" for symbol, info in shariah_stocks.items()]

# Stock selection
selected_stock_display = st.sidebar.selectbox(
    "Select Shariah Compliant Stock:",
    stock_names,
    index=0
)
selected_symbol = selected_stock_display.split(" - ")[0]

# Time period selection
time_period = st.sidebar.selectbox(
    "Historical Data Period:",
    ["1y", "2y", "5y", "max"],
    index=1
)

# Prediction parameters
st.sidebar.subheader("Prediction Parameters")
prediction_days = st.sidebar.slider("Prediction Days:", 1, 30, 7)
model_type = st.sidebar.selectbox(
    "Prediction Model:",
    ["Linear Regression", "Moving Average", "Both"],
    index=2
)

# Confidence interval
confidence_level = st.sidebar.slider("Confidence Level (%):", 80, 99, 95)

# Fetch and display data
if st.sidebar.button("Generate Prediction", type="primary"):
    with st.spinner(f"Fetching data for {selected_symbol}..."):
        try:
            # Fetch stock data
            stock_data = fetch_stock_data(selected_symbol, time_period)
            
            if stock_data is None or stock_data.empty:
                st.error(f"Unable to fetch data for {selected_symbol}. Please try another stock.")
                st.stop()
            
            # Display stock information
            stock_info = shariah_stocks[selected_symbol]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Company", stock_info['name'])
            with col2:
                st.metric("Sector", stock_info['sector'])
            with col3:
                current_price = stock_data['Close'].iloc[-1]
                prev_price = stock_data['Close'].iloc[-2]
                price_change = current_price - prev_price
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
            with col4:
                market_cap = stock_info.get('market_cap', 'N/A')
                st.metric("Market Cap", market_cap)
            
            # Create predictor instance
            predictor = StockPredictor(stock_data)
            
            # Generate predictions based on selected model
            predictions = {}
            
            if model_type in ["Linear Regression", "Both"]:
                lr_pred, lr_confidence = predictor.linear_regression_prediction(
                    prediction_days, confidence_level
                )
                predictions['Linear Regression'] = {
                    'predictions': lr_pred,
                    'confidence': lr_confidence
                }
            
            if model_type in ["Moving Average", "Both"]:
                ma_pred, ma_confidence = predictor.moving_average_prediction(
                    prediction_days, confidence_level
                )
                predictions['Moving Average'] = {
                    'predictions': ma_pred,
                    'confidence': ma_confidence
                }
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Historical Prices & Predictions', 'Technical Indicators', 
                              'Volume Analysis', 'Prediction Metrics'),
                specs=[[{"secondary_y": True, "colspan": 2}, None],
                       [{"secondary_y": True}, {"type": "table"}]],
                vertical_spacing=0.12
            )
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Historical Close Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Add predictions
            future_dates = pd.date_range(
                start=stock_data.index[-1] + timedelta(days=1),
                periods=prediction_days,
                freq='D'
            )
            
            colors = ['red', 'green', 'orange']
            color_idx = 0
            
            for model_name, pred_data in predictions.items():
                pred_prices = pred_data['predictions']
                confidence_intervals = pred_data['confidence']
                
                # Add prediction line
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=pred_prices,
                        mode='lines+markers',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[color_idx], width=2, dash='dash'),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                # Add confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=list(future_dates) + list(future_dates[::-1]),
                        y=list(confidence_intervals['upper']) + list(confidence_intervals['lower'][::-1]),
                        fill='toself',
                        fillcolor=f'rgba({255 if color_idx==0 else 0}, {255 if color_idx==1 else 0}, {255 if color_idx==2 else 0}, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model_name} {confidence_level}% CI',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                color_idx += 1
            
            # Technical indicators
            tech_indicators = get_technical_indicators(stock_data)
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index[-50:],
                    y=tech_indicators['sma_20'][-50:],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index[-50:],
                    y=tech_indicators['sma_50'][-50:],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index[-50:],
                    y=stock_data['Close'][-50:],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # Volume analysis
            fig.add_trace(
                go.Bar(
                    x=stock_data.index[-30:],
                    y=stock_data['Volume'][-30:],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1, secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title=f"{stock_info['name']} ({selected_symbol}) - Stock Analysis & Prediction",
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Price ($)", row=2, col=1)
            fig.update_yaxes(title_text="Volume", secondary_y=True, row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction summary
            st.subheader("📊 Prediction Summary")
            
            for model_name, pred_data in predictions.items():
                with st.expander(f"{model_name} Results"):
                    pred_prices = pred_data['predictions']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Predicted Price (Day 1)",
                            f"${pred_prices[0]:.2f}",
                            f"{pred_prices[0] - current_price:+.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            f"Predicted Price (Day {prediction_days})",
                            f"${pred_prices[-1]:.2f}",
                            f"{pred_prices[-1] - current_price:+.2f}"
                        )
                    
                    with col3:
                        total_return = ((pred_prices[-1] - current_price) / current_price) * 100
                        st.metric(
                            f"Expected Return ({prediction_days}d)",
                            f"{total_return:+.2f}%"
                        )
                    
                    with col4:
                        volatility = np.std(pred_prices)
                        st.metric(
                            "Prediction Volatility",
                            f"${volatility:.2f}"
                        )
                    
                    # Show prediction accuracy metrics if available
                    accuracy_metrics = predictor.get_model_accuracy(model_name.lower().replace(' ', '_'))
                    if accuracy_metrics:
                        st.write("**Model Accuracy Metrics:**")
                        acc_col1, acc_col2, acc_col3 = st.columns(3)
                        
                        with acc_col1:
                            st.metric("RMSE", f"{accuracy_metrics['rmse']:.2f}")
                        with acc_col2:
                            st.metric("MAE", f"{accuracy_metrics['mae']:.2f}")
                        with acc_col3:
                            st.metric("R² Score", f"{accuracy_metrics['r2']:.3f}")
            
            # Technical Analysis Summary
            st.subheader("🔍 Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Technical Indicators:**")
                current_sma_20 = tech_indicators['sma_20'][-1]
                current_sma_50 = tech_indicators['sma_50'][-1]
                current_rsi = tech_indicators['rsi'][-1]
                
                st.write(f"• 20-day SMA: ${current_sma_20:.2f}")
                st.write(f"• 50-day SMA: ${current_sma_50:.2f}")
                st.write(f"• RSI: {current_rsi:.1f}")
                
                # Simple trend analysis
                if current_price > current_sma_20 > current_sma_50:
                    st.success("📈 **Bullish Trend**: Price above both moving averages")
                elif current_price < current_sma_20 < current_sma_50:
                    st.error("📉 **Bearish Trend**: Price below both moving averages")
                else:
                    st.warning("📊 **Mixed Signals**: No clear trend direction")
            
            with col2:
                st.write("**Risk Assessment:**")
                
                # Calculate volatility
                returns = stock_data['Close'].pct_change().dropna()
                volatility_30d = returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100
                
                if volatility_30d < 20:
                    risk_level = "Low"
                    risk_color = "green"
                elif volatility_30d < 40:
                    risk_level = "Moderate"
                    risk_color = "orange"
                else:
                    risk_level = "High"
                    risk_color = "red"
                
                st.markdown(f"• **Volatility (30-day)**: {volatility_30d:.1f}%")
                st.markdown(f"• **Risk Level**: :{risk_color}[{risk_level}]")
                
                # Volume analysis
                avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
                recent_volume = stock_data['Volume'].iloc[-1]
                volume_ratio = recent_volume / avg_volume
                
                if volume_ratio > 1.5:
                    st.info("📊 **High Volume**: Above average trading activity")
                elif volume_ratio < 0.5:
                    st.info("📊 **Low Volume**: Below average trading activity")
                else:
                    st.info("📊 **Normal Volume**: Average trading activity")
            
        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            st.error("Please try selecting a different stock or check your internet connection.")

else:
    st.info("👈 Please select a Shariah compliant stock from the sidebar and click 'Generate Prediction' to begin analysis.")
    
    # Show available stocks
    st.subheader("🏢 Available Shariah Compliant Stocks")
    
    # Create a DataFrame for better display
    stocks_df = pd.DataFrame([
        {
            'Symbol': symbol,
            'Company': info['name'],
            'Sector': info['sector'],
            'Market Cap': info.get('market_cap', 'N/A')
        }
        for symbol, info in shariah_stocks.items()
    ])
    
    st.dataframe(stocks_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    ### About Shariah Compliant Investing
    
    This app uses stocks screened according to AAOIFI (Accounting and Auditing Organization for Islamic Financial Institutions) standards:
    
    **Prohibited Activities (Haram):**
    - ❌ Alcohol, tobacco, gambling, pork products
    - ❌ Interest-based banking and conventional insurance  
    - ❌ Adult entertainment and weapons manufacturing
    - ❌ Companies supporting unethical practices
    
    **Financial Requirements:**
    - Debt-to-equity ratio ≤ 30% of market cap
    - Interest income ≤ 5% of total revenue
    - Cash in interest-bearing accounts ≤ 30% of market cap
    
    **✅ Included Sectors:** Technology, healthcare, clean energy, manufacturing, halal consumer goods
    
    **Note:** Always verify current compliance status using certified platforms like Zoya, Islamicly, or Musaffa before investing.
    """)
