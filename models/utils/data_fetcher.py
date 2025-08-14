"""
Data fetching utilities for stock market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period="1y"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns or None if failed
    """
    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Fetch historical data
        stock_data = ticker.history(period=period)
        
        if stock_data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in stock_data.columns for col in required_columns):
            st.error(f"Missing required data columns for {symbol}")
            return None
        
        # Remove any rows with missing data
        stock_data = stock_data.dropna()
        
        if len(stock_data) < 30:  # Minimum data requirement
            st.warning(f"Insufficient data for {symbol}. Only {len(stock_data)} days available.")
            return None
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_technical_indicators(stock_data):
    """
    Calculate technical indicators for stock data
    
    Args:
        stock_data (pd.DataFrame): Stock data with OHLCV columns
    
    Returns:
        dict: Dictionary containing technical indicators
    """
    try:
        indicators = {}
        
        # Simple Moving Averages
        indicators['sma_5'] = stock_data['Close'].rolling(window=5).mean()
        indicators['sma_10'] = stock_data['Close'].rolling(window=10).mean()
        indicators['sma_20'] = stock_data['Close'].rolling(window=20).mean()
        indicators['sma_50'] = stock_data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        indicators['ema_12'] = stock_data['Close'].ewm(span=12).mean()
        indicators['ema_26'] = stock_data['Close'].ewm(span=26).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI (Relative Strength Index)
        indicators['rsi'] = calculate_rsi(stock_data['Close'])
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_ma = stock_data['Close'].rolling(window=bb_period).mean()
        bb_std_dev = stock_data['Close'].rolling(window=bb_period).std()
        
        indicators['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
        indicators['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
        indicators['bb_middle'] = bb_ma
        
        # Volume indicators
        indicators['volume_sma'] = stock_data['Volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = stock_data['Volume'] / indicators['volume_sma']
        
        # Price change indicators
        indicators['daily_return'] = stock_data['Close'].pct_change()
        indicators['volatility'] = indicators['daily_return'].rolling(window=30).std()
        
        # Support and Resistance (simplified)
        indicators['resistance'] = stock_data['High'].rolling(window=20).max()
        indicators['support'] = stock_data['Low'].rolling(window=20).min()
        
        return indicators
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return {}

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices (pd.Series): Price series
        period (int): RSI calculation period
    
    Returns:
        pd.Series: RSI values
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral value
        
    except Exception:
        return pd.Series(index=prices.index, data=50)  # Return neutral RSI if calculation fails

def get_company_info(symbol):
    """
    Get company information from Yahoo Finance
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Company information or empty dict if failed
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract key information
        company_info = {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': format_market_cap(info.get('marketCap', 0)),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'price': info.get('regularMarketPrice', 'N/A'),
            'previous_close': info.get('regularMarketPreviousClose', 'N/A'),
            'day_range': f"{info.get('regularMarketDayLow', 'N/A')} - {info.get('regularMarketDayHigh', 'N/A')}",
            '52_week_range': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}"
        }
        
        return company_info
        
    except Exception as e:
        st.warning(f"Could not fetch company info for {symbol}: {str(e)}")
        return {}

def format_market_cap(market_cap):
    """
    Format market capitalization in readable format
    
    Args:
        market_cap (float): Market cap value
    
    Returns:
        str: Formatted market cap string
    """
    try:
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.1f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.1f}M"
        else:
            return f"${market_cap:,.0f}"
    except:
        return "N/A"

def validate_stock_data(stock_data):
    """
    Validate stock data for completeness and quality
    
    Args:
        stock_data (pd.DataFrame): Stock data to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if stock_data is None or stock_data.empty:
        return False, "No data available"
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in stock_data.columns]
    
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    
    if len(stock_data) < 30:
        return False, f"Insufficient data: only {len(stock_data)} days available"
    
    # Check for data quality issues
    if stock_data['Close'].isna().sum() > len(stock_data) * 0.1:
        return False, "Too many missing values in price data"
    
    if (stock_data['Volume'] == 0).sum() > len(stock_data) * 0.5:
        return False, "Too many days with zero volume"
    
    return True, "Data validation passed"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_summary():
    """
    Get general market summary information
    
    Returns:
        dict: Market summary data
    """
    try:
        # Fetch data for major indices
        indices = {
            'S&P 500': '^GSPC',
            'Nasdaq': '^IXIC',
            'Dow Jones': '^DJI'
        }
        
        summary = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = current - previous
                    change_pct = (change / previous) * 100
                    
                    summary[name] = {
                        'current': current,
                        'change': change,
                        'change_pct': change_pct
                    }
            except:
                continue
        
        return summary
        
    except Exception:
        return {}
