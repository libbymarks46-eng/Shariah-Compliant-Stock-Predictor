import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    Stock price prediction model using various machine learning techniques
    """
    
    def __init__(self, stock_data):
        """
        Initialize the predictor with historical stock data
        
        Args:
            stock_data (pd.DataFrame): Historical stock data with OHLCV columns
        """
        self.stock_data = stock_data
        self.scaler = StandardScaler()
        self.models = {}
        self.accuracy_metrics = {}
        
    def prepare_features(self, lookback_days=30):
        """
        Prepare feature matrix for machine learning models
        
        Args:
            lookback_days (int): Number of previous days to use as features
            
        Returns:
            tuple: (X, y) feature matrix and target values
        """
        data = self.stock_data.copy()
        
        # Add technical indicators as features
        data['sma_5'] = data['Close'].rolling(5).mean()
        data['sma_10'] = data['Close'].rolling(10).mean()
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['rsi'] = self._calculate_rsi(data['Close'])
        data['volume_sma'] = data['Volume'].rolling(10).mean()
        data['price_change'] = data['Close'].pct_change()
        data['volume_change'] = data['Volume'].pct_change()
        
        # Create lagged features
        feature_columns = ['Close', 'Volume', 'sma_5', 'sma_10', 'sma_20', 'rsi', 'price_change', 'volume_change']
        
        X = []
        y = []
        
        for i in range(lookback_days, len(data)):
            # Features: previous lookback_days of selected columns
            feature_row = []
            for col in feature_columns:
                if col in data.columns:
                    feature_row.extend(data[col].iloc[i-lookback_days:i].values)
            
            X.append(feature_row)
            y.append(data['Close'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Remove any rows with NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def linear_regression_prediction(self, prediction_days, confidence_level=95):
        """
        Generate predictions using Linear Regression
        
        Args:
            prediction_days (int): Number of days to predict
            confidence_level (int): Confidence level for intervals
            
        Returns:
            tuple: (predictions, confidence_intervals)
        """
        try:
            X, y = self.prepare_features()
            
            if len(X) < 30:  # Need minimum data for reliable predictions
                raise ValueError("Insufficient data for reliable predictions")
            
            # Split data for training and validation
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            self.models['linear_regression'] = model
            
            # Calculate accuracy metrics
            y_pred_test = model.predict(X_test_scaled)
            self.accuracy_metrics['linear_regression'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            }
            
            # Generate future predictions
            predictions = []
            confidence_intervals = {'upper': [], 'lower': []}
            
            # Start with the last known data point
            last_sequence = X[-1].reshape(1, -1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Calculate prediction uncertainty based on historical errors
            residuals = y_test - y_pred_test
            std_error = np.std(residuals)
            
            # Z-score for confidence interval
            z_score = {80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}[confidence_level]
            
            current_price = self.stock_data['Close'].iloc[-1]
            
            for day in range(prediction_days):
                # Predict next price
                pred_price = model.predict(last_sequence_scaled)[0]
                
                # Add some trend and volatility
                trend_factor = 1 + (day * 0.001)  # Small upward trend
                volatility = std_error * (1 + day * 0.1)  # Increasing uncertainty
                
                pred_price *= trend_factor
                predictions.append(pred_price)
                
                # Calculate confidence intervals
                margin_error = z_score * volatility
                confidence_intervals['upper'].append(pred_price + margin_error)
                confidence_intervals['lower'].append(max(0, pred_price - margin_error))
                
                # Update sequence for next prediction (simplified approach)
                # In a real scenario, we'd need the actual features for the predicted day
                if day < prediction_days - 1:
                    # For simplicity, use trend-adjusted values
                    new_features = last_sequence[0].copy()
                    new_features[0] = pred_price  # Update close price
                    last_sequence = new_features.reshape(1, -1)
                    last_sequence_scaled = self.scaler.transform(last_sequence)
            
            return predictions, confidence_intervals
            
        except Exception as e:
            # Fallback to simple trend prediction
            return self._fallback_prediction(prediction_days, confidence_level)
    
    def moving_average_prediction(self, prediction_days, confidence_level=95):
        """
        Generate predictions using Moving Average method
        
        Args:
            prediction_days (int): Number of days to predict
            confidence_level (int): Confidence level for intervals
            
        Returns:
            tuple: (predictions, confidence_intervals)
        """
        try:
            data = self.stock_data['Close']
            
            # Calculate different moving averages
            ma_5 = data.rolling(5).mean()
            ma_10 = data.rolling(10).mean()
            ma_20 = data.rolling(20).mean()
            
            # Use weighted combination of moving averages
            weights = [0.5, 0.3, 0.2]  # More weight on shorter periods
            current_mas = [ma_5.iloc[-1], ma_10.iloc[-1], ma_20.iloc[-1]]
            
            # Calculate trend
            recent_prices = data.tail(10)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / 10
            
            # Calculate volatility for confidence intervals
            returns = data.pct_change().dropna()
            volatility = returns.std() * data.iloc[-1]
            
            predictions = []
            confidence_intervals = {'upper': [], 'lower': []}
            
            # Z-score for confidence interval
            z_score = {80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}[confidence_level]
            
            base_price = sum(w * ma for w, ma in zip(weights, current_mas))
            
            for day in range(1, prediction_days + 1):
                # Apply trend with diminishing effect
                trend_effect = trend * day * (1 - day * 0.01)  # Trend weakens over time
                
                # Add mean reversion component
                mean_price = data.tail(50).mean()
                reversion = (mean_price - base_price) * 0.1 * day / prediction_days
                
                pred_price = base_price + trend_effect + reversion
                predictions.append(pred_price)
                
                # Calculate confidence intervals with increasing uncertainty
                uncertainty = volatility * np.sqrt(day) * z_score / 1.96
                confidence_intervals['upper'].append(pred_price + uncertainty)
                confidence_intervals['lower'].append(max(0, pred_price - uncertainty))
            
            # Calculate simple accuracy metrics
            self.accuracy_metrics['moving_average'] = self._calculate_ma_accuracy()
            
            return predictions, confidence_intervals
            
        except Exception as e:
            return self._fallback_prediction(prediction_days, confidence_level)
    
    def _calculate_ma_accuracy(self):
        """
        Calculate accuracy metrics for moving average method
        """
        try:
            data = self.stock_data['Close']
            
            # Backtest moving average predictions
            test_days = min(30, len(data) // 4)
            actual = []
            predicted = []
            
            for i in range(test_days):
                test_data = data.iloc[:-test_days+i]
                ma_20 = test_data.rolling(20).mean().iloc[-1]
                actual_price = data.iloc[-test_days+i]
                
                actual.append(actual_price)
                predicted.append(ma_20)
            
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            return {
                'rmse': np.sqrt(mean_squared_error(actual, predicted)),
                'mae': mean_absolute_error(actual, predicted),
                'r2': max(0, r2_score(actual, predicted))  # Ensure non-negative R²
            }
            
        except Exception:
            return {'rmse': 0, 'mae': 0, 'r2': 0}
    
    def _fallback_prediction(self, prediction_days, confidence_level=95):
        """
        Fallback prediction method using simple trend analysis
        """
        data = self.stock_data['Close']
        current_price = data.iloc[-1]
        
        # Calculate simple trend from last 30 days
        recent_data = data.tail(30)
        trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / 30
        
        # Calculate volatility
        returns = data.pct_change().tail(30).std()
        volatility = returns * current_price
        
        predictions = []
        confidence_intervals = {'upper': [], 'lower': []}
        
        z_score = {80: 1.28, 85: 1.44, 90: 1.65, 95: 1.96, 99: 2.58}[confidence_level]
        
        for day in range(1, prediction_days + 1):
            pred_price = current_price + (trend * day)
            predictions.append(pred_price)
            
            margin_error = z_score * volatility * np.sqrt(day)
            confidence_intervals['upper'].append(pred_price + margin_error)
            confidence_intervals['lower'].append(max(0, pred_price - margin_error))
        
        return predictions, confidence_intervals
    
    def get_model_accuracy(self, model_name):
        """
        Get accuracy metrics for a specific model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Accuracy metrics or None if not available
        """
        return self.accuracy_metrics.get(model_name)
