import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_data(filepath):
    """Load the sales returns data"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def preprocess_data(df, product_id, target_col='Sales', seq_length=30):
    """
    Preprocess data for a specific product
    
    Args:
        df: DataFrame with sales data
        product_id: Product ID to filter
        target_col: Column to predict ('Sales' or 'Returns')
        seq_length: Length of input sequences
    
    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    # Filter for specific product
    product_df = df[df['Product_ID'] == product_id].copy()
    product_df = product_df.sort_values('Date')
    
    # Extract target values
    values = product_df[target_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Create sequences
    X, y = create_sequences(scaled_values, seq_length)
    
    # Split into train/test (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Test preprocessing
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'sales_returns_data.csv')
    
    df = load_data(data_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Products: {df['Product_ID'].unique()}")
    
    # Test preprocessing for one product
    X_train, y_train, X_test, y_test, scaler = preprocess_data(df, 'P001', 'Sales')
    print(f"\nSales Forecasting - Product P001:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Test for returns
    X_train_r, y_train_r, X_test_r, y_test_r, scaler_r = preprocess_data(df, 'P001', 'Returns')
    print(f"\nReturns Forecasting - Product P001:")
    print(f"X_train shape: {X_train_r.shape}")
    print(f"y_train shape: {y_train_r.shape}")
