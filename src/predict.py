import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

def load_trained_model(product_id, model_type='sales'):
    """
    Load a trained model and its scaler
    
    Args:
        product_id: Product ID (e.g., 'P001')
        model_type: 'sales' or 'returns'
    
    Returns:
        model, scaler
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    model_path = os.path.join(models_dir, f'{product_id}_{model_type}_model.h5')
    scaler_path = os.path.join(models_dir, f'{product_id}_{model_type}_scaler.pkl')
    
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_future(model, scaler, last_sequence, steps=30):
    """
    Predict future values
    
    Args:
        model: Trained LSTM model
        scaler: Fitted scaler
        last_sequence: Last sequence of data (scaled)
        steps: Number of steps to predict
    
    Returns:
        Array of predictions (inverse scaled)
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next value
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence
        current_sequence = np.append(current_sequence[1:], pred[0, 0])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def calculate_optimal_inventory(forecasted_sales, forecasted_returns, safety_stock_factor=1.2):
    """
    Calculate optimal inventory levels
    
    Args:
        forecasted_sales: Predicted sales
        forecasted_returns: Predicted returns
        safety_stock_factor: Multiplier for safety stock
    
    Returns:
        Recommended inventory level
    """
    net_demand = forecasted_sales - forecasted_returns
    optimal_inventory = net_demand * safety_stock_factor
    return np.maximum(optimal_inventory, 0)  # Ensure non-negative

if __name__ == "__main__":
    # Test prediction
    from preprocessing import load_data, preprocess_data
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'sales_returns_data.csv')
    
    # Check if models exist
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir):
        print("Models not found. Please run train.py first.")
    else:
        print("Testing prediction functionality...")
        # This will be tested after training
