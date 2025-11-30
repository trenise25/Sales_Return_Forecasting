import os
import pickle
import pandas as pd
from preprocessing import load_data, preprocess_data
from models import build_lstm_model, train_model, evaluate_model

def train_all_models(data_path, products, seq_length=30, epochs=50):
    """
    Train LSTM models for all products for both sales and returns forecasting
    
    Args:
        data_path: Path to the data CSV
        products: List of product IDs
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs
    """
    # Load data
    df = load_data(data_path)
    
    # Create models directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    results = []
    
    for product_id in products:
        print(f"\n{'='*60}")
        print(f"Training models for {product_id}")
        print(f"{'='*60}")
        
        # Train Sales Forecasting Model
        print(f"\n--- Sales Forecasting ---")
        X_train, y_train, X_test, y_test, scaler = preprocess_data(
            df, product_id, 'Sales', seq_length
        )
        
        sales_model = build_lstm_model(seq_length)
        history_sales = train_model(sales_model, X_train, y_train, X_test, y_test, epochs)
        loss_sales, mae_sales = evaluate_model(sales_model, X_test, y_test)
        
        # Save model and scaler
        sales_model_path = os.path.join(models_dir, f'{product_id}_sales_model.h5')
        sales_scaler_path = os.path.join(models_dir, f'{product_id}_sales_scaler.pkl')
        sales_model.save(sales_model_path)
        with open(sales_scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Train Returns Forecasting Model
        print(f"\n--- Returns Forecasting ---")
        X_train_r, y_train_r, X_test_r, y_test_r, scaler_r = preprocess_data(
            df, product_id, 'Returns', seq_length
        )
        
        returns_model = build_lstm_model(seq_length)
        history_returns = train_model(returns_model, X_train_r, y_train_r, X_test_r, y_test_r, epochs)
        loss_returns, mae_returns = evaluate_model(returns_model, X_test_r, y_test_r)
        
        # Save model and scaler
        returns_model_path = os.path.join(models_dir, f'{product_id}_returns_model.h5')
        returns_scaler_path = os.path.join(models_dir, f'{product_id}_returns_scaler.pkl')
        returns_model.save(returns_model_path)
        with open(returns_scaler_path, 'wb') as f:
            pickle.dump(scaler_r, f)
        
        results.append({
            'Product_ID': product_id,
            'Sales_MAE': mae_sales,
            'Returns_MAE': mae_returns
        })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(models_dir, 'training_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print("\nResults Summary:")
    print(results_df)
    print(f"\nModels saved to: {models_dir}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'sales_returns_data.csv')
    
    # Train for all products
    products = ['P001', 'P002', 'P003', 'P004', 'P005']
    train_all_models(data_path, products, seq_length=30, epochs=30)
