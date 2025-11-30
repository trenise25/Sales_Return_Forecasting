import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_products=5, days=730, start_date='2023-01-01'):
    np.random.seed(42)
    date_range = [datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=x) for x in range(days)]
    
    data = []
    
    for product_id in range(1, num_products + 1):
        # Base demand with some seasonality
        base_demand = np.random.randint(50, 150)
        seasonality = np.sin(np.linspace(0, 4 * np.pi, days)) * 20
        trend = np.linspace(0, 30, days)
        
        sales = base_demand + seasonality + trend + np.random.normal(0, 10, days)
        sales = np.maximum(sales, 0).astype(int)
        
        # Returns as a fraction of sales from a few days ago (simplified)
        # Assuming returns happen 1-7 days after sale
        returns = np.zeros(days)
        for i in range(days):
            if i > 7:
                # Returns are roughly 5-15% of sales from a week ago
                return_rate = np.random.uniform(0.05, 0.15)
                past_sales = sales[i-7:i]
                avg_past_sales = np.mean(past_sales) if len(past_sales) > 0 else 0
                returns[i] = int(avg_past_sales * return_rate)
        
        returns = np.maximum(returns, 0).astype(int)
        
        for i, date in enumerate(date_range):
            data.append({
                'Date': date,
                'Product_ID': f'P{product_id:03d}',
                'Sales': sales[i],
                'Returns': int(returns[i])
            })
            
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    import os
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sales_returns_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
