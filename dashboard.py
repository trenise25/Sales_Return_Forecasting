import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_data, preprocess_data
from predict import load_trained_model, predict_future, calculate_optimal_inventory

# Page configuration
st.set_page_config(
    page_title="Sales Return Forecasting & Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_project_data():
    """Load the sales returns data"""
    data_path = os.path.join('data', 'sales_returns_data.csv')
    return load_data(data_path)

def check_models_exist():
    """Check if trained models exist"""
    models_dir = 'models'
    return os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0

def main():
    st.title("📊 Sales Return Forecasting & Inventory Optimization")
    st.markdown("---")
    
    # Check if models are trained
    if not check_models_exist():
        st.warning("⚠️ Models not found! Please train the models first by running `python src/train.py`")
        st.info("This may take a few minutes. The dashboard will be fully functional after training.")
        
        if st.button("Show Sample Data"):
            df = load_project_data()
            st.subheader("Sample Data Preview")
            st.dataframe(df.head(20))
            
            # Show basic statistics
            st.subheader("Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Products", df['Product_ID'].nunique())
            with col3:
                st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        return
    
    # Load data
    df = load_project_data()
    products = sorted(df['Product_ID'].unique())
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    selected_product = st.sidebar.selectbox("Select Product", products)
    forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
    safety_stock_factor = st.sidebar.slider("Safety Stock Factor", 1.0, 2.0, 1.2, 0.1)
    
    # Load models
    try:
        sales_model, sales_scaler = load_trained_model(selected_product, 'sales')
        returns_model, returns_scaler = load_trained_model(selected_product, 'returns')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # Get historical data for selected product
    product_df = df[df['Product_ID'] == selected_product].copy()
    product_df = product_df.sort_values('Date')
    
    # Prepare last sequence for prediction
    seq_length = 30
    sales_values = product_df['Sales'].values[-seq_length:]
    returns_values = product_df['Returns'].values[-seq_length:]
    
    sales_scaled = sales_scaler.transform(sales_values.reshape(-1, 1)).flatten()
    returns_scaled = returns_scaler.transform(returns_values.reshape(-1, 1)).flatten()
    
    # Make predictions
    sales_forecast = predict_future(sales_model, sales_scaler, sales_scaled, forecast_days)
    returns_forecast = predict_future(returns_model, returns_scaler, returns_scaled, forecast_days)
    
    # Calculate optimal inventory
    optimal_inventory = calculate_optimal_inventory(sales_forecast, returns_forecast, safety_stock_factor)
    
    # Create forecast dates
    last_date = product_df['Date'].max()
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Metrics
    st.header(f"📈 Insights for {selected_product}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sales = sales_forecast.mean()
        st.metric("Avg Forecasted Sales", f"{avg_sales:.0f} units/day")
    
    with col2:
        avg_returns = returns_forecast.mean()
        st.metric("Avg Forecasted Returns", f"{avg_returns:.0f} units/day")
    
    with col3:
        return_rate = (avg_returns / avg_sales * 100) if avg_sales > 0 else 0
        st.metric("Return Rate", f"{return_rate:.1f}%")
    
    with col4:
        avg_inventory = optimal_inventory.mean()
        st.metric("Recommended Inventory", f"{avg_inventory:.0f} units")
    
    st.markdown("---")
    
    # Sales Forecast Chart
    st.subheader("📊 Sales Forecast")
    fig_sales = go.Figure()
    
    # Historical sales
    fig_sales.add_trace(go.Scatter(
        x=product_df['Date'].tail(90),
        y=product_df['Sales'].tail(90),
        mode='lines',
        name='Historical Sales',
        line=dict(color='#3498db', width=2)
    ))
    
    # Forecasted sales
    fig_sales.add_trace(go.Scatter(
        x=forecast_dates,
        y=sales_forecast,
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig_sales.update_layout(
        title=f"Sales Forecast - {selected_product}",
        xaxis_title="Date",
        yaxis_title="Sales (units)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # Returns Forecast Chart
    st.subheader("🔄 Returns Forecast")
    fig_returns = go.Figure()
    
    # Historical returns
    fig_returns.add_trace(go.Scatter(
        x=product_df['Date'].tail(90),
        y=product_df['Returns'].tail(90),
        mode='lines',
        name='Historical Returns',
        line=dict(color='#9b59b6', width=2)
    ))
    
    # Forecasted returns
    fig_returns.add_trace(go.Scatter(
        x=forecast_dates,
        y=returns_forecast,
        mode='lines+markers',
        name='Forecasted Returns',
        line=dict(color='#f39c12', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig_returns.update_layout(
        title=f"Returns Forecast - {selected_product}",
        xaxis_title="Date",
        yaxis_title="Returns (units)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_returns, use_container_width=True)
    
    # Inventory Optimization
    st.subheader("📦 Inventory Optimization")
    
    fig_inventory = go.Figure()
    
    fig_inventory.add_trace(go.Scatter(
        x=forecast_dates,
        y=sales_forecast,
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='#3498db', width=2),
        fill='tonexty'
    ))
    
    fig_inventory.add_trace(go.Scatter(
        x=forecast_dates,
        y=returns_forecast,
        mode='lines',
        name='Forecasted Returns',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig_inventory.add_trace(go.Scatter(
        x=forecast_dates,
        y=optimal_inventory,
        mode='lines',
        name='Recommended Inventory',
        line=dict(color='#2ecc71', width=3, dash='dot')
    ))
    
    fig_inventory.update_layout(
        title=f"Inventory Planning - {selected_product}",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_inventory, use_container_width=True)
    
    # Detailed Forecast Table
    with st.expander("📋 View Detailed Forecast Data"):
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted Sales': sales_forecast.round(0).astype(int),
            'Forecasted Returns': returns_forecast.round(0).astype(int),
            'Net Demand': (sales_forecast - returns_forecast).round(0).astype(int),
            'Recommended Inventory': optimal_inventory.round(0).astype(int)
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"{selected_product}_forecast.csv",
            mime="text/csv"
        )
    
    # Alerts
    st.subheader("⚠️ Alerts & Recommendations")
    
    # Check for potential issues
    alerts = []
    
    if return_rate > 15:
        alerts.append(("🔴 High Return Rate", f"Return rate of {return_rate:.1f}% is above threshold. Investigate quality issues."))
    
    if sales_forecast.std() > sales_forecast.mean() * 0.5:
        alerts.append(("🟡 High Volatility", "Sales forecast shows high volatility. Consider increasing safety stock."))
    
    if optimal_inventory.max() > optimal_inventory.mean() * 1.5:
        alerts.append(("🟡 Inventory Spike", "Significant inventory spike detected. Plan for storage capacity."))
    
    if len(alerts) == 0:
        st.success("✅ No alerts. Inventory levels are optimal!")
    else:
        for alert_type, message in alerts:
            st.warning(f"{alert_type}: {message}")

if __name__ == "__main__":
    main()
