import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(seq_length, units=50, dropout_rate=0.2):
    """
    Build LSTM model for time series forecasting
    
    Args:
        seq_length: Length of input sequences
        units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train the LSTM model
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_test, y_test: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Training history
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    return loss, mae

if __name__ == "__main__":
    print("LSTM Model Architecture:")
    model = build_lstm_model(seq_length=30)
    model.summary()
