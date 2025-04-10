import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ====================== TRAINING XGBOOST ======================
def train_xgboost(X_train, y_train):
    model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=4,
                                              colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1,
                                              reg_lambda=0.5, random_state=42))
    model.fit(X_train, y_train)
    return model
# ====================== TRAINING RANDOM FOREST ======================
def train_random_forest(X_train, y_train):
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=5, 
                                                       min_samples_leaf=2, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    return model
# ====================== TRAINING LSTM ======================
def train_lstm(X_train, y_train, X_test, y_test, time_steps, num_features, future_days):

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(time_steps, num_features)),
        LSTM(50, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(future_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
              validation_data=(X_test, y_test), verbose=1)
    return model
