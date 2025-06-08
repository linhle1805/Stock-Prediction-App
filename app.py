import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import re
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from stock_utils import train_xgboost, train_random_forest, train_lstm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="Stock Prediction App", layout="wide")
st.sidebar.title("Stock Prediction App")
st.sidebar.image("logo.png", caption="")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Forecasting"],
        icons=["house", "eye"],
        menu_icon="cast",
        default_index=0
    )

symbol = st.sidebar.text_input("Nhập mã cổ phiếu:", value=st.session_state.get("selected_symbol", "VIC"))
start_date = st.sidebar.date_input("Chọn ngày bắt đầu:", value=st.session_state.get("start_date", datetime(2007, 9, 19)))
st.session_state["selected_symbol"] = symbol.upper()
st.session_state["start_date"] = start_date
start_date_str = start_date.strftime("%d/%m/%Y")
end_date_str = datetime.today().strftime("%d/%m/%Y")

@st.cache_data
def fetch_stock_data(symbol="VIC", start_date="19/09/2007", end_date="", output_file="VIC_stock_data.csv"):
    url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    params = {
        "Symbol": symbol,
        "StartDate": start_date,
        "EndDate": end_date,
        "PageIndex": 1,
        "PageSize": 10000
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://cafef.vn/",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data.get("Data") and data["Data"].get("Data"):
            df = pd.DataFrame(data["Data"]["Data"])
            df["Ngay"] = pd.to_datetime(df["Ngay"], format="%d/%m/%Y")
            df.set_index("Ngay", inplace=True)
            return df
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()

def clean_numeric(value):
    try:
        return float(re.sub(r"[^\d.-]", "", str(value)))
    except ValueError:
        return np.nan

def preprocess_stock_data(df):
    columns_to_drop = ["GiaTriKhopLenh", "KLThoaThuan", "GtThoaThuan", "ThayDoi"]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df = df.applymap(clean_numeric)
    df.dropna(inplace=True)
    if 'Ngay' in df.columns:
        df['Ngay'] = pd.to_datetime(df['Ngay'], errors="coerce")
        df.dropna(subset=['Ngay'], inplace=True)
        df.set_index('Ngay', inplace=True)
    return df

data = fetch_stock_data(symbol, start_date_str, end_date_str)
data_cleaned = preprocess_stock_data(data)

def get_current_price(df, target_column="GiaDongCua"):
    if df.empty or target_column not in df.columns:
        st.error("Không có dữ liệu hoặc cột GiaDongCua không tồn tại!")
        return None
    df = df.sort_index()
    valid_prices = df[target_column][(df[target_column].notna()) & (df[target_column] != 0)]
    if not valid_prices.empty:
        return valid_prices.iloc[-1]
    else:
        st.error("Không tìm thấy giá đóng cửa hợp lệ trong dữ liệu!")
        return None

# Dashboard
if selected == "Dashboard":
    st.title("Dashboard - Phân tích cổ phiếu")
    if data_cleaned.empty:
        st.warning("Không có dữ liệu để hiển thị. Vui lòng thử lại sau!")
    else:
        # Dữ liệu lịch sử
        st.subheader("**Dữ liệu lịch sử giá cổ phiếu**")
        data = fetch_stock_data(symbol=symbol, start_date=start_date.strftime("%d/%m/%Y"))
        data_cleaned = preprocess_stock_data(data)
        st.data_editor(data)
        
        # Hiển thị chỉ số
        st.subheader("**Chỉ số quan trọng**")
        gia_dong_cua_tb = data["GiaDongCua"].mean()
        bien_dong_gia_max = data["GiaDongCua"].max() - data["GiaDongCua"].min()
        tong_khoi_luong = data["KhoiLuongKhopLenh"].sum()
        bien_do_dao_dong = data["GiaCaoNhat"].max() - data["GiaThapNhat"].min()
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""
            <div style="text-align: center; border: 2px solid black; padding: 10px; border-radius: 5px;">
                <strong>Giá đóng cửa TB</strong><br>
                <span style="font-size: 20px; color: black; display: inline-block; text-align: center;">{gia_dong_cua_tb:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
            <div style="text-align: center; border: 2px solid black; padding: 10px; border-radius: 5px;">
                <strong>Biến động lớn nhất</strong><br>
                <span style="font-size: 20px; color: black; display: inline-block; text-align: center;">{bien_dong_gia_max:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
            <div style="text-align: center; border: 2px solid black; padding: 10px; border-radius: 5px;">
                <strong>Tổng KL khớp lệnh</strong><br>
                <span style="font-size: 20px; color: black; display: inline-block; text-align: center;">{tong_khoi_luong:,.0f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        col4.markdown(f"""
            <div style="text-align: center; border: 2px solid black; padding: 10px; border-radius: 5px;">
                <strong>Biên độ dao động</strong><br>
                <span style="font-size: 20px; color: black; display: inline-block; text-align: center;">{bien_do_dao_dong:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("**Visualization**")
        # Biểu đồ 1: Xu hướng giá cổ phiếu
        fig = go.Figure()
        price_columns = ["GiaDieuChinh", "GiaDongCua", "GiaMoCua", "GiaCaoNhat", "GiaThapNhat"]
        colors = ["#440154", "#3B528B", "#306A8E", "#2C728E", "#2A788E"]
        for i, col in enumerate(price_columns):
            fig.add_trace(go.Scatter(
                x=data_cleaned.index, y=data_cleaned[col], mode='lines', name=col,
                line=dict(color=colors[i], width=2)
            ))
        fig.update_layout(
            title="Xu hướng giá cổ phiếu theo thời gian",
            xaxis_title="Thời gian",
            yaxis_title="Giá cổ phiếu",
            template="plotly_white",
            plot_bgcolor="white",
            font=dict(family="Arial", size=14, color="black"),
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
            legend=dict(x=0.98, y=0.99, bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=0.5),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Biểu đồ 2: Khối lượng khớp lệnh
        fig = px.bar(data, x=data.index, y="KhoiLuongKhopLenh",
                     title="Khối lượng khớp lệnh theo thời gian", color="KhoiLuongKhopLenh", color_continuous_scale="viridis")
        st.plotly_chart(fig)

        # Biểu đồ 3: Mối quan hệ giá & khối lượng
        fig = px.scatter(data, x="GiaDongCua", y="KhoiLuongKhopLenh",
                         title="Mối quan hệ giá cổ phiếu & khối lượng giao dịch",
                         color_discrete_sequence=["#355F8D"])
        st.plotly_chart(fig)

        # Biểu đồ 4: Volume Profile
        data["GiaMid"] = (data["GiaCaoNhat"] + data["GiaThapNhat"]) / 2
        bins = np.linspace(data['GiaMid'].min(), data['GiaMid'].max(), 20)
        data['GiaGroup'] = pd.cut(data['GiaMid'], bins, include_lowest=True).astype(str)
        volume_profile = data.groupby('GiaGroup')['KhoiLuongKhopLenh'].sum().reset_index()
        fig = px.bar(volume_profile, x="KhoiLuongKhopLenh", y="GiaGroup", orientation="h",
                     title="Volume profile - Vùng giá quan trọng",
                     color="KhoiLuongKhopLenh", color_continuous_scale="viridis")
        st.plotly_chart(fig)

        # Biểu đồ 5: Heatmap tương quan
        corr = data[['GiaMoCua', 'GiaDongCua', 'GiaCaoNhat', 'GiaThapNhat', 'KhoiLuongKhopLenh']].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="viridis")
        fig.update_layout(
            title="Heatmap tương quan giữa các yếu tố",
            width=1200, height=600,
            coloraxis_colorbar=dict(title="Hệ số tương quan"),
            xaxis=dict(title="", tickangle=45),
            yaxis=dict(title=""),
            margin=dict(l=50, r=50, t=50, b=50))
        st.plotly_chart(fig)
 
# ====================== DỰ BÁO ======================
elif selected == "Forecasting":
    st.title("Dự báo giá cổ phiếu")
    if data_cleaned.empty:
        st.warning("Không có dữ liệu để dự báo!")
    else:
        data_cleaned = data_cleaned.sort_index()
        FEATURES = ["GiaDieuChinh", "KhoiLuongKhopLenh", "GiaMoCua", "GiaCaoNhat", "GiaThapNhat"]
        TARGET = "GiaDongCua"
        time_steps = 10
        future_days = st.slider("Số ngày dự báo:", 1, 30, 7)

        data_raw = data_cleaned.copy()
        data_scaled = data_cleaned.copy()

        scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
        data_scaled[FEATURES] = scaler_X.fit_transform(data_scaled[FEATURES])
        data_scaled[[TARGET]] = scaler_y.fit_transform(data_scaled[[TARGET]])

        def prepare_data(df, time_steps=10):
            X, y = [], []
            for i in range(len(df) - time_steps - future_days):
                X.append(df.iloc[i:i + time_steps][FEATURES].values.flatten())
                y.append(df.iloc[i + time_steps:i + time_steps + future_days][TARGET].values)
            return np.array(X), np.array(y)

        def prepare_data_lstm(df, features, target, time_steps, future_days):
            X, y = [], []
            for i in range(time_steps, len(df) - future_days):
                x_data = df[features].iloc[i-time_steps:i].values
                y_data = df[target].iloc[i:i+future_days].values
                # Kiểm tra kích thước đồng nhất
                if x_data.shape == (time_steps, len(features)) and y_data.shape == (future_days,):
                    X.append(x_data)
                    y.append(y_data)
            return np.array(X), np.array(y)

        X_raw, y_raw = prepare_data(data_raw)
        train_size = int(len(X_raw) * 0.8)
        X_train_raw, y_train_raw = X_raw[:train_size], y_raw[:train_size]
        X_test_raw, y_test_raw = X_raw[train_size:], y_raw[train_size:]

        X_scaled, y_scaled = prepare_data_lstm(data_scaled, FEATURES, TARGET, time_steps, future_days)
        train_size = int(len(X_scaled) * 0.8)
        X_train_scaled, y_train_scaled = X_scaled[:train_size], y_scaled[:train_size]
        X_test_scaled, y_test_scaled = X_scaled[train_size:], y_scaled[train_size:]

        def train_xgboost():
            model = MultiOutputRegressor(XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=4,
                                                      colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.5, random_state=42))
            model.fit(X_train_raw, y_train_raw)
            return model

        def train_rf():
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=5,
                                                               min_samples_leaf=2, random_state=42, n_jobs=-1))
            model.fit(X_train_raw, y_train_raw)
            return model

        def train_lstm():
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(time_steps, len(FEATURES))),
                Dropout(0.3),
                LSTM(25, return_sequences=False),
                Dropout(0.3),
                Dense(20, activation='relu'),
                Dense(future_days)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test_scaled), verbose=1)
            return model

        model_option = st.selectbox("Chọn mô hình dự báo:", ["XGBoost", "Random Forest", "LSTM"])
        if st.button("Dự báo"):
            if model_option == "XGBoost":
                model = train_xgboost()
                last_data = data_raw[FEATURES].iloc[-10:].values.flatten().reshape(1, -1)
                future_preds = model.predict(last_data)[0]

            elif model_option == "Random Forest":
                model = train_rf()
                last_data = data_raw[FEATURES].iloc[-10:].values.flatten().reshape(1, -1)
                future_preds = model.predict(last_data)[0]

            else:
                model = train_lstm()
                last_data = data_scaled[FEATURES].iloc[-time_steps:].values.reshape(1, time_steps, len(FEATURES))
                future_preds_scaled = model.predict(last_data)[0]
                future_preds = scaler_y.inverse_transform(future_preds_scaled.reshape(-1, 1)).flatten()

            future_dates = []
            current_date = data_cleaned.index[-1]
            while len(future_dates) < future_days:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5: 
                    future_dates.append(current_date)

            if model_option in ["Random Forest", "XGBoost"]:
                X_test, y_test = X_test_raw, y_test_raw
                y_pred = model.predict(X_test)
            else:
                X_test, y_test = X_test_scaled, y_test_scaled
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled)
                y_test = scaler_y.inverse_transform(y_test)

            y_pred_avg = np.mean(y_pred, axis=1)
            df_test_dates = data_cleaned.index[train_size + time_steps + future_days:]
            df_test_preds = pd.DataFrame({'Ngay': df_test_dates, 'GiaDuBao': y_pred_avg})
            df_test_preds.set_index('Ngay', inplace=True)

            st.subheader("Số liệu chi tiết")
            col1, col2, col3 = st.columns(3)
            
            # Cột 1: Giá hiện tại & Biến động
            with col1:
                current_price = get_current_price(data)
                if current_price is not None:
                    st.metric("Giá hiện tại", f"{current_price:.2f}")
                    valid_prices = data["GiaDongCua"][(data["GiaDongCua"].notna()) & (data["GiaDongCua"] != 0)]
                    if len(valid_prices) >= 2:
                        percent_change = (current_price - valid_prices.iloc[-2]) / valid_prices.iloc[-2] * 100
                        st.metric("Biến động lịch sử", f"{percent_change:.1f}%")
                    else:
                        st.metric("Biến động lịch sử", "N/A")
                else:
                    st.metric("Giá hiện tại", "N/A")
                    st.metric("Biến động lịch sử", "N/A")
               
            # Cột 2: Dự đoán ngày tiếp theo & RSI
            with col2:
                st.metric("Dự đoán ngày tiếp theo", f"{future_preds[0]:.2f}", f"{((future_preds[0] - current_price) / current_price) * 100:.2f}%")
                # RSI (Relative Strength Index) tính đơn giản cho ví dụ
                delta = data_cleaned[TARGET].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                st.metric("RSI", f"{rsi.iloc[-1]:.1f}")
            
            # Cột 3: MACD & Trend Strength
            with col3:
                ema12 = data_cleaned[TARGET].ewm(span=12, adjust=False).mean()
                ema26 = data_cleaned[TARGET].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_value = macd.iloc[-1]
                st.metric("MACD", f"{macd_value:.2f}")
            
                trend_strength = (current_price - data_cleaned[TARGET].mean()) / data_cleaned[TARGET].mean() * 100
                st.metric("Trend Strength", f"{trend_strength:.2f}%")

            metrics = {
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R² Score": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred)
            }

            st.sidebar.write("### Đánh giá mô hình")
            for k, v in metrics.items():
                st.sidebar.write(f"{k}: {v:.4f}")
                
            st.subheader("Biểu đồ và bảng dự đoán")
            if model_option in ["Random Forest", "XGBoost"]:
                fig = go.Figure([
                    go.Scatter(x=data_cleaned.index[train_size + time_steps:], y=data_cleaned[TARGET][train_size + time_steps:],
                               mode='lines', name='Giá thực tế', line=dict(color='black', width=2)),
                    go.Scatter(x=df_test_preds.index, y=df_test_preds['GiaDuBao'],
                               mode='lines', name=f'Dự báo {model_option}', line=dict(color='blue', width=2, dash='dot')),
                    go.Scatter(x=future_dates, y=future_preds, mode='lines+markers',
                               name='Dự báo tương lai', line=dict(color='red', width=2, dash='dot'))
                ])
            else:
                actual_prices = data_cleaned[TARGET].iloc[train_size + time_steps:].values
                fig = go.Figure([
                    go.Scatter(x=data_cleaned.index[train_size + time_steps:], y=actual_prices,
                               mode='lines', name='Giá thực tế', line=dict(color='black', width=2)),
                    go.Scatter(x=df_test_preds.index, y=df_test_preds['GiaDuBao'],
                               mode='lines', name='Dự báo LSTM', line=dict(color='blue', width=2, dash='dot')),
                    go.Scatter(x=future_dates, y=future_preds,
                               mode='lines+markers', name='Dự báo tương lai', line=dict(color='red', width=2, dash='dot'))
                ])

            fig.update_layout(title=f'Dự báo giá cổ phiếu - {model_option}', xaxis_title="Thời gian", yaxis_title="Giá đóng cửa",
                              template="plotly_white", hovermode="x unified")
            st.plotly_chart(fig)

            st.write("### Bảng dự báo giá cổ phiếu tương lai")
            pred_df = pd.DataFrame({"Ngày": future_dates, "Giá dự báo": future_preds})
            st.dataframe(pred_df)

            st.success("Dự báo hoàn tất!")
