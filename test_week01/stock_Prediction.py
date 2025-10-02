import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta

class StockPredictor:
    """
    Lớp dự đoán giá cổ phiếu sử dụng mô hình FB Prophet
    """
    
    def __init__(self):
        """Khởi tạo bộ dự đoán"""
        pass
    
    def prepare_data(self, price_data):
        """
        Chuẩn bị dữ liệu cho mô hình Prophet
        
        Prophet yêu cầu DataFrame với 2 cột: 'ds' (ngày) và 'y' (giá)
        """
        # Tạo bản sao của dữ liệu để tránh cảnh báo về chế độ xem
        price_data_copy = price_data.copy()
        
        # Chuyển đổi index thành cột và loại bỏ thông tin múi giờ
        dates = price_data_copy.index
        
        # Loại bỏ timezone nếu có
        if hasattr(dates, 'tz_localize'):
            dates = dates.tz_localize(None)
        
        # Chuyển đổi dữ liệu sang định dạng Prophet yêu cầu
        prophet_data = pd.DataFrame({
            'ds': dates,
            'y': price_data_copy.values
        })
        
        return prophet_data
    
    def predict(self, price_data, days=30):
        """
        Dự đoán giá cổ phiếu trong tương lai
        
        price_data: Series pandas chứa dữ liệu giá đóng cửa với index là ngày
        days: Số ngày cần dự đoán trong tương lai
        
        Trả về: Biểu đồ và DataFrame chứa giá dự đoán
        """
        try:
            # Chuẩn bị dữ liệu
            prophet_data = self.prepare_data(price_data)
            
            # Tạo và huấn luyện mô hình
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            model.fit(prophet_data)
            
            # Tạo khung dự đoán cho tương lai
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Tạo DataFrame cho kết quả dự đoán
            # Trong phương thức predict(), chỉnh sửa phần tạo future_dates
            last_date = price_data.index[-1]
            if hasattr(last_date, 'tz_localize'):
                last_date = last_date.tz_localize(None)
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Lấy giá trị dự đoán cho các ngày tương lai
            future_predictions = forecast.iloc[-days:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            future_predictions.set_index('ds', inplace=True)
            
            # Vẽ biểu đồ
            fig = plt.figure(figsize=(12, 6))
            # Thêm SMA vào biểu đồ dự đoán
            if len(price_data) >= 50:
                sma50 = price_data.rolling(window=50).mean()
                plt.plot(price_data.index, sma50, 'g--', label='SMA 50')

            if len(price_data) >= 200:
                sma200 = price_data.rolling(window=200).mean()
                plt.plot(price_data.index, sma200, 'b--', label='SMA 200')

            # Thêm EMA vào biểu đồ
            if len(price_data) >= 20:
                ema20 = price_data.ewm(span=20, adjust=False).mean()
                plt.plot(price_data.index, ema20, 'purple', label='EMA 20')
            # Vẽ dữ liệu lịch sử
            plt.plot(price_data.index, price_data.values, label='Dữ liệu lịch sử')
            
            # Vẽ dự đoán
            plt.plot(future_predictions.index, future_predictions['yhat'], 'r--', label='Dự đoán')
            
            # Vẽ khoảng tin cậy
            plt.fill_between(
                future_predictions.index,
                future_predictions['yhat_lower'],
                future_predictions['yhat_upper'],
                color='red',
                alpha=0.2,
                label='Khoảng tin cậy 95%'
            )
            
            plt.title('Dự đoán giá cổ phiếu')
            plt.xlabel('Ngày')
            plt.ylabel('Giá đóng cửa')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            return fig, future_predictions
        except Exception as e:
            print(f"Lỗi khi dự đoán giá với Prophet: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def plot_components(self, price_data, days=30):
        """
        Vẽ các thành phần của mô hình Prophet
        """
        try:
            # Chuẩn bị dữ liệu
            prophet_data = self.prepare_data(price_data)
            
            # Tạo và huấn luyện mô hình
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            model.fit(prophet_data)
            
            # Tạo khung dự đoán
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Vẽ các thành phần
            fig = model.plot_components(forecast)
            return fig
        except Exception as e:
            print(f"Lỗi khi vẽ thành phần: {e}")
            return None