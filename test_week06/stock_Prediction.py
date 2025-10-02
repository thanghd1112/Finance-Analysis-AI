import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime, timedelta

class StockPredictor:
    """
    Lớp dự đoán giá cổ phiếu sử dụng mô hình FB Prophet
    Hỗ trợ hai chế độ seasonality: Additive và Multiplicative
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
    
    def predict(self, price_data, days=30, seasonality_mode='additive'):
        """
        Dự đoán giá cổ phiếu trong tương lai
        
        price_data: Series pandas chứa dữ liệu giá đóng cửa với index là ngày
        days: Số ngày cần dự đoán trong tương lai
        seasonality_mode: Chế độ seasonality ('additive' hoặc 'multiplicative')
        
        Trả về: Biểu đồ và DataFrame chứa giá dự đoán
        """
        try:
            # Chuẩn bị dữ liệu
            prophet_data = self.prepare_data(price_data)
            
            # Tạo và huấn luyện mô hình với chế độ seasonality được chọn
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode=seasonality_mode,  # Thêm chế độ seasonality
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            model.fit(prophet_data)
            
            # Tạo khung dự đoán cho tương lai
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Tạo DataFrame cho kết quả dự đoán
            last_date = price_data.index[-1]
            if hasattr(last_date, 'tz_localize'):
                last_date = last_date.tz_localize(None)
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            
            # Lấy giá trị dự đoán cho các ngày tương lai
            future_predictions = forecast.iloc[-days:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            future_predictions.set_index('ds', inplace=True)
            
            # Vẽ biểu đồ
            fig = plt.figure(figsize=(14, 7))
            
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
            
            plt.xlabel('Ngày')
            plt.ylabel('Giá đóng cửa')
            plt.title(f'Dự đoán giá cổ phiếu (Chế độ: {seasonality_mode.capitalize()})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            return fig, future_predictions
        except Exception as e:
            print(f"Lỗi khi dự đoán giá với Prophet: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def compare_seasonality_modes(self, price_data, days=30):
        """
        So sánh hai chế độ seasonality: additive và multiplicative
        
        Trả về: Biểu đồ so sánh và nhận xét
        """
        try:
            # Dự đoán với chế độ additive
            fig_add, pred_add = self.predict(price_data, days=days, seasonality_mode='additive')
            
            # Dự đoán với chế độ multiplicative
            fig_mul, pred_mul = self.predict(price_data, days=days, seasonality_mode='multiplicative')
            
            # Tạo figure so sánh
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            
            # Vẽ dữ liệu lịch sử, dự đoán additive
            ax1.plot(price_data.index, price_data.values, label='Dữ liệu lịch sử')
            ax1.plot(pred_add.index, pred_add['yhat'], 'r--', label='Dự đoán (Additive)')
            ax1.fill_between(
                pred_add.index,
                pred_add['yhat_lower'],
                pred_add['yhat_upper'],
                color='red',
                alpha=0.2,
                label='Khoảng tin cậy 95%'
            )
            ax1.set_title('Mô hình Additive')
            ax1.set_xlabel('Ngày')
            ax1.set_ylabel('Giá đóng cửa')
            ax1.legend()
            
            # Vẽ dữ liệu lịch sử, dự đoán multiplicative
            ax2.plot(price_data.index, price_data.values, label='Dữ liệu lịch sử')
            ax2.plot(pred_mul.index, pred_mul['yhat'], 'g--', label='Dự đoán (Multiplicative)')
            ax2.fill_between(
                pred_mul.index,
                pred_mul['yhat_lower'],
                pred_mul['yhat_upper'],
                color='green',
                alpha=0.2,
                label='Khoảng tin cậy 95%'
            )
            ax2.set_title('Mô hình Multiplicative')
            ax2.set_xlabel('Ngày')
            ax2.set_ylabel('Giá đóng cửa')
            ax2.legend()
            
            plt.tight_layout()
            
            # Tạo nhận xét so sánh
            comparison_text = f"""
            So sánh hai chế độ seasonality:

            1. Mô hình Additive (Cộng):
               - Xu hướng được mô hình hóa bằng cách cộng các thành phần xu hướng, mùa vụ và chu kỳ
               - Phù hợp với dữ liệu có biến động ổn định
               - Các thành phần mùa vụ có độ lớn không đổi
               
            2. Mô hình Multiplicative (Nhân):
               - Xu hướng được mô hình hóa bằng cách nhân các thành phần xu hướng, mùa vụ và chu kỳ
               - Phù hợp với dữ liệu có biến động tăng dần theo thời gian (VD: giá cổ phiếu có xu hướng tăng theo cấp số nhân)
               - Các thành phần mùa vụ thay đổi tỷ lệ với xu hướng chung

            Lưu ý: Độ phù hợp của mô hình phụ thuộc vào đặc điểm dữ liệu cụ thể.
            Khuyến nghị: Thử cả hai mô hình và so sánh kết quả dự đoán với dữ liệu thực tế.
            """
            
            return fig, comparison_text
        except Exception as e:
            print(f"Lỗi khi so sánh các chế độ seasonality: {e}")
            return None, None