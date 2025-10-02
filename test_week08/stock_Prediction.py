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
    
    def predict(self, price_data, days=30, multiplicative=False):
        """
        Dự đoán giá cổ phiếu trong tương lai với Prophet với khoảng tin cậy được điều chỉnh.
        
        Parameters:
        -----------
        price_data : pandas.Series
            Dữ liệu giá đóng cửa lịch sử
        days : int, default=30
            Số ngày dự đoán
        multiplicative : bool, default=False
            Sử dụng mô hình nhân (True) hay mô hình cộng (False)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Biểu đồ dự đoán
        future_predictions : pandas.DataFrame
            Dữ liệu dự đoán với các cột điều chỉnh
        """
        try:
            # Sao chép dữ liệu gốc
            original_data = price_data.copy()
            
            # Áp dụng log-transform nếu sử dụng mô hình nhân
            is_log_transformed = False
            if multiplicative:
                if (price_data > 0).all():  # Đảm bảo không có giá trị âm
                    price_data = np.log(price_data)
                    is_log_transformed = True
                else:
                    raise ValueError("Dữ liệu chứa giá trị <= 0, không thể log-transform.")
            
            # Chuẩn bị dữ liệu cho Prophet
            prophet_data = self.prepare_data(price_data)
            
            # Khởi tạo mô hình
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95,
                seasonality_mode='multiplicative' if multiplicative else 'additive'
            )
            
            model.fit(prophet_data)
            
            # Tạo khung dự đoán cho tương lai
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            
            # Lưu trữ dự đoán gốc
            forecast['yhat_original'] = forecast['yhat'].copy()
            forecast['yhat_lower_original'] = forecast['yhat_lower'].copy()
            forecast['yhat_upper_original'] = forecast['yhat_upper'].copy()
            
            # Lấy ngày cuối cùng của dữ liệu lịch sử
            last_historical_date = prophet_data['ds'].max()
            
            # Tính số ngày kể từ ngày cuối cùng của dữ liệu lịch sử
            forecast['days_from_last'] = (forecast['ds'] - last_historical_date).dt.days
            forecast['days_from_last'] = forecast['days_from_last'].clip(lower=0)  # Đảm bảo không âm
            
            # Tạo mặt nạ cho dữ liệu trong tương lai
            future_mask = forecast['ds'] > last_historical_date
            
            # Chuyển đổi từ log và điều chỉnh khoảng tin cậy cho mô hình nhân
            if is_log_transformed:
                # Chuyển đổi từ log về giá trị thực
                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.exp(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
                
                # Hệ số mở rộng tăng dần, mạnh mẽ hơn cho mô hình nhân
                # Tăng từ 1% đến 2.5% mỗi ngày để tạo hiệu ứng mở rộng mạnh hơn
                expansion_base = 0.01  # Hệ số cơ bản 1%
                expansion_growth = 0.0005  # Tốc độ tăng hệ số mở rộng
                
                # Hệ số mở rộng tăng dần theo thời gian, đặc biệt cho những ngày xa hơn
                expansion_factor = 1 + forecast['days_from_last'] * (expansion_base + 
                                                                    forecast['days_from_last'] * expansion_growth)
                
                # Tính lại khoảng tin cậy
                mid_point = forecast['yhat']
                half_width_lower = mid_point - forecast['yhat_lower']
                half_width_upper = forecast['yhat_upper'] - mid_point
                
                # Điều chỉnh cho dữ liệu tương lai
                new_half_width_lower = half_width_lower * expansion_factor
                new_half_width_upper = half_width_upper * expansion_factor
                
                # Cập nhật khoảng tin cậy điều chỉnh
                forecast['yhat_lower_adjusted'] = mid_point - new_half_width_lower
                forecast['yhat_upper_adjusted'] = mid_point + new_half_width_upper
                # Giá trị yhat giữ nguyên cho mô hình nhân
                forecast['yhat_adjusted'] = forecast['yhat']
                
            else:
                # Đối với mô hình cộng, chúng ta vẫn cần điều chỉnh khoảng tin cậy nhưng ít hơn
                forecast['yhat_adjusted'] = forecast['yhat']
                
                # Hệ số mở rộng nhẹ hơn cho mô hình cộng
                expansion_factor_additive = 1 + forecast['days_from_last'] * 0.005  # 0.5% mỗi ngày
                
                # Tính lại khoảng tin cậy
                mid_point = forecast['yhat']
                half_width_lower = mid_point - forecast['yhat_lower']
                half_width_upper = forecast['yhat_upper'] - mid_point
                
                # Áp dụng hệ số mở rộng
                new_half_width_lower = half_width_lower * expansion_factor_additive
                new_half_width_upper = half_width_upper * expansion_factor_additive
                
                # Cập nhật khoảng tin cậy điều chỉnh
                forecast['yhat_lower_adjusted'] = mid_point - new_half_width_lower
                forecast['yhat_upper_adjusted'] = mid_point + new_half_width_upper
            
            # Lấy kết quả dự đoán cho các ngày tương lai
            future_predictions = forecast.iloc[-days:][['ds', 'yhat', 'yhat_lower', 'yhat_upper', 
                                                    'yhat_adjusted', 'yhat_lower_adjusted', 'yhat_upper_adjusted']]
            future_predictions.set_index('ds', inplace=True)
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Vẽ dữ liệu lịch sử
            ax.plot(original_data.index, original_data.values, label="Dữ liệu lịch sử", color="blue")
            
            # Vẽ đường dự đoán điều chỉnh
            ax.plot(future_predictions.index, future_predictions['yhat_adjusted'], label="Dự đoán (điều chỉnh)", 
                linestyle="dashed", color="red")
            
            # Vẽ khoảng tin cậy điều chỉnh
            ax.fill_between(
                future_predictions.index,
                future_predictions['yhat_lower_adjusted'],
                future_predictions['yhat_upper_adjusted'],
                color='orange', alpha=0.3, label="Khoảng tin cậy 95% (điều chỉnh)"
            )
            
            # Thêm dự đoán gốc để so sánh (đường đứt nét)
            ax.plot(future_predictions.index, future_predictions['yhat'], 
                label="Dự đoán (gốc)", linestyle="dotted", color="green")
            
            # Vẽ khoảng tin cậy gốc (độ mờ thấp)
            ax.fill_between(
                future_predictions.index,
                future_predictions['yhat_lower'],
                future_predictions['yhat_upper'],
                color='green', alpha=0.1, label="Khoảng tin cậy 95% (gốc)"
            )
            
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá đóng cửa")
            ax.legend()
            ax.grid(True)
            
            # Thêm tiêu đề và chú thích tùy vào loại mô hình
            if multiplicative:
                plt.figtext(0.5, 0.01, 
                        "Mô hình Nhân: Xử lý tốt hơn dữ liệu có tăng trưởng theo tỉ lệ phần trăm\n" + 
                        "Khoảng tin cậy mở rộng theo thời gian dự đoán (càng xa càng rộng)",
                        ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            else:
                plt.figtext(0.5, 0.01, 
                        "Mô hình Cộng: Xử lý tốt hơn dữ liệu có tăng trưởng tuyến tính\n" + 
                        "Khoảng tin cậy mở rộng nhẹ theo thời gian",
                        ha="center", fontsize=9, bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5})
            
            return fig, future_predictions
        except Exception as e:
            import traceback
            print(f"Lỗi trong quá trình dự đoán: {e}")
            traceback.print_exc()
            return None, None
    def plot_components(self, price_data, days=30, multiplicative=False):
        """
        Vẽ các thành phần của mô hình Prophet
        
        multiplicative: Nếu True, sử dụng mô hình nhân
        """
        try:
            # Kiểm tra và áp dụng log transform nếu cần
            is_log_transformed = False
            if multiplicative:
                if isinstance(price_data.iloc[0], (float, int)) and price_data.iloc[0] > 0:
                    price_data = np.log(price_data)
                    is_log_transformed = True
            
            # Chuẩn bị dữ liệu
            prophet_data = self.prepare_data(price_data)
            
            # Tạo và huấn luyện mô hình
            if multiplicative:
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    seasonality_mode='multiplicative'  # Mô hình nhân
                )
            else:
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