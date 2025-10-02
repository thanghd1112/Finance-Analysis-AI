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
        Dự đoán giá cổ phiếu trong tương lai với Prophet.
        Với điều chỉnh khoảng tin cậy phù hợp với từng mô hình.
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
            
            # Tạo cột để đánh dấu điểm bắt đầu dự báo
            last_historical_date = prophet_data['ds'].max()
            forecast['is_prediction'] = forecast['ds'] > last_historical_date
            forecast['days_from_last'] = (forecast['ds'] - last_historical_date).dt.days
            forecast['days_from_last'] = forecast['days_from_last'].clip(lower=0)
            
            # Thêm các cột điều chỉnh
            forecast['yhat_adjusted'] = forecast['yhat'].copy()
            forecast['yhat_lower_adjusted'] = forecast['yhat_lower'].copy()
            forecast['yhat_upper_adjusted'] = forecast['yhat_upper'].copy()
            
            # Điều chỉnh khoảng tin cậy dựa trên loại mô hình
            if is_log_transformed:
                # Chuyển đổi kết quả dự đoán từ log về giá trị thực
                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.exp(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
                forecast[['yhat_adjusted', 'yhat_lower_adjusted', 'yhat_upper_adjusted']] = np.exp(forecast[['yhat_adjusted', 'yhat_lower_adjusted', 'yhat_upper_adjusted']])
                
                # Điều chỉnh cho mô hình nhân
                # Khoảng tin cậy mở rộng theo thời gian và theo giá trị của dự đoán
                future_mask = forecast['is_prediction']
                
                # Hệ số mở rộng tăng dần theo thời gian - cấp số nhân
                # Với mô hình nhân, chúng ta sử dụng cả tốc độ tăng trưởng theo ngày 
                # và theo giá trị dự đoán
                base_expansion = 1.0
                time_factor = 0.015  # 1.5% mở rộng mỗi ngày
                
                # Tính hệ số mở rộng tổng hợp
                expansion_factor = base_expansion + (forecast['days_from_last'] * time_factor)
                
                # Tính toán độ rộng khoảng tin cậy mới dựa trên giá trị dự đoán và thời gian
                mid_point = forecast.loc[future_mask, 'yhat_adjusted']
                
                # Tính lại độ lệch từ điểm giữa
                current_half_width_lower = mid_point - forecast.loc[future_mask, 'yhat_lower_adjusted']
                current_half_width_upper = forecast.loc[future_mask, 'yhat_upper_adjusted'] - mid_point
                
                # Áp dụng hệ số mở rộng cho độ lệch
                new_half_width_lower = current_half_width_lower * expansion_factor.loc[future_mask]
                new_half_width_upper = current_half_width_upper * expansion_factor.loc[future_mask]
                
                # Cập nhật khoảng tin cậy mới
                forecast.loc[future_mask, 'yhat_lower_adjusted'] = (mid_point - new_half_width_lower).clip(lower=0)  # Đảm bảo không âm
                forecast.loc[future_mask, 'yhat_upper_adjusted'] = mid_point + new_half_width_upper
            else:
                # Điều chỉnh cho mô hình cộng
                # Khoảng tin cậy mở rộng tuyến tính nhưng chậm hơn
                future_mask = forecast['is_prediction']
                
                # Hệ số mở rộng tăng dần theo thời gian - tuyến tính
                time_factor = 0.005  # 0.5% mở rộng mỗi ngày
                expansion_factor = 1.0 + (forecast['days_from_last'] * time_factor)
                
                # Tính lại khoảng tin cậy
                mid_point = forecast.loc[future_mask, 'yhat_adjusted']
                current_half_width_lower = mid_point - forecast.loc[future_mask, 'yhat_lower_adjusted']
                current_half_width_upper = forecast.loc[future_mask, 'yhat_upper_adjusted'] - mid_point
                
                # Áp dụng hệ số mở rộng
                new_half_width_lower = current_half_width_lower * expansion_factor.loc[future_mask]
                new_half_width_upper = current_half_width_upper * expansion_factor.loc[future_mask]
                
                # Cập nhật khoảng tin cậy
                forecast.loc[future_mask, 'yhat_lower_adjusted'] = mid_point - new_half_width_lower
                forecast.loc[future_mask, 'yhat_upper_adjusted'] = mid_point + new_half_width_upper
            
            # Lấy kết quả dự đoán
            future_predictions = forecast.iloc[-days:][['ds', 'yhat_adjusted', 'yhat_lower_adjusted', 'yhat_upper_adjusted', 'days_from_last']]
            future_predictions.rename(columns={
                'yhat_adjusted': 'Dự đoán',
                'yhat_lower_adjusted': 'Giá thấp (95%)',
                'yhat_upper_adjusted': 'Giá cao (95%)',
                'days_from_last': 'Ngày từ hiện tại'
            }, inplace=True)
            future_predictions.set_index('ds', inplace=True)
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Vẽ dữ liệu lịch sử
            ax.plot(original_data.index, original_data.values, label="Dữ liệu lịch sử", color="blue")
            
            # Vẽ đường dự đoán
            ax.plot(future_predictions.index, future_predictions['Dự đoán'], label="Dự đoán", linestyle="dashed", color="red")
            
            # Vẽ khoảng tin cậy
            ax.fill_between(
                future_predictions.index,
                future_predictions['Giá thấp (95%)'],
                future_predictions['Giá cao (95%)'],
                color='orange', alpha=0.3, label="Khoảng tin cậy 95%"
            )
            
            # Chỉnh sửa trục x để hiển thị ngày rõ ràng hơn
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá đóng cửa")
            ax.legend()
            ax.grid(True)
            
            # Tính toán thống kê về độ rộng khoảng tin cậy để hiển thị
            ci_width_first = future_predictions['Giá cao (95%)'].iloc[0] - future_predictions['Giá thấp (95%)'].iloc[0]
            ci_width_last = future_predictions['Giá cao (95%)'].iloc[-1] - future_predictions['Giá thấp (95%)'].iloc[-1]
            ci_expansion = (ci_width_last / ci_width_first) - 1
            
            # Lấy giá trị đầu tiên và cuối cùng để minh họa sự thay đổi
            first_day_range = f"{future_predictions['Giá thấp (95%)'].iloc[0]:.2f} - {future_predictions['Giá cao (95%)'].iloc[0]:.2f}"
            last_day_range = f"{future_predictions['Giá thấp (95%)'].iloc[-1]:.2f} - {future_predictions['Giá cao (95%)'].iloc[-1]:.2f}"
            
            # Thêm tiêu đề và thông tin tùy thuộc vào loại mô hình
            if multiplicative:
                subtitle = (
                    f"Mô hình Nhân: Xử lý tốt hơn dữ liệu có tăng trưởng theo tỉ lệ phần trăm\n"
                    f"Khoảng tin cậy mở rộng theo thời gian: Tăng {ci_expansion*100:.1f}% sau {days} ngày\n"
                    f"Ngày 1: {first_day_range} | Ngày {days}: {last_day_range}"
                )
                plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9, 
                          bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
            else:
                subtitle = (
                    f"Mô hình Cộng: Xử lý tốt hơn dữ liệu có tăng trưởng tuyến tính\n"
                    f"Khoảng tin cậy mở rộng nhẹ theo thời gian: Tăng {ci_expansion*100:.1f}% sau {days} ngày\n"
                    f"Ngày 1: {first_day_range} | Ngày {days}: {last_day_range}"
                )
                plt.figtext(0.5, 0.01, subtitle, ha="center", fontsize=9, 
                          bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5})
            
            # Thêm thông tin về biên độ dao động
            for i in [0, min(7, days-1), min(14, days-1), days-1]:  # Các mốc ngày để hiển thị
                if i < len(future_predictions):
                    dt = future_predictions.index[i].strftime('%Y-%m-%d')
                    low = future_predictions['Giá thấp (95%)'].iloc[i]
                    high = future_predictions['Giá cao (95%)'].iloc[i]
                    pred = future_predictions['Dự đoán'].iloc[i]
                    width = high - low
                    percent = width / pred * 100
                    print(f"Ngày {dt}: Dự đoán {pred:.2f}, Biên độ dao động: {low:.2f} - {high:.2f} ({percent:.1f}%)")
            
            return fig, future_predictions
            
        except Exception as e:
            print(f"Lỗi khi dự đoán giá: {e}")
            import traceback
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