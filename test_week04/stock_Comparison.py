import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class StockComparison:
    """
    Lớp so sánh hiệu suất của nhiều cổ phiếu
    """
    
    def __init__(self):
        """Khởi tạo"""
        pass
    
    def fetch_multiple_stocks(self, tickers, period="1y"):
        """
        Lấy dữ liệu của nhiều cổ phiếu cùng một lúc
        
        tickers: Danh sách các mã cổ phiếu
        period: Khoảng thời gian lấy dữ liệu
        
        Trả về: Dictionary chứa dữ liệu giá đóng cửa của các cổ phiếu
        """
        try:
            data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker] = hist['Close']
            
            return data
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu nhiều cổ phiếu: {e}")
            return {}
    
    def normalize_prices(self, price_data):
        """
        Chuẩn hóa giá để dễ so sánh (giá bắt đầu = 100)
        """
        normalized = {}
        for ticker, prices in price_data.items():
            if not prices.empty:
                start_price = prices.iloc[0]
                normalized[ticker] = (prices / start_price) * 100
        
        return normalized
    
    def calculate_returns(self, price_data):
        """
        Tính toán lợi nhuận phần trăm cho các khoảng thời gian
        """
        returns = {}
        for ticker, prices in price_data.items():
            if len(prices) < 2:
                continue
                
            last_price = prices.iloc[-1]
            first_price = prices.iloc[0]
            
            # Tính lợi nhuận
            total_return = ((last_price / first_price) - 1) * 100
            
            # Tính lợi nhuận theo các khoảng thời gian
            returns[ticker] = {
                'total': total_return
            }
            
            # 1 tháng
            if len(prices) >= 30:
                month1_return = ((last_price / prices.iloc[-30]) - 1) * 100
                returns[ticker]['1_month'] = month1_return
            
            # 3 tháng
            if len(prices) >= 90:
                month3_return = ((last_price / prices.iloc[-90]) - 1) * 100
                returns[ticker]['3_month'] = month3_return
                
            # 6 tháng
            if len(prices) >= 180:
                month6_return = ((last_price / prices.iloc[-180]) - 1) * 100
                returns[ticker]['6_month'] = month6_return
        
        return returns
    
    def plot_comparison(self, tickers, period="1y"):
        """
        Vẽ biểu đồ so sánh hiệu suất của nhiều cổ phiếu
        """
        try:
            # Lấy dữ liệu
            price_data = self.fetch_multiple_stocks(tickers, period)
            
            if not price_data:
                return None, None
                
            # Chuẩn hóa giá
            normalized_prices = self.normalize_prices(price_data)
            
            # Tính toán lợi nhuận
            returns = self.calculate_returns(price_data)
            
            # Tạo DataFrame với giá chuẩn hóa
            df = pd.DataFrame(normalized_prices)
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for ticker in normalized_prices.keys():
                ax.plot(df.index, df[ticker], label=f"{ticker}")
            
            ax.set_title("So sánh hiệu suất (Giá khởi điểm = 100)")
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá chuẩn hóa")
            ax.grid(True)
            ax.legend()
            
            plt.tight_layout()
            
            # Tạo bảng so sánh lợi nhuận
            return_data = []
            for ticker, data in returns.items():
                row = {'Mã CK': ticker, 'Tổng lợi nhuận (%)': f"{data['total']:.2f}"}
                
                if '1_month' in data:
                    row['1 tháng (%)'] = f"{data['1_month']:.2f}"
                if '3_month' in data:
                    row['3 tháng (%)'] = f"{data['3_month']:.2f}"
                if '6_month' in data:
                    row['6 tháng (%)'] = f"{data['6_month']:.2f}"
                
                return_data.append(row)
            
            return_df = pd.DataFrame(return_data)
            
            return fig, return_df
        
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ so sánh: {e}")
            import traceback
            traceback.print_exc()
            return None, None