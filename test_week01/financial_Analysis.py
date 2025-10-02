import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from stock_Prediction import StockPredictor  # Import mô hình dự đoán

class AIAgent:
    """
    AI Agent for financial data analysis and reporting
    """
    
    def __init__(self, name="Finance Bot"):
        """Initialize the financial data agent"""
        self.name = name
        self.data = {}
        self.analysis_results = {}
        self.predictor = StockPredictor()
        print(f"{self.name} được khởi tạo và sẵn sàng phân tích.")
    
    def fetch_financial_data(self, ticker, period="1y"):
        """
        Lấy dữ liệu tài chính từ Yahoo Finance
        
        ticker: Mã chứng khoán (ví dụ: 'AAPL', 'MSFT')
        period: Khoảng thời gian ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
                      
            if hist.empty:
                print(f"Không tìm thấy dữ liệu cho mã chứng khoán: {ticker}")
                return None

            # Lấy dữ liệu tài chính cơ bản
            financial_data = {
                'balance_sheet': stock.balance_sheet,
                'income_stmt': stock.income_stmt,
                'cash_flow': stock.cashflow,
                'info': stock.info,
                'price_history': hist
            }
            
            self.data[ticker] = financial_data
            print(f"Đã lấy dữ liệu tài chính cho {ticker}")
            return financial_data
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu: {e}")
            return None
    
    def analyze_financial_ratios(self, ticker):
        """Phân tích các tỷ số tài chính cơ bản"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            financial_data = self.data[ticker]
            balance_sheet = financial_data['balance_sheet']
            income_stmt = financial_data['income_stmt']
            
            # Check if data exists
            if balance_sheet.empty or income_stmt.empty:
                print(f"Không đủ dữ liệu tài chính cho {ticker}.")
                return None
            
            # Tính toán các tỷ số tài chính
            latest_quarter = balance_sheet.columns[0]
            
            # Debug - print available keys
            print(f"Available balance sheet items for {ticker}:")
            for item in balance_sheet.index:
                print(f"  - {item}")
            
            # Find appropriate keys for financial metrics
            current_assets_keys = ['Total Current Assets', 'CurrentAssets', 'Current Assets']
            current_assets = None
            for key in current_assets_keys:
                if key in balance_sheet.index:
                    current_assets = balance_sheet.loc[key, latest_quarter]
                    break
            
            current_liabilities_keys = ['Total Current Liabilities', 'CurrentLiabilities', 'Current Liabilities']
            current_liabilities = None
            for key in current_liabilities_keys:
                if key in balance_sheet.index:
                    current_liabilities = balance_sheet.loc[key, latest_quarter]
                    break
            
            # Net Income
            net_income_keys = ['Net Income', 'NetIncome', 'Net Income Common Stockholders']
            net_income = None
            for key in net_income_keys:
                if key in income_stmt.index:
                    net_income = income_stmt.loc[key, latest_quarter]
                    break
            
            # Total Assets
            total_assets_keys = ['Total Assets', 'TotalAssets', 'Assets']
            total_assets = None
            for key in total_assets_keys:
                if key in balance_sheet.index:
                    total_assets = balance_sheet.loc[key, latest_quarter]
                    break
            
            # Total Equity
            total_equity_keys = ['Total Stockholder Equity', 'StockholdersEquity', 'Total Equity']
            total_equity = None
            for key in total_equity_keys:
                if key in balance_sheet.index:
                    total_equity = balance_sheet.loc[key, latest_quarter]
                    break
            
            # Total Liabilities
            total_liabilities_keys = ['Total Liabilities', 'TotalLiabilities', 'Liabilities']
            total_liabilities = None
            for key in total_liabilities_keys:
                if key in balance_sheet.index:
                    total_liabilities = balance_sheet.loc[key, latest_quarter]
                    break
            
            # Check if we have all necessary data
            if None in [current_assets, current_liabilities, net_income, total_assets, total_equity, total_liabilities]:
                missing = []
                if current_assets is None: missing.append("Current Assets")
                if current_liabilities is None: missing.append("Current Liabilities")
                if net_income is None: missing.append("Net Income")
                if total_assets is None: missing.append("Total Assets")
                if total_equity is None: missing.append("Total Equity")
                if total_liabilities is None: missing.append("Total Liabilities")
                
                print(f"Thiếu dữ liệu cần thiết cho {ticker}: {', '.join(missing)}")
                
                # Use fallback values if missing data
                if current_assets is None: current_assets = 1
                if current_liabilities is None: current_liabilities = 1
                if net_income is None: net_income = 0
                if total_assets is None: total_assets = 1
                if total_equity is None: total_equity = 1
                if total_liabilities is None: total_liabilities = 0
            
            # Calculate ratios
            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
            roa = net_income / total_assets if total_assets != 0 else 0
            roe = net_income / total_equity if total_equity != 0 else 0
            debt_to_assets = total_liabilities / total_assets if total_assets != 0 else 0
            
            # Lưu kết quả phân tích
            results = {
                'current_ratio': current_ratio,
                'roa': roa,
                'roe': roe,
                'debt_to_assets': debt_to_assets,
                'analysis_date': latest_quarter,
                'has_complete_data': None not in [current_assets, current_liabilities, net_income, total_assets, total_equity, total_liabilities]
            }
            
            self.analysis_results[ticker] = results
            print(f"Đã phân tích tỷ số tài chính cho {ticker}")
            return results
        except Exception as e:
            print(f"Lỗi khi phân tích tỷ số tài chính: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_financial_report(self, ticker):
        """Tạo báo cáo tài chính cơ bản"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        if ticker not in self.analysis_results:
            print(f"Chưa có kết quả phân tích cho {ticker}. Tiến hành phân tích...")
            self.analyze_financial_ratios(ticker)
        
        if ticker not in self.analysis_results:
            print(f"Không thể tạo báo cáo cho {ticker} do không có kết quả phân tích.")
            return None
        
        try:
            financial_data = self.data[ticker]
            ratios = self.analysis_results[ticker]
            
            company_info = financial_data['info']
            price_history = financial_data['price_history']
            
            # Extract company information with fallbacks
            company_name = company_info.get('longName', ticker)
            sector = company_info.get('sector', 'N/A')
            current_price = company_info.get('currentPrice', price_history['Close'].iloc[-1] if not price_history.empty else 'N/A')
            market_cap = company_info.get('marketCap', 'N/A')
            
            # Tạo báo cáo
            report = f"""
            BÁO CÁO TÀI CHÍNH: {company_name}
            ==========================================
            
            Thông tin cơ bản:
            - Tên công ty: {company_name}
            - Mã chứng khoán: {ticker}
            - Ngành: {sector}
            - Giá hiện tại: {current_price}
            - Vốn hóa thị trường: {market_cap}
            
            """
            
            if ratios['has_complete_data']:
                report += f"""
                Chỉ số tài chính chính (từ {ratios['analysis_date']}):
                - Tỷ số thanh toán hiện hành: {ratios['current_ratio']:.2f}
                - ROA (Tỷ suất sinh lời trên tài sản): {ratios['roa']*100:.2f}%
                - ROE (Tỷ suất sinh lời trên vốn chủ sở hữu): {ratios['roe']*100:.2f}%
                - Tỷ số nợ trên tài sản: {ratios['debt_to_assets']*100:.2f}%
                
                Phân tích:
                """
                
                # Phân tích tỷ số thanh toán hiện hành
                if ratios['current_ratio'] > 2:
                    report += "- Tỷ số thanh toán hiện hành cao, công ty có khả năng thanh toán tốt các khoản nợ ngắn hạn.\n"
                elif ratios['current_ratio'] > 1:
                    report += "- Tỷ số thanh toán hiện hành ở mức an toàn, công ty có thể đáp ứng các khoản nợ ngắn hạn.\n"
                else:
                    report += "- Tỷ số thanh toán hiện hành thấp, công ty có thể gặp khó khăn trong việc thanh toán các khoản nợ ngắn hạn.\n"
                
                # Phân tích ROE
                if ratios['roe'] > 0.2:
                    report += "- ROE cao, cho thấy hiệu quả sử dụng vốn tốt.\n"
                elif ratios['roe'] > 0.1:
                    report += "- ROE ở mức trung bình, hiệu quả sử dụng vốn hợp lý.\n"
                else:
                    report += "- ROE thấp, công ty cần cải thiện hiệu quả sử dụng vốn.\n"
                
                # Phân tích tỷ số nợ
                if ratios['debt_to_assets'] > 0.7:
                    report += "- Tỷ số nợ cao, công ty có rủi ro tài chính đáng kể.\n"
                elif ratios['debt_to_assets'] > 0.4:
                    report += "- Tỷ số nợ ở mức trung bình, đòn bẩy tài chính hợp lý.\n"
                else:
                    report += "- Tỷ số nợ thấp, công ty có cấu trúc vốn an toàn.\n"
            else:
                report += """
                LƯU Ý: Không đủ dữ liệu để thực hiện phân tích tài chính đầy đủ.
                Báo cáo này chỉ bao gồm thông tin cơ bản và biểu đồ giá.
                """
            
            # Thêm phân tích kỹ thuật đơn giản
            if not price_history.empty:
                # Tính các chỉ báo kỹ thuật
                if len(price_history) >= 50:
                    price_history['SMA50'] = price_history['Close'].rolling(window=50).mean()
                    report += self.analyze_sma(price_history['Close'].iloc[-1], price_history['SMA50'].iloc[-1])
                
                if len(price_history) >= 200:
                    price_history['SMA200'] = price_history['Close'].rolling(window=200).mean()
                    report += self.analyze_long_term_trend(price_history['Close'].iloc[-1], price_history['SMA200'].iloc[-1])
                
                # Price momentum
                price_change_30d = ((price_history['Close'].iloc[-1] / price_history['Close'].iloc[-30] if len(price_history) >= 30 else 1) - 1) * 100
                report += f"- Biến động giá 30 ngày: {price_change_30d:.2f}%\n"
            
            return report
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_sma(self, current_price, sma50):
        """Phân tích dựa trên đường trung bình động 50 ngày"""
        if current_price > sma50:
            return f"- Xu hướng ngắn hạn tăng: Giá hiện tại ({current_price:.2f}) cao hơn SMA50 ({sma50:.2f}).\n"
        else:
            return f"- Xu hướng ngắn hạn giảm: Giá hiện tại ({current_price:.2f}) thấp hơn SMA50 ({sma50:.2f}).\n"
    
    def analyze_long_term_trend(self, current_price, sma200):
        """Phân tích dựa trên đường trung bình động 200 ngày"""
        if current_price > sma200:
            return f"- Xu hướng dài hạn tăng: Giá hiện tại ({current_price:.2f}) cao hơn SMA200 ({sma200:.2f}).\n"
        else:
            return f"- Xu hướng dài hạn giảm: Giá hiện tại ({current_price:.2f}) thấp hơn SMA200 ({sma200:.2f}).\n"

    def save_report_to_file(self, ticker, report=None, output_dir="reports"):
        """Lưu báo cáo tài chính vào file"""
        if report is None:
            report = self.generate_financial_report(ticker)
            
        if report is None:
            print(f"Không thể lưu báo cáo cho {ticker} vì không thể tạo báo cáo.")
            return False
            
        try:
            # Đảm bảo thư mục tồn tại
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Đã tạo thư mục {output_dir}")
                
            # Tạo tên file với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{ticker}_report_{timestamp}.txt"
            
            # Ghi báo cáo vào file
            with open(filename, "w", encoding="utf-8") as file:
                file.write(report)
                
            print(f"Đã lưu báo cáo tài chính cho {ticker} vào file: {filename}")
            return filename
        except Exception as e:
            print(f"Lỗi khi lưu báo cáo vào file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_rsi(self, prices, period=14):
        """Tính RSI không sử dụng talib"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else float('inf')
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up/down if down != 0 else float('inf')
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi
    
    def plot_technical_indicators(self, ticker):
        """Vẽ biểu đồ chỉ báo kỹ thuật"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            price_data = self.data[ticker]['price_history']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None
            
            # Kiểm tra xem có dữ liệu khối lượng không
            if 'Volume' in price_data.columns and not price_data['Volume'].isnull().all():
                # Tạo biểu đồ với 3 subplot
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), 
                                                  gridspec_kw={'height_ratios': [3, 1, 1]})
                has_volume = True
            else:
                # Tạo biểu đồ với 2 subplot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                              gridspec_kw={'height_ratios': [3, 1]})
                has_volume = False
            
            # Subplot 1: Giá và các đường trung bình động
            ax1.plot(price_data.index, price_data['Close'], label='Giá đóng cửa')
            
            # Thêm SMA (Simple Moving Average)
            if len(price_data) >= 50:
                price_data['SMA50'] = price_data['Close'].rolling(window=50).mean()
                ax1.plot(price_data.index, price_data['SMA50'], label='SMA 50 ngày', color='orange')
            
            if len(price_data) >= 200:
                price_data['SMA200'] = price_data['Close'].rolling(window=200).mean()
                ax1.plot(price_data.index, price_data['SMA200'], label='SMA 200 ngày', color='red')
            
            # Thêm EMA (Exponential Moving Average)
            if len(price_data) >= 20:
                price_data['EMA20'] = price_data['Close'].ewm(span=20, adjust=False).mean()
                ax1.plot(price_data.index, price_data['EMA20'], label='EMA 20 ngày', color='purple')
            
            # Thêm dải Bollinger Band
            if len(price_data) >= 20:
                price_data['SMA20'] = price_data['Close'].rolling(window=20).mean()
                price_data['UpperBand'] = price_data['SMA20'] + (price_data['Close'].rolling(window=20).std() * 2)
                price_data['LowerBand'] = price_data['SMA20'] - (price_data['Close'].rolling(window=20).std() * 2)
                
                ax1.plot(price_data.index, price_data['SMA20'], label='SMA 20 ngày', color='green', alpha=0.6)
                ax1.plot(price_data.index, price_data['UpperBand'], label='Upper Bollinger', color='gray', linestyle='--')
                ax1.plot(price_data.index, price_data['LowerBand'], label='Lower Bollinger', color='gray', linestyle='--')
                
                # Tô màu khoảng giữa dải Bollinger
                ax1.fill_between(price_data.index, price_data['UpperBand'], price_data['LowerBand'], color='gray', alpha=0.1)
            
            # Tính RSI bằng hàm tự viết
            close_prices = price_data['Close'].values
            price_data['RSI'] = self.calculate_rsi(close_prices, period=14)
            
            # Subplot 2: RSI
            ax2.plot(price_data.index, price_data['RSI'], label='RSI 14 ngày', color='purple')
            ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
            ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
            ax2.fill_between(price_data.index, 70, 100, color='red', alpha=0.1)
            ax2.fill_between(price_data.index, 0, 30, color='green', alpha=0.1)
            ax2.set_ylabel('RSI')
            ax2.grid(True)
            ax2.legend()
            
            # Nếu có dữ liệu khối lượng, vẽ subplot thứ 3
            if has_volume:
                ax3.bar(price_data.index, price_data['Volume'], color='blue', alpha=0.5)
                
                # Thêm đường trung bình khối lượng giao dịch
                if len(price_data) >= 20:
                    volume_ma = price_data['Volume'].rolling(window=20).mean()
                    ax3.plot(price_data.index, volume_ma, color='red', label='MA Volume 20 ngày')
                
                ax3.set_ylabel('Khối lượng')
                ax3.grid(True)
                ax3.legend()
            
            # Thiết lập tiêu đề và nhãn
            ax1.set_title(f'Phân tích kỹ thuật cho {ticker}')
            ax1.set_ylabel('Giá')
            ax1.grid(True)
            ax1.legend()
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ chỉ báo kỹ thuật: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_future_price(self, ticker, days=30):
        """Dự đoán giá cổ phiếu sử dụng mô hình Prophet"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None, None
        
        try:
            price_data = self.data[ticker]['price_history']['Close']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None, None
            
            # Sử dụng lớp StockPredictor để dự đoán
            plt_predict, future_prices = self.predictor.predict(price_data, days=days)
            
            if plt_predict is not None:
                plt_predict.suptitle(f'Dự đoán giá cổ phiếu {ticker} cho {days} ngày tới')
            
            return plt_predict, future_prices
        except Exception as e:
            print(f"Lỗi khi dự đoán giá: {e}")
            import traceback
            traceback.print_exc()
            return None, None