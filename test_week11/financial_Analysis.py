import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os
from stock_Prediction import StockPredictor  # Import mô hình dự đoán
import time
import jinja2
import pdfkit  
import base64
from io import BytesIO

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
            # Ensure ticker exists
            stock = yf.Ticker(ticker)
            
            print(f"Đang lấy dữ liệu cho mã {ticker}...")
            
            # Get historical data
            hist = stock.history(period=period)
            
            if hist.empty:
                print(f"Không tìm thấy dữ liệu lịch sử cho mã chứng khoán: {ticker}")
                return None

            # Get basic financial statements
            try:
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                cashflow = stock.cashflow
            except Exception as e:
                print(f"Cảnh báo: Không thể lấy báo cáo tài chính cho {ticker}: {e}")
                balance_sheet = pd.DataFrame()
                income_stmt = pd.DataFrame()
                cashflow = pd.DataFrame()
            
            # Get company info
            try:
                info = stock.info
            except Exception as e:
                print(f"Cảnh báo: Không thể lấy thông tin công ty cho {ticker}: {e}")
                info = {}

            # Kiểm tra dữ liệu sau khi đã gán giá trị
            print("🔍 Kiểm tra dữ liệu:")
            if not balance_sheet.empty:
                print("📌 Các mục trong Balance Sheet:", balance_sheet.index.tolist())
            else:
                print("⚠️ Không có dữ liệu Balance Sheet")
            
            if not income_stmt.empty:
                print("📌 Các mục trong Income Statement:", income_stmt.index.tolist())
            else:
                print("⚠️ Không có dữ liệu Income Statement")

            if not cashflow.empty:
                print("📌 Các mục trong Cash Flow:", cashflow.index.tolist())
            else:
                print("⚠️ Không có dữ liệu Cash Flow")

            # Save data
            financial_data = {
                'balance_sheet': balance_sheet,
                'income_stmt': income_stmt,
                'cash_flow': cashflow,
                'info': info,
                'price_history': hist
            }
            
            self.data[ticker] = financial_data
            print(f"✅ Đã lấy dữ liệu tài chính cho {ticker}")

            return financial_data
        except Exception as e:
            print(f"❌ Lỗi khi lấy dữ liệu cho {ticker}: {e}")
            import traceback
            traceback.print_exc()
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
                print(f"Không đủ dữ liệu tài chính cho {ticker}, sẽ tính toán các tỷ số có thể.")
            
            # Get the latest quarter available
            latest_quarter = None
            if not balance_sheet.empty:
                latest_quarter = balance_sheet.columns[0]
                print(f"Đang phân tích số liệu từ: {latest_quarter}")
            
            # Initialize with default values
            current_assets = None
            current_liabilities = None
            net_income = None
            total_assets = None
            total_equity = None
            total_liabilities = None
            
            # Function to find the closest matching key
            def find_closest_key(possible_keys, index):
                for key in possible_keys:
                    for idx in index:
                        if key.lower() in idx.lower():
                            return idx
                return None
            
            # Try to get financial data from balance sheet if available
            if not balance_sheet.empty and latest_quarter:
                # Current Assets
                current_assets_keys = ['Total Current Assets', 'CurrentAssets', 'Current Assets', 'TotalCurrentAssets']
                key = find_closest_key(current_assets_keys, balance_sheet.index)
                if key:
                    current_assets = balance_sheet.loc[key, latest_quarter]
                
                # Current Liabilities
                current_liabilities_keys = ['Total Current Liabilities', 'CurrentLiabilities', 'Current Liabilities', 'TotalCurrentLiabilities']
                key = find_closest_key(current_liabilities_keys, balance_sheet.index)
                if key:
                    current_liabilities = balance_sheet.loc[key, latest_quarter]
                
                # Total Assets
                total_assets_keys = ['Total Assets', 'TotalAssets', 'Assets']
                key = find_closest_key(total_assets_keys, balance_sheet.index)
                if key:
                    total_assets = balance_sheet.loc[key, latest_quarter]
                
                # Total Equity
                total_equity_keys = ['Total Stockholder Equity', 'StockholdersEquity', 'Total Equity', 'TotalStockholderEquity']
                key = find_closest_key(total_equity_keys, balance_sheet.index)
                if key:
                    total_equity = balance_sheet.loc[key, latest_quarter]
                
                # Total Liabilities
                total_liabilities_keys = ['Total Liabilities', 'TotalLiabilities', 'Liabilities']
                key = find_closest_key(total_liabilities_keys, balance_sheet.index)
                if key:
                    total_liabilities = balance_sheet.loc[key, latest_quarter]
            
            # Try to get net income from income statement if available
            if not income_stmt.empty and latest_quarter:
                # Net Income
                net_income_keys = ['Net Income', 'NetIncome', 'Net Income Common Stockholders', 'NetIncomeCommonStockholders']
                key = find_closest_key(net_income_keys, income_stmt.index)
                if key:
                    net_income = income_stmt.loc[key, latest_quarter]
            
            # Set default values for missing data
            has_complete_data = True
            missing = []
            
            if current_assets is None: 
                missing.append("Current Assets")
                current_assets = 1
                has_complete_data = False
                
            if current_liabilities is None: 
                missing.append("Current Liabilities")
                current_liabilities = 1
                has_complete_data = False
                
            if net_income is None: 
                missing.append("Net Income")
                net_income = 0
                has_complete_data = False
                
            if total_assets is None: 
                missing.append("Total Assets")
                total_assets = 1
                has_complete_data = False
                
            if total_equity is None: 
                missing.append("Total Equity")
                total_equity = 1
                has_complete_data = False
                
            if total_liabilities is None: 
                missing.append("Total Liabilities")
                total_liabilities = 0
                has_complete_data = False
            
            if missing:
                print(f"Thiếu dữ liệu cho {ticker}: {', '.join(missing)}")
            
            # Calculate ratios
            current_ratio = current_assets / current_liabilities if current_liabilities != 0 else 0
            roa = net_income / total_assets if total_assets != 0 else 0
            roe = net_income / total_equity if total_equity != 0 else 0
            debt_to_assets = total_liabilities / total_assets if total_assets != 0 else 0
            
            # Save analysis results
            results = {
                'current_ratio': current_ratio,
                'roa': roa,
                'roe': roe,
                'debt_to_assets': debt_to_assets,
                'analysis_date': latest_quarter,
                'has_complete_data': has_complete_data
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
        """Tạo báo cáo tài chính với giao diện HTML đẹp mắt"""
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
            from jinja2 import Template
            import os
            from datetime import datetime
            
            # Đọc template HTML
            template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
            with open(template_path, 'r', encoding='utf-8') as file:
                template_content = file.read()
            
            template = Template(template_content)
            
            financial_data = self.data[ticker]
            ratios = self.analysis_results[ticker]
            
            # Ensure company_info is retrieved correctly
            company_info = financial_data.get('info', {})
            if not company_info:
                print(f"Cảnh báo: Không thể lấy thông tin công ty cho {ticker}.")
                company_info = {}
            
            price_history = financial_data['price_history']
            
            # Extract company information with fallbacks
            company_name = company_info.get('longName', ticker)
            sector = company_info.get('sector', 'N/A')
            current_price = company_info.get('currentPrice', price_history['Close'].iloc[-1] if not price_history.empty else 'N/A')
            market_cap = company_info.get('marketCap', 'N/A')
            
            # Format market cap to be more readable
            if isinstance(market_cap, (int, float)) and market_cap != 'N/A':
                if market_cap >= 1_000_000_000_000:  # Trillion
                    market_cap = f"{market_cap/1_000_000_000_000:.2f} T"
                elif market_cap >= 1_000_000_000:  # Billion
                    market_cap = f"{market_cap/1_000_000_000:.2f} B"
                elif market_cap >= 1_000_000:  # Million
                    market_cap = f"{market_cap/1_000_000:.2f} M"
                else:
                    market_cap = f"{market_cap:,.0f}"
            
            # Định dạng giá hiện tại
            if isinstance(current_price, (int, float)) and current_price != 'N/A':
                current_price = f"{current_price:.2f}"
            
            # Chuẩn bị dữ liệu phân tích tài chính
            analysis_items = []
            technical_items = []
            
            if ratios['has_complete_data']:
                # Format financial ratios
                current_ratio = f"{ratios['current_ratio']:.2f}"
                roa = f"{ratios['roa']*100:.2f}%"
                roe = f"{ratios['roe']*100:.2f}%"
                debt_to_assets = f"{ratios['debt_to_assets']*100:.2f}%"
                
                # Phân tích tỷ số thanh toán hiện hành
                if ratios['current_ratio'] > 2:
                    analysis_items.append("Tỷ số thanh toán hiện hành cao, công ty có khả năng thanh toán tốt các khoản nợ ngắn hạn.")
                elif ratios['current_ratio'] > 1:
                    analysis_items.append("Tỷ số thanh toán hiện hành ở mức an toàn, công ty có thể đáp ứng các khoản nợ ngắn hạn.")
                else:
                    analysis_items.append("Tỷ số thanh toán hiện hành thấp, công ty có thể gặp khó khăn trong việc thanh toán các khoản nợ ngắn hạn.")
                
                # Phân tích ROE
                if ratios['roe'] > 0.2:
                    analysis_items.append("ROE cao, cho thấy hiệu quả sử dụng vốn tốt.")
                elif ratios['roe'] > 0.1:
                    analysis_items.append("ROE ở mức trung bình, hiệu quả sử dụng vốn hợp lý.")
                else:
                    analysis_items.append("ROE thấp, công ty cần cải thiện hiệu quả sử dụng vốn.")
                
                # Phân tích tỷ số nợ
                if ratios['debt_to_assets'] > 0.7:
                    analysis_items.append("Tỷ số nợ cao, công ty có rủi ro tài chính đáng kể.")
                elif ratios['debt_to_assets'] > 0.4:
                    analysis_items.append("Tỷ số nợ ở mức trung bình, đòn bẩy tài chính hợp lý.")
                else:
                    analysis_items.append("Tỷ số nợ thấp, công ty có cấu trúc vốn an toàn.")
            else:
                current_ratio = "N/A"
                roa = "N/A"
                roe = "N/A"
                debt_to_assets = "N/A"
            
            # Thêm phân tích kỹ thuật
            if not price_history.empty:
                # Tính các chỉ báo kỹ thuật
                if len(price_history) >= 50:
                    price_history['SMA50'] = price_history['Close'].rolling(window=50).mean()
                    technical_items.append(
                        self.analyze_sma(price_history['Close'].iloc[-1], price_history['SMA50'].iloc[-1]).strip()
                    )
                
                if len(price_history) >= 200:
                    price_history['SMA200'] = price_history['Close'].rolling(window=200).mean()
                    technical_items.append(
                        self.analyze_long_term_trend(price_history['Close'].iloc[-1], price_history['SMA200'].iloc[-1]).strip()
                    )
                
                # Price momentum
                if len(price_history) >= 30:
                    price_change_30d = ((price_history['Close'].iloc[-1] / price_history['Close'].iloc[-30]) - 1) * 100
                    technical_items.append(f"Biến động giá 30 ngày: {price_change_30d:.2f}%")
            
            
            # Prepare detailed financial analysis
            financial_analysis = []
            if ratios['current_ratio'] > 2:
                financial_analysis.append("Công ty có khả năng thanh toán tốt các khoản nợ ngắn hạn.")
            elif ratios['current_ratio'] > 1:
                financial_analysis.append("Công ty có thể đáp ứng các khoản nợ ngắn hạn.")
            else:
                financial_analysis.append("Công ty có thể gặp khó khăn trong việc thanh toán các khoản nợ ngắn hạn.")
            
            if ratios['roe'] > 0.2:
                financial_analysis.append("Hiệu quả sử dụng vốn tốt.")
            elif ratios['roe'] > 0.1:
                financial_analysis.append("Hiệu quả sử dụng vốn hợp lý.")
            else:
                financial_analysis.append("Cần cải thiện hiệu quả sử dụng vốn.")
            
            if ratios['debt_to_assets'] > 0.7:
                financial_analysis.append("Rủi ro tài chính đáng kể.")
            elif ratios['debt_to_assets'] > 0.4:
                financial_analysis.append("Đòn bẩy tài chính hợp lý.")
            else:
                financial_analysis.append("Cấu trúc vốn an toàn.")
            
           
            
            # Generate price and volume chart
            chart_image_base64 = self.plot_price_volume_chart(ticker)
            technical_chart_base64 = self.plot_technical_indicators(ticker)
            
            # Format the analysis date
            analysis_date = ratios['analysis_date']
            if analysis_date:
                analysis_date_start = analysis_date.strftime("%Y-%m-%d")
                # Assuming the data is for a quarter, calculate the end date
                analysis_date_end = (analysis_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
            else:
                analysis_date_start = "N/A"
                analysis_date_end = "N/A"
            
            # Render HTML template with data
            report_html = template.render(
                company_name=company_name,
                ticker=ticker,
                sector=sector,
                current_price=current_price,
                market_cap=market_cap,
                company_info=company_info,
                has_complete_data=ratios['has_complete_data'],
                analysis_date_start=analysis_date_start,
                analysis_date_end=analysis_date_end,
                current_ratio=current_ratio,
                roa=roa,
                roe=roe,
                debt_to_assets=debt_to_assets,
                financial_analysis="; ".join(financial_analysis),
                analysis_items=analysis_items,
                technical_items=technical_items,
                chart_image=f"data:image/png;base64,{chart_image_base64}",
                technical_chart=f"data:image/png;base64,{technical_chart_base64}",

                price_change_30d=price_change_30d,
                report_date=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
            
            return report_html
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_sma(self, current_price, sma50):
        """Phân tích dựa trên đường trung bình động 50 ngày"""
        if current_price > sma50:
            return f"Xu hướng ngắn hạn tăng: Giá hiện tại ({current_price:.2f}) cao hơn SMA50 ({sma50:.2f}).\n"
        else:
            return f"Xu hướng ngắn hạn giảm: Giá hiện tại ({current_price:.2f}) thấp hơn SMA50 ({sma50:.2f}).\n"
    
    def analyze_long_term_trend(self, current_price, sma200):
        """Phân tích dựa trên đường trung bình động 200 ngày"""
        if current_price > sma200:
            return f"Xu hướng dài hạn tăng: Giá hiện tại ({current_price:.2f}) cao hơn SMA200 ({sma200:.2f}).\n"
        else:
            return f"Xu hướng dài hạn giảm: Giá hiện tại ({current_price:.2f}) thấp hơn SMA200 ({sma200:.2f}).\n"

    def save_report_to_file(self, ticker, report=None, output_dir="reports"):
        """Lưu báo cáo tài chính vào file HTML"""
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
            
            # Tạo thư mục assets để lưu CSS và JS
            assets_dir = os.path.join(output_dir, "assets")
            if not os.path.exists(assets_dir):
                os.makedirs(assets_dir)
            
            # Copy CSS và JS vào thư mục assets nếu chưa có
            css_source = os.path.join(os.path.dirname(__file__), 'style.css')
            css_dest = os.path.join(assets_dir, 'style.css')
            
            js_source = os.path.join(os.path.dirname(__file__), 'report.js')
            js_dest = os.path.join(assets_dir, 'report.js')
            
         
            # Tạo tên file với timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"{ticker}_report_{timestamp}.html")

            
            
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
        """Calculate RSI without using talib"""
        if len(prices) <= period:
            print(f"Không đủ dữ liệu để tính RSI (cần ít nhất {period+1} điểm dữ liệu)")
            return np.array([np.nan] * len(prices))
            
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
    
    def fig_to_base64(self, fig):
        """Convert a Matplotlib figure to a base64-encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def plot_price_volume_chart(self, ticker):
        """Draw price and volume chart and return as base64 string."""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            price_data = self.data[ticker]['price_history']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None
            
            # Create chart with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                            gridspec_kw={'height_ratios': [3, 1]})
            
            # Subplot 1: Price
            ax1.plot(price_data.index, price_data['Close'], label='Close Price', color='blue')
            ax1.set_title(f'Price Chart for {ticker}')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Subplot 2: Volume
            ax2.bar(price_data.index, price_data['Volume'], color='gray', alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Convert figure to base64
            return self.fig_to_base64(fig)
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ giá và khối lượng: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def plot_technical_indicators(self, ticker):
        """Draw technical indicator charts and return as base64 string."""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            price_data = self.data[ticker]['price_history']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None
            
            # Create chart with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), 
                                                  gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Subplot 1: Price and moving averages
            ax1.plot(price_data.index, price_data['Close'], label='Close Price')
            
            # Add SMA (Simple Moving Average)
            if len(price_data) >= 50:
                price_data['SMA50'] = price_data['Close'].rolling(window=50).mean()
                ax1.plot(price_data.index, price_data['SMA50'], label='SMA 50 days', color='orange')
            
            if len(price_data) >= 200:
                price_data['SMA200'] = price_data['Close'].rolling(window=200).mean()
                ax1.plot(price_data.index, price_data['SMA200'], label='SMA 200 days', color='red')
            
            # Add EMA (Exponential Moving Average)
            if len(price_data) >= 20:
                price_data['EMA20'] = price_data['Close'].ewm(span=20, adjust=False).mean()
                ax1.plot(price_data.index, price_data['EMA20'], label='EMA 20 days', color='purple')
            
            # Add Bollinger Bands
            if len(price_data) >= 20:
                price_data['SMA20'] = price_data['Close'].rolling(window=20).mean()
                price_data['UpperBand'] = price_data['SMA20'] + (price_data['Close'].rolling(window=20).std() * 2)
                price_data['LowerBand'] = price_data['SMA20'] - (price_data['Close'].rolling(window=20).std() * 2)
                
                ax1.plot(price_data.index, price_data['SMA20'], label='SMA 20 days', color='green', alpha=0.6)
                ax1.plot(price_data.index, price_data['UpperBand'], label='Upper Bollinger', color='gray', linestyle='--')
                ax1.plot(price_data.index, price_data['LowerBand'], label='Lower Bollinger', color='gray', linestyle='--')
                
                # Fill area between Bollinger Bands
                ax1.fill_between(price_data.index, price_data['UpperBand'], price_data['LowerBand'], color='gray', alpha=0.1)
            
            # Calculate RSI
            close_prices = price_data['Close'].values
            price_data['RSI'] = self.calculate_rsi(close_prices, period=14)
            
            # Subplot 2: RSI
            ax2.plot(price_data.index, price_data['RSI'], label='RSI 14 days', color='purple')
            ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
            ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
            ax2.fill_between(price_data.index, 70, 100, color='red', alpha=0.1)
            ax2.fill_between(price_data.index, 0, 30, color='green', alpha=0.1)
            ax2.set_ylabel('RSI')
            ax2.grid(True)
            ax2.legend()
            
            # If volume data exists, draw subplot 3
            if 'Volume' in price_data.columns and not price_data['Volume'].isnull().all():
                ax3.bar(price_data.index, price_data['Volume'], color='blue', alpha=0.5)
                
                # Add volume moving average
                if len(price_data) >= 20:
                    volume_ma = price_data['Volume'].rolling(window=20).mean()
                    ax3.plot(price_data.index, volume_ma, color='red', label='Volume MA 20 days')
                
                ax3.set_ylabel('Volume')
                ax3.grid(True)
                ax3.legend()
            
            # Set title and labels
            ax1.set_title(f'Technical Analysis for {ticker}')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            plt.tight_layout()
            
            # Convert figure to base64
            return self.fig_to_base64(fig)
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ chỉ báo kỹ thuật: {e}")
            import traceback
            traceback.print_exc()
            return None
    def predict_future_price(self, ticker, days=30, multiplicative=False):
        """
        Dự đoán giá cổ phiếu sử dụng mô hình Prophet với khoảng tin cậy điều chỉnh
        
        Parameters:
        -----------
        ticker : str
            Mã cổ phiếu cần dự đoán
        days : int, default=30
            Số ngày dự đoán
        multiplicative : bool, default=False
            Sử dụng mô hình nhân (True) hay mô hình cộng (False)
            
        Returns:
        --------
        plt_predict : matplotlib.figure.Figure
            Biểu đồ dự đoán
        future_prices : pandas.DataFrame
            Dữ liệu dự đoán với các cột điều chỉnh
        """
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None, None
        
        try:
            price_data = self.data[ticker]['price_history']['Close']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None, None
            
            # Sử dụng StockPredictor để dự đoán
            plt_predict, future_prices = self.predictor.predict(price_data, days=days, multiplicative=multiplicative)
            
            if plt_predict is not None:
                model_type = "Nhân" if multiplicative else "Cộng"
                plt_predict.suptitle(f'Dự đoán giá cổ phiếu {ticker} cho {days} ngày tới (Mô hình {model_type})')
                
                # Thêm chú thích để phân biệt mô hình
                if multiplicative:
                    plt.figtext(0.5, 0.01, 
                            "Mô hình Nhân: Xử lý tốt hơn dữ liệu có tăng trưởng theo tỉ lệ phần trăm\n" + 
                            "Khoảng tin cậy mở rộng theo thời gian (càng xa càng rộng)",
                            ha="center", fontsize=9, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
                else:
                    plt.figtext(0.5, 0.01, 
                            "Mô hình Cộng: Xử lý tốt hơn dữ liệu có tăng trưởng tuyến tính\n" + 
                            "Khoảng tin cậy mở rộng nhẹ theo thời gian",
                            ha="center", fontsize=9, bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5})
            
            return plt_predict, future_prices
        except Exception as e:
            print(f"Lỗi khi dự đoán giá: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    def answer_question(self, ticker, question, financial_data=None):
        """
        Trả lời câu hỏi của người dùng dựa trên dữ liệu tài chính và phân tích chứng khoán.
        
        Tham số:
        - ticker (str): Mã chứng khoán
        - question (str): Câu hỏi của người dùng
        - financial_data (dict, optional): Dữ liệu tài chính đã thu thập
        
        Trả về:
        - str: Câu trả lời
        """
        # Kiểm tra nếu là câu hỏi về giới thiệu ứng dụng hoặc trợ lý
        question_lower = question.lower()
        
        # 📌 0️⃣ Câu hỏi về trợ lý và ứng dụng
        if "bạn là ai" in question_lower or "giới thiệu" in question_lower or "tính năng" in question_lower:
            return f"""Tôi là trợ lý tài chính AI, được phát triển để hỗ trợ phân tích và cung cấp thông tin về chứng khoán.
            
        Ứng dụng của chúng tôi cung cấp các tính năng:
        - Phân tích cơ bản
        - Phân tích kỹ thuật: RSI, Bollinger Bands, SMA, EMA
        - Dự đoán xu hướng giá tương lai 
        - Tự động tạo báo cáo tài chính cơ bản
        - Hỗ trợ trả lời câu hỏi về {ticker} mà bạn quan tâm

    Hãy thử nhập mã chứng khoán và đặt câu hỏi để tôi có thể giúp bạn! 🤖 
    """
        
        if "giúp gì" in question_lower or "hỗ trợ gì" in question_lower or "làm được gì" in question_lower:
            return """Tôi có thể giúp bạn những việc sau:
            
    1. Tra cứu thông tin cơ bản về mã chứng khoán (giá, khối lượng, biến động)
    2. Cung cấp phân tích kỹ thuật (RSI, MACD, Bollinger Bands, đường trung bình động)
    3. Dự đoán xu hướng giá trong tương lai
    4. Tạo báo cáo phân tích về mã chứng khoán bạn quan tâm
    5. Đưa ra nhận định về thời điểm mua/bán dựa trên các chỉ báo (tại các điểm giao nhau của chỉ các số dài hạn và ngắn hạn)
    6. Cung cấp thông tin về cổ tức, lợi nhuận của doanh nghiệp

    Để bắt đầu, hãy nhập mã chứng khoán bạn muốn phân tích! 🚀"""
        
        if "cách sử dụng" in question_lower or "hướng dẫn" in question_lower:
            return """Hướng dẫn sử dụng ứng dụng:
            
    1. Nhập mã chứng khoán bạn muốn phân tích vào ô trên thanh menu bên trái
    2. Chọn khoảng thời gian bạn muốn xem dữ liệu (1 tháng, 3 tháng, 1 năm,...)
    3. Nếu có các tab để truy cập gồm các tính năng khác nhau:
    - "Báo cáo tài chính: Xem Xem thông tin cơ bản, tỷ số tài chính, báo cáo 
    - "Phân tích kỹ thuật": Xem các chỉ báo RSI, MACD, Bollinger bands
    - "Dự đoán giá": Xem dự báo xu hướng giá trong tương lai
    - "Hỏi đáp": Đặt câu hỏi trực tiếp về mã chứng khoán

    4. Đặt câu hỏi tại tab "Hỏi đáp" để tôi có thể giúp bạn hiểu rõ hơn về mã chứng khoán 📌

    Chúc bạn có trải nghiệm tốt với ứng dụng của chúng tôi!"""
        
        # Kiểm tra nếu không có dữ liệu tài chính, thì tải về
        if financial_data is None:
            financial_data = self.fetch_financial_data(ticker)
                
        if not financial_data:
            return f"Không thể trả lời câu hỏi vì không có dữ liệu cho {ticker}."

        # 📌 1️⃣ Giá hiện tại và biến động
        if "giá hiện tại" in question_lower or "giá hôm nay" in question_lower:
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                latest_price = financial_data['price_history']['Close'].iloc[-1]
                return f"Giá đóng cửa gần nhất của {ticker} là {latest_price:.2f}."

        if "biến động" in question_lower or "tăng" in question_lower or "giảm" in question_lower:
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                prices = financial_data['price_history']['Close']
                change = prices.iloc[-1] - prices.iloc[-2]
                pct_change = (change / prices.iloc[-2]) * 100
                change_type = "tăng" if change > 0 else "giảm"
                return f"{ticker} {change_type} {abs(change):.2f} điểm ({abs(pct_change):.2f}%) trong phiên giao dịch gần nhất."

        # 📌 2️⃣ Khối lượng giao dịch
        if "khối lượng" in question_lower or "volume" in question_lower:
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                latest_volume = financial_data['price_history']['Volume'].iloc[-1]
                avg_volume = financial_data['price_history']['Volume'].mean()
                return f"Khối lượng giao dịch gần nhất của {ticker} là {latest_volume:,.0f} cổ phiếu. Trung bình khối lượng giao dịch: {avg_volume:,.0f} cổ phiếu."

        # 📌 3️⃣ Giá cao nhất / thấp nhất theo khoảng thời gian
        if "cao nhất" in question_lower or "thấp nhất" in question_lower:
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                highest_price = financial_data['price_history']['High'].max()
                lowest_price = financial_data['price_history']['Low'].min()
                return f"Giá cao nhất của {ticker} là {highest_price:.2f}, giá thấp nhất là {lowest_price:.2f} trong khoảng thời gian đã chọn."

        # 📌 4️⃣ Phân tích kỹ thuật
        if "rsi" in question_lower:
            # Tính toán RSI nếu chưa có
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                prices = financial_data['price_history']['Close'].values
                rsi_values = self.calculate_rsi(prices)
                rsi_value = rsi_values[-1]
                trend = "quá mua (overbought)" if rsi_value > 70 else "quá bán (oversold)" if rsi_value < 30 else "trung lập"
                return f"RSI của {ticker} hiện tại là {rsi_value:.2f}, cho thấy thị trường đang {trend}."

        if "macd" in question_lower:
            if 'MACD' in financial_data:
                macd_value = financial_data['MACD'].iloc[-1]
                signal_value = financial_data['MACD_signal'].iloc[-1]
                trend = "tín hiệu mua" if macd_value > signal_value else "tín hiệu bán"
                return f"MACD của {ticker} là {macd_value:.2f}, MACD Signal là {signal_value:.2f}, cho thấy {trend}."

        if "bollinger bands" in question_lower or "dải bollinger" in question_lower or "bollinger" in question_lower:
            if 'price_history' in financial_data and not financial_data['price_history'].empty:
                price_data = financial_data['price_history']
                if len(price_data) >= 20:
                    # Tính toán Bollinger Bands nếu chưa có
                    price_data['SMA20'] = price_data['Close'].rolling(window=20).mean()
                    price_data['Upper_Band'] = price_data['SMA20'] + (price_data['Close'].rolling(window=20).std() * 2)
                    price_data['Lower_Band'] = price_data['SMA20'] - (price_data['Close'].rolling(window=20).std() * 2)
                    
                    upper_band = price_data['Upper_Band'].iloc[-1]
                    middle_band = price_data['SMA20'].iloc[-1]
                    lower_band = price_data['Lower_Band'].iloc[-1]
                    current_price = price_data['Close'].iloc[-1]
                    
                    position = ""
                    if current_price > upper_band:
                        position = "vượt trên dải trên, cổ phiếu có thể đang quá mua."
                    elif current_price < lower_band:
                        position = "dưới dải dưới, cổ phiếu có thể đang quá bán."
                    else:
                        position = "nằm trong khoảng dải Bollinger, thị trường đang giao dịch ổn định."
                    
                    return f"Dải Bollinger của {ticker}: Dải trên = {upper_band:.2f}, Dải giữa = {middle_band:.2f}, Dải dưới = {lower_band:.2f}. Giá hiện tại ({current_price:.2f}) {position}"

        # 📌 5️⃣ Dự đoán giá tương lai
        if "dự đoán" in question_lower or "xu hướng" in question_lower or "dự báo" in question_lower:
            prediction_days = 30  # Dự báo 30 ngày
            prophet_fig, future_prices = self.predict_future_price(ticker, days=prediction_days)
            if future_prices is not None:
                predicted_price = future_prices['yhat_adjusted'].iloc[-1]
                current_price = financial_data['price_history']['Close'].iloc[-1]
                change = predicted_price - current_price
                pct_change = (change / current_price) * 100
                direction = "tăng" if change > 0 else "giảm"
                
                return f"Dự đoán xu hướng giá của {ticker} sau {prediction_days} ngày là {predicted_price:.2f}, {direction} {abs(pct_change):.2f}% so với giá hiện tại ({current_price:.2f})."
            else:
                return f"Không thể tạo dự đoán giá cho {ticker} lúc này."

        # 📌 6️⃣ Thông tin doanh nghiệp
        if "công ty" in question_lower or "thông tin" in question_lower or "doanh nghiệp" in question_lower:
            info = financial_data.get('info', {})
            summary = f"Thông tin về {ticker}:\n"
            summary += f"- Tên công ty: {info.get('longName', 'Không có dữ liệu')}\n"
            summary += f"- Ngành: {info.get('sector', 'Không có dữ liệu')}\n"
            summary += f"- Vốn hóa thị trường: {info.get('marketCap', 'Không có dữ liệu'):,}\n" if 'marketCap' in info else f"- Vốn hóa thị trường: Không có dữ liệu\n"
            summary += f"- Mô tả công ty: {info.get('longBusinessSummary', 'Không có dữ liệu')}\n"
            return summary
        
        # 📌 7️⃣ Phân tích cơ bản và chỉ số tài chính
        if "cổ tức" in question_lower or "dividend" in question_lower:
            dividend_yield = financial_data.get('info', {}).get('dividendYield', None)
            if dividend_yield:
                return f"Tỷ suất cổ tức của {ticker} là {dividend_yield * 100:.2f}%."
            else:
                return f"Không có dữ liệu cổ tức cho {ticker} hoặc công ty không chi trả cổ tức."
        
        if "eps" in question_lower or "thu nhập trên cổ phiếu" in question_lower:
            eps = financial_data.get('info', {}).get('trailingEps', None)
            return f"EPS hiện tại của {ticker} là {eps:.2f}." if eps else f"Không có dữ liệu EPS cho {ticker}."
        
        if "p/e" in question_lower or "pe" in question_lower:
            pe = financial_data.get('info', {}).get('trailingPE', None)
            if pe:
                return f"Tỷ số P/E hiện tại của {ticker} là {pe:.2f}. " + ("Tỷ số này tương đối cao, cổ phiếu có thể đang được định giá cao." if pe > 25 else "Tỷ số này ở mức hợp lý." if 10 <= pe <= 25 else "Tỷ số này tương đối thấp, cổ phiếu có thể đang được định giá thấp.")
            else:
                return f"Không có dữ liệu P/E cho {ticker}."
        
        if "p/b" in question_lower or "pb" in question_lower or "giá trên sổ sách" in question_lower:
            pb = financial_data.get('info', {}).get('priceToBook', None)
            if pb:
                return f"Tỷ số P/B hiện tại của {ticker} là {pb:.2f}. " + ("Tỷ số này tương đối cao, cổ phiếu có thể đang được định giá cao so với giá trị sổ sách." if pb > 3 else "Tỷ số này ở mức hợp lý." if 1 <= pb <= 3 else "Tỷ số này tương đối thấp, cổ phiếu có thể đang được định giá thấp so với giá trị sổ sách.")
            else:
                return f"Không có dữ liệu P/B cho {ticker}."
        
        # 📌 8️⃣ Cung cấp lời khuyên đầu tư
        if "nên mua" in question_lower or "nên bán" in question_lower or "khuyến nghị" in question_lower or "lời khuyên" in question_lower:
            return f"""Về {ticker}, tôi không thể đưa ra khuyến nghị đầu tư cụ thể vì các quyết định đầu tư nên dựa trên:
            
    1. Phân tích cơ bản và kỹ thuật
    2. Mục tiêu tài chính cá nhân của bạn
    3. Khả năng chịu đựng rủi ro
    4. Thời gian đầu tư dự kiến
            
    Tôi khuyên bạn nên tham khảo các báo cáo phân tích và tư vấn từ chuyên gia tài chính trước khi đưa ra quyết định."""
        
        # 📌 9️⃣ Câu hỏi về tin tức và sự kiện
        if "tin tức" in question_lower or "sự kiện" in question_lower or "tin mới" in question_lower:
            info = financial_data.get('info', {})
            return f"""Tôi không có khả năng truy cập dữ liệu tin tức thời gian thực cho {ticker}. 
            
    Để cập nhật tin tức mới nhất về công ty {info.get('longName')}, bạn nên kiểm tra:
    1. Trang web chính thức của công ty
    2. Các nền tảng tin tức tài chính như Bloomberg, CNBC, hoặc Báo Đầu tư Chứng khoán
    3. Mục tin tức trên các nền tảng giao dịch chứng khoán

    Ngoài ra, bạn cũng có thể theo dõi các thông báo chính thức từ công ty về báo cáo tài chính, chia cổ tức, hoặc các sự kiện quan trọng khác."""
        
        return f"Tôi không có đủ thông tin để trả lời câu hỏi của bạn về {ticker}. Vui lòng thử đặt câu hỏi cụ thể hơn về giá, khối lượng giao dịch, phân tích hoặc dự đoán giá cơ bản."

