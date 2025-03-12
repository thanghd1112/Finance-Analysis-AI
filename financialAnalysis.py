import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import os

class AIAgent:
    """
    AI Agent for financial data analysis and reporting
    """
    
    def __init__(self, name="Finance Bot"):
        """Initialize the financial data agent"""
        self.name = name
        self.data = {}
        self.analysis_results = {}
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
            # These may vary by stock/data source, so we'll try to find them
            
            # Current Assets and Liabilities
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
                

                if ratios['current_ratio'] > 2:
                    report += "- Tỷ số thanh toán hiện hành cao, công ty có khả năng thanh toán tốt các khoản nợ ngắn hạn.\n"
                elif ratios['current_ratio'] > 1:
                    report += "- Tỷ số thanh toán hiện hành ở mức an toàn, công ty có thể đáp ứng các khoản nợ ngắn hạn.\n"
                else:
                    report += "- Tỷ số thanh toán hiện hành thấp, công ty có thể gặp khó khăn trong việc thanh toán các khoản nợ ngắn hạn.\n"
                
                # ROE
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
            

            if not price_history.empty:
                # Calculate simple moving averages
                price_history['SMA50'] = price_history['Close'].rolling(window=50).mean()
                price_history['SMA200'] = price_history['Close'].rolling(window=200).mean()
                
                latest_close = price_history['Close'].iloc[-1]
                latest_sma50 = price_history['SMA50'].iloc[-1] if not pd.isna(price_history['SMA50'].iloc[-1]) else None
                latest_sma200 = price_history['SMA200'].iloc[-1] if not pd.isna(price_history['SMA200'].iloc[-1]) else None
                
                report += "\nPhân tích kỹ thuật:\n"
                
                # Trend analysis based on SMAs
                if latest_sma50 and latest_sma200:
                    if latest_close > latest_sma50 and latest_close > latest_sma200:
                        report += "- Xu hướng tăng: Giá hiện tại cao hơn cả SMA50 và SMA200.\n"
                    elif latest_close < latest_sma50 and latest_close < latest_sma200:
                        report += "- Xu hướng giảm: Giá hiện tại thấp hơn cả SMA50 và SMA200.\n"
                    else:
                        report += "- Xu hướng trung lập: Giá hiện tại nằm giữa SMA50 và SMA200.\n"
                
                # Price momentum
                price_change_30d = ((latest_close / price_history['Close'].iloc[-30] if len(price_history) >= 30 else 1) - 1) * 100
                report += f"- Biến động giá 30 ngày: {price_change_30d:.2f}%\n"
                
                # Volume analysis
                avg_volume = price_history['Volume'].mean() if 'Volume' in price_history.columns else 0
                latest_volume = price_history['Volume'].iloc[-1] if 'Volume' in price_history.columns else 0
                
                if avg_volume > 0 and latest_volume > 0:
                    volume_ratio = latest_volume / avg_volume
                    if volume_ratio > 1.5:
                        report += "- Khối lượng giao dịch hiện tại cao hơn đáng kể so với trung bình.\n"
                    elif volume_ratio < 0.5:
                        report += "- Khối lượng giao dịch hiện tại thấp hơn đáng kể so với trung bình.\n"
                    else:
                        report += "- Khối lượng giao dịch ở mức trung bình.\n"
            
            return report
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo: {e}")
            import traceback
            traceback.print_exc()
            return None

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
    
    def plot_price_history(self, ticker):
        """Vẽ biểu đồ lịch sử giá"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            price_data = self.data[ticker]['price_history']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None
            
            plt.figure(figsize=(12, 6))
            plt.plot(price_data.index, price_data['Close'])
            plt.title(f'Lịch sử giá cổ phiếu {ticker}')
            plt.xlabel('Ngày')
            plt.ylabel('Giá đóng cửa')
            plt.grid(True)
            plt.tight_layout()
            
            return plt
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_stocks(self, tickers, metric='Close', period='1y'):
        """So sánh nhiều cổ phiếu theo một chỉ số nhất định"""
        data_frames = []
        
        for ticker in tickers:
            if ticker not in self.data:
                self.fetch_financial_data(ticker, period)
            
            if ticker in self.data:
                price_data = self.data[ticker]['price_history']
                if not price_data.empty and metric in price_data.columns:
                    data_frames.append(price_data[metric].rename(ticker))
        
        if data_frames:
            comparison_df = pd.concat(data_frames, axis=1)
            
            # Handle NaN values
            comparison_df = comparison_df.fillna(method='ffill').fillna(method='bfill')
            
            if comparison_df.empty or comparison_df.isnull().all().all():
                print("Không đủ dữ liệu để so sánh các cổ phiếu.")
                return None
            
            normalized_df = comparison_df / comparison_df.iloc[0] * 100
            
            plt.figure(figsize=(12, 6))
            normalized_df.plot(figsize=(12, 6))
            plt.title('So sánh hiệu suất cổ phiếu (Chuẩn hóa 100)')
            plt.xlabel('Ngày')
            plt.ylabel(f'{metric} (Chuẩn hóa)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            return plt
        
        print("Không đủ dữ liệu để so sánh các cổ phiếu.")
        return None

    def predict_future_price(self, ticker, days=30):
        """Dự đoán giá cổ phiếu đơn giản sử dụng hồi quy tuyến tính"""
        if ticker not in self.data:
            print(f"Không tìm thấy dữ liệu cho {ticker}. Vui lòng lấy dữ liệu trước.")
            return None
        
        try:
            from sklearn.linear_model import LinearRegression
            
            price_data = self.data[ticker]['price_history']['Close']
            
            if price_data.empty:
                print(f"Không có dữ liệu giá cho {ticker}.")
                return None, None
            
            # Chuẩn bị dữ liệu
            X = np.array(range(len(price_data))).reshape(-1, 1)
            y = price_data.values
            
            # Tạo mô hình hồi quy
            model = LinearRegression()
            model.fit(X, y)
            
            # Dự đoán giá tương lai
            last_day = len(price_data)
            future_days = np.array(range(last_day, last_day + days)).reshape(-1, 1)
            future_prices = model.predict(future_days)
            
            # Tạo dữ liệu dự đoán
            last_date = price_data.index[-1]
            future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
            future_df = pd.Series(future_prices.flatten(), index=future_dates)
            
            # Vẽ biểu đồ
            plt.figure(figsize=(12, 6))
            plt.plot(price_data.index, price_data.values, label='Dữ liệu lịch sử')
            plt.plot(future_dates, future_prices, 'r--', label='Dự đoán')
            plt.title(f'Dự đoán giá cổ phiếu {ticker} cho {days} ngày tới')
            plt.xlabel('Ngày')
            plt.ylabel('Giá đóng cửa')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            return plt, future_df
        except Exception as e:
            print(f"Lỗi khi dự đoán giá: {e}")
            import traceback
            traceback.print_exc()
            return None, None



if __name__ == "__main__":
    finance_agent = AIAgent("FinReporter")
    
    reports_dir = "finance_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Đã tạo thư mục {reports_dir} để lưu báo cáo")
    
    # Lấy dữ liệu
    while True:
        ticker = input("Bạn muốn tìm hiểu về mã chứng khoán nào? ").strip().upper()
        if not ticker:
            print("Mã chứng khoán không được để trống. Vui lòng nhập lại!")
            continue
        
        data = finance_agent.fetch_financial_data(ticker, period="1y")
        if data:
            break
        else:
            print("Mã chứng khoán không hợp lệ hoặc không có dữ liệu. Vui lòng nhập lại!")
    
    ratios = finance_agent.analyze_financial_ratios(ticker)
    print("Các tỷ số tài chính:", ratios)
    
    # Tạo, lưu báo cáo
    report = finance_agent.generate_financial_report(ticker)
    if report:
        print("Báo cáo tài chính:")
        print(report)
        
        reports_dir = "finance_reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        finance_agent.save_report_to_file(ticker, report, reports_dir)
    
    plt_price = finance_agent.plot_price_history(ticker)
    if plt_price:
        price_chart_path = f"{reports_dir}/{ticker}_price_history.png"
        plt_price.savefig(price_chart_path)
        print(f"Đã lưu biểu đồ giá vào {price_chart_path}")
    
    # So sánh nhiều cổ phiếu
    finance_agent.fetch_financial_data("MSFT", period="1y")
    finance_agent.fetch_financial_data("GOOGL", period="1y")
    plt_compare = finance_agent.compare_stocks([ticker, "MSFT", "GOOGL"])
    if plt_compare:
        comparison_chart_path = f"{reports_dir}/stock_comparison.png"
        plt_compare.savefig(comparison_chart_path)
        print(f"Đã lưu biểu đồ so sánh vào {comparison_chart_path}")
    
    # Dự đoán
    plt_predict, future_prices = finance_agent.predict_future_price(ticker, days=30)
    if plt_predict and future_prices is not None:
        prediction_chart_path = f"{reports_dir}/{ticker}_price_prediction.png"
        plt_predict.savefig(prediction_chart_path)
        print(f"Đã lưu biểu đồ dự đoán vào {prediction_chart_path}")
        print(f"Dự đoán giá tương lai cho {future_prices.index[-1].strftime('%d/%m/%Y')}: {future_prices.iloc[-1]:.2f}")
        
        prediction_report_path = f"{reports_dir}/{ticker}_price_prediction.txt"
        with open(prediction_report_path, "w", encoding="utf-8") as file:
            file.write(f"Dự đoán giá cổ phiếu {ticker}\n")
            file.write(f"========================\n\n")
            file.write(f"Ngày dự đoán | Giá dự đoán\n")
            file.write(f"-------------|------------\n")
            for date, price in future_prices.items():
                file.write(f"{date.strftime('%d/%m/%Y')} | {price:.2f}\n")
        print(f"Đã lưu dự đoán giá vào {prediction_report_path}")