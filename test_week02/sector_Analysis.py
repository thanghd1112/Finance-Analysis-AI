import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

class SectorAnalysis:
    """
    Lớp phân tích ngành và đối thủ cạnh tranh
    """
    
    def __init__(self):
        """Khởi tạo"""
        self.sector_etfs = {
            'Technology': 'XLK',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Healthcare': 'XLV',
            'Consumer Cyclical': 'XLY',
            'Consumer Defensive': 'XLP',
            'Industrial': 'XLI',
            'Basic Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC',
            'Utilities': 'XLU'
        }
    
    def get_competitors(self, ticker):
        """
        Lấy danh sách đối thủ cạnh tranh dựa trên ngành
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Lấy thông tin ngành
            sector = info.get('sector', None)
            industry = info.get('industry', None)
            
            if not sector or not industry:
                return None, None, None
            
            # Tìm các công ty trong cùng ngành
            competitors = {}
            peers = stock.info.get('recommendationKey', [])
            
            # Tạo danh sách ngành và mô tả
            industry_info = {
                'sector': sector,
                'industry': industry,
                'description': info.get('longBusinessSummary', ''),
                'company_name': info.get('longName', ticker)
            }
            
            # Lấy dữ liệu ETF ngành nếu có
            sector_etf = self.sector_etfs.get(sector, None)
            if sector_etf:
                etf = yf.Ticker(sector_etf)
                etf_hist = etf.history(period="1y")
                industry_data = etf_hist['Close'] if not etf_hist.empty else None
            else:
                industry_data = None
            
            return industry_info, peers, industry_data
        
        except Exception as e:
            print(f"Lỗi khi lấy thông tin đối thủ cạnh tranh: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def analyze_sector_performance(self, ticker, period="1y"):
        """
        Phân tích hiệu suất so với ngành
        """
        try:
            # Lấy thông tin ngành
            industry_info, peers, industry_data = self.get_competitors(ticker)
            
            if not industry_info:
                return None, None
            
            # Lấy dữ liệu cổ phiếu
            stock = yf.Ticker(ticker)
            stock_hist = stock.history(period=period)
            
            if stock_hist.empty:
                return None, None
            
            stock_data = stock_hist['Close']
            
            # So sánh với ngành nếu có dữ liệu
            if industry_data is not None and not industry_data.empty:
                # Chuẩn hóa giá để dễ so sánh
                start_price_stock = stock_data.iloc[0]
                normalized_stock = (stock_data / start_price_stock) * 100
                
                start_price_industry = industry_data.iloc[0]
                normalized_industry = (industry_data / start_price_industry) * 100
                
                # Tính lợi nhuận
                stock_return = ((stock_data.iloc[-1] / stock_data.iloc[0]) - 1) * 100
                industry_return = ((industry_data.iloc[-1] / industry_data.iloc[0]) - 1) * 100
                
                # Tạo biểu đồ
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(normalized_stock.index, normalized_stock, label=f"{ticker}")
                ax.plot(normalized_industry.index, normalized_industry, label=f"Ngành {industry_info['sector']}")
                
                ax.set_title(f"So sánh {ticker} với ngành {industry_info['sector']}")
                ax.set_xlabel("Ngày")
                ax.set_ylabel("Giá chuẩn hóa (Bắt đầu = 100)")
                ax.grid(True)
                ax.legend()
                
                plt.tight_layout()
                
                # Tạo báo cáo ngắn
                outperform = stock_return > industry_return
                performance_text = f"Hiệu suất tốt hơn ngành" if outperform else "Hiệu suất kém hơn ngành"
                
                report = f"""
                PHÂN TÍCH SO SÁNH VỚI NGÀNH
                
                Công ty: {industry_info['company_name']} ({ticker})
                Ngành: {industry_info['sector']} / {industry_info['industry']}
                
                Hiệu suất {period}:
                - {ticker}: {stock_return:.2f}%
                - Ngành {industry_info['sector']}: {industry_return:.2f}%
                - Chênh lệch: {stock_return - industry_return:.2f}%
                - Kết luận: {performance_text}
                
                Mô tả công ty:
                {industry_info['description']}
                """
                
                return fig, report
            else:
                return None, f"""
                THÔNG TIN NGÀNH
                
                Công ty: {industry_info['company_name']} ({ticker})
                Ngành: {industry_info['sector']} / {industry_info['industry']}
                
                Mô tả công ty:
                {industry_info['description']}
                
                Không có đủ dữ liệu để so sánh với ngành.
                """
        
        except Exception as e:
            print(f"Lỗi khi phân tích hiệu suất ngành: {e}")
            import traceback
            traceback.print_exc()
            return None, None