import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta
import time
from deep_translator import GoogleTranslator
from textblob import TextBlob
import nltk
from newspaper import Article
import os
import matplotlib.pyplot as plt
import dateparser

class NewsSentimentAnalyzer:
    """
    Lớp phân tích tin tức và sentiment cho dữ liệu tài chính
    """
    
    def __init__(self):
        """Khởi tạo bộ phân tích tin tức"""
        self.translator = GoogleTranslator(source='auto', target='vi')
        
        # Tải các tài nguyên cần thiết cho NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def translate_text(self, text):
        """
        Dịch văn bản sang tiếng Việt
        """
        try:
            if not text:
                return ""
                
            # Giới hạn độ dài văn bản để tránh lỗi từ API
            if len(text) > 5000:
                text = text[:5000] + "..."
                
            return self.translator.translate(text)
        except Exception as e:
            print(f"Lỗi khi dịch văn bản: {e}")
            return f"{text} (Không thể dịch)"
    
    def fetch_news_google(self, ticker, days=7, max_results=10, sources=None):
        """
        Thu thập tin tức từ Google News
        
        ticker: Mã chứng khoán
        days: Số ngày gần đây để lấy tin tức
        max_results: Số lượng tin tức tối đa
        sources: Danh sách các nguồn tin cần lấy (None = tất cả)
        """
        try:
            # Tạo URL cho Google News
            query = f"{ticker} stock"
            url = f"https://www.google.com/search?q={query}&tbm=nws&source=lnt&tbs=qdr:w"
            
            # Giả lập trình duyệt web
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # Gửi yêu cầu
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Phân tích HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tìm kiếm các bài viết tin tức
            news_results = []
            news_elements = soup.select("div.SoaBEf")

            for element in news_elements[:max_results]:
                try:
                    # Trích xuất tiêu đề
                    title_element = element.select_one("div.mCBkyc")
                    title = title_element.text if title_element else "Không có tiêu đề"
                    
                    # Trích xuất link
                    link_element = element.select_one("a")
                    link = link_element['href'] if link_element else ""
                    if link.startswith("/url?q="):
                        link = link.split("/url?q=")[1].split("&sa=")[0]
                    
                    # Trích xuất nguồn và thời gian
                    source_time = element.select_one("div.CEMjEf")
                    source_text = source_time.text if source_time else "Unknown · Không rõ thời gian"
                    
                    # Nâng cấp việc trích xuất nguồn và thời gian
                    source_parts = source_text.split(" · ")
                    source = source_parts[0] if source_parts else "Unknown"
                    
                    # Xử lý thời gian một cách mạnh mẽ hơn
                    published_time_str = source_parts[1] if len(source_parts) > 1 else "Không rõ thời gian"
                    
                    # Sử dụng dateparser để chuyển đổi chuỗi thời gian
                    try:
                        parsed_time = dateparser.parse(published_time_str, settings={'RELATIVE_BASE': datetime.now()})
                        formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S") if parsed_time else published_time_str
                    except Exception as time_parse_error:
                        formatted_time = published_time_str
                        print(f"Lỗi phân tích thời gian: {time_parse_error}")
                    
                    # Trích xuất tóm tắt
                    summary_element = element.select_one("div.GI74Re")
                    summary = summary_element.text if summary_element else "Không có tóm tắt"
                    
                    news_results.append({
                        "title": title,
                        "link": link,
                        "source": source,
                        "published": formatted_time,
                        "summary": summary
                    })
                except Exception as e:
                    print(f"Lỗi khi xử lý một tin tức: {e}")
                    continue

                # Thêm lọc theo nguồn tin (nếu có)
            if sources and len(sources) > 0:
                filtered_news = []
                for news in news_results:
                    for source in sources:
                        if source.lower() in news['source'].lower():
                            filtered_news.append(news)
                            break
                return filtered_news if filtered_news else news_results
            
            return news_results
        except Exception as e:
            print(f"Lỗi khi lấy tin từ Google News: {e}")
            return []
    
    def fetch_article_content(self, url):
        """
        Trích xuất nội dung bài báo từ URL
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            return {
                "text": article.text,
                "summary": article.summary
            }
        except Exception as e:
            print(f"Lỗi khi trích xuất nội dung bài báo: {e}")
            return {"text": "", "summary": ""}
    
    def translate_text(self, text):
        """
        Dịch văn bản sang tiếng Việt khi cần thiết
        """
        try:
            if not text or len(text) == 0:
                return ""
            
            # Chỉ dịch khi người dùng yêu cầu hoặc cần thiết
            return text  # Mặc định không dịch
        except Exception as e:
            print(f"Lỗi khi xử lý văn bản: {e}")
            return text
    
    def analyze_sentiment(self, text):
        """
        Phân tích sentiment của văn bản
        Trả về: float (từ -1 đến 1, -1 là rất tiêu cực, 1 là rất tích cực)
        """
        try:
            if not text:
                return 0.0
            
            # Loại bỏ các từ không liên quan
            stop_words = ['apple', 'stock', 'market', 'company', 'inc']
            words = [word for word in text.lower().split() if word not in stop_words]
            text_cleaned = ' '.join(words)
            
            # Phân tích sentiment
            analysis = TextBlob(text_cleaned)
            sentiment = analysis.sentiment.polarity
            
            # Điều chỉnh sentiment để nhạy cảm hơn với tin tức tài chính
            if abs(sentiment) < 0.1:
                return 0.0  # Giữ sentiment trung lập nếu rất yếu
            
            return sentiment
        except Exception as e:
            print(f"Lỗi khi phân tích sentiment: {e}")
            return 0.0
    
    def analyze_news_sentiment(self, ticker, max_news=10, translate=False, sources=None):
        """
        Phân tích tin tức và sentiment cho một mã chứng khoán
        """
        try:
            # Thu thập tin tức với số lượng được chỉ định
            news_data = self.fetch_news_google(ticker, max_results=max_news, sources=sources)
            
            if not news_data:
                print(f"Không tìm thấy tin tức cho {ticker}")
                return None, 0
            
            # Chuẩn bị DataFrame
            news_df = pd.DataFrame(news_data)
            
            # Thêm cột cho sentiment và nội dung tiếng Việt nếu cần
            news_df['sentiment'] = 0.0
            
            # Tùy chỉnh định dạng ngày
            news_df['published_formatted'] = news_df['published'].apply(self._format_news_date)
            
            if translate:
                news_df['title_vi'] = ""
                news_df['summary_vi'] = ""
            
            total_sentiment = 0.0
            count = 0
            
            # Xử lý từng tin tức
            for idx, row in news_df.iterrows():
                # Phân tích sentiment dựa trên tiêu đề và tóm tắt
                text_to_analyze = row['title'] + " " + row['summary']
                sentiment = self.analyze_sentiment(text_to_analyze)
                news_df.at[idx, 'sentiment'] = sentiment
                
                # Dịch tiêu đề và tóm tắt nếu cần
                if translate:
                    news_df.at[idx, 'title_vi'] = self.translate_text(row['title'])
                    news_df.at[idx, 'summary_vi'] = self.translate_text(row['summary'])
                else:
                    # Nếu không dịch, sao chép nội dung gốc
                    news_df['title_vi'] = news_df['title']
                    news_df['summary_vi'] = news_df['summary']
                
                # Cộng dồn sentiment
                total_sentiment += sentiment
                count += 1
            
            # Tính sentiment trung bình
            avg_sentiment = total_sentiment / count if count > 0 else 0
            
            return news_df, avg_sentiment
        except Exception as e:
            print(f"Lỗi khi phân tích tin tức và sentiment: {e}")
            return None, 0
    
    def _format_news_date(self, date_string):
        """
        Định dạng lại chuỗi ngày để hiển thị rõ ràng
        """
        try:
            # Sử dụng dateparser để phân tích thời gian
            parsed_date = dateparser.parse(date_string, settings={'RELATIVE_BASE': datetime.now()})
            
            if parsed_date:
                # Trả về định dạng ngày/tháng/năm
                return parsed_date.strftime("%d/%m/%Y")
            else:
                return date_string
        except Exception as e:
            print(f"Lỗi khi định dạng ngày: {e}")
            return date_string

    def generate_news_report(self, ticker, max_news=10, sources=None):
        """
        Tạo báo cáo tin tức cho một mã chứng khoán
        """
        try:
            news_df, avg_sentiment = self.analyze_news_sentiment(ticker, max_news=max_news, sources=sources)
            
            if news_df is None or news_df.empty:
                return "Không tìm thấy tin tức cho mã chứng khoán này."
            
            # Sắp xếp theo ngày đăng tin, tin mới nhất ở trên cùng
            news_df = news_df.sort_values('published', ascending=False)
            
            # Tạo báo cáo
            report = f"BÁO CÁO TIN TỨC: {ticker}\n"
            report += "=" * 50 + "\n\n"
            
            # Đánh giá sentiment tổng thể
            report += f"Sentiment tổng thể: {avg_sentiment:.2f} "
            if sources and len(sources) > 0:
                report += f"\nNguồn tin đã lọc: {', '.join(sources)}\n"
            if avg_sentiment > 0.2:
                report += "(Tích cực)\n"
            elif avg_sentiment < -0.2:
                report += "(Tiêu cực)\n"
            else:
                report += "(Trung lập)\n"
            
            report += "\nTin tức gần đây:\n\n"
            
            # Thêm từng tin tức vào báo cáo
            for idx, row in news_df.iterrows():
                # Xác định loại sentiment
                if row['sentiment'] > 0.2:
                    sentiment_text = "Tích cực"
                elif row['sentiment'] < -0.2:
                    sentiment_text = "Tiêu cực"
                else:
                    sentiment_text = "Trung lập"
                
                # Xử lý tiêu đề
                title = row['title_vi'] if row['title_vi'] and row['title_vi'].strip() else "Không có tiêu đề"
                
                report += f"{idx+1}. {title}\n"
                report += f"   Nguồn: {row['source']} - Ngày đăng: {row['published_formatted']}\n"
                report += f"   Tóm tắt: {row['summary_vi']}\n"
                report += f"   Sentiment: {row['sentiment']:.2f} ({sentiment_text})\n"
                report += f"   Link: {row['link']}\n\n"
            
            return report
        except Exception as e:
            print(f"Lỗi khi tạo báo cáo tin tức: {e}")
            return f"Lỗi khi tạo báo cáo tin tức: {e}"
    
    def plot_sentiment_summary(self, ticker, max_news=10, sources=None):
        """
        Tạo biểu đồ tóm tắt sentiment
        """
        try:
            news_df, avg_sentiment = self.analyze_news_sentiment(ticker, max_news=max_news, sources=sources)
            
            if news_df is None or news_df.empty or len(news_df) < 1:
                print(f"Không đủ dữ liệu để tạo biểu đồ sentiment cho {ticker}")
                return None
            
            # Tạo biểu đồ
            plt.close('all')  # Đóng tất cả các figure trước đó
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Biểu đồ cột cho từng tin tức
            news_df = news_df.sort_values('sentiment')
            bar_colors = ['red' if x < -0.2 else 'green' if x > 0.2 else 'gray' for x in news_df['sentiment']]
            
            ax1.barh(range(len(news_df)), news_df['sentiment'], color=bar_colors)
            ax1.set_yticks(range(len(news_df)))
            ax1.set_yticklabels([f"Tin {i+1}" for i in range(len(news_df))])
            ax1.set_xlabel('Sentiment (-1 tiêu cực, 1 tích cực)')
            ax1.set_title(f'Sentiment của các tin tức về {ticker}')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Biểu đồ tròn cho phân phối sentiment
            positive = len(news_df[news_df['sentiment'] > 0.2])
            negative = len(news_df[news_df['sentiment'] < -0.2])
            neutral = len(news_df) - positive - negative
            
            sentiment_counts = [positive, neutral, negative]
            labels = ['Tích cực', 'Trung lập', 'Tiêu cực']
            colors = ['green', 'gray', 'red']
            
            # Chỉ vẽ biểu đồ tròn nếu có dữ liệu
            if sum(sentiment_counts) > 0:
                ax2.pie(sentiment_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'Phân phối sentiment cho {ticker}')
            else:
                ax2.text(0.5, 0.5, 'Không có dữ liệu', horizontalalignment='center', verticalalignment='center')
                ax2.set_title(f'Phân phối sentiment cho {ticker}')
                ax2.axis('off')
            
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Lỗi khi tạo biểu đồ sentiment: {e}")
            return None