import streamlit as st
import matplotlib.pyplot as plt
from financial_Analysis import AIAgent
import pandas as pd
import base64
from io import BytesIO
from stock_Comparison import StockComparison
from sector_Analysis import SectorAnalysis
from news_sentiment import NewsSentimentAnalyzer
import os

# Tiêu đề ứng dụng
st.set_page_config(page_title="Phân tích Tài chính", layout="wide")
st.title("Ứng dụng Phân tích Tài chính")

# Khởi tạo các đối tượng
@st.cache_resource
def get_agent():
    return AIAgent(name="Streamlit Finance Bot")

@st.cache_resource
def get_stock_comparison():
    return StockComparison()

@st.cache_resource
def get_sector_analysis():
    return SectorAnalysis()

@st.cache_resource
def get_news_analyzer():
    return NewsSentimentAnalyzer()

agent = get_agent()
stock_comparison = get_stock_comparison()
sector_analysis = get_sector_analysis()
news_analyzer = get_news_analyzer()

# Hàm chuyển đổi biểu đồ matplotlib thành hình ảnh cho streamlit
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# Hàm hiển thị báo cáo tài chính
def display_financial_report(ticker):
    """Hiển thị báo cáo tài chính và tải xuống HTML với CSS và JS tích hợp"""
    agent = AIAgent()
    agent.fetch_financial_data(ticker)
    report_html = agent.generate_financial_report(ticker)
    
    if report_html:
        # Đọc file CSS
        css_content = ""
        if os.path.exists("style.css"):
            with open("style.css", "r", encoding="utf-8") as css_file:
                css_content = css_file.read()
        
        # Đọc file JavaScript
        js_content = ""
        if os.path.exists("report.js"):
            with open("report.js", "r", encoding="utf-8") as js_file:
                js_content = js_file.read()
        
        # Chèn CSS và JS vào report_html nếu chưa có
        if "<style>" not in report_html and css_content:
            report_html = report_html.replace("</head>", f"<style>{css_content}</style></head>")
        
        if "<script>" not in report_html and js_content:
            report_html = report_html.replace("</body>", f"<script>{js_content}</script></body>")
        
        # Lưu file HTML hoàn chỉnh
        html_filename = f"{ticker}_report.html"
        with open(html_filename, "w", encoding="utf-8") as html_file:
            html_file.write(report_html)
        
        if os.path.exists(html_filename):
            with open(html_filename, "r", encoding="utf-8") as html_file:
                html_content = html_file.read()
            
            # Cho phép tải xuống file HTML
            st.download_button(
                label="Tải xuống báo cáo",
                data=html_content,
                file_name=f"{ticker}_report.html",
                mime="text/html"
            )
        else:
            st.error("Không thể tạo file HTML.")
    else:
        st.error(f"Không thể tạo báo cáo cho {ticker}.")

# Nâng cao giao diện với CSS tùy chỉnh
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 5px;
        overflow: hidden;
    }
    .stHeader {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar để nhập thông tin
with st.sidebar:
    st.markdown('<div class="stHeader"><h2>Thông tin Phân tích</h2></div>', unsafe_allow_html=True)
    
    # Tạo tabs trong sidebar
    analysis_type = st.selectbox(
        "Chọn loại phân tích:", 
        ["Phân tích Cơ bản", "So sánh Cổ phiếu", "Phân tích Ngành", "Phân tích Tin tức"], 
        index=0
    )
    
    if analysis_type == "Phân tích Cơ bản":
        ticker = st.text_input("Nhập mã chứng khoán (Ví dụ: AAPL, MSFT, GOOGL):", "AAPL")
        period = st.selectbox(
            "Chọn khoảng thời gian dữ liệu:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
        prediction_days = st.slider("Số ngày dự đoán:", min_value=7, max_value=90, value=30, step=1)
        
        analyze_button = st.button("Phân tích", use_container_width=True)
    
    elif analysis_type == "So sánh Cổ phiếu":
        st.subheader("So sánh hiệu suất cổ phiếu")
        default_tickers = "AAPL,MSFT,GOOGL"
        tickers_input = st.text_input(
            "Nhập các mã chứng khoán (cách nhau bằng dấu phẩy):",
            default_tickers
        )
        comparison_period = st.selectbox(
            "Chọn khoảng thời gian so sánh:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        compare_button = st.button("So sánh", use_container_width=True)
    
    elif analysis_type == "Phân tích Ngành":
        st.subheader("Phân tích ngành và đối thủ cạnh tranh")
        sector_ticker = st.text_input("Nhập mã chứng khoán cần phân tích:", "AAPL")
        sector_period = st.selectbox(
            "Chọn khoảng thời gian phân tích:",
            options=["1mo", "3mo", "6mo", "1y", "2y"],
            index=3
        )
        
        sector_button = st.button("Phân tích Ngành", use_container_width=True)
    
    elif analysis_type == "Phân tích Tin tức":
        st.subheader("Phân tích tin tức và sentiment")
        news_ticker = st.text_input("Nhập mã chứng khoán:", "AAPL")
        max_news = st.slider("Số lượng tin tức tối đa:", min_value=5, max_value=20, value=10, step=1)
        
        news_button = st.button("Phân tích Tin tức", use_container_width=True)

# Khu vực chính để hiển thị kết quả
if analysis_type == "Phân tích Cơ bản" and analyze_button:
    if not ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ.")
    else:
        with st.spinner(f"Đang phân tích {ticker}..."):
            # Hiển thị thông báo đang xử lý
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Lấy dữ liệu
            progress_text.text("Đang lấy dữ liệu tài chính...")
            progress_bar.progress(20)
            financial_data = agent.fetch_financial_data(ticker, period=period)
            
            if financial_data is None:
                st.error(f"Không thể lấy dữ liệu cho mã chứng khoán: {ticker}. Vui lòng kiểm tra lại mã chứng khoán.")
            else:
                # Phân tích và tạo báo cáo
                progress_text.text("Đang phân tích dữ liệu...")
                progress_bar.progress(40)
                agent.analyze_financial_ratios(ticker)
                report = agent.generate_financial_report(ticker)
                
                # Vẽ biểu đồ chỉ báo kỹ thuật
                progress_text.text("Đang tạo biểu đồ phân tích kỹ thuật...")
                progress_bar.progress(60)
                tech_fig = agent.plot_technical_indicators(ticker)
                
                # Dự đoán giá
                progress_text.text(f"Đang dự đoán giá cho {prediction_days} ngày tới...")
                progress_bar.progress(80)
                predict_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days)
                progress_bar.progress(100)
                
                # Tạo các tab với giao diện cải tiến
                st.markdown(f"""
                <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                    <h2 style="color: #2c3e50; margin: 0;">Kết quả phân tích cho {ticker}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                tabs = st.tabs(["Phân tích Kỹ thuật", "Dự đoán Giá", "Dữ liệu Dự đoán", "Báo cáo Tài chính Chi tiết"])
                
                with tabs[0]:
                    if tech_fig is not None:
                        st.pyplot(tech_fig)
                    else:
                        st.warning("Không thể tạo biểu đồ phân tích kỹ thuật.")
                
                with tabs[1]:
                    if predict_fig is not None:
                        st.pyplot(predict_fig)
                        st.info("""
                        **Chú thích:**
                        - Đường nét đứt màu xanh lá: SMA 50 (Đường trung bình động 50 ngày)
                        - Đường nét đứt màu xanh dương: SMA 200 (Đường trung bình động 200 ngày)
                        - Đường màu tím: EMA 20 (Đường trung bình động hàm mũ 20 ngày)
                        - Đường nét đứt màu đỏ: Đường dự đoán giá
                        - Vùng đỏ nhạt: Khoảng tin cậy 95% cho dự đoán
                        """)
                    else:
                        st.warning("Không thể tạo biểu đồ dự đoán giá.")
                
                with tabs[2]:
                    if future_prices is not None:
                        # Định dạng dữ liệu dự đoán
                        future_prices_display = pd.DataFrame({
                            'Ngày': future_prices.index,
                            'Giá dự đoán': future_prices['yhat'].round(2),
                            'Giá thấp nhất (95%)': future_prices['yhat_lower'].round(2),
                            'Giá cao nhất (95%)': future_prices['yhat_upper'].round(2)
                        })
                        future_prices_display = future_prices_display.reset_index(drop=True)
                        
                        st.dataframe(future_prices_display, use_container_width=True)
                        
                        # Thêm nút tải xuống dữ liệu
                        csv = future_prices_display.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        st.download_button(
                            label="Tải dữ liệu dự đoán (CSV)",
                            data=csv,
                            file_name=f"{ticker}_prediction.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Không có dữ liệu dự đoán.")
                
                with tabs[3]:
                    st.subheader(f"Báo cáo Tài chính cho {ticker}")
                    
                    # Sử dụng HTML với CSS tích hợp
                    if report:
                        st.components.v1.html(report, height=800, scrolling=True)
                        display_financial_report(ticker)
                    else:
                        st.warning("Không thể tạo báo cáo tài chính chi tiết.")

                # Xóa thông báo đang xử lý
                progress_text.empty()
                progress_bar.empty()

# Xử lý cho chức năng so sánh cổ phiếu
elif analysis_type == "So sánh Cổ phiếu" and compare_button:
    # Xử lý danh sách cổ phiếu
    tickers_list = [ticker.strip() for ticker in tickers_input.split(',')]
    
    if not tickers_list or all(ticker == "" for ticker in tickers_list):
        st.error("Vui lòng nhập ít nhất một mã chứng khoán hợp lệ.")
    else:
        with st.spinner(f"Đang so sánh các cổ phiếu: {', '.join(tickers_list)}..."):
            # Hiển thị thanh tiến trình
            progress_bar = st.progress(30)
            
            # Thực hiện so sánh
            comparison_fig, returns_df = stock_comparison.plot_comparison(tickers_list, period=comparison_period)
            progress_bar.progress(100)
            
            if comparison_fig is not None:
                # Hiển thị tiêu đề với định dạng đẹp hơn
                st.markdown(f"""
                <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                    <h2 style="color: #2c3e50; margin: 0;">So sánh hiệu suất cổ phiếu: {', '.join(tickers_list)}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Hiển thị biểu đồ so sánh
                st.pyplot(comparison_fig)
                
                # Hiển thị bảng lợi nhuận với định dạng cải tiến
                if returns_df is not None:
                    st.subheader("Bảng so sánh lợi nhuận")
                    st.dataframe(returns_df, use_container_width=True)
                
                    # Thêm chức năng tải xuống dữ liệu với nút đẹp hơn
                    csv = returns_df.to_csv(index=False)
                    st.download_button(
                        label="Tải dữ liệu so sánh (CSV)",
                        data=csv,
                        file_name="stock_comparison.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Không thể so sánh các mã chứng khoán đã chọn. Vui lòng kiểm tra lại các mã chứng khoán.")
            
            # Xóa thanh tiến trình khi hoàn thành
            progress_bar.empty()

# Xử lý cho chức năng phân tích ngành
elif analysis_type == "Phân tích Ngành" and sector_button:
    if not sector_ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ để phân tích ngành.")
    else:
        with st.spinner(f"Đang phân tích ngành cho {sector_ticker}..."):
            # Hiển thị thanh tiến trình
            progress_bar = st.progress(30)
            
            # Thực hiện phân tích ngành
            sector_fig, sector_report = sector_analysis.analyze_sector_performance(sector_ticker, period=sector_period)
            progress_bar.progress(100)
            
            # Hiển thị kết quả với giao diện cải tiến
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <h2 style="color: #2c3e50; margin: 0;">Phân tích ngành cho {sector_ticker}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if sector_report:
                st.markdown(f"""
                <div style="background-color: white; padding: 1.5rem; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 1rem;">
                    <pre style="white-space: pre-wrap; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">{sector_report}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                # Tạo nút tải xuống báo cáo với định dạng đẹp hơn
                report_text = f"Báo cáo Phân tích Ngành - {sector_ticker}\n\n{sector_report}"
                st.download_button(
                    label="Tải báo cáo phân tích ngành (TXT)",
                    data=report_text,
                    file_name=f"{sector_ticker}_sector_analysis.txt",
                    mime="text/plain"
                )
            else:
                st.warning(f"Không thể lấy thông tin ngành cho {sector_ticker}.")
            
            if sector_fig:
                st.pyplot(sector_fig)
            
            # Xóa thanh tiến trình khi hoàn thành
            progress_bar.empty()

# Xử lý cho chức năng phân tích tin tức
elif analysis_type == "Phân tích Tin tức" and news_button:
    if not news_ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ để phân tích tin tức.")
    else:
        with st.spinner(f"Đang phân tích tin tức cho {news_ticker}..."):
            # Hiển thị thanh tiến trình
            progress_bar = st.progress(30)
            
            # Tạo báo cáo tin tức
            news_report = news_analyzer.generate_news_report(news_ticker)
            progress_bar.progress(60)
            
            # Tạo biểu đồ sentiment
            sentiment_fig = news_analyzer.plot_sentiment_summary(news_ticker)
            progress_bar.progress(80)
            
            # Lấy dữ liệu tin tức chi tiết
            news_df, avg_sentiment = news_analyzer.analyze_news_sentiment(news_ticker, max_news=max_news)
            progress_bar.progress(100)
            
            # Hiển thị kết quả với giao diện cải tiến
            st.markdown(f"""
            <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <h2 style="color: #2c3e50; margin: 0;">Báo cáo Tin tức và Sentiment cho {news_ticker}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị tóm tắt sentiment
            if avg_sentiment is not None:
                sentiment_status = "Tích cực" if avg_sentiment > 0.2 else "Tiêu cực" if avg_sentiment < -0.2 else "Trung lập"
                sentiment_color = "green" if avg_sentiment > 0.2 else "red" if avg_sentiment < -0.2 else "gray"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid {sentiment_color}; margin-bottom: 1rem;">
                    <h3 style="margin-top: 0;">Phân tích sentiment tổng thể: <span style="color: {sentiment_color};">{sentiment_status} ({avg_sentiment:.2f})</span></h3>
                    <ul>
                        <li>Chỉ số sentiment > 0.2: Tích cực</li>
                        <li>Chỉ số sentiment < -0.2: Tiêu cực</li>
                        <li>Chỉ số sentiment -0.2 đến 0.2: Trung lập</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Hiển thị biểu đồ sentiment
            if sentiment_fig:
                st.pyplot(sentiment_fig)
            
            # Hiển thị báo cáo chi tiết
            tabs = st.tabs(["Báo cáo Tóm tắt", "Tin tức Chi tiết"])
            
            with tabs[0]:
                st.text_area("Báo cáo Chi tiết", news_report, height=400)
                
                # Nút tải xuống báo cáo
                st.download_button(
                    label="Tải báo cáo tin tức (TXT)",
                    data=news_report,
                    file_name=f"{news_ticker}_news_report.txt",
                    mime="text/plain"
                )
            
            with tabs[1]:
                if news_df is not None and not news_df.empty:
                    # Hiển thị từng tin tức trong các expander với định dạng đẹp hơn
                    for idx, row in news_df.iterrows():
                        # Xác định màu sắc dựa trên sentiment
                        sentiment_color = "green" if row['sentiment'] > 0.2 else "red" if row['sentiment'] < -0.2 else "gray"
                        sentiment_text = "Tích cực" if row['sentiment'] > 0.2 else "Tiêu cực" if row['sentiment'] < -0.2 else "Trung lập"
                        
                        # Tạo expander cho từng tin tức với giao diện cải tiến
                        with st.expander(f"{row['title_vi']} - {sentiment_text}"):
                            st.markdown(f"""
                            <div style="border-left: 3px solid {sentiment_color}; padding-left: 1rem;">
                                <p><strong>Nguồn:</strong> {row['source']} - {row['published']}</p>
                                <p><strong>Tóm tắt (Tiếng Việt):</strong></p>
                                <blockquote style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px;">
                                    {row['summary_vi']}
                                </blockquote>
                                <p><strong>Tóm tắt (Tiếng Anh):</strong></p>
                                <blockquote style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px;">
                                    {row['summary']}
                                </blockquote>
                                <p><strong>Sentiment:</strong> <span style="color: {sentiment_color};">{row['sentiment']:.2f} ({sentiment_text})</span></p>
                                <a href="{row['link']}" target="_blank">Đọc thêm</a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Thêm nút xuất CSV cho dữ liệu tin tức
                    csv_news = news_df.to_csv(index=False)
                    st.download_button(
                        label="Tải dữ liệu phân tích tin tức (CSV)",
                        data=csv_news,
                        file_name=f"{news_ticker}_news_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning(f"Không tìm thấy tin tức cho {news_ticker}")
            
            # Xóa thanh tiến trình khi hoàn thành
            progress_bar.empty()

else:
    # Hiển thị hướng dẫn ban đầu với giao diện cải tiến
    st.markdown("""
    <div style="background-color: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="color: #2c3e50; margin-top: 0;">Hướng dẫn sử dụng</h2>
        <ol>
            <li>Chọn loại phân tích từ menu bên trái</li>
            <li>Nhập các thông tin cần thiết</li>
            <li>Nhấn nút tương ứng để bắt đầu phân tích</li>
        </ol>
        
        <h3 style="color: #2c3e50;">Các chức năng hiện có:</h3>
        <ul>
            <li><strong>Phân tích Cơ bản:</strong> Phân tích tài chính, hiển thị các chỉ báo kỹ thuật và dự đoán giá trong tương lai</li>
            <li><strong>So sánh Cổ phiếu:</strong> So sánh hiệu suất nhiều cổ phiếu cùng lúc</li>
            <li><strong>Phân tích Ngành:</strong> Phân tích cổ phiếu so với ngành và đối thủ cạnh tranh</li>
            <li><strong>Phân tích Tin tức:</strong> Phân tích tin tức và sentiment liên quan đến cổ phiếu</li>
        </ul>
        
        <div style="background-color: #eaf7fb; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
            <p style="margin: 0;"><strong>Lưu ý:</strong> Ứng dụng sử dụng dữ liệu từ Yahoo Finance. Đảm bảo bạn nhập đúng mã chứng khoán.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Thêm footer với giao diện cải tiến
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #777;">
    <p>Ứng dụng phân tích tài chính và dự đoán giá cổ phiếu © 2025</p>
</div>
""", unsafe_allow_html=True)