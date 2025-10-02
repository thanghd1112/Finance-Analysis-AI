import streamlit as st
import matplotlib.pyplot as plt
from financial_Analysis import AIAgent
import pandas as pd
import base64
from io import BytesIO
from stock_Comparison import StockComparison  # Import lớp so sánh cổ phiếu
from sector_Analysis import SectorAnalysis  # Import lớp phân tích ngành
from news_sentiment import NewsSentimentAnalyzer
import os
import base64

if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = 'add'


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
    """Hiển thị báo cáo tài chính và tải xuống HTML"""
    agent = AIAgent()
    agent.fetch_financial_data(ticker)
    report_html = agent.generate_financial_report(ticker)
    
    if report_html:
        # Đọc nội dung CSS và JS để nhúng trực tiếp vào HTML
        try:
            with open("style.css", "r", encoding="utf-8") as css_file:
                css_content = css_file.read()
            
            with open("report.js", "r", encoding="utf-8") as js_file:
                js_content = js_file.read()
                
            # Nhúng CSS và JS trực tiếp vào HTML
            report_html = report_html.replace('</head>', f'<style>{css_content}</style></head>')
            report_html = report_html.replace('</body>', f'<script>{js_content}</script></body>')
            
            # Lưu file HTML với CSS và JS đã nhúng
            html_filename = agent.save_report_to_file(ticker, report_html)
            
            if html_filename and os.path.exists(html_filename):
                with open(html_filename, "r", encoding="utf-8") as html_file:
                    html_content = html_file.read()
                
                # Cho phép tải xuống file HTML
                st.download_button(
                    label="Tải xuống báo cáo đầy đủ",
                    data=html_content,
                    file_name=f"{ticker}_report.html",
                    mime="text/html"
                )
            else:
                st.error("Không thể tạo file HTML.")
        except Exception as e:
            st.error(f"Lỗi khi nhúng CSS và JS: {str(e)}")
    else:
        st.error(f"Không thể tạo báo cáo cho {ticker}.")
    

# Sidebar để nhập thông tin
with st.sidebar:
    st.header("Thông tin Phân tích")
    
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
        seasonality_mode = st.selectbox(
        "Chọn chế độ mô hình:",
        ["Additive (Cộng)", "Multiplicative (Nhân)", "So sánh cả hai"],
        index=0
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
        max_news = st.slider("Số lượng tin tức tối đa:", min_value=1, max_value=10, value=10, step=1)
        
        # Thêm tùy chọn nguồn tin
        news_sources = st.multiselect(
            "Nguồn tin:",
            ["Google News", "Yahoo Finance", "Bloomberg", "CNBC", "Reuters"],
            default=["Google News"]
        ) 
        news_button = st.button("Phân tích Tin tức", use_container_width=True)

# Khu vực chính để hiển thị kết quả
if analysis_type == "Phân tích Cơ bản" and analyze_button:
    if not ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ.")
    else:
        with st.spinner(f"Đang phân tích {ticker}..."):
            # Hiển thị thông báo đang xử lý
            progress_text = st.empty()
            
            # Lấy dữ liệu
            progress_text.text("Đang lấy dữ liệu tài chính...")
            financial_data = agent.fetch_financial_data(ticker, period=period)
            
            if financial_data is None:
                st.error(f"Không thể lấy dữ liệu cho mã chứng khoán: {ticker}. Vui lòng kiểm tra lại mã chứng khoán.")
            else:
                # Phân tích và tạo báo cáo
                progress_text.text("Đang phân tích dữ liệu...")
                agent.analyze_financial_ratios(ticker)
                report = agent.generate_financial_report(ticker)
                
                # Vẽ biểu đồ chỉ báo kỹ thuật
                progress_text.text("Đang tạo biểu đồ phân tích kỹ thuật...")
                tech_fig = agent.plot_technical_indicators(ticker)
                
                # Dự đoán giá
                progress_text.text(f"Đang dự đoán giá cho {prediction_days} ngày tới...")
                predictions_add, future_prices_add = agent.predict_future_price(ticker, days=prediction_days, mode='add')
                predictions_mul, future_prices_mul = agent.predict_future_price(ticker, days=prediction_days, mode='mul')
                # Tạo các tab để hiển thị biểu đồ
                tabs = st.tabs(["Phân tích Kỹ thuật", "Dự đoán Giá", "Dữ liệu Dự đoán", "Báo cáo Tài chính Chi tiết"])
                # Lấy chế độ dự đoán từ session state
                # Xác định chế độ seasonality
                if seasonality_mode == "Additive (Cộng)":
                    mode = 'additive'
                elif seasonality_mode == "Multiplicative (Nhân)":
                    mode = 'multiplicative'
                else:
                    mode = 'both'
                
                # Nếu chọn so sánh cả hai
                if mode == 'both':
                    # Sử dụng phương thức mới compare_seasonality_modes
                    comparison_fig, comparison_text = agent.predictor.compare_seasonality_modes(
                        financial_data['price_history']['Close'], 
                        days=prediction_days
                    )
                    with tabs[0]:
                        if tech_fig is not None:
                            st.pyplot(tech_fig)
                        else:
                            st.warning("Không thể tạo biểu đồ phân tích kỹ thuật.")
                
                    with tabs[1]:
                        if comparison_fig is not None:
                            st.pyplot(comparison_fig)
                            
                            # Hiển thị nhận xét so sánh
                            st.text_area("So sánh các chế độ mô hình", comparison_text, height=300)
                        else:
                            st.warning("Không thể so sánh các chế độ mô hình.")
                else:
                    # Dự đoán với chế độ được chọn
                    predict_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days, seasonality_mode=mode)
                    
                    # Hiển thị như bình thường
                    with tabs[1]:
                        if predict_fig is not None:
                            st.pyplot(predict_fig)
                        else:
                            st.warning("Không thể tạo biểu đồ dự đoán.")

                # Tab Dữ liệu Dự đoán
                with tabs[2]:
                    st.subheader(f"Dữ liệu Dự đoán - Mô hình {st.session_state.prediction_mode.upper()}")
                    
                    # Nút chuyển đổi chế độ
                    col1, col2 = st.columns([0.9, 0.1])
                    with col2:
                        toggle_data_button = st.button("🔄", key="toggle_prediction_data")
                        if toggle_data_button:
                            st.session_state.prediction_mode = 'mul' if st.session_state.prediction_mode == 'add' else 'add'
                    
                    # Lấy dữ liệu dự đoán
                    _, future_prices_add = agent.predict_future_price(ticker, days=prediction_days, mode='add')
                    _, future_prices_mul = agent.predict_future_price(ticker, days=prediction_days, mode='mul')
                    
                    # So sánh hai bảng dữ liệu
                    expand_data = st.checkbox("So sánh dữ liệu 2 mô hình")
                    
                    if expand_data:
                        col_add, col_mul = st.columns(2)
                        
                        with col_add:
                            st.subheader("Mô hình Cộng")
                            if future_prices_add is not None:
                                future_prices_add_display = pd.DataFrame({
                                    'Ngày': future_prices_add.index,
                                    'Giá dự đoán': future_prices_add['yhat'].round(2),
                                    'Giá thấp nhất (95%)': future_prices_add['yhat_lower'].round(2),
                                    'Giá cao nhất (95%)': future_prices_add['yhat_upper'].round(2)
                                }).reset_index(drop=True)
                                st.dataframe(future_prices_add_display)
                        
                        with col_mul:
                            st.subheader("Mô hình Nhân")
                            if future_prices_mul is not None:
                                future_prices_mul_display = pd.DataFrame({
                                    'Ngày': future_prices_mul.index,
                                    'Giá dự đoán': future_prices_mul['yhat'].round(2),
                                    'Giá thấp nhất (95%)': future_prices_mul['yhat_lower'].round(2),
                                    'Giá cao nhất (95%)': future_prices_mul['yhat_upper'].round(2)
                                }).reset_index(drop=True)
                                st.dataframe(future_prices_mul_display)
                    else:
                        # Mặc định hiển thị dữ liệu của mô hình được chọn
                        future_prices = future_prices_add if st.session_state.prediction_mode == 'add' else future_prices_mul
                        
                        if future_prices is not None:
                            future_prices_display = pd.DataFrame({
                                'Ngày': future_prices.index,
                                'Giá dự đoán': future_prices['yhat'].round(2),
                                'Giá thấp nhất (95%)': future_prices['yhat_lower'].round(2),
                                'Giá cao nhất (95%)': future_prices['yhat_upper'].round(2)
                            }).reset_index(drop=True)
                            
                            st.dataframe(future_prices_display)
                            
                            # Nút tải xuống dữ liệu
                            csv = future_prices_display.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction_{st.session_state.prediction_mode}.csv">Tải dữ liệu dự đoán (CSV)</a>'
                            st.markdown(href, unsafe_allow_html=True)
                
                with tabs[3]:
                    st.subheader(f"Báo cáo Tài chính cho {ticker}")
                    
                    # Nhúng CSS để styled components hiển thị đúng
                    with open("style.css", "r", encoding="utf-8") as css_file:
                        css_content = css_file.read()
                    st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
                    
                    # Hiển thị báo cáo
                    st.components.v1.html(report, height=800, scrolling=True)
                    
                    # Thêm JavaScript động 
                    with open("report.js", "r", encoding="utf-8") as js_file:
                        js_content = js_file.read()
                    st.markdown(f'<script>{js_content}</script>', unsafe_allow_html=True)
                    
                    display_financial_report(ticker)

                    # Xóa thông báo đang xử lý
                    progress_text.empty()
                
                # Hiển thị kết quả
                #st.subheader(f"Báo cáo Tài chính cho {ticker}")
                #st.text(report)
                
                

# Xử lý cho chức năng so sánh cổ phiếu
elif analysis_type == "So sánh Cổ phiếu" and compare_button:
    # Xử lý danh sách cổ phiếu
    tickers_list = [ticker.strip() for ticker in tickers_input.split(',')]
    
    if not tickers_list or all(ticker == "" for ticker in tickers_list):
        st.error("Vui lòng nhập ít nhất một mã chứng khoán hợp lệ.")
    else:
        with st.spinner(f"Đang so sánh các cổ phiếu: {', '.join(tickers_list)}..."):
            # Thực hiện so sánh
            comparison_fig, returns_df = stock_comparison.plot_comparison(tickers_list, period=comparison_period)
            
            if comparison_fig is not None:
                # Hiển thị biểu đồ so sánh
                st.subheader("So sánh hiệu suất các cổ phiếu")
                st.pyplot(comparison_fig)
                
                # Hiển thị bảng lợi nhuận
                if returns_df is not None:
                    st.subheader("Bảng so sánh lợi nhuận")
                    st.dataframe(returns_df)
                
                # Thêm chức năng tải xuống dữ liệu
                if returns_df is not None:
                    csv = returns_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="stock_comparison.csv">Tải dữ liệu so sánh (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("Không thể so sánh các mã chứng khoán đã chọn. Vui lòng kiểm tra lại các mã chứng khoán.")

# Xử lý cho chức năng phân tích ngành
elif analysis_type == "Phân tích Ngành" and sector_button:
    if not sector_ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ để phân tích ngành.")
    else:
        with st.spinner(f"Đang phân tích ngành cho {sector_ticker}..."):
            # Thực hiện phân tích ngành
            sector_fig, sector_report = sector_analysis.analyze_sector_performance(sector_ticker, period=sector_period)
            
            # Hiển thị kết quả
            st.subheader(f"Phân tích ngành cho {sector_ticker}")
            
            if sector_report:
                st.text(sector_report)
                
                # Tạo nút tải xuống báo cáo
                report_text = f"Báo cáo Phân tích Ngành - {sector_ticker}\n\n{sector_report}"
                b64 = base64.b64encode(report_text.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="{sector_ticker}_sector_analysis.txt">Tải báo cáo phân tích ngành (TXT)</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning(f"Không thể lấy thông tin ngành cho {sector_ticker}.")
            
            if sector_fig:
                st.pyplot(sector_fig)

# Xử lý cho chức năng phân tích tin tức
elif analysis_type == "Phân tích Tin tức" and news_button:
    if not news_ticker:
        st.error("Vui lòng nhập mã chứng khoán hợp lệ để phân tích tin tức.")
    else:
        with st.spinner(f"Đang phân tích tin tức cho {news_ticker}..."):
            # Tạo báo cáo tin tức - truyền max_news vào đây
            news_report = news_analyzer.generate_news_report(news_ticker, max_news=max_news, sources=news_sources)
            
            # Tạo biểu đồ sentiment
            sentiment_fig = news_analyzer.plot_sentiment_summary(news_ticker, max_news=max_news, sources=news_sources)
            
            # Hiển thị kết quả
            st.subheader(f"Báo cáo Tin tức và Sentiment cho {news_ticker}")
            
            if sentiment_fig:
                st.pyplot(sentiment_fig)
            
            st.text_area("Báo cáo Chi tiết", news_report, height=500)
            
            # Lấy dữ liệu tin tức để hiển thị chi tiết - QUAN TRỌNG: truyền max_news từ slider vào đây
            news_df, avg_sentiment = news_analyzer.analyze_news_sentiment(news_ticker, max_news=max_news, sources=news_sources)
            
            if avg_sentiment is not None:
                sentiment_status = "Tích cực" if avg_sentiment > 0.2 else "Tiêu cực" if avg_sentiment < -0.2 else "Trung lập"
                st.info(f"""
                **Phân tích sentiment tổng thể: {sentiment_status} ({avg_sentiment:.2f})**
                
                Chỉ số sentiment > 0.2: Tích cực
                Chỉ số sentiment < -0.2: Tiêu cực
                Chỉ số sentiment -0.2 đến 0.2: Trung lập
                """)
            
            if news_df is not None and not news_df.empty:
                st.subheader("Tin tức đã phân tích")
                
                # Hiển thị từng tin tức trong các expander
                for idx, row in news_df.iterrows():
                    # Xác định màu sắc dựa trên sentiment
                    sentiment_color = "green" if row['sentiment'] > 0.2 else "red" if row['sentiment'] < -0.2 else "gray"
                    sentiment_text = "Tích cực" if row['sentiment'] > 0.2 else "Tiêu cực" if row['sentiment'] < -0.2 else "Trung lập"
                    
                    # Tạo expander cho từng tin tức
                    with st.expander(f"{row['title_vi']} - Sentiment: {sentiment_text}"):
                        st.markdown(f"**Nguồn:** {row['source']} - {row['published']}")
                        st.markdown(f"**Tóm tắt (Tiếng Việt):**")
                        st.markdown(f"> {row['summary_vi']}")
                        st.markdown(f"**Tóm tắt (Tiếng Anh):**")
                        st.markdown(f"> {row['summary']}")
                        st.markdown(f"**Sentiment:** {row['sentiment']:.2f} ({sentiment_text})")
                        st.markdown(f"[Đọc thêm]({row['link']})")
                
                # Thêm nút xuất CSV cho dữ liệu tin tức
                csv_news = news_df.to_csv(index=False)
                b64_news = base64.b64encode(csv_news.encode()).decode()
                href_news = f'<a href="data:file/csv;base64,{b64_news}" download="{news_ticker}_news_analysis.csv">Tải dữ liệu phân tích tin tức (CSV)</a>'
                st.markdown(href_news, unsafe_allow_html=True)
            else:
                st.warning(f"Không tìm thấy tin tức cho {news_ticker}")

else:
    # Hiển thị hướng dẫn ban đầu
    st.info("""
    ### Hướng dẫn sử dụng
    1. Chọn loại phân tích từ menu bên trái
    2. Nhập các thông tin cần thiết
    3. Nhấn nút tương ứng để bắt đầu phân tích
    
    **Các chức năng hiện có:**
    - **Phân tích Cơ bản**: Phân tích tài chính, hiển thị các chỉ báo kỹ thuật và dự đoán giá trong tương lai
    - **So sánh Cổ phiếu**: So sánh hiệu suất nhiều cổ phiếu cùng lúc
    - **Phân tích Ngành**: Phân tích cổ phiếu so với ngành và đối thủ cạnh tranh
    - **Phân tích Tin tức**: Phân tích tin tức và sentiment liên quan đến cổ phiếu
    
    Lưu ý: Ứng dụng sử dụng dữ liệu từ Yahoo Finance. Đảm bảo bạn nhập đúng mã chứng khoán.
    """)

# Thêm footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #777;">
    <p>Ứng dụng phân tích tài chính và dự đoán giá cổ phiếu © 2025</p>
</div>
""", unsafe_allow_html=True)