import streamlit as st
import matplotlib.pyplot as plt
from financial_Analysis import AIAgent
import pandas as pd
import base64
from io import BytesIO
from stock_Comparison import StockComparison  # Import lớp so sánh cổ phiếu
from sector_Analysis import SectorAnalysis  # Import lớp phân tích ngành
from news_sentiment import NewsSentimentAnalyzer

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

# Khu vực chính để hiển thị kết quả
if analysis_type == "Phân tích Cơ bản" and analyze_button:
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
            predict_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days)
            
            # Xóa thông báo đang xử lý
            progress_text.empty()
            
            # Hiển thị kết quả
            st.subheader(f"Báo cáo Tài chính cho {ticker}")
            st.text(report)
            
            # Tạo các tab để hiển thị biểu đồ
            tabs = st.tabs(["Phân tích Kỹ thuật", "Dự đoán Giá", "Dữ liệu Dự đoán"])
            
            with tabs[0]:
                if tech_fig is not None:
                    st.pyplot(tech_fig)
                else:
                    st.warning("Không thể tạo biểu đồ phân tích kỹ thuật.")
            
            # Trong phần hiển thị biểu đồ dự đoán, thêm thông tin về các chỉ số kỹ thuật
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
                    
                    st.dataframe(future_prices_display)
                    
                    # Thêm nút tải xuống dữ liệu
                    csv = future_prices_display.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction.csv">Tải dữ liệu dự đoán (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    st.warning("Không có dữ liệu dự đoán.")

# Xử lý cho chức năng so sánh cổ phiếu
elif analysis_type == "So sánh Cổ phiếu" and compare_button:
    # Xử lý danh sách cổ phiếu
    tickers_list = [ticker.strip() for ticker in tickers_input.split(',')]
    
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
    with st.spinner(f"Đang phân tích ngành cho {sector_ticker}..."):
        # Thực hiện phân tích ngành
        sector_fig, sector_report = sector_analysis.analyze_sector_performance(sector_ticker, period=sector_period)
        
        # Hiển thị kết quả
        st.subheader(f"Phân tích ngành cho {sector_ticker}")
        
        if sector_report:
            st.text(sector_report)
        else:
            st.warning(f"Không thể lấy thông tin ngành cho {sector_ticker}.")
        
        if sector_fig:
            st.pyplot(sector_fig)

# Xử lý cho chức năng phân tích tin tức
elif analysis_type == "Phân tích Tin tức":
    with st.sidebar:
        st.subheader("Phân tích tin tức và sentiment")
        news_ticker = st.text_input("Nhập mã chứng khoán:", "AAPL")
        max_news = st.slider("Số lượng tin tức tối đa:", min_value=5, max_value=20, value=10, step=1)
        
        news_button = st.button("Phân tích Tin tức", use_container_width=True)
    
    if news_button:
        with st.spinner(f"Đang phân tích tin tức cho {news_ticker}..."):
            # Tạo báo cáo tin tức
            news_report = news_analyzer.generate_news_report(news_ticker)
            
            # Tạo biểu đồ sentiment
            sentiment_fig = news_analyzer.plot_sentiment_summary(news_ticker)
            
            # Hiển thị kết quả
            st.subheader(f"Báo cáo Tin tức và Sentiment cho {news_ticker}")
            
            if sentiment_fig:
                st.pyplot(sentiment_fig)
            
            st.text_area("Báo cáo Chi tiết", news_report, height=500)
            
            # Lấy dữ liệu tin tức để hiển thị chi tiết
            news_df, avg_sentiment = news_analyzer.analyze_news_sentiment(news_ticker, max_news=max_news)
            
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
    
    Lưu ý: Ứng dụng sử dụng dữ liệu từ Yahoo Finance. Đảm bảo bạn nhập đúng mã chứng khoán.
    """)

# Thêm footer
st.markdown("---")
st.caption("Ứng dụng phân tích tài chính và dự đoán giá cổ phiếu")