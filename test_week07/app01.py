import streamlit as st
import matplotlib.pyplot as plt
from financial_Analysis import AIAgent
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from stock_Comparison import StockComparison  # Import lớp so sánh cổ phiếu
from sector_Analysis import SectorAnalysis  # Import lớp phân tích ngành
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
                
                # Lưu nội dung vào session state
                st.session_state['html_content'] = html_content
                
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
    return report_html

# Sidebar để nhập thông tin
with st.sidebar:
    st.header("Thông tin Phân tích")
    
    # Tạo tabs trong sidebar
    analysis_type = st.selectbox(
        "Chọn loại phân tích:", 
        ["Phân tích Cơ bản", "So sánh Cổ phiếu", "Phân tích Ngành", "Phân tích Tin tức", "Dự đoán Xu hướng"], 
        index=0
    )
    
    if analysis_type == "Phân tích Cơ bản":
        ticker = st.text_input("Nhập mã chứng khoán (Ví dụ: AAPL, MSFT, GOOGL):", "AAPL")
        period = st.selectbox(
            "Chọn khoảng thời gian dữ liệu:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
        # Kiểm tra xem session_state đã có giá trị của 'question' chưa
        if "question" not in st.session_state:
            st.session_state.question = ""

        # Cập nhật nội dung câu hỏi mỗi khi ticker thay đổi
        default_question = f"Giá hôm nay của {ticker} là bao nhiêu?"
        if st.session_state.question == "" or ticker not in st.session_state.question:
            st.session_state.question = default_question

        # Hiển thị text_area với giá trị mặc định từ session_state
        question = st.text_area("Đặt câu hỏi về chứng khoán:", st.session_state.question, height=100)

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

    elif analysis_type == "Dự đoán Xu hướng":
        ticker = st.text_input("Nhập mã chứng khoán:", "AAPL")
        period = st.selectbox(
            "Chọn khoảng thời gian:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        prediction_days = st.slider("Số ngày dự đoán:", min_value=7, max_value=90, value=30, step=1)
        #prediction_models = st.multiselect(
        #    ["Prophet", "ARIMA", "Machine Learning", "Neural Network"],
        #    default=["Prophet"]
        #)
        prediction_method = st.selectbox(
            "Chọn phương pháp dự đoán:",
            ["Prophet (Cộng)", "Prophet (Nhân)", "So sánh các mô hình"]
        )
        predict_button = st.button("Dự đoán", use_container_width=True)

# Khu vực chính để hiển thị kết quả
# Xử lý phân tích cơ bản
if analysis_type == "Phân tích Cơ bản" and analyze_button:
    agent = get_agent()
    
    with st.spinner(f"Đang phân tích {ticker}..."):
        # Lấy dữ liệu
        financial_data = agent.fetch_financial_data(ticker, period=period)
        
        if financial_data is None:
            st.error(f"Không thể lấy dữ liệu cho {ticker}")
        else:
            # Tạo tabs
            tabs = st.tabs(["Báo cáo Tài chính", "Phân tích Kỹ thuật", "Dữ liệu Thu thập", "Hỏi Đáp"])
            
            with tabs[0]:
                # Báo cáo tài chính
                report = agent.generate_financial_report(ticker)
                
                st.components.v1.html(report, height=800, scrolling=True)
                display_financial_report(ticker)
            
            with tabs[1]:
                # Phân tích kỹ thuật
                tech_fig = agent.plot_technical_indicators(ticker)
                if tech_fig:
                    st.pyplot(tech_fig)
                else:
                    st.warning("Không thể tạo biểu đồ phân tích kỹ thuật.")
            with tabs[2]:
                # Hiển thị dữ liệu đã thu thập
                st.subheader(f"Dữ liệu thu thập cho {ticker}")
                
                if financial_data:
                    # Hiển thị dữ liệu giá
                    st.write("### Lịch sử giá")
                    st.dataframe(financial_data['price_history'])
                    
                    # Hiển thị thông tin tài chính
                    if 'financial_info' in financial_data:
                        st.write("### Thông tin tài chính")
                        st.json(financial_data['financial_info'])
                    
                    # Hiển thị thông tin công ty
                    if 'company_info' in financial_data:
                        st.write("### Thông tin công ty")
                        st.json(financial_data['company_info'])
                else:
                    st.warning(f"Không có dữ liệu nào được thu thập cho {ticker}")

            with tabs[3]:
                #    # Xử lý phần hỏi đáp
                st.subheader(f"Hỏi đáp về {ticker}")
                
                # Khởi tạo chat history trong session state nếu chưa có
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                # Khởi tạo submitted flag trong session state
                if "submitted" not in st.session_state:
                    st.session_state.submitted = False
                
                # Container chính cho chat
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                
                # Khu vực tin nhắn
                st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    if msg["type"] == "question":
                        st.markdown(
                            f'<div class="chat-message chat-question">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-message chat-answer">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Khu vực nhập liệu
                st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
                st.markdown('<div class="chat-input-box">', unsafe_allow_html=True)
                
                # Callback function để xử lý khi người dùng nhấn Enter
                def handle_question_submit():
                    if st.session_state.question_input.strip():
                        question = st.session_state.question_input
                        st.session_state.chat_history.append({
                            "type": "question",
                            "content": question
                        })
                        answer = agent.answer_question(ticker, question, financial_data)
                        st.session_state.chat_history.append({
                            "type": "answer",
                            "content": answer
                        })
                        st.session_state.submitted = True
                
                # Text area với placeholder và nút gửi
                if st.session_state.submitted:
                    st.session_state.question_input = ""
                    st.session_state.submitted = False
                    
                question = st.text_area(
                    "",
                    key="question_input",
                    placeholder=f"Nhập câu hỏi của bạn về {ticker}...",
                    height=50,
                    label_visibility="collapsed"
                )
                
                # Nút gửi nằm trong box
                st.markdown(
                    '<button class="send-button" onclick="document.querySelector(\'textarea\').dispatchEvent(new Event(\'change\'))">➤</button>',
                    unsafe_allow_html=True
                )
                
                # Xử lý khi nhấn Enter hoặc nút gửi
                if question and question.strip():
                    handle_question_submit()
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif analysis_type == "Dự đoán Xu hướng" and predict_button:
    agent = get_agent()
    
    with st.spinner(f"Đang dự đoán xu hướng cho {ticker}..."):
        # Lấy dữ liệu
        financial_data = agent.fetch_financial_data(ticker, period=period)
        
        if financial_data is None:
            st.error(f"Không thể lấy dữ liệu cho {ticker}")
        else:
            if prediction_method == "Prophet (Cộng)":
                # Dự đoán với mô hình cộng (additive)
                prophet_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days, multiplicative=False)
                
                tabs = st.tabs(["Dự đoán Giá", "Chi tiết Dự đoán"])
                with tabs[0]:    
                    if prophet_fig is not None:
                        st.subheader("Dự đoán Giá (Mô hình Cộng)")
                        st.pyplot(prophet_fig)
                        
                        st.info("""
                        **Giải thích:**
                        - **Dự đoán (điều chỉnh)** (đường đỏ đứt nét): Dự đoán với khoảng tin cậy mở rộng nhẹ
                        - **Dự đoán (gốc)** (đường xanh lá chấm chấm): Dự đoán gốc từ Prophet
                        - **Khoảng tin cậy màu cam**: Khoảng tin cậy 95% điều chỉnh, mở rộng nhẹ theo thời gian
                        """)
                with tabs[1]:
                    # Hiển thị dữ liệu dự đoán
                    st.dataframe(future_prices)
            
            elif prediction_method == "Prophet (Nhân)":               
                # Thực hiện mô hình nhân (multiplicative)
                prophet_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days, multiplicative=True)
                
                tabs = st.tabs(["Dự đoán Giá", "Chi tiết Dự đoán"])
                with tabs[0]: 
                    if prophet_fig is not None:
                        st.subheader("Dự đoán Giá (Mô hình Nhân)")
                        st.pyplot(prophet_fig)
                        
                        st.info("""
                        **Giải thích:**
                        - **Dự đoán (điều chỉnh)** (đường đỏ đứt nét): Dự đoán với khoảng tin cậy mở rộng mạnh
                        - **Dự đoán (gốc)** (đường xanh lá chấm chấm): Dự đoán gốc từ Prophet
                        - **Khoảng tin cậy màu cam**: Khoảng tin cậy 95% điều chỉnh, mở rộng mạnh theo thời gian (càng xa càng rộng)
                        """)
                with tabs[1]:        
                    # Hiển thị dữ liệu dự đoán
                    st.dataframe(future_prices)
            
            elif prediction_method == "So sánh các mô hình":
                # Mô hình cộng (additive)
                prophet_fig_add, future_prices_add = agent.predict_future_price(ticker, days=prediction_days, multiplicative=False)
                
                # Mô hình nhân (multiplicative)
                prophet_fig_mult, future_prices_mult = agent.predict_future_price(ticker, days=prediction_days, multiplicative=True)
                
                tabs = st.tabs(["So sánh Biểu đồ", "So sánh Dự đoán", "Phân tích"])

                with tabs[0]:  
                    st.subheader("So sánh mô hình dự đoán")
                    
                    col1, col2 = st.columns(2)  # Chia giao diện thành 2 cột
                    
                    with col1:
                        st.write("### Prophet (Cộng)")
                        st.pyplot(prophet_fig_add)

                    with col2:
                        st.write("### Prophet (Nhân)")
                        st.pyplot(prophet_fig_mult)

                with tabs[1]:  
                    st.subheader("So sánh dữ liệu dự đoán")
                    
                    col1, col2 = st.columns(2)  # Chia giao diện thành 2 cột
                    
                    with col1:
                        st.write("### Dự đoán Prophet (Cộng)")
                        st.dataframe(future_prices_add)

                    with col2:
                        st.write("### Dự đoán Prophet (Nhân)")
                        st.dataframe(future_prices_mult)
                
                with tabs[2]:
                    st.subheader("Phân tích sự khác biệt giữa hai mô hình")
                    
                    # Tính toán sự khác biệt giữa hai mô hình
                    if future_prices_add is not None and future_prices_mult is not None:
                        # Đổi tên cột trong từng dataframe để tránh nhầm lẫn khi merge
                        add_df = future_prices_add.copy()
                        mult_df = future_prices_mult.copy()
                        
                        for col in add_df.columns:
                            if col != 'ds':  # Không đổi tên cột ds (ngày)
                                add_df.rename(columns={col: f"{col}_add"}, inplace=True)
                        
                        for col in mult_df.columns:
                            if col != 'ds':  # Không đổi tên cột ds (ngày)
                                mult_df.rename(columns={col: f"{col}_mult"}, inplace=True)
                        
                        # Tạo dataframe chứa cả hai mô hình
                        comparison_df = pd.merge(add_df, mult_df, left_index=True, right_index=True)
                        
                        # Tính phần trăm chênh lệch giữa các dự đoán
                        comparison_df['diff_percent'] = ((comparison_df['yhat_adjusted_mult'] - 
                                                         comparison_df['yhat_adjusted_add']) / 
                                                        comparison_df['yhat_adjusted_add'] * 100).round(2)
                        
                        comparison_df['width_add'] = (comparison_df['yhat_upper_adjusted_add'] - 
                                                     comparison_df['yhat_lower_adjusted_add']).round(2)
                        
                        comparison_df['width_mult'] = (comparison_df['yhat_upper_adjusted_mult'] - 
                                                      comparison_df['yhat_lower_adjusted_mult']).round(2)
                        
                        comparison_df['width_ratio'] = (comparison_df['width_mult'] / 
                                                       comparison_df['width_add']).round(2)
                        
                        # Hiển thị bảng so sánh
                        st.write("### So sánh chênh lệch giữa hai mô hình")
                        comparison_display = comparison_df[[
                            'yhat_adjusted_add', 'yhat_adjusted_mult', 'diff_percent',
                            'width_add', 'width_mult', 'width_ratio'
                        ]].copy()
                        
                        # Đổi tên cột để dễ đọc
                        comparison_display.columns = [
                            'Dự đoán (Cộng)', 'Dự đoán (Nhân)', 'Chênh lệch (%)',
                            'Độ rộng KTC (Cộng)', 'Độ rộng KTC (Nhân)', 'Tỉ lệ độ rộng (Nhân/Cộng)'
                        ]
                        
                        st.dataframe(comparison_display)
                        
                        # Vẽ biểu đồ so sánh độ rộng khoảng tin cậy
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(comparison_df.index, comparison_df['width_add'], 
                               label='Độ rộng khoảng tin cậy (Cộng)', color='lightblue')
                        ax.plot(comparison_df.index, comparison_df['width_mult'], 
                               label='Độ rộng khoảng tin cậy (Nhân)', color='orange')
                        ax.set_xlabel('Ngày')
                        ax.set_ylabel('Độ rộng khoảng tin cậy')
                        ax.set_title('So sánh độ rộng khoảng tin cậy giữa hai mô hình')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        # Đề xuất mô hình phù hợp
                        st.write("### Đề xuất mô hình phù hợp")
                        
                        # Phân tích dữ liệu lịch sử
                        price_data = agent.data[ticker]['price_history']['Close']
                        
                        # Tính tỷ lệ tăng trưởng
                        growth_rate = (price_data.iloc[-1] / price_data.iloc[0] - 1) * 100
                        
                        # Kiểm tra biến động
                        price_volatility = price_data.pct_change().std() * 100
                        
                        st.write(f"**Tỷ lệ tăng trưởng lịch sử:** {growth_rate:.2f}%")
                        st.write(f"**Độ biến động giá:** {price_volatility:.2f}%")
                        
                        if abs(growth_rate) > 20 or price_volatility > 3:
                            st.write("**Đề xuất:** Nên sử dụng **Mô hình Nhân** do cổ phiếu có tỷ lệ tăng trưởng cao hoặc biến động lớn.")
                        else:
                            st.write("**Đề xuất:** Nên sử dụng **Mô hình Cộng** do cổ phiếu có tỷ lệ tăng trưởng ổn định và biến động thấp.")
            
            # Thêm nút tải xuống dữ liệu
            st.subheader("Tải xuống dữ liệu dự đoán")
            
            col1, col2 = st.columns(2)
            
            if 'future_prices_add' in locals() and future_prices_add is not None:
                with col1:
                    csv = future_prices_add.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction_additive.csv">Tải dữ liệu dự đoán (Cộng) (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            if 'future_prices_mult' in locals() and future_prices_mult is not None:
                with col2:
                    csv = future_prices_mult.to_csv()
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction_multiplicative.csv">Tải dữ liệu dự đoán (Nhân) (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            elif 'future_prices' in locals() and future_prices is not None:
                csv = future_prices.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction.csv">Tải dữ liệu dự đoán (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
            

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