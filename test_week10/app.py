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

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân tích Tài chính",
    layout="wide"

)

def load_all_css():
    # Load main CSS
    with open("style.css", "r", encoding="utf-8") as f:
        main_css = f.read()
    
    # Load chat interface CSS
    with open("chat_Interface.css", "r", encoding="utf-8") as f:
        chat_css = f.read()
    
    # Combine all CSS
    all_css = main_css + chat_css
    
    # Additional styles to fix overlapping
    additional_css = """
    /* Fix dropdown text overlapping */
    .stSelectbox {
        margin-bottom: 25px;
    }
    
    /* Fix sidebar spacing */
    [data-testid="stSidebar"] {
        padding: 2rem 1rem;
    }
    
    /* Fix selectbox options */
    .stSelectbox div[data-baseweb="select"] {
        max-height: 300px;
        overflow-y: auto;
    }
    """
    
    # Apply all CSS
    st.markdown(f'<style>{all_css}{additional_css}</style>', unsafe_allow_html=True)

# Call this function at the top of your app
load_all_css()

# Xóa các phần markdown CSS riêng lẻ sau đó
# Thêm CSS tùy chỉnh
st.markdown("""
    <style>
    /* Fix text overlapping */
    .stMarkdown {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    /* Fix dataframe overflow */
    .dataframe {
        width: 100%;
        overflow-x: auto;
    }
    
    /* Fix text input width */
    .stTextInput > div > div > input {
        width: 100%;
    }
    
    /* Fix sidebar width */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Fix main content area */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Fix tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        min-height: 50px;
        white-space: pre-wrap;
        overflow-wrap: break-word;
        word-wrap: break-word;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    
    /* Fix markdown text */
    .stMarkdown p {
        margin-bottom: 1rem;
    }
    
    /* Fix code blocks */
    .stCodeBlock {
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    /* Fix tables */
    .stTable {
        width: 100%;
        overflow-x: auto;
    }
    
    /* Fix expander */
    .streamlit-expanderHeader {
        font-size: 1rem;
        padding: 1rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
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

if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

def handle_tab_change(tab_index):
    st.session_state.active_tab = tab_index
    if st.session_state.question_input.strip():
        # Set active tab to Q&A tab (index 3)
        st.session_state.active_tab = 3

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
def setup_chat_interface(ticker, agent, financial_data):
    """Set up the Q&A chat interface with improved styling"""
    st.subheader(f"Hỏi đáp về {ticker}")
    
    # Khởi tạo chat history trong session state nếu chưa có
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Hiển thị hướng dẫn khi không có lịch sử chat
    if not st.session_state.chat_history:
        st.info(f"Hãy nhập câu hỏi của bạn về {ticker} vào khung dưới đây.")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        # Xóa div không cần thiết và thay bằng container đơn giản hơn
        for msg in st.session_state.chat_history:
            if msg["type"] == "question":
                st.markdown(
                    f'<div style="background-color: #DCF8C6; padding: 10px; border-radius: 8px; margin: 5px 0; text-align: right; max-width: 80%; float: right; clear: both;">{msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Xử lý code blocks trong câu trả lời
                content = msg["content"]
                if "```python" in content:
                    parts = content.split("```python")
                    formatted_content = parts[0]
                    for i in range(1, len(parts)):
                        if "```" in parts[i]:
                            code, rest = parts[i].split("```", 1)
                            formatted_content += f'<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; border-left: 3px solid #2196F3; margin: 10px 0;"><pre><code>{code}</code></pre></div>{rest}'
                        else:
                            formatted_content += parts[i]
                    content = formatted_content
                
                st.markdown(
                    f'<div style="background-color: #FFFFFF; padding: 10px; border-radius: 8px; margin: 5px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.1); max-width: 80%; float: left; clear: both;">{content}</div>',
                    unsafe_allow_html=True
                )
        
        # Thêm phần tử xóa float để tránh các vấn đề bố cục
        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)
    
    # Thêm biến kiểm tra để xử lý việc gửi câu hỏi
    if "submit_question" not in st.session_state:
        st.session_state.submit_question = False
    
    # Callback function for handling user input
    def handle_question_submit():
        if st.session_state.question_input.strip():
            question = st.session_state.question_input
            # Add question to chat history
            st.session_state.chat_history.append({
                "type": "question",
                "content": question
            })
            # Get answer with try-except để bắt lỗi
            try:
                answer = agent.answer_question(ticker, question, financial_data)
                if not answer or answer.strip() == "":
                    answer = f"Xin lỗi, tôi không tìm thấy thông tin phù hợp cho câu hỏi của bạn về {ticker}."
            except Exception as e:
                answer = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
            
            # Add answer to chat history
            st.session_state.chat_history.append({
                "type": "answer",
                "content": answer
            })
            # Clear input và đánh dấu đã gửi
            st.session_state.question_input = ""
            st.session_state.submit_question = True
    
    # Tạo container cho input field
    question = st.text_input(
        "",
        key="question_input",
        placeholder=f"Nhập câu hỏi của bạn về {ticker}",
        on_change=handle_question_submit
    )
    
    # Thêm JavaScript để tự động cuộn xuống cuối khi có tin nhắn mới
    st.markdown("""
    <script>
    // Scroll to bottom of chat container
    function scrollChatToBottom() {
        const messages = document.querySelectorAll('[data-testid="stMarkdown"]');
        if (messages.length > 0) {
            messages[messages.length - 1].scrollIntoView();
        }
    }
    
    // Run after page load and after each update
    setTimeout(scrollChatToBottom, 500);
    </script>
    """, unsafe_allow_html=True)
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
            ["Prophet Additive", "Prophet Multiplicative", "So sánh mô hình"]
        )
        predict_button = st.button("Dự đoán", use_container_width=True)

# Khu vực chính để hiển thị kết quả
# Xử lý phân tích cơ bản
if analysis_type == "Phân tích Cơ bản" and (analyze_button or st.session_state.analyzed_data is not None):
    agent = get_agent()
    
    # Chỉ fetch dữ liệu mới khi nhấn nút phân tích
    if analyze_button:
        with st.spinner(f"Đang phân tích {ticker}..."):
            financial_data = agent.fetch_financial_data(ticker, period=period)
            st.session_state.analyzed_data = financial_data
    else:
        financial_data = st.session_state.analyzed_data
    
    if financial_data is None:
        st.error(f"Không thể lấy dữ liệu cho {ticker}")
    else:
        tab_options = ["Báo cáo Tài chính", "Phân tích Kỹ thuật", "Dữ liệu Thu thập", "Hỏi Đáp"]
        active_tab = st.session_state.active_tab

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(tab_options)
            
        with tab1:
            # Báo cáo tài chính
            report = agent.generate_financial_report(ticker)
            
            st.components.v1.html(report, height=800, scrolling=True)
            display_financial_report(ticker)
    
        with tab2:
            # Phân tích kỹ thuật
            tech_fig = agent.plot_technical_indicators(ticker)
            if tech_fig:
                st.pyplot(tech_fig)
            else:
                st.warning("Không thể tạo biểu đồ phân tích kỹ thuật.")

        with tab3:
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

        with tab4:
            
            # Hiển thị giao diện chat
            setup_chat_interface(ticker, agent, financial_data)

elif analysis_type == "Dự đoán Xu hướng" and predict_button:
    agent = get_agent()
    
    with st.spinner(f"Đang dự đoán xu hướng cho {ticker}..."):
        # Lấy dữ liệu
        financial_data = agent.fetch_financial_data(ticker, period=period)
        
        if financial_data is None:
            st.error(f"Không thể lấy dữ liệu cho {ticker}")
        else:
            if prediction_method == "Prophet Additive":
                # Dự đoán với mô hình cộng (additive)
                prophet_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days, multiplicative=False)
                
                tabs = st.tabs(["Dự đoán Giá", "Chi tiết Dự đoán"])
                with tabs[0]:    
                    if prophet_fig is not None:
                        st.subheader("Dự đoán Giá (Mô hình Cộng)")
                        st.pyplot(prophet_fig)
                        
                
                with tabs[1]:
                    # Hiển thị dữ liệu dự đoán
                    st.dataframe(future_prices)
            
            elif prediction_method == "Prophet Multiplicative":               
                # Thực hiện mô hình nhân (multiplicative)
                prophet_fig, future_prices = agent.predict_future_price(ticker, days=prediction_days, multiplicative=True)
                
                tabs = st.tabs(["Dự đoán Giá", "Chi tiết Dự đoán"])
                with tabs[0]: 
                    if prophet_fig is not None:
                        st.subheader("Dự đoán Giá (Mô hình Nhân)")
                        st.pyplot(prophet_fig)
                        
                        
                        
                
                with tabs[1]:        
                    # Hiển thị dữ liệu dự đoán
                    st.dataframe(future_prices)
            
            elif prediction_method == "So sánh mô hình":
                # Mô hình cộng (additive)
                prophet_fig_add, future_prices_add = agent.predict_future_price(ticker, days=prediction_days, multiplicative=False)
                
                # Mô hình nhân (multiplicative)
                prophet_fig_mult, future_prices_mult = agent.predict_future_price(ticker, days=prediction_days, multiplicative=True)
                
                tabs = st.tabs(["So sánh Biểu đồ", "So sánh Dự đoán", "So sánh Biên độ"])

                with tabs[0]:  
                    st.subheader("So sánh mô hình dự đoán")
                    
                    col1, col2 = st.columns(2)  # Chia giao diện thành 2 cột
                    
                    with col1:
                        st.write("### Prophet Additive")
                        st.pyplot(prophet_fig_add)

                    with col2:
                        st.write("### Prophet Multiplicative")
                        st.pyplot(prophet_fig_mult)

                with tabs[1]:  
                    st.subheader("So sánh dữ liệu dự đoán")
                    
                    col1, col2 = st.columns(2)  # Chia giao diện thành 2 cột
                    
                    with col1:
                        st.write("### Dự đoán Prophet Additive")
                        st.dataframe(future_prices_add)

                    with col2:
                        st.write("### Dự đoán Prophet Multiplicative")
                        st.dataframe(future_prices_mult)
                
                with tabs[2]:
                    st.subheader("So sánh biên độ dao động theo thời gian")
                    
                    # Tính biên độ cho cả hai mô hình
                    add_ranges = []
                    mult_ranges = []
                    
                    sample_days = [0, prediction_days // 4, prediction_days // 2, prediction_days - 1]
                    for day_idx in sample_days:
                        if day_idx < len(future_prices_add) and day_idx < len(future_prices_mult):
                            # Mô hình cộng
                            date = future_prices_add.index[day_idx].strftime('%Y-%m-%d')
                            pred_add = future_prices_add['yhat'].iloc[day_idx]
                            low_add = future_prices_add['yhat_lower'].iloc[day_idx]
                            high_add = future_prices_add['yhat_upper'].iloc[day_idx]
                            width_add = high_add - low_add
                            percent_add = width_add / pred_add * 100
                            
                            # Mô hình nhân
                            pred_mult = future_prices_mult['yhat'].iloc[day_idx]
                            low_mult = future_prices_mult['yhat_lower'].iloc[day_idx]
                            high_mult = future_prices_mult['yhat_upper'].iloc[day_idx]
                            width_mult = high_mult - low_mult
                            percent_mult = width_mult / pred_mult * 100
                            
                            add_ranges.append({
                                'Ngày': date,
                                'Dự đoán': pred_add,
                                'Biên độ': width_add,
                                'Phần trăm': percent_add
                            })
                            
                            mult_ranges.append({
                                'Ngày': date,
                                'Dự đoán': pred_mult,
                                'Biên độ': width_mult,
                                'Phần trăm': percent_mult
                            })
                    
                    # Hiển thị bảng so sánh
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Biên độ - Mô hình Cộng")
                        st.table(pd.DataFrame(add_ranges).set_index('Ngày').round(2))
                        
                        # Tính tỷ lệ tăng
                        if len(add_ranges) > 1:
                            increase = add_ranges[-1]['Phần trăm'] / add_ranges[0]['Phần trăm'] - 1
                            st.info(f"Biên độ tăng {increase*100:.1f}% sau {prediction_days} ngày")
                    
                    with col2:
                        st.write("### Biên độ - Mô hình Nhân")
                        st.table(pd.DataFrame(mult_ranges).set_index('Ngày').round(2))
                        
                        # Tính tỷ lệ tăng
                        if len(mult_ranges) > 1:
                            increase = mult_ranges[-1]['Phần trăm'] / mult_ranges[0]['Phần trăm'] - 1
                            st.info(f"Biên độ tăng {increase*100:.1f}% sau {prediction_days} ngày")
            
            # Thêm nút tải xuống dữ liệu
            st.subheader("Tải xuống dữ liệu dự đoán")
            
            col1, col2 = st.columns(2)
            
            if 'future_prices_add' in locals():
                csv = future_prices_add.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                with col1:
                    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction_add.csv">Tải dữ liệu dự đoán (Cộng) (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            if 'future_prices_mult' in locals():
                csv = future_prices_mult.to_csv()
                b64 = base64.b64encode(csv.encode()).decode()
                with col2:
                    href = f'<a href="data:file/csv;base64,{b64}" download="{ticker}_prediction_mult.csv">Tải dữ liệu dự đoán (Nhân) (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            elif 'future_prices' in locals():
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