import streamlit as st
import matplotlib.pyplot as plt
from financial_Analysis import AIAgent
import pandas as pd
import base64
from io import BytesIO

# Tiêu đề ứng dụng
st.set_page_config(page_title="Phân tích Tài chính", layout="wide")
st.title("Ứng dụng Phân tích Tài chính")

# Khởi tạo AI Agent
@st.cache_resource
def get_agent():
    return AIAgent(name="Streamlit Finance Bot")

agent = get_agent()

# Hàm chuyển đổi biểu đồ matplotlib thành hình ảnh cho streamlit
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# Sidebar để nhập thông tin
with st.sidebar:
    st.header("Thông tin Phân tích")
    ticker = st.text_input("Nhập mã chứng khoán (Ví dụ: AAPL, MSFT, GOOGL):", "AAPL")
    period = st.selectbox(
        "Chọn khoảng thời gian dữ liệu:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    prediction_days = st.slider("Số ngày dự đoán:", min_value=7, max_value=90, value=30, step=1)
    
    analyze_button = st.button("Phân tích", use_container_width=True)

# Khu vực chính để hiển thị kết quả
if analyze_button:
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
else:
    # Hiển thị hướng dẫn ban đầu
    st.info("""
    ### Hướng dẫn sử dụng
    1. Nhập mã chứng khoán (ví dụ: AAPL cho Apple, MSFT cho Microsoft) vào ô bên trái.
    2. Chọn khoảng thời gian dữ liệu bạn muốn phân tích.
    3. Điều chỉnh số ngày bạn muốn dự đoán giá trong tương lai.
    4. Nhấn nút "Phân tích" để xem kết quả.
    
    Lưu ý: Ứng dụng sử dụng dữ liệu từ Yahoo Finance. Đảm bảo bạn nhập đúng mã chứng khoán.
    """)

# Thêm footer
st.markdown("---")
st.caption("Ứng dụng phân tích tài chính và dự đoán giá cổ phiếu")