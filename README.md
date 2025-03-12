# Finance Analysis Project

## Overview
This project is a basic tool for stock market analysis, financial reporting, and trend prediction. Built with Streamlit, the application provides an intuitive interface for investors and financial analysts to analyze stocks, compare market performance, examine sector trends, and evaluate news sentiment. Designed as an **AI-powered agent**, it can be extended with new features, making it a collaborative and evolving tool for the community.
**Real-time** market data is fetched directly from Yahoo Finance, ensuring that users always have the most up-to-date stock prices, financial statements, and market trends.

## Features

### Basic Analysis
- **Financial Reports**: Generate detailed financial reports for any ticker
- **Technical Analysis**: View technical indicators with customizable time periods
- **Data Collection**: Access and download historical price data, financial information, and company details
- **Q&A Interface**: Ask specific questions about a stock and get AI-powered answers

### Stock Comparison
- Compare performance metrics of multiple stocks
- Visualize relative returns over various time periods
- Export comparison data for further analysis

### Sector Analysis
- Analyze a stock's performance relative to its sector
- Identify competitors and sector trends
- Generate sector performance reports

### News Sentiment Analysis
- Analyze news sentiment for specific stocks
- Visual representation of sentiment trends
- Detailed news summaries with sentiment scores
- Support for multiple news sources including Google News, Yahoo Finance, Bloomberg, CNBC, and Reuters

### Trend Prediction
- Forecast stock prices using Prophet models
- Compare additive and multiplicative prediction models
- Visualize prediction confidence intervals
- Export prediction data for further analysis

## Technical Details

### Data Sources
- Yahoo Finance API: Fetches real-time and historical stock market data, including stock prices, financial statements, and company details.
- News APIs: Aggregates real-time financial news from sources like Google News, Yahoo Finance, Bloomberg, CNBC, and Reuters.
- Historical price data: Used for technical analysis

### Technologies Used
- **Streamlit**: Frontend web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Prophet**: Time series forecasting
- **Natural Language Processing**: NLTK for news sentiment analysis
- **CSS & JavaScript**: UI/UX experience

## Installation

1. Clone the repository:
```bash
git clone https://github.com/thanghd1112/Finance-Analysis-AI.git
cd finance-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Requirements
See [requirements.txt](requirements.txt) for a full list of dependencies. Main requirements include:
- streamlit
- pandas
- numpy
- matplotlib
- yfinance
- prophet
- plotly
- nltk

## Project Structure
```
finance-analysis-app/
├── app.py                    # Main Streamlit application
├── financial_Analysis.py     # Core financial analysis functionality
├── stock_Comparison.py       # Stock comparison tools
├── sector_Analysis.py        # Sector analysis functionality
├── news_sentiment.py         # News sentiment analysis
├── stock_Prediction.py       # Stock price prediction models
├── report_template.html      # HTML template for finance reports
├── chat_Interface.html       # HTML for Chat Q&A interface
├── style.css                 # Main CSS styling
├── chat_Interface.css        # CSS for Q&A interface
├── report.js                 # JavaScript for report functionality
├── chat_Interface.js         # Chat interface functionality
├── charts/                   # Generated charts storage
├── reports/                  # Generated reports storage
└── __pycache__/              # Python cache files
```

## Usage Examples

### Basic Stock Analysis
Enter a stock ticker (e.g., AAPL, MSFT, GOOGL) and select a time period to view comprehensive financial analysis including price history, technical indicators, and financial reports.

### Stock Comparison
Enter multiple stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL) to compare their performance over different time periods with visual charts and performance metrics.

### News Sentiment
Analyze the sentiment of recent news articles about a specific stock to gauge market perception and potential impact on stock price.

### Price Prediction
Use the trend prediction feature to forecast future stock prices based on historical data using different forecasting models.

## Contributing
Contributions to improve the Finance Analysis Application are welcome. Please feel free to submit a Pull Request.

## Acknowledgements
- Yahoo Finance for providing stock data APIs
- Streamlit for the web application framework
- Prophet for time series forecasting capabilities
- All other open-source libraries that made this project possible

## Disclaimer
This application is for educational and informational purposes only. The financial analysis and predictions should not be considered as investment advice. Always consult with a qualified financial advisor before making investment decisions.
