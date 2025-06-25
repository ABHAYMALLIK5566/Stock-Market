# 📊 Indian Stock Market Analysis Agent

A comprehensive Python-based stock market analysis tool specifically designed for the Indian stock market. This agent provides real-time market analysis, technical indicators, fundamental analysis, and investment recommendations across multiple sectors.

## 🚀 Features

### 📈 Market Analysis
- **Real-time Market Overview**: Track NIFTY50, SENSEX, and BANKNIFTY indices
- **Sector-wise Performance**: Analyze 30+ sectors with comprehensive stock coverage
- **Technical Indicators**: RSI, MACD, Moving Averages, Volume Analysis
- **Fundamental Analysis**: PE Ratio, Market Cap, ROE, Debt-to-Equity

### 🎯 Investment Recommendations
- **Short-term Opportunities** (1-7 days)
- **Mid-term Selections** (1-3 months)
- **Long-term Wealth Creators** (6+ months)
- **Sector-specific Insights**: Defense, Aerospace, Renewable Energy, etc.

### 📰 News & Sentiment Analysis
- **Market News Integration**: Real-time news from multiple sources
- **Sentiment Analysis**: AI-powered sentiment scoring
- **Sector-specific News**: Targeted news for specific sectors

### 🛡️ Specialized Analysis
- **Defense & Aerospace**: Government contracts and strategic importance
- **Emerging Sectors**: EV, Fintech, E-commerce, Renewable Energy
- **PSU Analysis**: Government-owned companies
- **Sector Correlations**: Identify correlated sector movements

## 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Market
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
npm install
```

3. **Set up API keys** (optional):
   - Get free API key from [NewsAPI](https://newsapi.org/)
   - Get free API key from [Alpha Vantage](https://alphavantage.co/)

## 🔧 Configuration

**Keep your API keys private!**
- Do **not** hardcode your API keys in the codebase.
- Use a `.env` file or environment variables to store your API keys securely.
- Add `.env` to your `.gitignore` so it is never pushed to GitHub.

Example `.env` file:
```
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
```

Edit the API keys in `stock_market_agent.py` to load from environment variables:
```python
import os
self.news_api_key = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "YOUR_ALPHAVANTAGE_KEY")
```

## 🗂️ Clean Project Structure

Project folder should look like this:

```
Market/
  whatsapp_api.py
  index.js
  db.py
  requirements.txt
  package.json
  stock_market_agent.py
  streamlit_app.py
  README.md
```

**Optional files/folders:**
- `marketbot.db` (SQLite database, keep if you want to preserve user data)
- `node_modules/` (Node.js dependencies, can be regenerated with `npm install`)
- `__pycache__/` (Python bytecode, can be deleted, will be regenerated)
- `_IGNORE_session/` (browser/automation session files, can be deleted for a truly clean repo)

## 🏁 Running the Project

1. **Start the Python backend:**
   ```bash
   python whatsapp_api.py
   ```
2. **Start the Node.js WhatsApp bot:**
   ```bash
   node index.js
   ```
3. **(Optional) Run the Streamlit UI:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🧹 Cleaning Up
- Remove any `.db`, `.pyc`, `node_modules/`, `__pycache__/`, or `_IGNORE_session/` folders/files for a production deployment or to keep your repo clean.

## 📚 Usage
- Interact with the WhatsApp bot using the commands listed in the `help` command.
- Use the Streamlit UI for a web-based experience.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code comments for detailed explanations

## 🔄 Updates

The agent is regularly updated with:
- New stock symbols
- Enhanced technical indicators
- Improved sentiment analysis
- Additional sector coverage
- Better recommendation algorithms

## 👤 Author
- Abhay Mallik