import streamlit as st
from stock_market_agent import StockMarketAgent

st.set_page_config(page_title="Indian Stock Market Analysis", layout="wide")
st.title("📈 Indian Stock Market Analysis & Insights")

st.markdown("""
Enter an NSE stock symbol (e.g., `HAL.NS`, `TCS.NS`, `BEL.NS`, `ZENTEC.NS`).
""")

agent = StockMarketAgent(verbose=False)

# General Market News Section
st.subheader("📰 General Market News")
general_news = agent.get_general_news_data()

show_raw_news = st.checkbox("Show Raw News Data")

if not general_news:
    st.info("No recent general news found.")
else:
    for n in general_news:
        st.markdown(f"**[{n['date']}]** {n['sentiment_emoji']} {n['title']}{n['lang_note']}")
        st.markdown(f"*{n['description']}*")
        st.markdown(f"Source: {n['source']} | Sentiment: {n['sentiment'].upper()}")
        st.markdown(f"[Read more]({n['url']})")
        if show_raw_news:
            st.json(n)
        st.write("---")

symbol = st.text_input("Stock Symbol", value="HAL.NS")

if st.button("Analyze") or symbol:
    with st.spinner(f"Analyzing {symbol}..."):
        data = agent.get_stock_report_data(symbol)
    if not data:
        st.error(f"No data available for {symbol}.")
    else:
        st.header(f"{data['symbol']} - {data['longName']}")
        st.markdown(f"**Sector:** {data['sector']} | **Industry:** {data['industry']}")
        st.markdown(f"**Current Price:** ₹{data['current_price']:.2f} ({data['price_change']:+.2f}%)")
        st.markdown(f"**Status:** {data['recommendation']['action']} (Confidence: {data['recommendation']['confidence']}) {data['stars']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Technicals")
            t = data['technical']
            st.write(f"RSI: {t.get('rsi', 'N/A'):.2f}")
            st.write(f"MACD: {t.get('macd', 'N/A'):.2f}")
            st.write(f"MA(20): {t.get('ma_20', 'N/A'):.2f}")
            st.write(f"MA(50): {t.get('ma_50', 'N/A'):.2f}")
            st.write(f"Volume Ratio: {t.get('volume_ratio', 'N/A'):.2f}")
            st.write(f"Signals: {', '.join(data['signals']) if data['signals'] else 'N/A'}")
        with col2:
            st.subheader("Fundamentals")
            f = data['fundamental']
            st.write(f"PE Ratio: {f.get('pe_ratio', 'N/A')}")
            st.write(f"ROE: {f.get('roe', 'N/A')}")
            st.write(f"Market Cap: {f.get('market_cap', 'N/A')}")
            st.write(f"Dividend Yield: {f.get('dividend_yield', 'N/A')}")
            st.write(f"Debt/Equity: {f.get('debt_to_equity', 'N/A')}")
            st.write(f"EPS: {f.get('eps', 'N/A')}")
        
        st.subheader("📰 News & Sentiment")
        if not data['news']:
            st.info("No recent news found for this stock.")
        else:
            for n in data['news']:
                st.markdown(f"**[{n['date']}]** {n['sentiment_emoji']} {n['title']}{n['lang_note']}")
                st.markdown(f"*{n['description']}*")
                st.markdown(f"Source: {n['source']} | Sentiment: {n['sentiment'].upper()}")
                st.markdown(f"[Read more]({n['url']})")
                st.write("---")
        
        st.subheader("🎯 Price Targets")
        pt = data['price_targets']
        st.write(f"Short-term (1-7d): ₹{pt['short']['target']:.0f} (Target), ₹{pt['short']['stop_loss']:.0f} (Stop Loss)")
        st.write(f"Mid-term (1-3mo): ₹{pt['mid']['target']:.0f} (Target), ₹{pt['mid']['stop_loss']:.0f} (Stop Loss)")
        st.write(f"Long-term (6mo+): ₹{pt['long']['target']:.0f} (Target), ₹{pt['long']['stop_loss']:.0f} (Stop Loss)")
        
        st.subheader("🤖 AI Outlook")
        st.success(data['ai_outlook']) 