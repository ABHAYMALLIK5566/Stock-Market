from flask import Flask, request, jsonify
from stock_market_agent import StockMarketAgent
from textblob import TextBlob
import threading
import time
import db
import requests

app = Flask(__name__)
agent = StockMarketAgent()

SEND_MESSAGE_URL = 'http://localhost:3000/send_message'  # Node.js bot endpoint

# Track user news offsets in memory (session-based, not persistent)
user_news_offset = {}

# --- Helper Functions ---
def format_news_item(n):
    # Compose a detailed, AI-style news summary with link
    summary = f"\n📰 *{n['title']}*\n"
    if n.get('description'):
        summary += f"_{n['description']}_\n"
    summary += f"Source: {n['source']} | Sentiment: {n['sentiment'].upper()}\n"
    if n.get('url'):
        summary += f"[Read more]({n['url']})\n"
    return summary

def get_top_gainers_losers():
    # Stub: Use agent.sectors to get a few stocks and random changes
    # In production, fetch real data
    gainers = [
        {'symbol': 'TCS.NS', 'change': '+3.2%'},
        {'symbol': 'RELIANCE.NS', 'change': '+2.8%'},
        {'symbol': 'INFY.NS', 'change': '+2.5%'}
    ]
    losers = [
        {'symbol': 'SBIN.NS', 'change': '-2.1%'},
        {'symbol': 'ITC.NS', 'change': '-1.7%'},
        {'symbol': 'HDFCBANK.NS', 'change': '-1.5%'}
    ]
    return gainers, losers

def get_ipo_events():
    # Stub: Sample IPO/events
    return [
        {'name': 'ABC Tech IPO', 'date': '2024-07-10', 'desc': 'Upcoming tech IPO.'},
        {'name': 'XYZ Infra IPO', 'date': '2024-07-15', 'desc': 'Infrastructure sector IPO.'}
    ]

def send_whatsapp_message(phone, message):
    try:
        requests.post(SEND_MESSAGE_URL, json={"to": phone, "message": message})
    except Exception as e:
        print(f"Failed to send WhatsApp message to {phone}: {e}")

def price_alert_worker():
    while True:
        alerts = db.get_alerts()
        for alert in alerts:
            symbol = alert['symbol']
            price = alert['price']
            phone = alert['phone']
            # Get current price (use agent.get_stock_report_data)
            data = agent.get_stock_report_data(symbol)
            if data and 'current_price' in data:
                current = data['current_price']
                if (current >= price):
                    msg = f"🔔 Price Alert: {symbol} has reached ₹{current:.2f} (target: ₹{price:.2f})"
                    send_whatsapp_message(phone, msg)
                    db.mark_alert_triggered(alert['id'])
        time.sleep(300)  # Check every 5 minutes

# --- Main WhatsApp Handler ---
@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    data = request.get_json()
    msg = data.get('body', '').strip()
    user = data.get('from', 'default')
    msg_lower = msg.lower()
    reply = None

    # --- News with pagination ---
    if msg_lower == "news":
        news = agent.get_general_news_data()
        if not news:
            reply = "Sorry, I couldn't find any recent stock market news right now."
        else:
            offset = user_news_offset.get(user, 0)
            articles = news[offset:offset+3]
            if not articles:
                reply = "No more news available."
            else:
                reply = "Here are the latest stock market news headlines with sentiment analysis and links:\n"
                for n in articles:
                    reply += format_news_item(n)
                if offset + 3 < len(news):
                    reply += "\nReply with 'news more' or 'more news' to see more articles."
                user_news_offset[user] = offset + 3
    elif msg_lower in ["news more", "more news"]:
        news = agent.get_general_news_data()
        offset = user_news_offset.get(user, 0)
        articles = news[offset:offset+3]
        if not articles:
            reply = "No more news available."
        else:
            reply = "More news headlines:\n"
            for n in articles:
                reply += format_news_item(n)
            if offset + 3 < len(news):
                reply += "\nReply with 'news more' or 'more news' to see more articles."
            user_news_offset[user] = offset + 3

    # --- Watchlist ---
    elif msg_lower.startswith('add watchlist '):
        symbol = msg.split('add watchlist ', 1)[1].strip().upper()
        db.add_watchlist(user, symbol)
        reply = f"Added {symbol} to your watchlist."
    elif msg_lower.startswith('remove watchlist '):
        symbol = msg.split('remove watchlist ', 1)[1].strip().upper()
        db.remove_watchlist(user, symbol)
        reply = f"Removed {symbol} from your watchlist."
    elif msg_lower == 'show watchlist':
        wl = db.get_watchlist(user)
        if not wl:
            reply = "Your watchlist is empty. Add stocks with 'add watchlist SYMBOL'."
        else:
            reply = "Your watchlist: " + ", ".join(wl)

    # --- Price Alerts ---
    elif msg_lower.startswith('alert '):
        try:
            parts = msg.split()
            symbol = parts[1].upper()
            price = float(parts[2])
            db.add_alert(user, symbol, price)
            reply = f"Alert set for {symbol} at ₹{price:.2f}. You will be notified when the price is reached."
        except Exception:
            reply = "Usage: alert SYMBOL PRICE (e.g., alert TCS.NS 4000)"

    # --- Daily Summary Subscription ---
    elif msg_lower == 'subscribe daily summary':
        db.subscribe_daily(user)
        reply = "You are now subscribed to the daily market summary. (Demo: No actual messages will be sent.)"
    elif msg_lower == 'unsubscribe daily summary':
        db.unsubscribe_daily(user)
        reply = "You have unsubscribed from the daily market summary."

    # --- Top Gainers/Losers ---
    elif msg_lower == 'top gainers':
        gainers, _ = get_top_gainers_losers()
        reply = "Top Gainers Today:\n" + "\n".join([f"{g['symbol']} ({g['change']})" for g in gainers])
    elif msg_lower == 'top losers':
        _, losers = get_top_gainers_losers()
        reply = "Top Losers Today:\n" + "\n".join([f"{l['symbol']} ({l['change']})" for l in losers])

    # --- Sector Performance ---
    elif msg_lower.startswith('sector '):
        sector = msg.split('sector ', 1)[1].strip()
        # Use agent.get_sector_specific_news if available
        news = agent.get_sector_specific_news(sector)
        if not news:
            reply = f"No news found for sector '{sector}'."
        else:
            reply = f"Sector: {sector}\n"
            for n in news:
                reply += f"- {n['title']} (Sentiment: {n['sentiment'].upper()})\n"

    # --- Fundamentals ---
    elif msg_lower.startswith('fundamentals '):
        symbol = msg.split('fundamentals ', 1)[1].strip().upper()
        data = agent.get_stock_report_data(symbol)
        if not data:
            reply = f"No data available for {symbol}."
        else:
            f = data['fundamental']
            reply = (
                f"Fundamentals for {symbol}:\n"
                f"PE Ratio: {f.get('pe_ratio', 'N/A')}\n"
                f"ROE: {f.get('roe', 'N/A')}\n"
                f"Market Cap: {f.get('market_cap', 'N/A')}\n"
                f"Dividend Yield: {f.get('dividend_yield', 'N/A')}\n"
                f"Debt/Equity: {f.get('debt_to_equity', 'N/A')}\n"
                f"EPS: {f.get('eps', 'N/A')}\n"
            )

    # --- Technical Signals ---
    elif msg_lower.startswith('signals '):
        symbol = msg.split('signals ', 1)[1].strip().upper()
        data = agent.get_stock_report_data(symbol)
        if not data:
            reply = f"No data available for {symbol}."
        else:
            t = data['technical']
            reply = (
                f"Technical signals for {symbol}:\n"
                f"RSI: {t.get('rsi', 'N/A')}\n"
                f"MACD: {t.get('macd', 'N/A')}\n"
                f"MA(20): {t.get('ma_20', 'N/A')}\n"
                f"MA(50): {t.get('ma_50', 'N/A')}\n"
                f"Volume Ratio: {t.get('volume_ratio', 'N/A')}\n"
                f"Signals: {', '.join(data['signals']) if data['signals'] else 'N/A'}\n"
            )

    # --- IPO/Events ---
    elif msg_lower in ['ipo', 'events']:
        events = get_ipo_events()
        reply = "Upcoming IPOs/Events:\n"
        for e in events:
            reply += f"- {e['name']} on {e['date']}: {e['desc']}\n"

    # --- Sentiment Analysis ---
    elif msg_lower.startswith('sentiment '):
        text = msg.split('sentiment ', 1)[1].strip()
        if not text:
            reply = "Please provide text for sentiment analysis."
        else:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            reply = f"Sentiment: {sentiment.upper()} (score: {polarity:.2f})"

    # --- News & Analysis (existing) ---
    elif msg_lower.startswith("analyze "):
        symbol = msg.split("analyze ", 1)[1].strip().upper()
        data = agent.get_stock_report_data(symbol)
        if not data:
            reply = f"Sorry, I couldn't find any data for {symbol}. Please check the symbol and try again."
        else:
            reply = (
                f"Here's a detailed analysis for *{data['symbol']} - {data['longName']}*:\n"
                f"Current Price: ₹{data['current_price']:.2f} ({data['price_change']:+.2f}%)\n"
                f"Status: {data['recommendation']['action']} (Confidence: {data['recommendation']['confidence']})\n"
                f"Sector: {data['sector']} | Industry: {data['industry']}\n\n"
                f"*AI Outlook:* {data['ai_outlook']}\n\n"
                f"Short-term Target: ₹{data['price_targets']['short']['target']:.0f}, Stop Loss: ₹{data['price_targets']['short']['stop_loss']:.0f}\n"
                f"Mid-term Target: ₹{data['price_targets']['mid']['target']:.0f}, Stop Loss: ₹{data['price_targets']['mid']['stop_loss']:.0f}\n"
                f"Long-term Target: ₹{data['price_targets']['long']['target']:.0f}, Stop Loss: ₹{data['price_targets']['long']['stop_loss']:.0f}\n\n"
            )
            if data['news']:
                n = data['news'][0]
                reply += f"Latest News: *{n['title']}*\n"
                if n.get('description'):
                    reply += f"_{n['description']}_\n"
                reply += f"Source: {n['source']} | Sentiment: {n['sentiment'].upper()}\n"
                if n.get('url'):
                    reply += f"[Read more]({n['url']})\n"

    # --- Help ---
    elif msg_lower == 'help':
        reply = (
            "*Stock Market Bot Commands:*\n"
            "- news: Latest market news (reply 'news more' for more)\n"
            "- analyze SYMBOL: Detailed stock analysis\n"
            "- add watchlist SYMBOL: Add stock to your watchlist\n"
            "- remove watchlist SYMBOL: Remove stock from your watchlist\n"
            "- show watchlist: Show your watchlist\n"
            "- alert SYMBOL PRICE: Set a price alert\n"
            "- subscribe daily summary: Get daily market summary\n"
            "- unsubscribe daily summary: Stop daily summary\n"
            "- top gainers: Top gainers today\n"
            "- top losers: Top losers today\n"
            "- sector SECTORNAME: Sector news\n"
            "- fundamentals SYMBOL: Key stats\n"
            "- signals SYMBOL: Technical signals\n"
            "- ipo / events: Upcoming IPOs/events\n"
            "- sentiment TEXT: Analyze sentiment of any text\n"
            "- help: Show this help message\n"
        )

    # --- Fallback ---
    else:
        # Ignore unrecognized messages (do not reply)
        return ('', 204)

    return jsonify({"reply": reply})

# --- Start price alert worker thread ---
threading.Thread(target=price_alert_worker, daemon=True).start()

if __name__ == "__main__":
    app.run(port=5005) 