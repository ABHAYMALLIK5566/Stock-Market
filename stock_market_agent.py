import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import warnings
import os
import sys
import contextlib
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class StockMarketAgent:
    def __init__(self, verbose=False):
        # Load API keys from environment variables for privacy
        # Recommend using a .env file for local development
        self.news_api_key = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "YOUR_ALPHAVANTAGE_KEY")
        self.verbose = verbose
        
        # Indian Stock Market Indices and Major Stocks
        self.indices = {
            'NIFTY50': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK'
        }
        
        # Major Indian stocks by sector (Comprehensive Coverage)
        self.sectors = {
            # Core Banking & Financial Services
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS', 'FEDERALBNK.NS'],
            'NBFC': ['BAJFINANCE.NS', 'SHRIRAMFIN.NS', 'LICHSGFIN.NS', 'PFC.NS', 'RECLTD.NS', 'MUTHOOTFIN.NS'],
            'Insurance': ['LICI.NS', 'ICICIGI.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS'],
            
            # Technology & Digital
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTI.NS', 'COFORGE.NS'],
            'Telecom': ['BHARTIARTL.NS', 'RJIO.NS', 'IDEA.NS', 'INDUS.NS', 'TEJAS.NS'],
            
            # Healthcare & Pharmaceuticals
            'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS', 'BIOCON.NS', 'TORNTPHARM.NS'],
            'Healthcare': ['APOLLOHOSP.NS', 'FORTIS.NS', 'MAXHEALTH.NS', 'NARAYHEALTH.NS', 'METROPOLIS.NS'],
            
            # Manufacturing & Industrial
            'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'EICHERMOT.NS', 'ASHOKLEY.NS'],
            'Auto_Ancillary': ['BOSCHLTD.NS', 'MOTHERSON.NS', 'BALKRISIND.NS', 'MRF.NS', 'APOLLOTYRE.NS', 'BHARATFORG.NS'],
            'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'COALINDIA.NS', 'SAIL.NS', 'NMDC.NS'],
            'Mining': ['COALINDIA.NS', 'NMDC.NS', 'VEDL.NS', 'HINDZINC.NS', 'MOIL.NS'],
            
            # Defense & Aerospace
            'Defense': ['HAL.NS', 'BEL.NS', 'BHEL.NS', 'BEML.NS', 'GRSE.NS', 'MAZAGON.NS', 'MIDHANI.NS'],
            'Aeronautics': ['HAL.NS', 'TANLA.NS', 'CENTUM.NS', 'ASTRAZEN.NS', 'DYNAMATECH.NS'],
            'Railways': ['IRCTC.NS', 'RVNL.NS', 'RAILTEL.NS', 'IRFC.NS', 'CONCOR.NS', 'TEXRAIL.NS'],
            
            # Energy & Utilities
            'Energy': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'NTPC.NS', 'POWERGRID.NS', 'GAIL.NS'],
            'Power': ['NTPC.NS', 'POWERGRID.NS', 'TATAPOWER.NS', 'ADANIPOWER.NS', 'NHPC.NS', 'SJVN.NS'],
            'Renewable_Energy': ['ADANIGREEN.NS', 'SUZLON.NS', 'INOXWIND.NS', 'WAAREE.NS', 'RENEWPOWER.NS'],
            
            # Consumer & Retail
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS'],
            'Retail': ['DMART.NS', 'RELIANCE.NS', 'TRENT.NS', 'SHOPRITE.NS', 'VMART.NS'],
            'Food_Processing': ['BRITANNIA.NS', 'NESTLEIND.NS', 'VADILAL.NS', 'HATSUN.NS', 'VBL.NS'],
            
            # Infrastructure & Construction
            'Construction': ['L&T.NS', 'UBL.NS', 'JKCEMENT.NS', 'SHREECEM.NS', 'RAMCOCEM.NS', 'ACC.NS', 'AMBUJACEMENT.NS'],
            'Infrastructure': ['L&T.NS', 'ADANIPORTS.NS', 'GMR.NS', 'GVK.NS', 'IRB.NS', 'SADBHAV.NS'],
            'Real_Estate': ['DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'SOBHA.NS', 'BRIGADE.NS', 'PRESTIGE.NS'],
            
            # Chemicals & Materials
            'Chemicals': ['RELIANCE.NS', 'UPL.NS', 'PIDILITIND.NS', 'AARTI.NS', 'BALRAMCHIN.NS', 'DEEPAK.NS', 'SRF.NS'],
            'Fertilizers': ['UPL.NS', 'COROMANDEL.NS', 'GSFC.NS', 'CHAMBLFERT.NS', 'KRIBHCO.NS'],
            'Paints': ['ASIANPAINT.NS', 'BERGER.NS', 'AKZONOBEL.NS', 'INDIGO.NS', 'KANSAINER.NS'],
            
            # Media & Entertainment
            'Media': ['ZEEL.NS', 'SUNTV.NS', 'HATHWAY.NS', 'NETWORK18.NS', 'TV18BRDCST.NS'],
            'Entertainment': ['PVR.NS', 'INOXLEISUR.NS', 'EROS.NS', 'BALAJITELE.NS', 'TIPS.NS'],
            
            # Services & Logistics
            'Logistics': ['BLUEDART.NS', 'GATI.NS', 'MAHLOG.NS', 'CONCOR.NS', 'AARTIIND.NS'],
            'Aviation': ['INTERGLOBE.NS', 'SPICEJET.NS', 'JETAIRWAYS.NS'],
            'Shipping': ['SCI.NS', 'SHREYAS.NS', 'ESABINDIA.NS', 'GESHIP.NS'],
            
            # Textiles & Apparel
            'Textiles': ['ARVIND.NS', 'WELSPUN.NS', 'TRIDENT.NS', 'RAYMOND.NS', 'PAGEINDS.NS', 'RTNPOWER.NS'],
            'Apparel': ['ARVIND.NS', 'RAYMOND.NS', 'GRASIM.NS', 'SPANDANA.NS'],
            
            # Agriculture & Allied
            'Agriculture': ['UPL.NS', 'RALLIS.NS', 'PIIND.NS', 'DHANUKA.NS', 'INSECTICIDES.NS'],
            'Sugar': ['BALRAMCHIN.NS', 'BAJAJHIND.NS', 'DHAMPUR.NS', 'EID.NS'],
            
            # Emerging Sectors
            'Renewable_Tech': ['SUZLON.NS', 'INOXWIND.NS', 'WEBSOL.NS', 'BOROSIL.NS'],
            'EV_Battery': ['EXIDE.NS', 'AMARON.NS', 'TATAELXSI.NS', 'TATACHEM.NS'],
            'Fintech': ['PAYTM.NS', 'POLICYBZR.NS', 'NYKAA.NS'],
            'E_Commerce': ['NYKAA.NS', 'ZOMATO.NS', 'PAYTM.NS'],
            
            # Government & PSU
            'PSU_Banks': ['SBIN.NS', 'PNB.NS', 'BANKBARODA.NS', 'CANFINHOME.NS', 'UNIONBANK.NS'],
            'PSU_Energy': ['ONGC.NS', 'IOC.NS', 'BPCL.NS', 'GAIL.NS', 'HINDPETRO.NS'],
            'PSU_Others': ['IRCTC.NS', 'COALINDIA.NS', 'NMDC.NS', 'SAIL.NS', 'BHEL.NS']
        }

    def fetch_market_news(self, query, language='en'):
        """Fetch latest news from NewsAPI for a specific query and language (default English)."""
        news_data = []
        if self.news_api_key != "YOUR_NEWS_API_KEY":
            try:
                url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language={language}&apiKey={self.news_api_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles:
                        news_data.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', '')
                        })
            except Exception as e:
                print(f"Error fetching news: {e}")
        return news_data

    def analyze_sentiment(self, text):
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def fetch_stock_data(self, symbol, period='1mo'):
        """Fetch stock data using yfinance, suppressing annoying output"""
        try:
            with suppress_stdout_stderr():
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                info = stock.info
            if data is None or data.empty:
                if self.verbose:
                    print(f"[Info] No data for {symbol} (possibly delisted or illiquid)")
                return None, None
            return data, info
        except Exception as e:
            if self.verbose:
                print(f"[Error] Could not fetch data for {symbol}: {e}")
            return None, None

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        if data is None or len(data) < 20:
            return {}
        
        # Moving Averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean() if len(data) >= 50 else data['Close'].rolling(window=len(data)).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Volume trend
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        latest = data.iloc[-1]
        return {
            'current_price': latest['Close'],
            'ma_20': latest['MA_20'],
            'ma_50': latest['MA_50'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_Signal'],
            'volume_ratio': latest['Volume'] / latest['Volume_MA'] if latest['Volume_MA'] > 0 else 1
        }

    def get_fundamental_data(self, info):
        """Extract fundamental data from stock info"""
        if not info:
            return {}
        
        return {
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'roe': info.get('returnOnEquity', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }

    def analyze_stock(self, symbol):
        """Comprehensive stock analysis"""
        data, info = self.fetch_stock_data(symbol)
        if data is None:
            return None
        
        technical = self.calculate_technical_indicators(data)
        fundamental = self.get_fundamental_data(info)
        
        # Price change analysis
        current_price = technical.get('current_price', 0)
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        
        # Fix for ZeroDivisionError
        if prev_close == 0 or prev_close is None:
            price_change = 0.0
        else:
            price_change = ((current_price - prev_close) / prev_close) * 100
        
        # Generate signals
        signals = self.generate_signals(technical, fundamental, price_change)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'technical': technical,
            'fundamental': fundamental,
            'signals': signals,
            'recommendation': self.get_recommendation(technical, fundamental, signals)
        }

    def generate_signals(self, technical, fundamental, price_change):
        """Generate buy/sell/hold signals"""
        signals = []
        
        # Technical signals
        rsi = technical.get('rsi', 50)
        if rsi < 30:
            signals.append('RSI Oversold - Potential Buy')
        elif rsi > 70:
            signals.append('RSI Overbought - Consider Sell')
        
        # MACD signal
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        if macd > macd_signal:
            signals.append('MACD Bullish')
        else:
            signals.append('MACD Bearish')
        
        # Moving average signals
        current_price = technical.get('current_price', 0)
        ma_20 = technical.get('ma_20', 0)
        ma_50 = technical.get('ma_50', 0)
        
        if current_price > ma_20 > ma_50:
            signals.append('Price above MAs - Bullish')
        elif current_price < ma_20 < ma_50:
            signals.append('Price below MAs - Bearish')
        
        # Volume signal
        volume_ratio = technical.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            signals.append('High Volume - Strong Move')
        
        return signals

    def get_recommendation(self, technical, fundamental, signals):
        """Generate investment recommendation"""
        score = 0
        
        # Technical scoring
        rsi = technical.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 1
        elif rsi < 30:
            score += 2  # Oversold is good for buying
        
        current_price = technical.get('current_price', 0)
        ma_20 = technical.get('ma_20', 0)
        if current_price > ma_20:
            score += 1
        
        # Fundamental scoring
        pe_ratio = fundamental.get('pe_ratio', 'N/A')
        if isinstance(pe_ratio, (int, float)) and 10 <= pe_ratio <= 25:
            score += 1
        
        # Generate recommendation with price targets
        current_price = technical.get('current_price', 0)
        
        if score >= 4:
            return {
                'action': 'BUY',
                'entry_price': current_price,
                'target_price': current_price * 1.15,  # 15% upside
                'stop_loss': current_price * 0.92,     # 8% downside
                'confidence': 'High'
            }
        elif score >= 2:
            return {
                'action': 'HOLD',
                'entry_price': current_price,
                'target_price': current_price * 1.08,  # 8% upside
                'stop_loss': current_price * 0.95,     # 5% downside
                'confidence': 'Medium'
            }
        else:
            return {
                'action': 'AVOID',
                'entry_price': current_price,
                'target_price': current_price,
                'stop_loss': current_price * 0.90,
                'confidence': 'Low'
            }

    def get_sector_specific_news(self, sector):
        """Get sector-specific news and analysis"""
        sector_keywords = {
            'Defense': ['defense', 'military', 'army', 'navy', 'air force', 'border', 'security'],
            'Aeronautics': ['aviation', 'aircraft', 'helicopter', 'aerospace', 'HAL', 'flight'],
            'Railways': ['railways', 'train', 'metro', 'IRCTC', 'railway budget'],
            'Energy': ['oil', 'gas', 'crude', 'petroleum', 'energy', 'fuel'],
            'Banking': ['RBI', 'repo rate', 'interest rate', 'banking', 'credit'],
            'IT': ['technology', 'software', 'IT', 'digital', 'cyber'],
            'Auto': ['automobile', 'car', 'vehicle', 'EV', 'electric vehicle'],
            'Pharma': ['pharmaceutical', 'drug', 'medicine', 'FDA', 'health'],
            'Power': ['electricity', 'power', 'solar', 'wind', 'renewable'],
            'Renewable_Energy': ['solar', 'wind', 'renewable', 'green energy', 'climate']
        }
        
        keywords = sector_keywords.get(sector, [sector.lower()])
        sector_news = []
        
        # This would integrate with news APIs to fetch sector-specific news
        # For now, returning sample sector-specific news
        sample_news = {
            'Defense': [
                {'title': 'India increases defense budget by 13% for modernization', 'sentiment': 'positive'},
                {'title': 'HAL receives major order for helicopter manufacturing', 'sentiment': 'positive'}
            ],
            'Aeronautics': [
                {'title': 'Civil aviation sector shows strong recovery post-pandemic', 'sentiment': 'positive'},
                {'title': 'New aircraft orders boost aerospace manufacturing', 'sentiment': 'positive'}
            ],
            'Banking': [
                {'title': 'RBI maintains accommodative stance, keeps rates unchanged', 'sentiment': 'neutral'},
                {'title': 'Bank credit growth accelerates to 16% YoY', 'sentiment': 'positive'}
            ]
        }
        
        return sample_news.get(sector, [])

    def analyze_sector_correlation(self):
        """Analyze correlation between different sectors"""
        sector_returns = {}
        
        # Calculate sector-wise average returns
        for sector, stocks in self.sectors.items():
            sector_data = []
            for stock in stocks[:3]:  # Top 3 stocks per sector
                data, _ = self.fetch_stock_data(stock, period='1mo')
                if data is not None and len(data) > 1:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    
                    # Fix for ZeroDivisionError
                    if start_price == 0 or start_price is None:
                        returns = 0.0
                    else:
                        returns = ((end_price - start_price) / start_price) * 100
                    sector_data.append(returns)
            
            if sector_data:
                sector_returns[sector] = np.mean(sector_data)
        
        # Find correlated sectors
        correlations = []
        sector_list = list(sector_returns.keys())
        
        for i, sector1 in enumerate(sector_list):
            for sector2 in sector_list[i+1:]:
                # Simple correlation based on similar performance
                perf_diff = abs(sector_returns[sector1] - sector_returns[sector2])
                if perf_diff < 2:  # Less than 2% difference indicates correlation
                    correlations.append((sector1, sector2, perf_diff))
        
        return sector_returns, correlations

    def get_defense_aerospace_insights(self):
        """Specialized analysis for Defense and Aerospace sectors"""
        defense_stocks = self.sectors.get('Defense', []) + self.sectors.get('Aeronautics', [])
        insights = []
        
        for stock in defense_stocks:
            analysis = self.analyze_stock(stock)
            if analysis:
                # Special considerations for defense stocks
                fundamental = analysis['fundamental']
                
                # Defense stocks often have government contracts - check order book
                special_factors = []
                if 'HAL' in stock:
                    special_factors.append('Major helicopter & aircraft manufacturer')
                elif 'BEL' in stock:
                    special_factors.append('Electronics for defense applications')
                elif 'BHEL' in stock:
                    special_factors.append('Heavy engineering & power equipment')
                
                analysis['special_factors'] = special_factors
                insights.append(analysis)
        
        return insights

    def get_emerging_sector_analysis(self):
        """Analyze emerging sectors like EV, Renewable Energy, Fintech"""
        emerging_sectors = ['Renewable_Energy', 'EV_Battery', 'Fintech', 'E_Commerce']
        analysis = {}
        
        for sector in emerging_sectors:
            if sector in self.sectors:
                sector_stocks = []
                for stock in self.sectors[sector]:
                    stock_analysis = self.analyze_stock(stock)
                    if stock_analysis:
                        # Add growth potential scoring for emerging sectors
                        growth_score = self.calculate_growth_potential(stock_analysis)
                        stock_analysis['growth_score'] = growth_score
                        sector_stocks.append(stock_analysis)
                
                if sector_stocks:
                    # Sort by growth potential
                    sector_stocks.sort(key=lambda x: x.get('growth_score', 0), reverse=True)
                    analysis[sector] = sector_stocks
        
        return analysis

    def calculate_growth_potential(self, stock_analysis):
        """Calculate growth potential score for emerging sector stocks"""
        score = 0
        
        # Revenue growth (if available in fundamental data)
        pe_ratio = stock_analysis['fundamental'].get('pe_ratio', 0)
        if isinstance(pe_ratio, (int, float)):
            if 15 <= pe_ratio <= 30:  # Moderate PE suggests growth with value
                score += 2
            elif pe_ratio > 30:  # High PE might indicate high growth expectations
                score += 1
        
        # Price momentum
        price_change = stock_analysis.get('price_change', 0)
        if price_change > 5:
            score += 2
        elif price_change > 0:
            score += 1
        
        # Technical signals
        signals = stock_analysis.get('signals', [])
        bullish_signals = sum(1 for signal in signals if 'Bullish' in signal or 'Buy' in signal)
        score += bullish_signals
        
        return score

    def analyze_sectors(self):
        """Analyze performance of different sectors"""
        sector_performance = {}
        
        for sector, stocks in self.sectors.items():
            sector_data = []
            for stock in stocks:
                analysis = self.analyze_stock(stock)
                if analysis:
                    sector_data.append(analysis)
            
            if sector_data:
                avg_change = np.mean([s['price_change'] for s in sector_data])
                sector_performance[sector] = {
                    'avg_change': avg_change,
                    'top_performer': max(sector_data, key=lambda x: x['price_change']),
                    'stocks_analyzed': len(sector_data)
                }
        
        return sector_performance

    def get_market_overview(self):
        """Get overall market overview"""
        market_data = {}
        
        for name, symbol in self.indices.items():
            data, info = self.fetch_stock_data(symbol, period='5d')
            if data is not None and len(data) >= 2:
                current = data['Close'].iloc[-1]
                previous = data['Close'].iloc[-2]
                
                # Fix for ZeroDivisionError
                if previous == 0 or previous is None:
                    change = 0.0
                else:
                    change = ((current - previous) / previous) * 100
                    
                market_data[name] = {
                    'current': current,
                    'change': change,
                    'trend': 'UP' if change > 0 else 'DOWN'
                }
        
        return market_data

    def generate_daily_report(self):
        """Generate comprehensive daily market report"""
        print("📊 DAILY STOCK MARKET ANALYSIS REPORT")
        print("=" * 50)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Market Overview
        print("🏛️ MARKET OVERVIEW")
        print("-" * 20)
        market_data = self.get_market_overview()
        for index, data in market_data.items():
            trend_emoji = "📈" if data['trend'] == 'UP' else "📉"
            print(f"{trend_emoji} {index}: {data['current']:.2f} ({data['change']:+.2f}%)")
        print()
        
        # 2. News & Sentiment Analysis
        print("📰 NEWS & SENTIMENT ANALYSIS")
        print("-" * 30)
        news_data = self.fetch_market_news(query="indian stock market")
        for news in news_data[:3]:  # Top 3 news
            sentiment = self.analyze_sentiment(news.get('description', ''))
            sentiment_emoji = "😊" if sentiment == 'positive' else "😟" if sentiment == 'negative' else "😐"
            print(f"{sentiment_emoji} {news['title']}")
            print(f"   Source: {news['source']} | Sentiment: {sentiment.upper()}")
        print()
        
        # 3. Comprehensive Sector Performance
        print("🏭 COMPREHENSIVE SECTOR PERFORMANCE")
        print("-" * 35)
        sector_data = self.analyze_sectors()
        sector_returns, correlations = self.analyze_sector_correlation()
        
        # Group sectors by performance
        top_performers = sorted(sector_data.items(), key=lambda x: x[1]['avg_change'], reverse=True)[:8]
        bottom_performers = sorted(sector_data.items(), key=lambda x: x[1]['avg_change'])[:5]
        
        print("🚀 TOP PERFORMING SECTORS:")
        for sector, data in top_performers:
            trend_emoji = "🚀" if data['avg_change'] > 3 else "📈" if data['avg_change'] > 0 else "📉"
            print(f"{trend_emoji} {sector}: {data['avg_change']:+.2f}% avg")
            top_stock = data['top_performer']
            print(f"   Leader: {top_stock['symbol']} ({top_stock['price_change']:+.2f}%)")
        
        print(f"\n⚠️ UNDERPERFORMING SECTORS:")
        for sector, data in bottom_performers:
            print(f"📉 {sector}: {data['avg_change']:+.2f}% avg")
        
        # Sector correlations
        if correlations:
            print(f"\n🔗 CORRELATED SECTORS (Similar Performance):")
            for sector1, sector2, diff in correlations[:3]:
                print(f"• {sector1} ↔ {sector2} (Δ{diff:.1f}%)")
        
        # 4. Defense & Aerospace Special Analysis
        print(f"\n🛡️ DEFENSE & AEROSPACE INSIGHTS")
        print("-" * 30)
        defense_insights = self.get_defense_aerospace_insights()
        for insight in defense_insights[:3]:
            rec = insight['recommendation']
            factors = ', '.join(insight.get('special_factors', ['Strategic sector stock']))
            print(f"• {insight['symbol']}: ₹{rec['entry_price']:.2f} ({rec['action']})")
            print(f"  Context: {factors}")
            print(f"  Target: ₹{rec['target_price']:.2f} | SL: ₹{rec['stop_loss']:.2f}")
        
        # 5. Emerging Sectors Analysis
        print(f"\n🌱 EMERGING SECTORS WATCH")
        print("-" * 25)
        emerging_analysis = self.get_emerging_sector_analysis()
        for sector, stocks in emerging_analysis.items():
            if stocks:
                best_stock = stocks[0]  # Highest growth score
                print(f"🔮 {sector.replace('_', ' ')}: {best_stock['symbol']}")
                print(f"   Growth Score: {best_stock.get('growth_score', 0)}/5")
                print(f"   Price: ₹{best_stock['current_price']:.2f} ({best_stock['price_change']:+.2f}%)")
        
        print()
        
        # 6. Enhanced Stock Recommendations
        print("💡 COMPREHENSIVE INVESTMENT RECOMMENDATIONS")
        print("-" * 40)
        
        # Get stocks from all sectors
        all_stocks = []
        sector_representation = {}
        
        for sector, stocks in self.sectors.items():
            sector_stocks = []
            for stock in stocks[:2]:  # Top 2 from each sector
                analysis = self.analyze_stock(stock)
                if analysis:
                    analysis['sector_name'] = sector
                    sector_stocks.append(analysis)
            
            if sector_stocks:
                # Add best stock from each sector
                best_sector_stock = max(sector_stocks, key=lambda x: x['price_change'])
                all_stocks.append(best_sector_stock)
                sector_representation[sector] = best_sector_stock
        
        # Categorize recommendations
        buy_stocks = [s for s in all_stocks if s['recommendation']['action'] == 'BUY']
        hold_stocks = [s for s in all_stocks if s['recommendation']['action'] == 'HOLD']
        
        buy_stocks.sort(key=lambda x: (x['recommendation']['confidence'] == 'High', x['price_change']), reverse=True)
        
        print("🎯 SHORT-TERM OPPORTUNITIES (1-7 days):")
        for i, stock in enumerate(buy_stocks[:5], 1):
            rec = stock['recommendation']
            sector = stock['sector_name']
            print(f"{i}. {stock['symbol']} ({sector}): ₹{rec['entry_price']:.2f}")
            print(f"   Target: ₹{rec['target_price']:.2f} | SL: ₹{rec['stop_loss']:.2f}")
            print(f"   Reason: {stock['price_change']:+.2f}% momentum, {rec['confidence']} confidence")
            main_signals = [s for s in stock['signals'] if 'Bullish' in s or 'Buy' in s or 'above' in s][:2]
            if main_signals:
                print(f"   Signals: {', '.join(main_signals)}")
            print()
        
        print("🎯 MID-TERM SELECTIONS (1-3 months):")
        mid_term_candidates = [s for s in all_stocks if s['recommendation']['confidence'] in ['High', 'Medium']]
        mid_term_candidates.sort(key=lambda x: x['fundamental'].get('pe_ratio', 100) if isinstance(x['fundamental'].get('pe_ratio'), (int, float)) else 100)
        
        for i, stock in enumerate(mid_term_candidates[:5], 1):
            rec = stock['recommendation']
            fundamental = stock['fundamental']
            sector = stock['sector_name']
            pe_ratio = fundamental.get('pe_ratio', 'N/A')
            market_cap = fundamental.get('market_cap', 'N/A')
            
            print(f"{i}. {stock['symbol']} ({sector}): ₹{rec['entry_price']:.2f}")
            print(f"   Target: ₹{rec['target_price']:.2f} | SL: ₹{rec['stop_loss']:.2f}")
            print(f"   Fundamentals: PE {pe_ratio}, MCap: {market_cap}")
            print()
        
        print("🎯 LONG-TERM WEALTH CREATORS (6+ months):")
        # Focus on fundamentally strong companies
        long_term_stocks = []
        for stock in all_stocks:
            pe_ratio = stock['fundamental'].get('pe_ratio', None)
            roe = stock['fundamental'].get('roe', None)
            if isinstance(pe_ratio, (int, float)) and pe_ratio < 30:
                if isinstance(roe, (int, float)) and roe > 0.1:  # ROE > 10%
                    long_term_stocks.append(stock)
        
        long_term_stocks.sort(key=lambda x: x['fundamental'].get('roe', 0), reverse=True)
        
        for i, stock in enumerate(long_term_stocks[:5], 1):
            rec = stock['recommendation']
            fundamental = stock['fundamental']
            sector = stock['sector_name']
            
            print(f"{i}. {stock['symbol']} ({sector}): ₹{rec['entry_price']:.2f}")
            print(f"   Sector: {fundamental.get('sector', 'N/A')} | Industry: {fundamental.get('industry', 'N/A')}")
            print(f"   Quality: PE {fundamental.get('pe_ratio', 'N/A')}, ROE {fundamental.get('roe', 'N/A')}")
            print(f"   Long-term Target: ₹{rec['target_price'] * 1.5:.2f} (18-month horizon)")
            print()
        
        # 7. Sector-wise Investment Themes
        print("🎨 SECTOR-WISE INVESTMENT THEMES")
        print("-" * 32)
        
        themes = {
            'Defense & Aerospace': 'Government modernization drive, geopolitical tensions',
            'Renewable Energy': 'Climate commitments, energy transition policies',
            'Banking & NBFC': 'Credit growth recovery, interest rate cycle',
            'IT & Telecom': 'Digital transformation, 5G rollout',
            'Healthcare & Pharma': 'Aging population, medical infrastructure',
            'Infrastructure': 'Government capex push, urbanization',
            'FMCG & Retail': 'Rural recovery, consumption normalization'
        }
        
        for theme, description in themes.items():
            relevant_sectors = [s for s in self.sectors.keys() if any(word in s for word in theme.split())]
            if relevant_sectors:
                avg_performance = np.mean([sector_returns.get(s, 0) for s in relevant_sectors if s in sector_returns])
                trend_emoji = "📈" if avg_performance > 1 else "📊" if avg_performance > -1 else "📉"
                print(f"{trend_emoji} {theme}: {description}")
                print(f"   Current Performance: {avg_performance:+.1f}% avg")
                print()
        
        # 8. Enhanced Final Verdict
        print("🎯 COMPREHENSIVE MARKET VERDICT")
        print("-" * 30)
        market_trend = "BULLISH" if sum(d['change'] for d in market_data.values()) > 0 else "BEARISH"
        positive_news = sum(1 for news in news_data if self.analyze_sentiment(news.get('description', '')) == 'positive')
        
        print(f"Market Trend: {market_trend}")
        print(f"News Sentiment: {positive_news}/{len(news_data)} Positive")
        
        # Get sorted sectors for best performer
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True) if sector_returns else []
        print(f"Best Performing Sector: {sorted_sectors[0][0] if sorted_sectors else 'N/A'}")
        
        if market_trend == "BULLISH" and positive_news >= len(news_data) // 2:
            print("✅ GOOD DAY TO INVEST - Market conditions favorable")
        elif market_trend == "BEARISH":
            print("⚠️ CAUTIOUS APPROACH - Wait for better entry points")
        else:
            print("⚡ MIXED SIGNALS - Selective stock picking recommended")

    def detect_language(self, text):
        """Detect the language of a given text using TextBlob (fallback: 'en')"""
        try:
            blob = TextBlob(text)
            return blob.detect_language()
        except Exception:
            return 'en'

    def translate_to_english(self, text):
        """Translate text to English using TextBlob (if not already English)"""
        try:
            blob = TextBlob(text)
            if blob.detect_language() != 'en':
                return str(blob.translate(to='en'))
            return text
        except Exception:
            return text

    def detailed_stock_report(self, symbol):
        """Print a detailed, formatted report for a given stock symbol"""
        analysis = self.analyze_stock(symbol)
        if not analysis:
            print(f"No data available for {symbol}.")
            return
        info = analysis['fundamental']
        technical = analysis['technical']
        signals = analysis['signals']
        rec = analysis['recommendation']
        confidence_map = {'High': 5, 'Medium': 3, 'Low': 1}
        stars = confidence_map.get(rec['confidence'], 1)
        star_str = '★' * stars + '☆' * (5 - stars)
        print(f"\n===== {symbol} - {info.get('longName', symbol)} =====")
        print(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
        print(f"Current Price: ₹{analysis['current_price']:.2f} ({analysis['price_change']:+.2f}%)")
        print(f"Status: {rec['action']} (Confidence: {rec['confidence']}) | {star_str}")
        print("\n--- Technicals ---")
        print(f"RSI: {technical.get('rsi', 'N/A'):.2f} | MACD: {'Bullish' if 'Bullish' in signals else 'Bearish'} | MA(20): {technical.get('ma_20', 'N/A'):.2f} | MA(50): {technical.get('ma_50', 'N/A'):.2f}")
        main_signals = ', '.join(signals[:3])
        print(f"Signals: {main_signals if main_signals else 'N/A'}")
        print("\n--- Fundamentals ---")
        print(f"PE: {info.get('pe_ratio', 'N/A')} | ROE: {info.get('roe', 'N/A')} | Market Cap: {info.get('market_cap', 'N/A')} | Dividend Yield: {info.get('dividend_yield', 'N/A')}")
        print(f"Debt/Equity: {info.get('debt_to_equity', 'N/A')} | EPS: {info.get('eps', 'N/A')}")
        print("\n--- News & Sentiment ---")
        # Build NewsAPI query for this stock
        news_query = symbol.split('.')[0]
        if info.get('longName') and isinstance(info.get('longName'), str):
            news_query += f" OR {info['longName']}"
        news = self.fetch_market_news(query=news_query, language='en')
        if not news:
            print("No recent news found for this stock.")
        else:
            for n in news[:3]:
                title = n.get('title', '')
                desc = n.get('description', '')
                lang = self.detect_language(title)
                show_title = title
                show_desc = desc
                lang_note = ''
                if lang != 'en':
                    show_title = self.translate_to_english(title)
                    show_desc = self.translate_to_english(desc)
                    lang_note = f" [Translated from {lang}]"
                sentiment = self.analyze_sentiment(show_desc)
                sentiment_emoji = "😊" if sentiment == 'positive' else "😟" if sentiment == 'negative' else "😐"
                date_str = n.get('publishedAt', '')
                if date_str:
                    date_str = date_str.split('T')[0]
                    print(f"[{date_str}] {sentiment_emoji} {show_title}{lang_note}")
                else:
                    print(f"{sentiment_emoji} {show_title}{lang_note}")
                print(f"   Source: {n['source']} | Sentiment: {sentiment.upper()}")
        print("\n--- Price Targets ---")
        cp = analysis['current_price']
        print(f"Short-term (1-7d): ₹{cp*1.03:.0f} (Target), ₹{cp*0.96:.0f} (Stop Loss)")
        print(f"Mid-term (1-3mo): ₹{cp*1.08:.0f} (Target), ₹{cp*0.95:.0f} (Stop Loss)")
        print(f"Long-term (6mo+): ₹{cp*1.5:.0f} (Target), ₹{cp*0.90:.0f} (Stop Loss)")
        print("\n--- AI Outlook ---")
        outlook = []
        if rec['action'] == 'BUY':
            outlook.append("Strong technical and/or fundamental signals. Good for accumulation.")
        elif rec['action'] == 'HOLD':
            outlook.append("Stable performance. Hold for now, watch for breakout or reversal.")
        else:
            outlook.append("Weak signals or overbought. Avoid new positions or wait for correction.")
        if info.get('roe', 0) and isinstance(info.get('roe'), (int, float)) and info['roe'] > 0.15:
            outlook.append("High ROE indicates efficient management and profitability.")
        if technical.get('rsi', 50) < 35:
            outlook.append("RSI indicates possible oversold condition.")
        elif technical.get('rsi', 50) > 70:
            outlook.append("RSI indicates possible overbought condition.")
        if 'Bullish' in signals:
            outlook.append("Momentum is currently bullish.")
        print(' '.join(outlook))
        print("=============================================")

    def get_stock_report_data(self, symbol):
        """Return all detailed report data for a given stock symbol as a dict (for UI)."""
        analysis = self.analyze_stock(symbol)
        if not analysis:
            return None
        info = analysis['fundamental']
        technical = analysis['technical']
        signals = analysis['signals']
        rec = analysis['recommendation']
        confidence_map = {'High': 5, 'Medium': 3, 'Low': 1}
        stars = confidence_map.get(rec['confidence'], 1)
        star_str = '★' * stars + '☆' * (5 - stars)
        # News
        news_query = symbol.split('.')[0]
        if info.get('longName') and isinstance(info.get('longName'), str):
            news_query += f" OR {info['longName']}"
        news = self.fetch_market_news(query=news_query, language='en')
        news_items = []
        for n in news[:3]:
            title = n.get('title', '')
            desc = n.get('description', '')
            lang = self.detect_language(title)
            show_title = title
            show_desc = desc
            lang_note = ''
            if lang != 'en':
                show_title = self.translate_to_english(title)
                show_desc = self.translate_to_english(desc)
                lang_note = f" [Translated from {lang}]"
            sentiment = self.analyze_sentiment(show_desc)
            date_str = n.get('publishedAt', '')
            if date_str:
                date_str = date_str.split('T')[0]
            news_items.append({
                'title': show_title,
                'lang_note': lang_note,
                'description': show_desc,
                'sentiment': sentiment,
                'sentiment_emoji': "😊" if sentiment == 'positive' else "😟" if sentiment == 'negative' else "😐",
                'date': date_str,
                'source': n.get('source', ''),
                'url': n.get('url', '')
            })
        # Price targets
        cp = analysis['current_price']
        price_targets = {
            'short': {'target': cp*1.03, 'stop_loss': cp*0.96},
            'mid': {'target': cp*1.08, 'stop_loss': cp*0.95},
            'long': {'target': cp*1.5, 'stop_loss': cp*0.90}
        }
        # AI Outlook
        outlook = []
        if rec['action'] == 'BUY':
            outlook.append("Strong technical and/or fundamental signals. Good for accumulation.")
        elif rec['action'] == 'HOLD':
            outlook.append("Stable performance. Hold for now, watch for breakout or reversal.")
        else:
            outlook.append("Weak signals or overbought. Avoid new positions or wait for correction.")
        if info.get('roe', 0) and isinstance(info.get('roe'), (int, float)) and info['roe'] > 0.15:
            outlook.append("High ROE indicates efficient management and profitability.")
        if technical.get('rsi', 50) < 35:
            outlook.append("RSI indicates possible oversold condition.")
        elif technical.get('rsi', 50) > 70:
            outlook.append("RSI indicates possible overbought condition.")
        if 'Bullish' in signals:
            outlook.append("Momentum is currently bullish.")
        return {
            'symbol': symbol,
            'longName': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': cp,
            'price_change': analysis['price_change'],
            'technical': technical,
            'fundamental': info,
            'signals': signals,
            'recommendation': rec,
            'stars': star_str,
            'news': news_items,
            'price_targets': price_targets,
            'ai_outlook': ' '.join(outlook)
        }

    def get_general_news_data(self):
        """Return top 3 most recent general Indian stock market news (with translation/lang info) for UI."""
        news = self.fetch_market_news(query="indian stock market", language='en')
        news_items = []
        for n in news[:3]:
            title = n.get('title', '')
            desc = n.get('description', '')
            lang = self.detect_language(title)
            show_title = title
            show_desc = desc
            lang_note = ''
            if lang != 'en':
                show_title = self.translate_to_english(title)
                show_desc = self.translate_to_english(desc)
                lang_note = f" [Translated from {lang}]"
            sentiment = self.analyze_sentiment(show_desc)
            date_str = n.get('publishedAt', '')
            if date_str:
                date_str = date_str.split('T')[0]
            news_items.append({
                'title': show_title,
                'lang_note': lang_note,
                'description': show_desc,
                'sentiment': sentiment,
                'sentiment_emoji': "😊" if sentiment == 'positive' else "😟" if sentiment == 'negative' else "😐",
                'date': date_str,
                'source': n.get('source', ''),
                'url': n.get('url', '')
            })
        return news_items

# Usage Example
if __name__ == "__main__":
    # Initialize the agent
    agent = StockMarketAgent()
    
    # Generate daily report
    agent.generate_daily_report()
    
    # For individual stock analysis
    # stock_analysis = agent.analyze_stock('TCS.NS')
    # print(stock_analysis) 