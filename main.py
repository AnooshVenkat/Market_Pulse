import pandas as pd
import pandas_ta as ta
import yfinance as yf
import xgboost as xgb
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta, date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import json
import sqlite3
from flask import Flask, request, jsonify, render_template

NEWS_API_KEY = ''
STOCKS_BY_INDUSTRY = {
    "Technology & AI": {"Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL", "NVIDIA": "NVDA", "AMD": "AMD"},
    "E-Commerce & Cloud": {"Amazon": "AMZN", "Shopify": "SHOP"},
    "Automotive & Energy": {"Tesla": "TSLA", "Ford": "F"},
    "Finance & Payments": {"JPMorgan Chase": "JPM", "Visa": "V"}
}
app = Flask(__name__)

def init_db():
    with sqlite3.connect('predictions_cache.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS predictions (ticker TEXT, prediction_date TEXT, results TEXT, PRIMARY KEY (ticker, prediction_date))')
init_db()


def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty: return None
    if isinstance(stock_data.columns, pd.MultiIndex): stock_data.columns = stock_data.columns.droplevel(1)
    return stock_data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

def get_daily_news_df(query, api_key):
    newsapi = NewsApiClient(api_key=api_key)
    analyzer = SentimentIntensityAnalyzer()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    try:
        articles = newsapi.get_everything(q=query, from_param=start_date.strftime('%Y-%m-%d'), to=end_date.strftime('%Y-%m-%d'), language='en', sort_by='relevancy', page_size=100)
    except Exception as e:
        print(f"News API Error: {e}")
        return pd.DataFrame()
    if not articles['articles']: return pd.DataFrame()
    news_df = pd.DataFrame(articles['articles'])[['publishedAt', 'title']]
    news_df.columns = ['date', 'headline']
    news_df['sentiment_score'] = news_df['headline'].apply(lambda title: analyzer.polarity_scores(title)['compound'])
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    return news_df

def create_and_train_model(stock_df):
    df = stock_df.copy()
    df.ta.rsi(close='Close', length=14, append=True); df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True); df.ta.sma(close='Close', length=50, append=True)
    df['Price_Change'] = df['Close'].diff()
    df['Target'] = (df['Close'] > df['Close'].shift(1)).astype(int)
    df.dropna(inplace=True)
    if len(df) < 50: return None
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'SMA_50']
    X, y = df[features], df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_accuracy, test_accuracy = accuracy_score(y_train, model.predict(X_train)), accuracy_score(y_test, model.predict(X_test))
    prediction, probability = model.predict(X.iloc[[-1]])[0], model.predict_proba(X.iloc[[-1]])[0]
    return {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "validation_gap": train_accuracy - test_accuracy, "prediction": int(prediction), "probability": probability.tolist()}

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_price_history(ticker, stock_data):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    baseline_price = stock_data['Close'].iloc[0]
    
    ax.plot(stock_data['Date'], stock_data['Close'], label=f'{ticker} Adjusted Close', color='navy', lw=2)
    
    ax.axhline(y=baseline_price, color='gray', linestyle='--', lw=1, label=f'Starting Price: ${baseline_price:.2f}')

    ax.fill_between(stock_data['Date'], stock_data['Close'], baseline_price, where=(stock_data['Close'] >= baseline_price), facecolor='green', alpha=0.3)
    ax.fill_between(stock_data['Date'], stock_data['Close'], baseline_price, where=(stock_data['Close'] < baseline_price), facecolor='red', alpha=0.3)
    
    ax.set_title(f'{ticker} Price Change (Last 30 Days)', fontsize=14)
    ax.set_ylabel('Adjusted Close Price (USD)')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.autofmt_xdate()
    plt.tight_layout()
    return plot_to_base64(fig)

def plot_sentiment_history(ticker, aggregated_sentiment):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in aggregated_sentiment['sentiment_score']]
    ax.bar(aggregated_sentiment['date'], aggregated_sentiment['sentiment_score'], color=colors, alpha=0.7)
    ax.set_title(f'{ticker} Daily News Sentiment (Last 30 Days)', fontsize=14); ax.set_ylabel('Aggregated Sentiment Score'); ax.grid(True, linestyle='--', alpha=0.6); fig.autofmt_xdate(); plt.tight_layout()
    return plot_to_base64(fig)

def plot_merged_graph(ticker, combined_data):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_title(f'{ticker} Price vs. Sentiment Analysis', fontsize=14)
    ax1.set_ylabel('Stock Price (USD)', color='navy')
    line = ax1.plot(combined_data['Date'], combined_data['Close'], color='navy', label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='navy')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Aggregated Sentiment Score', color='gray')
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in combined_data['sentiment_score']]
    bars = ax2.bar(combined_data['Date'], combined_data['sentiment_score'], color=colors, alpha=0.6, label='Sentiment Score')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax1.legend(
        handles=[line[0], bars], 
        labels=['Stock Price', 'Sentiment Score'], 
        loc='upper left', 
        bbox_to_anchor=(0, 1.15), 
        ncol=1,                   
        frameon=False             
    )
    
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    return plot_to_base64(fig)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/get_industries')
def get_industries(): return jsonify(list(STOCKS_BY_INDUSTRY.keys()))

@app.route('/get_stocks')
def get_stocks():
    industry = request.args.get('industry')
    stocks = STOCKS_BY_INDUSTRY.get(industry, {})
    return jsonify([{'name': n, 'ticker': t} for n, t in stocks.items()])

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.get_json().get('ticker')
    if not ticker: return jsonify({"error": "Ticker symbol is required."}), 400

    today_str = str(date.today())
    with sqlite3.connect('predictions_cache.db') as conn:
        cached_result = conn.execute("SELECT results FROM predictions WHERE ticker=? AND prediction_date=?", (ticker, today_str)).fetchone()
    if cached_result:
        return jsonify(json.loads(cached_result[0]))

    stock_data_hist = get_stock_data(ticker, datetime.now() - timedelta(days=3*365), datetime.now())
    if stock_data_hist is None: return jsonify({"error": "Could not fetch historical stock data."}), 404
    model_results = create_and_train_model(stock_data_hist)
    if not model_results: return jsonify({"error": "Not enough data to train the model."}), 400
        
    daily_news_df = get_daily_news_df(ticker, NEWS_API_KEY)
    if daily_news_df.empty: return jsonify({"error": "Could not fetch news for sentiment analysis."}), 404
    
    stock_data_30d = yf.download(ticker, start=datetime.now() - timedelta(days=30), end=datetime.now(), auto_adjust=True)
    if stock_data_30d.empty: return jsonify({"error": "Could not fetch recent stock data for plots."}), 404
    if isinstance(stock_data_30d.columns, pd.MultiIndex): stock_data_30d.columns = stock_data_30d.columns.droplevel(1)
    
    aggregated_sentiment = daily_news_df.groupby('date')['sentiment_score'].sum().reset_index()
    
    stock_data_30d.reset_index(inplace=True)
    stock_data_30d['Date'] = pd.to_datetime(stock_data_30d['Date']).dt.date
    combined_data = pd.merge(stock_data_30d, aggregated_sentiment, left_on='Date', right_on='date', how='inner')

    if combined_data.empty:
        return jsonify({"error": "Could not generate plots. No overlapping dates found between stock data and news articles for the last 30 days."}), 400
    
    plot_price_b64, plot_sentiment_b64, plot_merged_b64 = None, None, None
    try:
        plot_price_b64 = plot_price_history(ticker, stock_data_30d)
    except Exception as e:
        print(f"!!! Error generating price plot: {e}")
    try:
        plot_sentiment_b64 = plot_sentiment_history(ticker, aggregated_sentiment)
    except Exception as e:
        print(f"!!! Error generating sentiment plot: {e}")
    try:
        plot_merged_b64 = plot_merged_graph(ticker, combined_data)
    except Exception as e:
        print(f"!!! Error generating merged plot: {e}")

    final_results = {
        "model_performance": model_results,
        "prediction": {"direction": "HIGHER" if model_results['prediction'] == 1 else "LOWER", "confidence": model_results['probability'][1] if model_results['prediction'] == 1 else model_results['probability'][0]},
        "plot_price_image": plot_price_b64,
        "plot_sentiment_image": plot_sentiment_b64,
        "plot_merged_image": plot_merged_b64
    }

    with sqlite3.connect('predictions_cache.db') as conn:
        conn.execute("INSERT OR REPLACE INTO predictions (ticker, prediction_date, results) VALUES (?, ?, ?)", (ticker, today_str, json.dumps(final_results)))

    return jsonify(final_results)

if __name__ == '__main__':
    app.run(debug=True)