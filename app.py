import os
import tempfile
import base64
from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend for matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

app = Flask(__name__)

# Create a static directory for saving images
if not os.path.exists('static'):
    os.makedirs('static')

# Suppress debug-level logs
logging.basicConfig(level=logging.INFO)

# Suppress matplotlib and other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Function to identify peaks in the data using NumPy
def find_peaks(data):
    return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

# Function to fetch intraday stock data for the past week
def fetch_intraday_data(symbol):
    ticker = yf.Ticker(symbol)

    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')
    one_week_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

    # Fetch historical data for the past week
    historical_data_df = ticker.history(start=one_week_ago, end=today, interval='1d')
    historical_close = historical_data_df['Close'].to_numpy()

    # Find peaks in historical closing prices
    peaks = find_peaks(historical_close)

    # Fetch live intraday data for today
    intraday_data_df = ticker.history(period="1d", interval="1m")
    if intraday_data_df.empty:
        return None

    # Calculate the current price and price change
    current_price = intraday_data_df['Close'].iloc[-1]
    previous_close = historical_close[-1]
    price_change = current_price - previous_close
    percentage_change = (price_change / previous_close) * 100

    intraday_data = intraday_data_df[['Close']].reset_index().to_dict(orient='records')

    return {
        "symbol": symbol,
        "intraday_data": intraday_data,
        "peaks": peaks.tolist(),
        "historical_data": historical_data_df[['Close']].reset_index().to_dict(orient='records'),
        "current_price": current_price,
        "price_change": price_change,
        "percentage_change": percentage_change
    }

def create_plot(data):
    symbol = data['symbol']
    intraday_data = data['intraday_data']
    historical_data = data['historical_data']

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='black')

    historical_df = pd.DataFrame(historical_data)
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    ax.fill_between(historical_df['Date'], historical_df['Close'], color='green', alpha=0.5, label='Historical (1 Week)')

    intraday_df = pd.DataFrame(intraday_data)
    intraday_df['Datetime'] = pd.to_datetime(intraday_df['Datetime'])
    ax.plot(intraday_df['Datetime'], intraday_df['Close'], color='blue', label='Intraday (Today)')

    peak_dates = historical_df['Date'].iloc[data['peaks']]
    peak_prices = historical_df['Close'].iloc[data['peaks']]
    ax.scatter(peak_dates, peak_prices, color='white', s=0.2, zorder=5)

    ax.set_title(f'{symbol} Stock Price (Intraday + Historical)', color='white')
    ax.set_xlabel('Date/Time', color='white')
    ax.set_ylabel('Price (INR)', color='white')
    ax.grid(color='dimgray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    ax.set_xlim(left=historical_df['Date'].iloc[0], right=intraday_df['Datetime'].iloc[-1])
    ax.set_ylim(bottom=min(historical_df['Close'].min(), intraday_df['Close'].min()) - 5, 
                top=max(historical_df['Close'].max(), intraday_df['Close'].max()) + 5)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Define the path to save the image
    image_filename = f'static/{symbol}_stock_chart.png'
    plt.savefig(image_filename, dpi=100, bbox_inches='tight')  # Save the image
    
    plt.close()
    
    # Return the URL to the image
    return f'/static/{symbol}_stock_chart.png'

@app.route('/fetch-stock-chart', methods=['GET'])
def fetch_stock_chart():
    symbol = request.args.get('symbol')

    if not symbol:
        return jsonify({"error": "Invalid input. Expecting 'symbol' query parameter."}), 400

    try:
        stock_data = fetch_intraday_data(symbol)

        if stock_data is None:
            return jsonify({"error": f"No data found for symbol: {symbol}. Check if it may be delisted."}), 404

        # Create the plot and save it to a static directory
        image_url = create_plot(stock_data)

        # Prepare JSON response
        response_data = {
            "symbol": stock_data['symbol'],
            "current_price": stock_data['current_price'],
            "price_change": stock_data['price_change'],
            "percentage_change": stock_data['percentage_change'],
            "image_url": image_url  # URL to the image
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Set host and port for production
