import yfinance as yf
import pandas as pd
from textblob import TextBlob
from datetime import datetime

# Function to read and calculate daily sentiment scores from a text file of headlines
def read_news_file(file_path):
    daily_sentiments = {}

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Assume each line is "YYYY-MM-DD headline text"
                date_str, headline = line.strip().split(' ', 1)
                date = datetime.strptime(date_str, '%Y-%m-%d').date()

                # Calculate sentiment score for the headline
                sentiment = TextBlob(headline).sentiment.polarity

                # Store sentiment by date
                if date not in daily_sentiments:
                    daily_sentiments[date] = []
                daily_sentiments[date].append(sentiment)
            except ValueError:
                print(f"Skipping malformed line: {line}")

    # Average sentiment score per day
    daily_sentiment_scores = {date: sum(scores) / len(scores) for date, scores in daily_sentiments.items()}
    sentiment_df = pd.DataFrame(list(daily_sentiment_scores.items()), columns=['Date', 'Sentiment_Score'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)  # Remove timezone info
    return sentiment_df

# 1. Fetch Historical Stock Data
symbol = 'AAPL'
data = yf.download(symbol, start='2020-01-01', end='2023-01-01')
data = data[['Close']].reset_index()  # Reset index to ensure `Date` is a column, not an index

# Remove timezone info from Date column in `data`
data['Date'] = data['Date'].dt.tz_localize(None)

# Flatten the MultiIndex columns
data.columns = ['Date', 'Close']

# 2. Calculate Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Load sentiment data from a text file
news_file_path = 'news_headlines.txt'  # Replace with your file path
sentiment_data = read_news_file(news_file_path).reset_index(drop=True)  # Ensure `Date` is a column, not an index

# Merge stock data with sentiment data on the 'Date' column
data = data.merge(sentiment_data, on='Date', how='left')
data['Sentiment_Score'] = data['Sentiment_Score'].fillna(0)  # Fill missing scores with 0 (neutral)

# Debug: Check the structure before merging
print("Data columns:", data.columns)
print("Sentiment data columns:", sentiment_data.columns)

# 3. Generate Buy/Sell Signals incorporating sentiment with adjusted conditions
data['Signal'] = 0
data.loc[(data['MA50'] > data['MA200']) & (data['Sentiment_Score'] >= 0), 'Signal'] = 1  # Buy signal
data.loc[(data['MA50'] < data['MA200']) & (data['Sentiment_Score'] <= 0), 'Signal'] = -1  # Sell signal

# Shift signals by one day to reflect the previous day's signal
signals = data['Signal'].shift(1)

# Check the number of buy/sell signals generated
print("Buy signals:", (data['Signal'] == 1).sum())
print("Sell signals:", (data['Signal'] == -1).sum())

# 4. Calculate Returns Based on Signals
daily_returns = data['Close'].pct_change()
data['Strategy_Return'] = signals * daily_returns

# Compute cumulative returns
cumulative_return = (1 + data['Strategy_Return'].fillna(0)).cumprod() - 1
print(f"Final Cumulative Return with News Sentiment: {cumulative_return.iloc[-1]:.2%}")

# Print sentiment score summary
print("Sentiment Score Summary:")
print(data['Sentiment_Score'].describe())

# Print sample rows where signals are generated
print("Sample rows with signals:")
print(data[data['Signal'] != 0].head(10))
