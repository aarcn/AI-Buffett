import pandas as pd
from stock_data import fetch_stock_data, calculate_moving_averages
from sentiment_analysis import read_news_file
from strategy import generate_signals, calculate_returns
from backtest import backtest_strategy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'
news_file_path = 'news_headlines.txt'

stock_data = fetch_stock_data(symbol, start_date, end_date)

stock_data_with_ma = calculate_moving_averages(stock_data)

if isinstance(stock_data_with_ma.columns, pd.MultiIndex):
    stock_data_with_ma.columns = ['Date', 'Close', 'MA50', 'MA200']

sentiment_data = read_news_file(news_file_path)

sentiment_data.reset_index(drop=True, inplace=True)

print("Stock Data Columns:", stock_data_with_ma.columns)
print("Sentiment Data Columns:", sentiment_data.columns)
print("Is Stock Data MultiIndex?", isinstance(stock_data_with_ma.columns, pd.MultiIndex))
print("Is Sentiment Data MultiIndex?", isinstance(sentiment_data.columns, pd.MultiIndex))

merged_data = stock_data_with_ma.merge(sentiment_data, on='Date', how='left')
merged_data['Sentiment_Score'] = merged_data['Sentiment_Score'].fillna(0)  # Fill missing scores with 0 (neutral)

final_data = generate_signals(merged_data)

print("Buy signals:", (final_data['Signal'] == 1).sum())
print("Sell signals:", (final_data['Signal'] == -1).sum())

cumulative_return = calculate_returns(final_data)

print(f"Final Cumulative Return with News Sentiment: {cumulative_return.iloc[-1]:.2%}")

def prepare_data_for_prediction(data):
    features = data[['Close', 'MA50', 'MA200', 'Sentiment_Score']]
    labels = data['Close'].shift(-1)
    features = features[:-1]
    labels = labels[:-1]

    features = features.dropna()
    labels = labels.loc[features.index]  # Align labels with features after dropping NaN
    return features, labels

def train_and_predict(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    next_day_prediction = model.predict(X_test)
    return next_day_prediction, y_test

features, labels = prepare_data_for_prediction(final_data)

if features.empty or labels.empty:
    print("No valid data available for prediction.")
else:
    next_day_prediction, true_values = train_and_predict(features, labels)

    print("Predicted next day closing prices:", next_day_prediction)
    print("True next day closing prices:", true_values.values)

backtest_strategy(final_data)

print("Sentiment Score Summary:")
print(final_data['Sentiment_Score'].describe())

print("Sample rows with signals:")
print(final_data[final_data['Signal'] != 0].head(10))
