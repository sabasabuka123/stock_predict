import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Replace 'YOUR_API_KEY' with your Alpha Vantage API key
api_key = 'XRonBj27UJfFaHp2CsCOfuWepkUTEp8p'

# Define the stock symbol and desired output size
symbol = 'AAPL'  # Replace with the desired stock symbol
output_size = 'compact'  # Options: 'compact' (last 100 data points) or 'full' (all available data)

# Set the Alpha Vantage API endpoint
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize={output_size}&apikey={api_key}'

# Send a GET request to the API endpoint
response = requests.get(url)

# Parse the JSON response
data = response.json()

# Extract the stock data
time_series_data = data['Time Series (1min)']

# Convert the data into a DataFrame
df = pd.DataFrame.from_dict(time_series_data, orient='index')

# Sort the DataFrame by date in ascending order
df.sort_index(ascending=True, inplace=True)

# Rename the columns
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Convert 'close' column to numeric type
df['close'] = pd.to_numeric(df['close'])

# Calculate the price movement
df['price_movement'] = df['close'].diff().shift(-1)
df['price_movement'] = df['price_movement'].apply(lambda x: 'increase' if x >= 0 else 'decrease')

# Save the DataFrame to a CSV file
df.to_csv('stockdata.csv', index_label='timestamp')

print("Stock data saved to stock_data.csv")

# Load the stock data from the CSV file
df = pd.read_csv('stockdata.csv')
print(df.columns)

# Extract features and target variable
X = df[['open', 'high', 'low', 'close', 'volume']]  # Replace with the actual feature column names
y = df['price_movement']  # Replace with the actual target column name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the machine learning model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute probabilities for each class
y_prob = model.predict_proba(X_test)

# Define a threshold for probability-based trading decisions
threshold = 0.7

# Create a strategy based on predicted probabilities
strategy = []
for prob in y_prob[:, 1]:  # Assuming class 1 indicates a buy signal
    if prob > threshold:
        strategy.append('Buy')
    else:
        strategy.append('Hold/Sell')

# Evaluate the strategy's performance
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Backtest the strategy on historical data
portfolio_value = 100000  # Initial portfolio value
num_shares = 0  # Number of shares held
buy_price = 0  # Price at which shares were bought

buy_prices = []
sell_prices = []
profits = []
for i in range(len(strategy)):
    if strategy[i] == 'Buy' and num_shares == 0:
        num_shares = portfolio_value / df.iloc[i]['close']
        buy_price = df.iloc[i]['close']
        buy_prices.append((df.iloc[i]['timestamp'], buy_price))
        print("Buying at price:", buy_price)
    elif strategy[i] == 'Hold/Sell' and num_shares > 0:
        sell_price = df.iloc[i]['close']
        sell_prices.append((df.iloc[i]['timestamp'], sell_price))
        portfolio_value = num_shares * sell_price
        gain = (sell_price - buy_price) * num_shares
        num_shares = 0
        profits.append(gain)
        print("Selling at price:", sell_price)
        print("Gain:", gain)


# Calculate final portfolio value and gain/loss
final_portfolio_value = portfolio_value
gain_loss = final_portfolio_value - 100000

print("Final portfolio value:", final_portfolio_value)
print("Gain/Loss:", gain_loss)

# Visualization: Plot the closing prices
dates = pd.to_datetime(df['timestamp'])
closing_prices = df['close']

plt.figure(figsize=(12, 6))
plt.plot(dates, closing_prices, label='Closing Price')
plt.scatter([pd.to_datetime(timestamp) for timestamp, _ in buy_prices], [price for _, price in buy_prices], color='green', marker='^', label='Buy')
plt.scatter([pd.to_datetime(timestamp) for timestamp, _ in sell_prices], [price for _, price in sell_prices], color='red', marker='v', label='Sell')
plt.title(f'{symbol} Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
buy_sell_data = pd.DataFrame({'Buy Timestamp': [timestamp for timestamp, _ in buy_prices],
                              'Buy Price': [price for _, price in buy_prices],
                              'Sell Timestamp': [timestamp for timestamp, _ in sell_prices],
                              'Sell Price': [price for _, price in sell_prices],
                              'Profit': profits})

buy_sell_data.to_csv('buy_sell_profit.csv', index=False)
print("Buying, selling prices, and profits saved to buy_sell_profit.csv")