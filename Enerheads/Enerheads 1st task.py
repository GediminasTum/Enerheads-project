import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

meteological_df=pd.read_csv(
    "weather_location_Vilnius.csv",
    parse_dates=["Unnamed: 0"],
    index_col="Unnamed: 0")

market_df = pd.read_csv(
    "market_data.csv",
    parse_dates=["Unnamed: 0"],
    index_col="Unnamed: 0")

meteological_df.index = pd.to_datetime(meteological_df.index, utc=True)
market_df.index = pd.to_datetime(market_df.index, utc=True)

price_data = input("Enter 'Nord Pool' or 'mFRR': ").strip().lower()

if price_data in ["nord pool", "nordpool", "nord"]:
    price_column = "10YLT-1001A0008Q_DA_eurmwh"
    filtered_market = market_df[[price_column]]

    df = filtered_market.join(
        meteological_df[
            [
                "wind_speed_80m_previous_day1",
                "temperature_2m_previous_day1",
                "relative_humidity_2m_previous_day1",
                "direct_radiation_previous_day1",
                "diffuse_radiation_previous_day1",
                "cloud_cover_previous_day1"
            ]
        ],
        how='inner'
    )
    
    
elif price_data in ["mfrr", "m frr"]:
    price_column = "LT_up_sa_cbmp"
    filtered_market = market_df[market_df.index.minute == 30][[price_column]]

    shifted_factor = meteological_df[
        [
            "wind_speed_80m_previous_day1",
            "temperature_2m_previous_day1",
            "relative_humidity_2m_previous_day1",
            "direct_radiation_previous_day1",
            "diffuse_radiation_previous_day1",
            "cloud_cover_previous_day1"
        ]
    ].copy()
    shifted_factor.index = shifted_factor.index + pd.Timedelta(minutes=30)

    df = filtered_market.join(shifted_factor, how='inner')
    df = df.ffill().bfill()
    # df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')

else:
    raise ValueError("Invalid input: choose 'Nord Pool' or 'mFRR'")

features_to_scale = df.columns.drop(price_column)
scaler_features = StandardScaler()
scaler_price = StandardScaler()

df[features_to_scale] = scaler_features.fit_transform(df[features_to_scale])
df[price_column] = scaler_price.fit_transform(df[[price_column]])


seq_length = 30
def create_sequences(data, seq_length):

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data[price_column].iloc[i+seq_length])
    return np.array(X), np.array(y)

data_x, data_y = create_sequences(df, seq_length)

train_size = int(0.8 * len(data_x))
X_train = torch.from_numpy(data_x[:train_size].astype(np.float32)).to(device)
y_train = torch.from_numpy(data_y[:train_size].astype(np.float32)).unsqueeze(1).to(device)
X_test = torch.from_numpy(data_x[train_size:].astype(np.float32)).to(device)
y_test = torch.from_numpy(data_y[train_size:].astype(np.float32)).unsqueeze(1).to(device)

input_dim = data_x.shape[2]

class ForecastingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ForecastingModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
    
model = ForecastingModel(
    input_dim=data_x.shape[2], 
    hidden_dim=32, 
    num_layers=2, 
    output_dim=1
).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def forecast_future(model, last_sequence, n_steps, scaler_price):
    
    model.eval()
    predictions = []
    seq = last_sequence.copy()
    
    for _ in range(n_steps):
        input_seq = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(input_seq).cpu().numpy()[0, 0]
            pred_real = scaler_price.inverse_transform([[pred_scaled]])[0, 0]
        
        predictions.append(pred_real)

        new_row = seq[-1].copy()
        new_row[0] = pred_scaled
        seq = np.vstack([seq[1:], new_row])

    return predictions

def evaluate_price_extremes(y_actual, y_pred, timestamps, spread_threshold=200):
    
    df = pd.DataFrame({
        'actual': y_actual.flatten(),
        'predicted': y_pred.flatten(),
        'timestamp': pd.to_datetime(timestamps)
    })
    df['date'] = df['timestamp'].dt.date
    
    min_correct = max_correct = total_days = 0
    actual_high_spreads = predicted_high_spreads = correctly_identified = 0
    
    for date, day_data in df.groupby('date'):
        if len(day_data) < 2:
            continue
            
        actual_min_time = day_data.loc[day_data['actual'].idxmin(), 'timestamp']
        actual_max_time = day_data.loc[day_data['actual'].idxmax(), 'timestamp']
        pred_min_time = day_data.loc[day_data['predicted'].idxmin(), 'timestamp']
        pred_max_time = day_data.loc[day_data['predicted'].idxmax(), 'timestamp']
        
        if abs((actual_min_time - pred_min_time).total_seconds()) <= 7200:  # 2 hours
            min_correct += 1
        if abs((actual_max_time - pred_max_time).total_seconds()) <= 7200:
            max_correct += 1
        total_days += 1
        
        actual_spread = day_data['actual'].max() - day_data['actual'].min()
        pred_spread = day_data['predicted'].max() - day_data['predicted'].min()
        
        if actual_spread > spread_threshold:
            actual_high_spreads += 1
        if pred_spread > spread_threshold:
            predicted_high_spreads += 1
        if actual_spread > spread_threshold and pred_spread > spread_threshold:
            correctly_identified += 1
    
    min_accuracy = min_correct / total_days if total_days > 0 else 0
    max_accuracy = max_correct / total_days if total_days > 0 else 0
    precision = correctly_identified / predicted_high_spreads if predicted_high_spreads > 0 else 0
    recall = correctly_identified / actual_high_spreads if actual_high_spreads > 0 else 0
    
    print(f"\nPrice Extreme Timing (±2 hours):")
    print(f"  Min price timing accuracy: {min_accuracy:.1%}")
    print(f"  Max price timing accuracy: {max_accuracy:.1%}")
    
    print(f"\nHigh Spreads (>{spread_threshold} EUR/MWh):")
    print(f"  Actual high-spread days: {actual_high_spreads}/{total_days}")
    print(f"  Correctly identified: {correctly_identified}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")

def evaluate_and_forecast():

    model.eval()
    
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred = model(X_test).cpu().numpy()
    
    y_train_actual = scaler_price.inverse_transform(y_train.cpu().numpy())
    y_test_actual = scaler_price.inverse_transform(y_test.cpu().numpy())
    y_train_pred = scaler_price.inverse_transform(y_train_pred)
    y_test_pred = scaler_price.inverse_transform(y_test_pred)
    
    train_rmse = root_mean_squared_error(y_train_actual, y_train_pred)
    test_rmse = root_mean_squared_error(y_test_actual, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Train RMSE: {train_rmse:.2f} EUR/MWh")
    print(f"Test RMSE: {test_rmse:.2f} EUR/MWh")
    
    test_start_idx = len(df) - len(y_test_actual)
    test_timestamps = df.index[test_start_idx:test_start_idx + len(y_test_actual)]
    evaluate_price_extremes(y_test_actual, y_test_pred, test_timestamps)
    
    n_future_hours = 24
    last_seq = df.iloc[-seq_length:].values
    future_prices = forecast_future(model, last_seq, n_future_hours, scaler_price)
    
    return y_test_actual, y_test_pred, test_rmse, future_prices

y_test_actual, y_test_pred, test_rmse, future_prices = evaluate_and_forecast()

def create_visualizations():
    
    test_dates = df.iloc[-len(y_test_actual):].index
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), 
        periods=24, 
        freq="H"
    )
    
    historical_data = scaler_price.inverse_transform(
        df[[price_column]]
    ).flatten()
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    axes[0].plot(test_dates, y_test_actual.flatten(), 'b-', label='Actual', linewidth=2)
    axes[0].plot(test_dates, y_test_pred.flatten(), 'g--', label='Predicted', linewidth=2)
    axes[0].set_title('Model Performance on Test Set')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (EUR/MWh)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    errors = np.abs(y_test_actual.flatten() - y_test_pred.flatten())
    axes[1].plot(test_dates, errors, 'r-', label='Absolute Error', alpha=0.7)
    axes[1].axhline(test_rmse, color='blue', linestyle='--', label=f'RMSE: {test_rmse:.2f}')
    axes[1].set_title('Prediction Errors')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Error (EUR/MWh)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df.index[-100:], historical_data[-100:], 'b-', 
                   label='Historical', linewidth=2, alpha=0.8)
    axes[2].plot(future_dates, future_prices, 'orange', 
                   label='24h Forecast', linewidth=2, marker='o', markersize=4)
    axes[2].axvline(df.index[-1], color='gray', linestyle='--', alpha=0.7)
    axes[2].set_title('Historical Data and 24-Hour Forecast')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Price (EUR/MWh)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n24-Hour Forecast Summary:")
    print(f"Average price: {np.mean(future_prices):.2f} EUR/MWh")
    print(f"Min price: {np.min(future_prices):.2f} EUR/MWh")
    print(f"Max price: {np.max(future_prices):.2f} EUR/MWh")
    print(f"Price volatility (std): {np.std(future_prices):.2f} EUR/MWh")

create_visualizations()

forecast_df = pd.DataFrame({
    'datetime': pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=24, freq="H"),
    'predicted_price_eur_mwh': future_prices
})