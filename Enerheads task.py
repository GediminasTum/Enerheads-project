import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

df = market_df[["10YLT-1001A0008Q_DA_eurmwh"]].join(
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

features_to_scale = df.columns.drop("10YLT-1001A0008Q_DA_eurmwh")
scaler_features = StandardScaler()
scaler_price = StandardScaler()

df[features_to_scale] = scaler_features.fit_transform(df[features_to_scale])
df["10YLT-1001A0008Q_DA_eurmwh"] = scaler_price.fit_transform(df[["10YLT-1001A0008Q_DA_eurmwh"]])


seq_length = 30
def create_sequences(data, seq_length):

    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data["10YLT-1001A0008Q_DA_eurmwh"].iloc[i+seq_length])
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
    
    n_future_hours = 24
    last_seq = df.iloc[-seq_length:].values
    future_prices = forecast_future(model, last_seq, n_future_hours, scaler_price)
    
    return y_test_actual, y_test_pred, test_rmse, future_prices

y_test_actual, y_test_pred, test_rmse, future_prices = evaluate_and_forecast()

# Create comprehensive visualization
def create_visualizations():
    """Create all visualizations in one function."""
    
    # Test dates and future dates
    test_dates = df.iloc[-len(y_test_actual):].index
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), 
        periods=24, 
        freq="H"
    )
    
    # Historical data for context (last 100 hours)
    historical_data = scaler_price.inverse_transform(
        df[["10YLT-1001A0008Q_DA_eurmwh"]]
    ).flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(test_dates, y_test_actual.flatten(), 'b-', label='Actual', linewidth=2)
    axes[0, 0].plot(test_dates, y_test_pred.flatten(), 'g--', label='Predicted', linewidth=2)
    axes[0, 0].set_title('Model Performance on Test Set')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price (EUR/MWh)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    errors = np.abs(y_test_actual.flatten() - y_test_pred.flatten())
    axes[0, 1].plot(test_dates, errors, 'r-', label='Absolute Error', alpha=0.7)
    axes[0, 1].axhline(test_rmse, color='blue', linestyle='--', label=f'RMSE: {test_rmse:.2f}')
    axes[0, 1].set_title('Prediction Errors')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Error (EUR/MWh)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df.index[-100:], historical_data[-100:], 'b-', 
                   label='Historical', linewidth=2, alpha=0.8)
    axes[1, 0].plot(future_dates, future_prices, 'orange', 
                   label='24h Forecast', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axvline(df.index[-1], color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Historical Data and 24-Hour Forecast')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Price (EUR/MWh)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    hours = list(range(1, 25))
    axes[1, 1].bar(hours, future_prices, alpha=0.7, color='lightcoral', edgecolor='darkred')
    axes[1, 1].set_title('24-Hour Forecast Distribution')
    axes[1, 1].set_xlabel('Hours Ahead')
    axes[1, 1].set_ylabel('Predicted Price (EUR/MWh)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
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

print(f"\nFirst 12 hours of forecast:")
print(forecast_df.head(12))