import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set device to GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Load data
sp500 = yf.Ticker("^GSPC").history(period="max")
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Add technical indicators
def add_technical_indicators(sp500, horizons):
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        sp500[f"Close_Ratio_{horizon}"] = sp500["Close"] / rolling_averages["Close"]
        sp500[f"Trend_{horizon}"] = sp500.shift(1).rolling(horizon).sum()["Target"]
        sp500[f"EMA_COLUMN_{horizon}"] = sp500["Close"].ewm(span=horizon, adjust=False).mean()
        sp500[f'SMA_{horizon}'] = sp500['Close'].rolling(window=horizon).mean()
        sp500[f'EMA_{horizon}'] = sp500['Close'].ewm(span=horizon, adjust=False).mean()
        
        delta = sp500['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=horizon).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=horizon).mean()
        rs = gain / loss
        sp500[f'RSI_{horizon}'] = 100 - (100 / (1 + rs))
        
        ema_fast = sp500['Close'].ewm(span=horizon//2, adjust=False).mean()
        ema_slow = sp500['Close'].ewm(span=horizon, adjust=False).mean()
        sp500[f'MACD_{horizon}'] = ema_fast - ema_slow
        sp500[f'MACD_signal_{horizon}'] = sp500[f'MACD_{horizon}'].ewm(span=9, adjust=False).mean()
        sp500[f'MACD_diff_{horizon}'] = sp500[f'MACD_{horizon}'] - sp500[f'MACD_signal_{horizon}']
        
        rolling_std = sp500['Close'].rolling(window=horizon).std()
        sp500[f'BB_high_{horizon}'] = sp500[f'SMA_{horizon}'] + (rolling_std * 2)
        sp500[f'BB_low_{horizon}'] = sp500[f'SMA_{horizon}'] - (rolling_std * 2)
        
        high_low = sp500['High'] - sp500['Low']
        high_close = (sp500['High'] - sp500['Close'].shift()).abs()
        low_close = (sp500['Low'] - sp500['Close'].shift()).abs()
        tr = high_low.combine(high_close, max).combine(low_close, max)
        sp500[f'ATR_{horizon}'] = tr.rolling(window=horizon).mean()
        
        obv = (sp500['Volume'] * ((sp500['Close'] - sp500['Close'].shift()) / sp500['Close'].shift())).fillna(0)
        sp500[f'OBV_{horizon}'] = obv.cumsum()
    
    sp500.dropna(inplace=True)
    return sp500

horizons = [2, 5, 60, 250, 1000]
sp500 = add_technical_indicators(sp500, horizons)

# Split data using numpy
train_size = int(len(sp500) * 0.9)
test_size = len(sp500) - train_size

train_data, test_data = sp500[:train_size], sp500[train_size:]

X_train = train_data.drop(columns=['Close', 'Tomorrow', 'Target']).values
y_train = train_data['Close'].values
X_test = test_data.drop(columns=['Close', 'Tomorrow', 'Target']).values
y_test = test_data['Close'].values

# Convert to torch tensors and reshape to match LSTM input shape (batch_size, seq_length, num_features)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

x_pred = torch.tensor(X_train[-1], dtype=torch.float32).view(1,-1).to(device)

# Check dimensions
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, x_pred.shape)

hidden_size = 256
num_layers = 2
learning_rate = 0.005
batch_size = 64
num_epochs = 3
input_size = 71

class stonks(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(stonks, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layers
        #self.pool = nn.MaxPool1d(kernel_size = 2)   #same formula to keep the dimension size
        self.conv = nn.Conv1d(in_channels = input_size, out_channels = input_size,kernel_size = (3), padding = "same")  
        self.LSTMInitial = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fcMid = nn.Linear(hidden_size, input_size)

        self.fcLast = nn.Linear(input_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        print(x.shape)

        x = x.permute(1,0)
        print(x.shape)

        x = F.relu(self.conv(x))
        print(x.shape)

        # Forward propagate LSTM
        x = x.permute(1,0)
        print(x.shape)
        x, _ = self.LSTMInitial(x, (h0, c0))
        #fc
        x = self.fcMid(x)

        print(x.shape)
        out = self.fcLast(x)
        return out

# Instantiate the model and move it to the GPU
model = stonks(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
epochs = 4
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1 == 0:  # Print every n epochs
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # Toggle model back to train
    model.train()
    return torch.mean(torch.true_divide(predictions,y_test))

print(f"Accuracy on training set: {check_accuracy(X_train, model) * 100:.2f}%")
print(f"Accuracy on test set: {check_accuracy(X_test, model) * 100:.2f}%")


with torch.no_grad():
    predictions = model(x_pred)
    print("Predicted Labels:", predictions)


# Saving the model
torch.save(model.state_dict(), 'stock_market_prediction_model.pth')
