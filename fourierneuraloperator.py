import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from timeit import default_timer
import math

# ==========================================
# 1. THE FOURIER NEURAL OPERATOR (FNO) MODEL
# ==========================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform on low fourier modes, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # 1. Compute Fourier coefficients up to a factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)

        # We process the lowest 'modes' frequencies
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        # 3. Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to a higher dimension channel space
        2. 4 layers of integral operators (SpectralConv) + activation
        3. Project back to the target output dimension
        """
        self.modes = modes
        self.width = width
        self.padding = 2 # pad the domain if input is not non-periodic

        # Input is usually (x, a(x)) -> we map 1 channel to width
        self.p = nn.Linear(2, self.width) # input channel is 2: (u(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.q = nn.Linear(self.width, 1) # output channel is 1: u(x, T)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1) # Concatenate spatial coordinate

        x = x.permute(0, 2, 1)
        x = self.p(x)
        x = x.permute(0, 2, 1)

        # Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 3
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 4
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)

# ==========================================
# 2. PHYSICAL APPLICATION: BURGERS EQUATION
# ==========================================
# We generate data for the 1D Burgers equation: du/dt + u*du/dx = v*d^2u/dx^2

def generate_burgers_data(ntrain=100, ntest=20, s=128, T=1.0, dt=0.001): # Changed dt from 0.01 to 0.001
    """
    Generates synthetic real-physics data using a numerical solver.
    """
    print(f"Generating {ntrain+ntest} samples of Burgers' Equation physics...")
    nu = 0.01 # Viscosity

    inputs = []
    outputs = []

    for _ in range(ntrain + ntest):
        # Initial condition: Random sum of sin waves
        x = np.linspace(0, 1, s)
        u0 = np.zeros(s)
        for k in range(1, 5):
            u0 += np.sin(2 * np.pi * k * x) * np.random.randn() * 0.5

        # Solve using Finite Difference
        u = u0.copy()
        nx = len(x)
        dx = 1.0 / (nx - 1)
        # Ensure steps match the new dt
        steps = int(T / dt)

        # Evolution loop
        for _ in range(steps):
            un = u.copy()
            # Explicit finite difference scheme
            for i in range(1, nx-1):
                u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1]) + \
                       nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
            # Periodic BC
            u[0] = u[-1]

        inputs.append(u0)
        outputs.append(u)

    inputs = torch.tensor(np.array(inputs), dtype=torch.float).unsqueeze(1)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float).unsqueeze(1)

    return inputs[:ntrain], outputs[:ntrain], inputs[ntrain:], outputs[ntrain:]
# ==========================================
# 3. FINANCIAL APPLICATION: STOCK DATA
# ==========================================

def get_financial_data(ticker="SPY", n_samples=1000, seq_len=64):
    print(f"Downloading real financial data for {ticker}...")
    import pandas as pd
    try:
        data = yf.download(ticker, start="2000-01-01", end="2024-01-01", progress=False)
        if data.empty: raise ValueError("Empty Data")

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
             prices = data['Close'].iloc[:, 0].values.flatten()
        else:
             prices = data['Close'].values.flatten()
    except:
        print("Download failed, using synthetic random walk.")
        prices = np.cumsum(np.random.randn(5000)) + 100

    X, Y = [], []
    # Create sliding windows
    max_len = len(prices) - 2 * seq_len

    for i in range(min(max_len, n_samples * 2)): # Get enough data
        # Raw sequences
        x_raw = prices[i : i+seq_len]
        y_raw = prices[i+seq_len : i+2*seq_len]

        # --- KEY FIX: INSTANCE NORMALIZATION ---
        # Normalize each sample by its OWN mean/std to focus on the 'shape'
        mu = np.mean(x_raw)
        sigma = np.std(x_raw) + 1e-5 # avoid div by zero

        x_norm = (x_raw - mu) / sigma
        y_norm = (y_raw - mu) / sigma # Normalize target using INPUT stats to preserve relative scale

        X.append(x_norm)
        Y.append(y_norm)

    X = np.array(X)
    Y = np.array(Y)

    # Convert to Tensor
    limit = min(len(X), n_samples)
    X = torch.tensor(X[:limit], dtype=torch.float).unsqueeze(1)
    Y = torch.tensor(Y[:limit], dtype=torch.float).unsqueeze(1)

    split = int(0.8 * limit)
    return X[:split], Y[:split], X[split:], Y[split:]

# ==========================================
# 4. TRAINING UTILITY
# ==========================================

def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=50, lr=0.001, name="Task"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = nn.MSELoss()

    print(f"\n--- Training {name} ---")
    t0 = default_timer()

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        loss = loss_fn(out, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 10 == 0:
            print(f"Epoch {ep}: Loss {loss.item():.6f}")

    t1 = default_timer()
    print(f"Training completed in {t1-t0:.2f}s")

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        test_loss = loss_fn(pred, test_y)

    print(f"Test MSE: {test_loss.item():.6f}")

    # Plotting first test sample
    plt.figure(figsize=(10, 4))
    plt.title(f"{name}: Prediction vs Ground Truth (Sample 0)")
    # Flatten for plotting
    truth = test_y[0,0,:].cpu().numpy()
    prediction = pred[0,0,:].cpu().numpy()
    input_seq = test_x[0,0,:].cpu().numpy()

    if name == "Financial":
        # For finance, plotting input (history) and output (future)
        x_axis_in = np.arange(len(input_seq))
        x_axis_out = np.arange(len(input_seq), len(input_seq) + len(truth))
        plt.plot(x_axis_in, input_seq, label="Input (History)", color='gray', linestyle='--')
        plt.plot(x_axis_out, truth, label="Ground Truth (Future)", color='blue')
        plt.plot(x_axis_out, prediction, label="FNO Prediction", color='red')
    else:
        plt.plot(truth, label="Ground Truth", color='blue')
        plt.plot(prediction, label="FNO Prediction", color='red', linestyle='--')

    plt.legend()
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # --- 1. PHYSICAL TEST (Burgers Equation) ---
    modes = 16
    width = 64
    x_train, y_train, x_test, y_test = generate_burgers_data(ntrain=200, ntest=40, s=64)

    fno_phys = FNO1d(modes, width).to(device)
    train_and_evaluate(fno_phys,
                       x_train.to(device), y_train.to(device),
                       x_test.to(device), y_test.to(device),
                       epochs=50, name="Physical (Burgers Eq)")

    # --- 2. FINANCIAL TEST (SPY Stock Data) ---
    # Note: Financial data is noisier and harder to predict.
    # We use a smaller mode count to avoid overfitting high-frequency noise.
    modes = 12
    width = 32
    fx_train, fy_train, fx_test, fy_test = get_financial_data("SPY", n_samples=1000, seq_len=64)

    fno_fin = FNO1d(modes, width).to(device)
    train_and_evaluate(fno_fin,
                       fx_train.to(device), fy_train.to(device),
                       fx_test.to(device), fy_test.to(device),
                       epochs=50, name="Financial (SPY Stock)")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from timeit import default_timer
import pandas as pd

# ==========================================
# 1. THE FOURIER NEURAL OPERATOR (FNO) MODEL
# ==========================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.p = nn.Linear(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.q = nn.Linear(self.width, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 1)
        x = self.p(x)
        x = x.permute(0, 2, 1)

        for conv, w in zip([self.conv0, self.conv1, self.conv2, self.conv3], [self.w0, self.w1, self.w2, self.w3]):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)

# ==========================================
# 2. PHYSICAL APPLICATION: BURGERS EQUATION (FIXED)
# ==========================================

def generate_burgers_data(ntrain=100, ntest=20, s=128, T=1.0):
    """
    Generates synthetic data with reduced amplitude and smaller dt for stability.
    """
    print(f"Generating {ntrain+ntest} samples of Burgers' Equation physics...")

    # PARAMETERS FOR STABILITY
    nu = 0.01          # Viscosity
    dt = 0.0005        # Reduced time step (Crucial for stability)
    dx = 1.0 / (s - 1) # Grid size
    steps = int(T / dt)

    inputs = []
    outputs = []

    for _ in range(ntrain + ntest):
        # Initial condition: Smaller amplitude random waves
        x_grid = np.linspace(0, 1, s)
        u0 = np.zeros(s)
        for k in range(1, 4):
            # Reduced amplitude from 0.5 to 0.2 to prevent shockwave explosion
            u0 += np.sin(2 * np.pi * k * x_grid) * np.random.randn() * 0.2

        u = u0.copy()

        # Evolution loop with explicit Finite Difference
        # If this still overflows, the amplitude is too high for the resolution.
        valid_sim = True
        for _ in range(steps):
            un = u.copy()
            for i in range(1, s-1):
                # Standard discretized Burgers equation
                convection = un[i] * dt / dx * (un[i] - un[i-1])
                diffusion = nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
                u[i] = un[i] - convection + diffusion

            # Periodic BC
            u[0] = u[-1]

            # Safety check: if values explode, stop
            if np.max(np.abs(u)) > 10.0:
                valid_sim = False
                break

        if valid_sim:
            inputs.append(u0)
            outputs.append(u)
        else:
            # If simulation failed, just add a dummy zero-vec to keep indexing simple
            # (In production, you would regenerate)
            inputs.append(np.zeros(s))
            outputs.append(np.zeros(s))

    # Convert to Tensor
    inputs = torch.tensor(np.array(inputs), dtype=torch.float).unsqueeze(1)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float).unsqueeze(1)

    return inputs[:ntrain], outputs[:ntrain], inputs[ntrain:], outputs[ntrain:]

# ==========================================
# 3. FINANCIAL APPLICATION: STOCK DATA (FIXED)
# ==========================================

def get_financial_data(ticker="SPY", n_samples=1000, seq_len=64):
    print(f"Downloading real financial data for {ticker}...")
    try:
        data = yf.download(ticker, start="2000-01-01", end="2024-01-01", progress=False)
        if data.empty: raise ValueError("Empty Data")

        if isinstance(data.columns, pd.MultiIndex):
             prices = data['Close'].iloc[:, 0].values.flatten()
        else:
             prices = data['Close'].values.flatten()
    except:
        print("Download failed, using synthetic random walk.")
        prices = np.cumsum(np.random.randn(5000)) + 100

    X, Y = [], []
    max_len = len(prices) - 2 * seq_len

    for i in range(min(max_len, n_samples * 2)):
        # Raw sequences
        x_raw = prices[i : i+seq_len]
        y_raw = prices[i+seq_len : i+2*seq_len]

        # --- INSTANCE NORMALIZATION ---
        # We normalize y_raw using x_raw's stats.
        # This teaches the model to predict relative movement from the last seen window.
        mu = np.mean(x_raw)
        sigma = np.std(x_raw) + 1e-6

        x_norm = (x_raw - mu) / sigma
        y_norm = (y_raw - mu) / sigma

        X.append(x_norm)
        Y.append(y_norm)

    X = np.array(X)
    Y = np.array(Y)

    limit = min(len(X), n_samples)
    X = torch.tensor(X[:limit], dtype=torch.float).unsqueeze(1)
    Y = torch.tensor(Y[:limit], dtype=torch.float).unsqueeze(1)

    split = int(0.8 * limit)
    return X[:split], Y[:split], X[split:], Y[split:]

# ==========================================
# 4. TRAINING & PLOTTING
# ==========================================

def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=50, lr=0.001, name="Task"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"\n--- Training {name} ---")
    t0 = default_timer()

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_x)
        loss = loss_fn(out, train_y)
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:
            print(f"Epoch {ep}: Loss {loss.item():.6f}")

    t1 = default_timer()
    print(f"Training completed in {t1-t0:.2f}s")

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        test_loss = loss_fn(pred, test_y)

    print(f"Test MSE: {test_loss.item():.6f}")

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.title(f"{name}: Prediction vs Ground Truth (Sample 0)")

    # Extract sample 0
    truth = test_y[0,0,:].cpu().numpy()
    prediction = pred[0,0,:].cpu().numpy()

    if name.startswith("Financial"):
        input_seq = test_x[0,0,:].cpu().numpy()
        # Create x-axis for plotting continuity
        x_in = np.arange(len(input_seq))
        x_out = np.arange(len(input_seq), len(input_seq) + len(truth))

        plt.plot(x_in, input_seq, label="Input (History)", color='gray', linestyle='--')
        plt.plot(x_out, truth, label="Ground Truth (Future)", color='blue')
        plt.plot(x_out, prediction, label="FNO Prediction", color='red')
    else:
        plt.plot(truth, label="Ground Truth", color='blue')
        plt.plot(prediction, label="FNO Prediction", color='red', linestyle='--')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # 1. PHYSICAL TEST
    modes = 16
    width = 64
    x_train, y_train, x_test, y_test = generate_burgers_data(ntrain=200, ntest=40, s=64)

    fno_phys = FNO1d(modes, width).to(device)
    train_and_evaluate(fno_phys,
                       x_train.to(device), y_train.to(device),
                       x_test.to(device), y_test.to(device),
                       epochs=100, name="Physical (Burgers Eq)")

    # 2. FINANCIAL TEST
    modes = 12
    width = 32
    fx_train, fy_train, fx_test, fy_test = get_financial_data("SPY", n_samples=1000, seq_len=64)

    fno_fin = FNO1d(modes, width).to(device)
    train_and_evaluate(fno_fin,
                       fx_train.to(device), fy_train.to(device),
                       fx_test.to(device), fy_test.to(device),
                       epochs=50, name="Financial (SPY Stock)")

