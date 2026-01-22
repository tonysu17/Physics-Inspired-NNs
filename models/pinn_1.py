import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Configuration & Physics Parameters
# ==========================================
class Config:
    alpha = 0.1          # Thermal diffusivity (akin to volatility^2/2 in Black-Scholes)
    x_min, x_max = -1, 1 # Spatial domain
    t_min, t_max = 0, 1  # Time domain

    n_collocation = 5000 # Points inside the domain to test PDE
    n_boundary    = 500  # Points on the boundary
    epochs        = 3000
    lr            = 0.001

    # Neural Net Architecture
    hidden_layers = [40, 40, 40, 40]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. The PINN Model (Multi-Layer Perceptron)
# ==========================================
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(PINN, self).__init__()

        layers = []
        input_size = input_dim

        for hidden_size in Config.hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh()) # Tanh is standard for PINNs (smooth derivatives)
            input_size = hidden_size

        layers.append(nn.Linear(input_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # Concatenate x and t to form the input vector
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ==========================================
# 3. Physics Loss (The Core Logic)
# ==========================================
def physics_loss(model, x, t):
    """
    Calculates the PDE residual: f = u_t - alpha * u_xx
    We want f to be 0.
    """
    # Enable gradient tracking for inputs to compute partial derivatives
    x.requires_grad = True
    t.requires_grad = True

    u = model(x, t)

    # First derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    # Second derivative
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # The Heat Equation Residual
    residual = u_t - Config.alpha * u_xx
    return torch.mean(residual ** 2)

# ==========================================
# 4. Data Generation (Collocation Sampling)
# ==========================================
def get_training_data():
    # A. Collocation Points (Inside the domain)
    # Random sample of t and x
    x_col = np.random.uniform(Config.x_min, Config.x_max, (Config.n_collocation, 1))
    t_col = np.random.uniform(Config.t_min, Config.t_max, (Config.n_collocation, 1))

    # B. Initial Condition (t=0)
    # Let's say u(x,0) = sin(pi * x)
    x_ic = np.random.uniform(Config.x_min, Config.x_max, (Config.n_boundary, 1))
    t_ic = np.zeros_like(x_ic)
    u_ic = np.sin(np.pi * x_ic) # The actual initial values

    # C. Boundary Conditions (x=-1 and x=1)
    # Let's say u(-1, t) = 0 and u(1, t) = 0
    t_bc = np.random.uniform(Config.t_min, Config.t_max, (Config.n_boundary, 1))

    x_bc_left = np.full_like(t_bc, Config.x_min)
    u_bc_left = np.zeros_like(x_bc_left)

    x_bc_right = np.full_like(t_bc, Config.x_max)
    u_bc_right = np.zeros_like(x_bc_right)

    # Convert all to tensors
    pt = lambda x: torch.tensor(x, dtype=torch.float32).to(device)

    return {
        'col_x': pt(x_col), 'col_t': pt(t_col),
        'ic_x': pt(x_ic),   'ic_t': pt(t_ic),   'ic_u': pt(u_ic),
        'bc_left_x': pt(x_bc_left), 'bc_left_t': pt(t_bc), 'bc_left_u': pt(u_bc_left),
        'bc_right_x': pt(x_bc_right), 'bc_right_t': pt(t_bc), 'bc_right_u': pt(u_bc_right)
    }

# ==========================================
# 5. Training Loop
# ==========================================
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
data = get_training_data()

print("Starting Training...")
loss_history = []

for epoch in range(Config.epochs):
    optimizer.zero_grad()

    # 1. Loss at Collocation Points (PDE Consistency)
    loss_pde = physics_loss(model, data['col_x'], data['col_t'])

    # 2. Loss at Initial Condition (Data Consistency)
    pred_ic = model(data['ic_x'], data['ic_t'])
    loss_ic = torch.mean((pred_ic - data['ic_u']) ** 2)

    # 3. Loss at Boundary Conditions
    pred_bc_left = model(data['bc_left_x'], data['bc_left_t'])
    loss_bc_left = torch.mean((pred_bc_left - data['bc_left_u']) ** 2)

    pred_bc_right = model(data['bc_right_x'], data['bc_right_t'])
    loss_bc_right = torch.mean((pred_bc_right - data['bc_right_u']) ** 2)

    # Total Loss
    total_loss = loss_pde + loss_ic + loss_bc_left + loss_bc_right

    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss {total_loss.item():.6f} (PDE: {loss_pde.item():.5f})")


# ==========================================
# 6. RIGOROUS VALIDATION
# ==========================================

def exact_solution(x, t, alpha):
    """
    Analytical solution for: u_t = alpha * u_xx
    IC: u(x,0) = sin(pi*x)
    BC: u(-1,t)=0, u(1,t)=0
    Solution: u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)
    """
    return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

# --- A. Prepare Validation Data on a Grid ---
# Generate a fine grid for detailed validation
val_x = np.linspace(Config.x_min, Config.x_max, 200)
val_t = np.linspace(Config.t_min, Config.t_max, 100)
T_grid, X_grid = np.meshgrid(val_t, val_x)

# Convert grid to torch tensors for prediction
X_flat_val = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32).to(device)
T_flat_val = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32).to(device)

# 1. Get Neural Net Prediction
model.eval() # Set to evaluation mode
with torch.no_grad():
    U_pred_val = model(X_flat_val, T_flat_val).cpu().numpy().reshape(X_grid.shape)

# 2. Get Exact Solution
U_exact_val = exact_solution(X_grid, T_grid, Config.alpha)

# 3. Compute Absolute Error
abs_error = np.abs(U_pred_val - U_exact_val)

# --- B. Visualization: Time Slices (The most intuitive check) ---
# We verify the solution at t=0.0, t=0.5, and t=1.0
snapshots = [0.0, 0.50, 0.99] # Times to sample

plt.figure(figsize=(14, 5))

for i, t_snap in enumerate(snapshots):
    # Find the index in our grid closest to this time
    t_idx = (np.abs(val_t - t_snap)).argmin()

    plt.subplot(1, 3, i+1)

    # Plot Exact
    plt.plot(val_x, U_exact_val[:, t_idx], 'k--', linewidth=2, label="Exact (Analytical)")

    # Plot Neural Net
    plt.plot(val_x, U_pred_val[:, t_idx], 'r', linestyle=(0, (1, 1)), linewidth=2.5, label="PINN (Neural Net)")

    plt.title(f"Snapshot at t = {t_snap:.2f}s")
    plt.xlabel("Space (x)")
    plt.ylabel("u(x,t)")
    plt.grid(True, alpha=0.3)
    if i == 0: plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 7. Visualization (Simulated Dashboard)
# ==========================================
print("Training Complete. Generating Heatmap...")

# Create a grid to visualize the solution
t_vals = np.linspace(Config.t_min, Config.t_max, 100)
x_vals = np.linspace(Config.x_min, Config.x_max, 100)
T, X = np.meshgrid(t_vals, x_vals)

# Flatten for prediction
X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)

with torch.no_grad():
    U_pred = model(X_flat, T_flat).cpu().numpy().reshape(100, 100)

plt.figure(figsize=(10, 6))
sns.heatmap(U_pred, cmap="inferno", xticklabels=False, yticklabels=False)
plt.title(f"PINN Solution to Heat Equation (Alpha={Config.alpha})")
plt.xlabel("Time (t)")
plt.ylabel("Space (x)")
plt.show()

# --- C. Visualization: Error Heatmap ---
# This shows us exactly where the model struggles (usually boundaries)

plt.figure(figsize=(12, 5))

# Plot 1: The Exact Solution
plt.subplot(1, 2, 1)
plt.pcolormesh(T_grid, X_grid, U_exact_val, cmap='viridis', shading='auto')
plt.colorbar(label='u(x,t)')
plt.title("Exact Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Space")

# Plot 2: The Absolute Error
plt.subplot(1, 2, 2)
im = plt.pcolormesh(T_grid, X_grid, abs_error, cmap='magma', shading='auto')
plt.colorbar(im, label='|Prediction - Exact|')
plt.title("Absolute Error Map")
plt.xlabel("Time")
plt.ylabel("Space")

plt.tight_layout()
plt.show()

print(f"Max Absolute Error: {np.max(abs_error):.5f}")
print(f"Mean Squared Error: {np.mean(abs_error**2):.6f}")
