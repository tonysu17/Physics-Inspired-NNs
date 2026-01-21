import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation_fdm():
    """
    Solves the 1D Heat Equation u_t = alpha * u_xx using Finite Difference Method.
    Domain: x in [0, L]
    Boundary Conditions: Dirichlet u(0,t)=0, u(L,t)=0
    Initial Condition: Gaussian Pulse
    """

    # --- Physical Parameters ---
    L = 1.0           # Length of the domain (m)
    alpha = 0.01      # Thermal diffusivity (m^2/s)
    T_final = 0.5     # Total simulation time (s)

    # --- Discretization Parameters ---
    nx = 101          # Number of spatial grid points
    dx = L / (nx - 1) # Spatial grid spacing

    # Stability Check: CFL Condition for explicit scheme
    # r = alpha * dt / dx^2 <= 0.5
    # We choose r = 0.4 for safety margin
    r = 0.4
    dt = r * dx**2 / alpha
    nt = int(T_final / dt) + 1  # Number of time steps

    print(f"Discretization: dx={dx:.4f}, dt={dt:.5f}")
    print(f"Total Steps: {nt}")

    # --- Initialization ---
    x = np.linspace(0, L, nx)
    u = np.zeros(nx)       # Array to store current temperature distribution
    u_new = np.zeros(nx)   # Array to store next time step

    # Initial Condition: Gaussian pulse centered at L/2
    u = np.exp(-200 * (x - 0.5)**2)

    # Enforce Boundary Conditions at t=0
    u[0] = 0.0 # Fixed: Apply to the first element of the array
    u[-1] = 0.0

    # Store history for plotting
    history = [u.copy()]
    checkpoints = np.linspace(0, nt, 5, dtype=int)

    # --- Time Stepping Loop ---
    # Vectorized implementation avoids slow Python loops over spatial indices
    # The formula: u_i^{n+1} = u_i^n + r * (u_{i+1}^n - 2u_i^n + u_{i-1}^n)

    for n in range(nt):
        # Slicing:
        # u[1:-1] refers to indices 1 to nx-2 (internal nodes)
        # u[2:]   refers to indices 2 to nx-1 (right neighbors)
        # u[:-2]  refers to indices 0 to nx-3 (left neighbors)

        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

        # Enforce Boundary Conditions (Dirichlet)
        u_new[0] = 0.0 # Fixed: Apply to the first element of the array
        u_new[-1] = 0.0

        # Update solution
        u = u_new.copy()

        if n in checkpoints:
            history.append(u.copy())

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    for i, u_state in enumerate(history):
        # Calculate approximate time for label
        t_snapshot = checkpoints[i-1] * dt if i > 0 else 0
        if i == 0: t_snapshot = 0
        plt.plot(x, u_state, label=f't={t_snapshot:.3f}s')

    plt.title('1D Heat Diffusion using Finite Difference Method')
    plt.xlabel('Position x (m)')
    plt.ylabel('Temperature u')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()

if __name__ == "__main__":
    solve_heat_equation_fdm()

