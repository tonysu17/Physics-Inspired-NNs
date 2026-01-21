import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class LBMSolver:
    """
    D2Q9 Lattice Boltzmann Solver for Flow Past a Cylinder.
    Implements BGK collision and Half-way Bounce-Back boundaries.
    """
    def __init__(self, nx, ny, re, u_inlet, cyl_center, cyl_radius):
        self.nx = nx  # Lattice width
        self.ny = ny  # Lattice height

        # Simulation Parameters
        self.Re = re
        self.u_inlet = u_inlet
        self.radius = cyl_radius
        self.cx, self.cy = cyl_center

        # Derive Viscosity and Relaxation Time
        # Re = (u * D) / nu  => nu = (u * D) / Re
        # D = 2 * radius
        self.D = 2.0 * self.radius
        self.nu = (self.u_inlet * self.D) / self.Re
        self.tau = 3.0 * self.nu + 0.5

        # Stability Check
        if self.tau <= 0.51:
            print(f"WARNING: tau = {self.tau:.4f} is dangerously close to the stability limit 0.5.")

        print(f"Simulation Configuration:")
        print(f"  Grid: {nx}x{ny}")
        print(f"  Reynolds Number: {re}")
        print(f"  Lattice Velocity: {u_inlet}")
        print(f"  Lattice Viscosity: {self.nu:.6f}")
        print(f"  Relaxation Time (tau): {self.tau:.4f}")

        # D2Q9 Lattice Constants
        # Weights w_i
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        # Lattice Velocities c_i (x, y)
        # Index: 0, 1(E), 2(N), 3(W), 4(S), 5(NE), 6(NW), 7(SW), 8(SE)
        self.c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1]]
        ).T

        # Reverse indices for bounce-back (e.g., East(1) becomes West(3))
        self.opposite_idx = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Initialize Distribution Functions (f)
        # Shape: [9, ny, nx]
        self.f = np.zeros((9, ny, nx))
        self.rho = np.ones((ny, nx))
        self.u = np.zeros((2, ny, nx))

        # Initial Condition: Uniform flow to the right
        self.u[0, :, :] = self.u_inlet

        # Initialize f to equilibrium state
        self.f = self.equilibrium(self.rho, self.u)

        # Create Cylinder Obstacle Mask (Boolean: True inside obstacle)
        y_grid, x_grid = np.mgrid[0:ny, 0:nx]
        self.mask = (x_grid - self.cx)**2 + (y_grid - self.cy)**2 < self.radius**2

        # Optimize mask for bounce-back (pre-calculate indices)
        self.mask_idx = np.where(self.mask)

    def equilibrium(self, rho, u):
        """
        Computes the Equilibrium Distribution Function (f_eq).
        Formula: f_eq = w * rho * (1 + 3(c.u) + 4.5(c.u)^2 - 1.5u^2)
        """
        # Vectorized implementation for speed
        # c_dot_u calculation: Einstein summation
        # self.c shape: (9, 2), u shape: (2, ny, nx) -> Result: (9, ny, nx)
        c_dot_u = np.einsum('id, dXY -> iXY', self.c, u)

        # Velocity magnitude squared: u^2
        u_sq = u[0]**2 + u[1]**2

        # Broadcast weights to grid shape
        w_grid = self.w[:, np.newaxis, np.newaxis]

        # Expansion terms
        term1 = 3.0 * c_dot_u
        term2 = 4.5 * (c_dot_u ** 2)
        term3 = 1.5 * u_sq

        feq = rho * w_grid * (1 + term1 + term2 - term3)
        return feq

    def step(self):
        """
        Execute one time step of the LBM algorithm.
        Sequence: Macroscopic -> Collision -> Bounce-Back -> Streaming -> BCs
        """
        # 1. Compute Macroscopic Variables (rho, u)
        self.rho = np.sum(self.f, axis=0)
        # Momentum = sum(c_i * f_i)
        # Using tensordot for more direct calculation of u
        # This calculates sum(c_x * f_i) and sum(c_y * f_i)
        u_temp = np.tensordot(self.c, self.f, axes=([0], [0])) / self.rho
        self.u[0, :, :] = u_temp[0, :, :]
        self.u[1, :, :] = u_temp[1, :, :]

        # Force Inlet Condition (Macroscopic Override)
        # Left boundary (x=0) is fixed inlet
        self.u[0, :, 0] = self.u_inlet
        self.u[1, :, 0] = 0.0 # No y-velocity at inlet
        # Inlet density approximation (Zou-He simplified)
        self.rho[:, 0] = 1.0  # Simplified constant density inlet

        # 2. Collision (BGK Relaxation)
        feq = self.equilibrium(self.rho, self.u)
        # f_post = f - (f - feq) / tau
        f_post = self.f - (self.f - feq) / self.tau

        # 3. Obstacle Boundary (Half-way Bounce-Back)
        # At obstacle nodes, particles reflect back.
        # We perform this on the post-collision distribution *before* streaming.
        # This effectively reverses the populations at the boundary nodes.
        for i in range(9):
             # For every direction i, particles inside mask reflect to opposite_idx[i]
             f_post[i, self.mask] = self.f[self.opposite_idx[i], self.mask]

        # 4. Streaming
        # Shift populations in their respective directions.
        # np.roll wraps around, providing periodic BCs by default.
        # We will overwrite the non-periodic boundaries (Inlet/Outlet) next.
        for i in range(9):
            self.f[i] = np.roll(f_post[i], shift=self.c[i, 0], axis=1) # Shift x
            self.f[i] = np.roll(self.f[i], shift=self.c[i, 1], axis=0) # Shift y

        # 5. Domain Boundary Conditions

        # Inlet (Left, x=0): Set to Equilibrium for unknown populations
        # Compute rho for inlet using known populations and target u_inlet
        # Based on Zou-He boundary conditions for velocity inlet
        rho_inlet = (self.f[0, :, 0] + self.f[2, :, 0] + self.f[4, :, 0] +
                     2 * (self.f[3, :, 0] + self.f[7, :, 0] + self.f[6, :, 0])) / (1 - self.u_inlet)

        # Set inlet density to rho_inlet (derived for fixed velocity inlet)
        self.rho[:, 0] = rho_inlet
        # Set inlet velocity
        self.u[0, :, 0] = self.u_inlet
        self.u[1, :, 0] = 0.0

        # Calculate equilibrium for inlet density/velocity
        feq_inlet = self.equilibrium(self.rho[:, 0:1], self.u[:, :, 0:1])

        # Update unknown populations at inlet (Zou-He)
        self.f[1, :, 0] = feq_inlet[1, :, 0] + self.f[3, :, 0] - feq_inlet[3, :, 0]
        self.f[5, :, 0] = feq_inlet[5, :, 0] + self.f[7, :, 0] - feq_inlet[7, :, 0]
        self.f[8, :, 0] = feq_inlet[8, :, 0] + self.f[6, :, 0] - feq_inlet[6, :, 0]


        # Outlet (Right, x=-1): Zero Gradient (Neumann)
        # Copy populations from neighbor (x=-2) to boundary (x=-1)
        self.f[:, :, -1] = self.f[:, :, -2]

        # Top/Bottom: Periodic (handled implicitly by np.roll for the full domain)
        # For walls, we would need bounce-back, but assuming periodic for top/bottom here
        # If hard walls are desired, mask and bounce-back should be applied.

    def get_vorticity(self):
        """
        Compute scalar vorticity (curl of velocity field) using central differences.
        omega = dv/dx - du/dy
        """
        v = self.u[1, :, :]
        u = self.u[0, :, :]

        # np.gradient computes central differences
        dv_dx = np.gradient(v, axis=1)
        du_dy = np.gradient(u, axis=0)

        return dv_dx - du_dy

# --- Simulation Execution ---

# Configuration
NX = 420
NY = 180
RE = 200.0  # Reynolds Number (Vortex Shedding Regime)
U_LB = 0.04 # Lattice Velocity (Low enough for incompressibility)
CYL_R = 10  # Cylinder Radius (D=20)
CYL_POS = (NX//4, NY//2)

# Instantiate Solver
solver = LBMSolver(NX, NY, RE, U_LB, CYL_POS, CYL_R)

# Run Loop
steps = 15000  # Sufficient time for shedding to develop
probe_x, probe_y = NX//2, NY//2 + 20 # Probe location in the wake
u_history = [] # Initialize u_history as an empty list

print("Starting Time Integration...")
for step in range(steps):
    solver.step()

    # Monitor velocity at probe for Strouhal analysis
    u_history.append(solver.u[0, probe_y, probe_x])

    if step % 1000 == 0:
        print(f"  Step {step}/{steps} completed")

print("Simulation Finished.")

# --- Visualization and Verification ---
plt.figure(figsize=(14, 10))

# 1. Vorticity Field Visualization
vorticity = solver.get_vorticity()
vorticity[solver.mask] = np.nan # Mask obstacle

plt.subplot(2, 1, 1)
plt.title(f"Vorticity Field (Re={RE}) - Von Kármán Vortex Street")
# RdBu colormap: Red (Positive Spin), Blue (Negative Spin)
plt.imshow(vorticity, cmap='RdBu', origin='lower', vmin=-0.015, vmax=0.015)
plt.colorbar(label='Vorticity Magnitude')
# Draw Cylinder
circle = plt.Circle(CYL_POS, CYL_R, color='black')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.xlabel("Lattice X")
plt.ylabel("Lattice Y")

# 2. Time Series Analysis (Strouhal Number)
plt.subplot(2, 1, 2)
# Discard initial transient (first 2000 steps)
valid_history = np.array(u_history[2000:])
plt.plot(range(2000, steps), valid_history)
plt.title("Longitudinal Velocity Trace in Wake")
plt.xlabel("Time Steps")
plt.ylabel("u_x (lattice units)")
plt.grid(True, which='both', linestyle='--')

plt.tight_layout()
plt.show()

