from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from scipy.integrate import solve_ivp
import numpy as np
from pde import CartesianGrid, ScalarField, PDE



def create_gaussian_samples(sample_num, length_scale):
  # Set up gaussian process
  kernel = RBF(length_scale)
  gp = GaussianProcessRegressor(kernel=kernel)

  # Create coordinate space
  x = np.linspace(0, 1, 100, dtype=np.float32)
  t = np.linspace(0, 1, 100, dtype=np.float32)
  X, T = np.meshgrid(x, t, indexing='ij')
  coords = np.stack([X, T], axis=-1).reshape(-1, 2).astype(np.float32)
  u_sample = np.zeros((sample_num, 100 * 100), dtype=np.float32)

  # Generate samples
  for i in tqdm(range(sample_num)):
    n = np.random.randint(0, 10000)
    u = gp.sample_y(coords, random_state=n).astype(np.float32)
    u_sample[i, :] = u.flatten()

  return u_sample

# Linear Function
def create_linear_samples(sample_num):
  x = np.linspace(0, 1, 100)
  t = np.linspace(0, 1, 100)
  X, T = np.meshgrid(x, t, indexing='ij')

  u_sample = np.zeros((sample_num, 100 * 100))

  for i in tqdm(range(sample_num)):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    u = a * X + b * T
    u_sample[i, :] = u.flatten()

  return u_sample

# Sinusoidal Function
def create_sinusoidal_samples(sample_num):
  x = np.linspace(0, 1, 100)
  t = np.linspace(0, 1, 100)
  X, T = np.meshgrid(x, t, indexing='ij')

  u_sample = np.zeros((sample_num, 100 * 100))

  for i in tqdm(range(sample_num)):
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(-1, 1)
    u = np.sin(2*np.pi*a*X) + np.sin(2*np.pi*b*T)
    u_sample[i, :] = u.flatten()

  return u_sample

# Random Walk
def create_random_walk_samples(sample_num):
  x = np.linspace(0, 1, 100)
  t = np.linspace(0, 1, 100)
  X, T = np.meshgrid(x, t, indexing='ij')

  u_sample = np.zeros((sample_num, 100 * 100))

  for i in tqdm(range(sample_num)):
    walk_x = np.cumsum(np.random.randn(100), axis=0)
    walk_t = np.cumsum(np.random.randn(100), axis=0)
    u = np.outer(walk_x, walk_t)
  u_sample[i,:] = u.flatten()

  return u_sample

# Generate the dataset
def generate_dataset(N, type = 'gaussian', PDE_solve=False, length_scale=0.4):
    # Create random fields
    if type == 'gaussian':
      random_field = create_gaussian_samples(N, length_scale)
    elif type == 'linear':
      random_field = create_linear_samples(N)
    elif type == 'sinusoidal':
      random_field = create_sinusoidal_samples(N)
    elif type == 'random_walk':
      random_field = create_random_walk_samples(N)
    else:
      raise ValueError("Unsupported sample_type.")

    # Compile dataset
    x = np.linspace(0, 1, 100, dtype=np.float32).reshape(-1, 1)
    t = np.linspace(0, 1, 100, dtype=np.float32).reshape(-1, 1)
    X = np.zeros((N * 100, 2 + 100 + 1)) # N * 100 rows, X_grid + T_grid + u(x,t) + u(x=t) columns
    y = np.zeros((N * 100, 1))


    for i in tqdm(range(N)):
      u = random_field[i, :].reshape(100, 100)
      u_t = np.diag(u).reshape(-1, 1)

      X[i * 100:(i + 1) * 100, :] = np.concatenate((x, t, u, u_t), axis=1)

      # Solve PDE instead of ODE
      if PDE_solve:
          # Setup PDE grid and RHS
          grid = CartesianGrid([[0, 1]], [100])
          dt = 1.0 / 99
          u_xt = u  # shape (100, 100)

          def pde_rhs(state, t_val):
              idx = int(np.clip(t_val / dt, 0, 99))
              u_slice = u_xt[:, idx]
              return {"s": state.laplace(bc="natural") + u_slice}

          eq = PDE({"s": pde_rhs}, bc={"s": "natural"})
          s0 = ScalarField(grid, data=0.0)
          sol = eq.solve(s0, t_range=(0, 1), dt=dt, tracker=None)

          sol_data = np.stack([sol.get_field(t=ti).data for ti in np.linspace(0, 1, 100)], axis=1)
          y[i * 100:(i + 1) * 100, :] = sol_data.T.reshape(-1, 1)

    return X, y