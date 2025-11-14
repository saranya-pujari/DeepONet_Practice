
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from scipy.integrate import solve_ivp
import numpy as np


def create_samples(length_scale, sample_num):
    """Create synthetic data for u(·)

    Args:
    ----
    length_scale: float, length scale for RNF kernel
    sample_num: number of u(·) profiles to generate

    Outputs:
    --------
    u_sample: generated u(·) profiles
    """

    # Define kernel with given length scale
    kernel = RBF(length_scale)

    # Create Gaussian process regressor
    gp = GaussianProcessRegressor(kernel=kernel)

    # Collocation point locations
    X_sample = np.linspace(0, 1, 100).reshape(-1, 1) 

    # Create samples
    u_sample = np.zeros((sample_num, 100))
    for i in range(sample_num):
        # sampling from the prior directly
        n = np.random.randint(0, 10000)
        u_sample[i, :] = gp.sample_y(X_sample, random_state=n).flatten()  

    return u_sample

# Linear Function
def create_linear_samples(sample_num):
  # Collocation point locations
  X_sample = np.linspace(0, 1, 100).reshape(-1, 1) 

  # Create samples
  u_sample = np.zeros((sample_num, 100))
  for i in range(sample_num):
      a = np.random.uniform(-1, 1)
      b = np.random.uniform(-1, 1)
      u_sample[i, :] = a*X_sample.flatten() + b

  return u_sample

# Sinusoidal Function
def create_sinusoidal_samples(sample_num):
  # Collocation point locations
  X_sample = np.linspace(0, 1, 100).reshape(-1, 1) 

  # Create samples
  u_sample = np.zeros((sample_num, 100))
  for i in range(sample_num):
      a = np.random.uniform(-1, 1)
      b = np.random.uniform(-1, 1)
      u_sample[i, :] = a*np.sin(2*np.pi*b*X_sample.flatten())

  return u_sample

# Random Walk
def create_random_walk_samples(sample_num):
  # Collocation point locations
  X_sample = np.linspace(0, 1, 100).reshape(-1, 1) 

  # Create samples
  u_sample = np.zeros((sample_num, 100))
  for i in range(sample_num):
    steps = np.random.randn(99)  # 99 steps to get 100 points
    u = np.concatenate([[0], np.cumsum(steps)])  # start at 0
    u_sample[i, :] = u
  return u_sample

# Modified generate_dataset function
def generate_dataset_new(N, type, ODE_solve=False, length_scale=0.4):
  # Create random fields
    
  random_field = []
  if type == 'linear':
      random_field = create_linear_samples(N)
  elif type == 'sinusoidal':
      random_field = create_sinusoidal_samples(N)
  elif type == 'random_walk':
      random_field = create_random_walk_samples(N)
  else:
      create_samples(N, length_scale)
# Compile dataset
  X = np.zeros((N*100, 100+2))
  y = np.zeros((N*100, 1))

  for i in tqdm(range(N)):
      u = np.tile(random_field[i, :], (100, 1))
      t = np.linspace(0, 1, 100).reshape(-1, 1)

      # u(·) evaluated at t
      u_t = np.diag(u).reshape(-1, 1)

      # Update overall matrix
      X[i*100:(i+1)*100, :] = np.concatenate((t, u, u_t), axis=1)

      # Solve ODE
      if ODE_solve:
          sol = solve_ivp(lambda var_t, var_s: np.interp(var_t, t.flatten(), random_field[i, :]), 
                          t_span=[0, 1], y0=[0], t_eval=t.flatten(), method='RK45')
          y[i*100:(i+1)*100, :] = sol.y[0].reshape(-1, 1)

  return X, y
    


def generate_dataset(N, length_scale, ODE_solve=False):
    """Generate dataset for Physics-informed DeepONet training.

    Args:
    ----
    N: int, number of u(·) profiles
    length_scale: float, length scale for RNF kernel
    ODE_solve: boolean, indicate whether to compute the corresponding s(·)

    Outputs:
    --------
    X: the dataset for t, u(·) profiles, and u(t)
    y: the dataset for the corresponding ODE solution s(·)
    """

    # Create random fields
    random_field = create_samples(length_scale, N)

    # Compile dataset
    X = np.zeros((N*100, 100+2))
    y = np.zeros((N*100, 1))

    for i in tqdm(range(N)):
        u = np.tile(random_field[i, :], (100, 1))
        t = np.linspace(0, 1, 100).reshape(-1, 1)

        # u(·) evaluated at t
        u_t = np.diag(u).reshape(-1, 1)

        # Update overall matrix
        X[i*100:(i+1)*100, :] = np.concatenate((t, u, u_t), axis=1)

        # Solve ODE
        if ODE_solve:
            sol = solve_ivp(lambda var_t, var_s: np.interp(var_t, t.flatten(), random_field[i, :]), 
                            t_span=[0, 1], y0=[0], t_eval=t.flatten(), method='RK45')
            y[i*100:(i+1)*100, :] = sol.y[0].reshape(-1, 1)

    return X, y