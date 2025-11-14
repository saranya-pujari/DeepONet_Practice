from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

def create_model(mean, var, verbose=False):
    """Definition of a DeepONet with fully connected branch and trunk layers.

    Args:
    ----
    mean: dictionary, mean values of the inputs
    var: dictionary, variance values of the inputs
    verbose: boolean, indicate whether to show the model summary

    Outputs:
    --------
    model: the DeepONet model
    """
    # Branch net
    branch_input = tf.keras.Input(shape=(len(mean['forcing']),), name="forcing")
    branch = tf.keras.layers.Normalization(mean=mean['forcing'], variance=var['forcing'])(branch_input)
    for _ in range(3):
        branch = tf.keras.layers.Dense(50, activation="tanh")(branch)

    # Trunk net
    trunk_input = tf.keras.Input(shape=(len(mean['time']),), name="time")
    trunk = tf.keras.layers.Normalization(mean=mean['time'], variance=var['time'])(trunk_input)
    for _ in range(3):
        trunk = tf.keras.layers.Dense(50, activation="tanh")(trunk)

    # Compute the dot product between branch and trunk net
    dot_product = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True))([branch, trunk])

    # Add the bias
    output = BiasLayer()(dot_product)

    # Create the model
    model = tf.keras.models.Model(inputs=[branch_input, trunk_input], outputs=output)

    if verbose:
        model.summary()

    return model

class BiasLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1,),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)
    def call(self, inputs):
        return inputs + self.bias
    
def ODE_residual_calculator(t, u, u_t, model):
    """ODE residual calculation.

    Args:
    ----
    t: temporal coordinate
    u: input function evaluated at discrete temporal coordinates
    u_t: input function evaluated at t
    model: DeepONet model

    Outputs:
    --------
    ODE_residual: residual of the governing ODE
    """

    with tf.GradientTape() as tape:
        tape.watch(t)
        s = model({"forcing": u, "time": t})

    ds_dt = tape.gradient(s, t)
    ODE_residual = ds_dt - u_t
    return ODE_residual

def train_step(X, X_init, IC_weight, ODE_weight, model):
    """Calculate gradients of the total loss with respect to network model parameters.

    Args:
    ----
    X: training dataset for evaluating ODE residuals
    X_init: training dataset for evaluating initial conditions
    IC_weight: weight for initial condition loss
    ODE_weight: weight for ODE loss
    model: DeepONet model

    Outputs:
    --------
    ODE_loss: calculated ODE loss
    IC_loss: calculated initial condition loss
    total_loss: weighted sum of ODE loss and initial condition loss
    gradients: gradients of the total loss with respect to network model parameters.
    """
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights[-1].value)

        # Initial condition prediction
        y_pred_IC = model({"forcing": X_init[:, 1:-1], "time": X_init[:, :1]})

        # Equation residual
        ODE_residual = ODE_residual_calculator(t=X[:, :1], u=X[:, 1:-1], u_t=X[:, -1:], model=model)

        # Calculate loss
        MSE = keras.losses.MeanSquaredError()
        IC_loss = tf.reduce_mean(MSE(0, y_pred_IC))
        ODE_loss = tf.reduce_mean(tf.square(ODE_residual))

        # Total loss
        total_loss = IC_loss*IC_weight + ODE_loss*ODE_weight

    gradients = tape.gradient(total_loss, model.trainable_variables)

    return ODE_loss, IC_loss, total_loss, gradients

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

# Create training dataset
N_train = 2000
length_scale_train = 0.4
X_train, y_train = generate_dataset(N_train, length_scale_train)

# Create validation dataset
N_val = 100
length_scale_test = 0.4
X_val, y_val = generate_dataset(N_val, length_scale_test)

# Create testing dataset
N_test = 100
length_scale_test = 0.4
X_test, y_test = generate_dataset(N_test, length_scale_test, ODE_solve=True)

# Determine batch size
ini_batch_size = int(2000/100)
col_batch_size = 2000

# Create dataset object (initial conditions)
X_train_ini = tf.convert_to_tensor(X_train[X_train[:, 0]==0], dtype=tf.float32)
ini_ds = tf.data.Dataset.from_tensor_slices((X_train_ini))
ini_ds = ini_ds.shuffle(5000).batch(ini_batch_size)

# Create dataset object (collocation points)
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
train_ds = tf.data.Dataset.from_tensor_slices((X_train))
train_ds = train_ds.shuffle(100000).batch(col_batch_size)

# Scaling 
mean = {
    'forcing': tf.convert_to_tensor(np.mean(X_train[:, 1:-1], axis=0), dtype=tf.float32),
    'time': tf.convert_to_tensor(np.mean(X_train[:, :1], axis=0), dtype=tf.float32)
}

var = {
    'forcing': tf.convert_to_tensor(np.var(X_train[:, 1:-1], axis=0), dtype=tf.float32),
    'time': tf.convert_to_tensor(np.var(X_train[:, :1], axis=0), dtype=tf.float32)
}

class LossTracking:

    def __init__(self):
        self.mean_total_loss = keras.metrics.Mean()
        self.mean_IC_loss = keras.metrics.Mean()
        self.mean_ODE_loss = keras.metrics.Mean()
        self.loss_history = defaultdict(list)

    def update(self, total_loss, IC_loss, ODE_loss):
        self.mean_total_loss(total_loss)
        self.mean_IC_loss(IC_loss)
        self.mean_ODE_loss(ODE_loss)

    def reset(self):
        self.mean_total_loss.reset_state()
        self.mean_IC_loss.reset_state()
        self.mean_ODE_loss.reset_state()

    def print(self):
        print(f"IC={self.mean_IC_loss.result().numpy():.4e}, ODE={self.mean_ODE_loss.result().numpy():.4e}, total_loss={self.mean_total_loss.result().numpy():.4e}")

    def history(self):
        self.loss_history['total_loss'].append(self.mean_total_loss.result().numpy())
        self.loss_history['IC_loss'].append(self.mean_IC_loss.result().numpy())
        self.loss_history['ODE_loss'].append(self.mean_ODE_loss.result().numpy())

# Set up training configurations
n_epochs = 300
IC_weight= tf.constant(1.0, dtype=tf.float32)   
ODE_weight= tf.constant(1.0, dtype=tf.float32)
loss_tracker = LossTracking()
val_loss_hist = []

# Set up optimizer
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Instantiate the PINN model
PI_DeepONet= create_model(mean, var)
PI_DeepONet.compile(optimizer=optimizer)

# Configure callbacks
_callbacks = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=30),
             tf.keras.callbacks.ModelCheckpoint('NN_model.keras', monitor='val_loss', save_best_only=True)]
callbacks = tf.keras.callbacks.CallbackList(
                _callbacks, add_history=False, model=PI_DeepONet)

# Start training process
for epoch in range(1, n_epochs + 1):  
    print(f"Epoch {epoch}:")

    for X_init, X in zip(ini_ds, train_ds):

        # Calculate gradients
        ODE_loss, IC_loss, total_loss, gradients = train_step(X, X_init, 
                                                            IC_weight, ODE_weight,
                                                            PI_DeepONet)
        # Gradient descent
        PI_DeepONet.optimizer.apply_gradients(zip(gradients, PI_DeepONet.trainable_variables))

        # Loss tracking
        loss_tracker.update(total_loss, IC_loss, ODE_loss)

    # Loss summary
    loss_tracker.history()
    loss_tracker.print()
    loss_tracker.reset()

    ####### Validation
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    val_res = ODE_residual_calculator(X_val_tensor[:, :1], X_val_tensor[:, 1:-1], X_val_tensor[:, -1:], PI_DeepONet)
    val_ODE = tf.cast(tf.reduce_mean(tf.square(val_res)), tf.float32)

    X_val_ini = X_val[X_val[:, 0]==0]
    pred_ini_valid = PI_DeepONet.predict({"forcing": X_val_ini[:, 1:-1], "time": X_val_ini[:, :1]}, batch_size=12800)
    MSE = tf.keras.losses.MeanSquaredError()
    val_IC = tf.reduce_mean(MSE(0, pred_ini_valid))
    print(f"val_IC: {val_IC.numpy():.4e}, val_ODE: {val_ODE.numpy():.4e}, lr: {PI_DeepONet.optimizer.learning_rate.numpy():.2e}")

    # Callback at the end of epoch
    callbacks.on_epoch_end(epoch, logs={'val_loss': val_IC+val_ODE})
    val_loss_hist.append(val_IC+val_ODE)

    # Re-shuffle dataset
    ini_ds = tf.data.Dataset.from_tensor_slices((X_train_ini))
    ini_ds = ini_ds.shuffle(5000).batch(ini_batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train))
    train_ds = train_ds.shuffle(100000).batch(col_batch_size)

# History
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(range(n_epochs), loss_tracker.loss_history['IC_loss'])
ax[1].plot(range(n_epochs), loss_tracker.loss_history['ODE_loss'])
ax[2].plot(range(n_epochs), val_loss_hist)
ax[0].set_title('IC Loss')
ax[1].set_title('ODE Loss')
ax[2].set_title('Val Loss')
for axs in ax:
    axs.set_yscale('log')
plt.show()
