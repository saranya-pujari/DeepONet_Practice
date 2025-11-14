import numpy as np
from collections import defaultdict

import DeepONet_data as data
import DeepONet_model as model

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)


# Create training dataset
N_train = 2000
length_scale_train = 0.4
X_train, y_train = data.generate_dataset(N_train, length_scale_train)

# Create validation dataset
N_val = 100
length_scale_test = 0.4
X_val, y_val = data.generate_dataset(N_val, length_scale_test)

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
PI_DeepONet= model.create_model(mean, var)
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
        ODE_loss, IC_loss, total_loss, gradients = model.train_step(X, X_init, 
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
    val_res = model.ODE_residual_calculator(X_val_tensor[:, :1], X_val_tensor[:, 1:-1], X_val_tensor[:, -1:], PI_DeepONet)
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

PI_DeepONet.save_weights("PI_DeepONet.weights.h5")