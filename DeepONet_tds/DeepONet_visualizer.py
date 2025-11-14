# DeepONet_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import DeepONet_data as data
import DeepONet_model as model

# Load trained model
N_train = 2000
length_scale_train = 0.4
X_train, y_train = data.generate_dataset(N_train, length_scale_train)
mean = {
    'forcing': tf.convert_to_tensor(np.mean(X_train[:, 1:-1], axis=0), dtype=tf.float32),
    'time': tf.convert_to_tensor(np.mean(X_train[:, :1], axis=0), dtype=tf.float32)
}

var = {
    'forcing': tf.convert_to_tensor(np.var(X_train[:, 1:-1], axis=0), dtype=tf.float32),
    'time': tf.convert_to_tensor(np.var(X_train[:, :1], axis=0), dtype=tf.float32)
}

PI_DeepONet = model.create_model(mean, var)
PI_DeepONet.load_weights("/Users/saranyapujari/Documents/GitHub/DeepONet_Practice/DeepONet/PI_DeepONet.weights.h5")

# Random Samples
def plot_samples(X_test, y_test, N_test, title):
    profiles = np.random.randint(0, N_test, 3)
    t = X_test[0:N_test, 0:1]
    fig, ax = plt.subplots(2, 3, figsize=(12, 4))
    ax[0,0].plot(t, X_test[profiles[0] * N_test:(profiles[0] + 1) * N_test, 101])
    ax[0,0].set_xlabel("t")
    ax[0,0].set_ylabel("u(t)")
    ax[1,0].plot(t, y_test[profiles[0] * N_test:(profiles[0] + 1) * N_test, 0], label="Ground Truth")
    ax[1,0].set_xlabel("t")
    ax[1,0].set_ylabel("s(t)")
    s_pred = PI_DeepONet.predict({"forcing": X_test[profiles[0] * N_test:(profiles[0] + 1) * N_test, 1:101], "time": t}, verbose=0)
    s_pred = s_pred.flatten()
    ax[1,0].plot(t, s_pred, label="Predicted")
    ax[1,0].legend()

    ax[0,1].plot(t, X_test[profiles[1] * N_test:(profiles[1] + 1) * N_test, 101])
    ax[0,1].set_xlabel("t")
    ax[0,1].set_ylabel("u(t)")
    ax[1,1].plot(t, y_test[profiles[1] * N_test:(profiles[1] + 1) * N_test, 0], label="Ground Truth")
    ax[1,1].set_xlabel("t")
    ax[1,1].set_ylabel("s(t)")
    s_pred = PI_DeepONet.predict({"forcing": X_test[profiles[1] * N_test:(profiles[1] + 1) * N_test, 1:101], "time": t}, verbose=0)
    s_pred = s_pred.flatten()
    ax[1,1].plot(t, s_pred, label="Predicted")
    ax[1,1].legend()

    ax[0,2].plot(t, X_test[profiles[2] * N_test:(profiles[2] + 1) * N_test, 101])
    ax[0,2].set_xlabel("t")
    ax[0,2].set_ylabel("u(t)")
    ax[1,2].plot(t, y_test[profiles[2] * N_test:(profiles[2] + 1) * N_test, 0], label="Ground Truth")
    ax[1,2].set_xlabel("t")
    ax[1,2].set_ylabel("s(t)")
    s_pred = PI_DeepONet.predict({"forcing": X_test[profiles[2] * N_test:(profiles[2] + 1) * N_test, 1:101], "time": t}, verbose=0)
    s_pred = s_pred.flatten()
    ax[1,2].plot(t, s_pred, label="Predicted")
    ax[1,2].legend()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()


# Plot 3 random ODEs
N_test = 100
length_scale_test = 0.4
X_test, y_test = data.generate_dataset(N_test, length_scale_test, ODE_solve=True)
plot_samples(X_test, y_test, N_test, "Gaussian")

# Length Scale 0.2
X_test_oor, y_test_oor = data.generate_dataset(N_test, 0.2, ODE_solve=True)
plot_samples(X_test_oor, y_test_oor, N_test, "Length Scale 0.2")

# Length Scale 0.6
X_test_oor, y_test_oor = data.generate_dataset(N_test, 0.6, ODE_solve=True)
plot_samples(X_test_oor, y_test_oor, N_test, "Length Scale 0.6")

# Testing different functions
X_test_lin, y_test_lin = data.generate_dataset_new(N_test, 'linear', ODE_solve=True)
plot_samples(X_test_lin, y_test_lin, N_test, "Linear Functions")

X_test_sin, y_test_sin = data.generate_dataset_new(N_test, 'sinusoidal', ODE_solve=True)
plot_samples(X_test_sin, y_test_sin, N_test, "Sinusoidal Functions")

X_test_rw, y_test_rw = data.generate_dataset_new(N_test, 'random_walk', ODE_solve=True)
plot_samples(X_test_rw, y_test_rw, N_test, "Random Walk")