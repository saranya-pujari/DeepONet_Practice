import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)

def create_model(mean, var, verbose=False):
    # Branch Net
    branch_input = tf.keras.Input(shape=(len(mean['forcing']),), name="forcing")
    branch = tf.keras.layers.Normalization(mean=mean['forcing'], variance=var['forcing'])(branch_input)
    for _ in range(3):
        branch = tf.keras.layers.Dense(50, activation="tanh")(branch)

    # Trunk Net
    trunk_input = tf.keras.Input(shape=(len(mean['coords']),), name="coords")
    trunk = tf.keras.layers.Normalization(mean=mean['coords'], variance=var['coords'])(trunk_input)
    for _ in range(3):
        trunk = tf.keras.layers.Dense(50, activation="tanh")(trunk)

    # Dot Product
    dot_product = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True)
    )([branch, trunk])

    # Output + Bias
    output = BiasLayer()(dot_product)

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
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def PDE_residual_calculator(coords, forcing, u_func, model):
    coords = tf.convert_to_tensor(coords, dtype=tf.float32)
    coords = tf.Variable(coords)

    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(coords)
            tape2.watch(coords)
            s_pred = model({"forcing": forcing, "coords": coords})

        ds_dcoords = tape1.gradient(s_pred, coords) #[∂s/∂x, ∂s/∂t]
        ds_dx = ds_dcoords[:, 0:1]
        ds_dt = ds_dcoords[:, 1:2]

    d2s_dx2 = tape2.gradient(ds_dx, coords)[:, 0:1]

    residual = ds_dt - 0.01 * d2s_dx2 - 0.01 * s_pred - u_func
    return residual


def train_step(X_col, X_init, IC_weight, PDE_weight, model):
    with tf.GradientTape() as tape:
        # PDE residual loss
        coords_col = X_col[:, :2]  # [x, t]
        forcing_col = X_col[:, 2:-1]
        u_val_col = X_col[:, -1:]

        residual = PDE_residual_calculator(coords_col, forcing_col, u_val_col, model)
        PDE_loss = tf.reduce_mean(tf.square(residual))

        # Initial condition loss (s(x, 0) ≈ 0)
        coords_ini = X_init[:, :2]
        forcing_ini = X_init[:, 2:-1]
        s_pred_ini = model({"forcing": forcing_ini, "coords": coords_ini})
        IC_loss = tf.reduce_mean(tf.square(s_pred_ini))

        # Total loss
        total_loss = PDE_weight * PDE_loss + IC_weight * IC_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    return PDE_loss, IC_loss, total_loss, gradients
