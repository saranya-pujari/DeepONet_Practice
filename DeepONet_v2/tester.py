import DeepONet_v2_data as data
import numpy as np

X,y = data.generate_dataset(100, 'sinusoidal', True)
print(np.shape(X))
print(np.shape(y))