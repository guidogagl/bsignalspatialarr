import numpy as np
from bsignalspatialarr import rearrange_features
import matplotlib.pyplot as plt


# a 2D array with linearly increasing values on the diagonal

# Example features array with shape (num_samples, num_features, num_bands, num_electrodes)
features = np.ones((10, 1, 4, 19))

# Rearrange the features using default settings
rearranged_features = rearrange_features(features)

plt.matshow(rearranged_features[0])

plt.show()
print(rearranged_features.shape)  # Output: (10, 7, 45)