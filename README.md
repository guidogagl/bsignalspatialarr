# Brain Signal Spatial Arrangement

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/brain_signal_processing.svg)](https://pypi.org/project/brain_signal_processing/)

Brain Signal Spatial Arrangement (BSSA) is a Python package that provides utilities for preprocessing brain signal features. It includes a function `rearrange_features()` for rearranging the extracted features from brain signals.

The proposed rearrangment schema has proved to increase the performances of CNNs for different brain-signal classification tasks, such as [EEG based subject-dependent emotion recognition](https://10.1109/ACCESS.2023.3268233) or [Brain-Heart Interplay (BHI) based multiclass emotion recognition](https://10.1109/NER52421.2023.10123758).

## Features

- Rearrange extracted features from brain signals.
- Support for 19 and 32 electrodes based on the 10-20 system.
- Flexible padding options for creating spatial arrangements.

## Installation

You can install BSSA using pip:

```shell
pip install git+https://github.com/guidogagl/bsignalspatialarr
```

# Usage
Here's an example of how to use the rearrange_features() function:

```python
import numpy as np
from brain_signal_processing import rearrange_features

# Example features array with shape (num_samples, num_features, num_bands, num_electrodes)
features = np.ones((10, 1, 4, 19))

# Rearrange the features using default settings
rearranged_features = rearrange_features(features)

print(rearranged_features.shape)  # Output: (10, 9, 60)
```

You can customize the behavior of rearrange_features() using the following parameters:

- v_pad (int): Vertical padding for spatial arrangement (default: 3).
- o_pad (int): Horizontal padding for spatial arrangement (default: 3).
- mode (str): Rearrangement mode, either "multi" (default) or "mono".
- cols (int): Number of columns for single feature rearrangement (only used in "mono" mode).
- rows (int): Number of rows for single feature rearrangement (only used in "mono" mode).


## Contributing

Contributions to BSSA are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. In particular if you need to add support for different electrodes number please conctact me via email for collaboration.

# Cite Us
If you use this package in your research or work, please consider citing the following article where this approach was published:

```
@ARTICLE{10105240,
  author={Gagliardi, Guido and Alfeo, Antonio Luca and Catrambone, Vincenzo and Candia-Rivera, Diego and Cimino, Mario G. C. A. and Valenza, Gaetano},
  journal={IEEE Access},
  title={Improving Emotion Recognition Systems by Exploiting the Spatial Information of EEG Sensors},
  year={2023},
  volume={11},
  number={},
  pages={39544-39554},
  doi={10.1109/ACCESS.2023.3268233}
}
```





