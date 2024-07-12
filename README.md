# Data Handling Library

Created for use as an interface to MIO formulation for searching for Counterfactuals - [LiCE](https://github.com/Epanemu/LiCE).

Compared to other common implementations, it does not store the data itself. The class stores only meta-information about the data, normalization coefficients and other features of the data.

Thus, it is useful for experimentation, to ensure reproducibility.

## Capabilities

- Able to manage Binary, Categorical, and Contiguous feature types
  - also limited support for Mixed types - Contiguous value with some categorical values (e.g. missing value indicator)
  - Contiguous includes discrete continuous values
  - Categorical includes ordinal values
- Manages normalization and denormalization
- Manages one-hot encoding and decoding
- Works with NumPy and Pandas data representations
- Contains information about
  - feature value bounds
  - mutability of features
  - monotonicity of features
  - inter-feature causal relations
- Supports encoding into MIO formulation

### Future Capabilities

- Binarization of features
- log-scale transformation of data
- Interface to other dataset-handling libraries
  - already extensible to the CARLA library
