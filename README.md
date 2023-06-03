<h1> Rust Functions for Metric Space Analysis </h1>

This repository contains a set of functions for metric space analysis.

## Installation
This package is bundled with the [metricspace](https://github.com/NeuroPyPy/metricspace) repository.
It can also be installed separately using the following command:
```bash
pip install rs-distances
```
Using this method, make sure to activate a virtual environment compatable with python 3.7 or higher.

## Performance
The functions in this package are written in Rust and compiled to a shared library. The purpose is to 
make computationally intensive functions available to python. The following table shows the performance
gain from matlab, python and rust implementations. Note, the Matlab implementation is not optimized using mex, which 
would be the numba @jit equivalent, and translations from Matlab -> Python -> -> Rust are not 1:1. 

| Spike-train iterator   | Matlab  | Python  | Rust   |
|------------------------|---------|---------|--------|
| `raw function`         | 30.235s | 64.992s | 2.028s |
| `with numba @jit`      | 30.235s | 25.119s | 2.028s |
| `with @jit + parralel` | 24.050s | 18.067s | 0.945s |

