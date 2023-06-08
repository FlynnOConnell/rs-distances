<h1> Rust Functions for Metric Space Analysis </h1>

This repository contains a set of functions implemented in Rust for the purpose of metric space analysis. These functions aim to provide superior performance by taking advantage of Rust's efficient memory management and computational capabilities.

<br>

[![CI](https://github.com/NeuroPyPy/rs-distances/actions/workflows/CI.yml/badge.svg)](https://github.com/NeuroPyPy/rs-distances/actions/workflows/CI.yml)
[![version](https://img.shields.io/pypi/v/rs-distances)](https://img.shields.io/pypi/v/rs-distances?style=plastic)
[![implementation](https://img.shields.io/pypi/implementation/rs-distances)](https://img.shields.io/pypi/implementation/rs-distances?style=plastic)
[![license](https://img.shields.io/pypi/l/rs-distances)](https://img.shields.io/pypi/l/rs-distances?style=plastic)
[![format](https://img.shields.io/pypi/format/rs-distances)](https://img.shields.io/pypi/format/rs-distances?style=plastic)

<br>

## Installation
This package is bundled with the [metricspace](https://github.com/NeuroPyPy/metricspace) repository.
It can also be installed separately using the following command:
```bash
pip install rs-distances
```
**Note**: Be sure to activate your virtual environment with Python 3.7 or higher before installing this package via pip.

## Performance
The functions provided in this package are written in Rust and compiled into a shared library that can be utilized within Python. This approach is intended to boost the computational efficiency of metric space analysis operations.

Below is a comparative performance table of the spike-train iterator function implemented in Matlab, Python, and Rust. It should be noted that the Matlab version is not optimized using MEX (which would be comparable to Python's numba @jit), and the translations from Matlab to Python to Rust are not exact 1:1.

| Spike-train iterator   | Matlab  | Python  | Rust   |
| ---------------------- | ------- | ------- | ------ |
| `raw function`         | 30.235s | 64.992s | 2.028s |
| `with numba @jit`      | 30.235s | 25.119s | 2.028s |
| `with @jit + parralel` | 24.050s | 18.067s | 0.945s |

## Advantages of Rust Implementation 
Array manipulations, particularly those performed within computationally intensive tasks, are highly sensitive to memory allocation and cleanup. Rust, with its ownership model and automatic memory management, excels in this area. Rust automatically reclaims the memory when an object (like an array or a slice) goes out of scope. This is a stark contrast to languages like Python, where a garbage collector is relied upon to perform memory cleanup. This difference provides Rust implementations with a distinct edge in performance, which is reflected in the comparative analysis shown above.

With these Rust implementations, you can achieve the high-level expressiveness of Python while benefiting from the superior performance and efficiency of Rust.