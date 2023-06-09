from __future__ import annotations
import numpy as np


def calculate_spkd(cspks: np.ndarray | list, qvals: list | np.ndarray) -> np.ndarray: 
    """
        Internal function to compute pairwise spike train distances with variable time precision for multiple cost values.

        Rust implementation. 

        Args:
            cspks (nested iterable[list | np.ndarray]): Each inner list contains spike times for a single spike train.
            qvals (list of float | int): List of time precision values to use in the computation.

        Returns:
            ndarray: A 3D array containing pairwise spike train distances for each time precision value.

        Raises:
                TypeError: If cspks is not a list or numpy array.
    """
    cspks: np.ndarray | list
    qvals: np.ndarray | list 
    ...
