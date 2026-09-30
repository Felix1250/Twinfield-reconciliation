import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
import random

def scipy_toeplitz_hash_with_matrix(raw_key: np.ndarray, m: int, seed: np.ndarray):
    n = len(raw_key)
    p = n + m - 1
    seed_slice = seed[:p]
    
    # -------------------------------------------------------------
    # Reconstruct the explicit m x n Toeplitz matrix
    # First column: seed_slice[:m][::-1] (seed[m-1] down to seed[0])
    # First row:    seed_slice[m-1 : m-1+n] (seed[m-1] up to seed[m-1+n-1])
    # -------------------------------------------------------------
    col = seed_slice[:m][::-1]
    row = seed_slice[m-1 : m-1+n]
    T_matrix = toeplitz(col, row)

    # Print matrix details
    print(f"Matrix Shape: {T_matrix.shape} (m={m}, n={n})")
    print("Toeplitz Matrix T:\n", T_matrix)
    print("-" * 40)
    
    # Execute the FFT-based hash using the reversed seed slice
    # and extracting from index n-1 to n-1+m
    conv = fftconvolve(raw_key, seed_slice[::-1], mode='full')
    raw_sums = np.round(conv[n-1 : n-1 + m]).astype(np.int64)
    hashed_key = (raw_sums % 2).astype(np.uint8)
    
    # Verification: compute matrix multiplication directly to prove equivalence
    direct_key = (np.dot(T_matrix, raw_key) % 2).astype(np.uint8)
    print("FFT Hashed Key:    ", hashed_key)
    print("Direct Matrix Key: ", direct_key)
    print("Match:", np.array_equal(hashed_key, direct_key))
    
    return np.array_equal(hashed_key, direct_key)

# --- Example Run ---
while True:
    length_1 = random.randint(3000, 4000)  # Random length for raw_key
    raw_key = np.random.randint(0, 2, size=length_1, dtype=np.uint8)     # n = 8
    m = random.randint(1, length_1-1)                                                     # m = 5
    seed = np.random.randint(0, 2, size=m+length_1, dtype=np.uint8) # p = n + m - 1 = 12

    if not scipy_toeplitz_hash_with_matrix(raw_key, m, seed):
        break
