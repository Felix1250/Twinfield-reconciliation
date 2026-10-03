import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
import random
import math

def get_hashed(key,seed,n,m):
    conv = fftconvolve(key, seed[::-1], mode='full')
    raw_sums = np.round(conv[n-1 : n-1 + m]).astype(np.int64)
    return (raw_sums % 2).astype(np.uint8)

error =  math.power(10,-20)
def after_EC(alice_key: np.ndarray, bob_key: np.ndarray, seed: np.ndarray = np.empty(0, dtype=np.uint8)):
    if len(alice_key) != len(bob_key):
        raise ValueError("Alice and Bob keys must have the same length.")
    if seed.size == 0:
        seed = np.random.randint(0, 2, size=alice_key.size + m - 1, dtype=np.uint8)

    m = math.log2(1/error)
    n = len(alice_key)
    p = n + m - 1
    seed_slice = seed[:p]

    alice_hash = get_hashed(alice_key, seed_slice, n, m)
    bob_hash = get_hashed(bob_key, seed_slice, n, m)



    print("FFT Hashed Key:    ", alice_hash)
    print("Direct Matrix Key: ", bob_hash)
    print("Match:", np.array_equal(alice_hash, bob_hash))
    if
    return alice_key, bob_key


def calc_m(length_key_after_reconciliation,e_ph):
    H = - e_ph * math.log2(e_ph) - (1 - e_ph)  * math.log2(1 - e_ph)
    factor =   - math.log2(2/error) - 2 * math.log2( 1 / ( math.sqrt(2) * error * error))
    return length_key_after_reconciliation * (1-H) + factor 

def scipy_toeplitz_hash_with_matrix(alice_key: np.ndarray, bob_key: np.ndarray, m: int, seed: np.ndarray = np.empty(0, dtype=np.uint8)):
    if len(alice_key) != len(bob_key):
        raise ValueError("Alice and Bob keys must have the same length.")
    if seed.size == 0:
        seed = np.random.randint(0, 2, size=alice_key.size + m - 1, dtype=np.uint8)
    
    n = len(alice_key)
    p = n + m - 1
    seed_slice = seed[:p]

    alice_hash = get_hashed(alice_key, seed_slice, n, m)
    bob_hash = get_hashed(bob_key, seed_slice, n, m)

    print("Match:", np.array_equal(alice_hash, bob_hash))
    
    return alice_hash, bob_hash