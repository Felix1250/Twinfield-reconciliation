# Empirical test of a loss channel in Strawberry Fields
#
# pip install strawberryfields numpy matplotlib

import strawberryfields as sf
from strawberryfields.ops import Coherent, LossChannel, MeasureFock, Fock
import numpy as np
import matplotlib.pyplot as plt


alpha = 2.0
T = 0.65
shots = 5000
cutoff_dim = 10

eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})

samples = []

for _ in range(shots):

    prog = sf.Program(1)

    with prog.context as q:
        Fock(1) | q[0]
        LossChannel(T) | q[0]
        MeasureFock() | q[0]

    result = eng.run(prog)

    n = result.samples[0, 0]

    samples.append(n)

    eng.reset()

samples = np.array(samples)

empirical_mean = np.mean(samples)
empirical_var = np.var(samples)

theoretical_mean = T * abs(alpha) ** 2

print("=== Loss Channel Test ===")
print(f"alpha = {alpha}")
print(f"T     = {T}")
print()
print(f"Theoretical mean : {theoretical_mean:.4f}")
print(f"Empirical mean   : {empirical_mean:.4f}")
print(f"Empirical var    : {empirical_var:.4f}")

# --------------------------------
# Histogram
# --------------------------------
bins = np.arange(samples.max() + 2) - 0.5

plt.figure(figsize=(8, 5))
plt.hist(samples, bins=bins, density=True,
         color="royalblue", alpha=0.75)

plt.xlabel("Photon number")
plt.ylabel("Probability")
plt.title("Loss Channel Photon Statistics")

plt.grid(alpha=0.3)
plt.show()