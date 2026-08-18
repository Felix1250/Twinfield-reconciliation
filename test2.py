import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import math
import random
import tf_utils

def run_trial(alice_bits,bob_bits,start,finish):
    length = finish-start
    prog = sf.Program(2)

    phase_A = prog.params("alice_p")
    phase_B = prog.params("bob_p")
    amplitude_A = prog.params("alice_A")
    amplitude_B = prog.params("bob_A")

    theta = np.float32(np.pi / 4)
    phi = np.float32(np.pi / 2)

    with prog.context as q:
        # State preparation

        ops.Coherent(amplitude_A, phase_A) | q[0]
        ops.Coherent(amplitude_B, phase_B) | q[1]



        # Interference
        ops.BSgate(theta,phi) | (q[0], q[1]) 

        # Detection
        ops.MeasureFock() | q

    #eng = sf.Engine("fock", backend_options={"cutoff_dim": 10}) # use fock to access single Photons
    eng = sf.Engine("tf", backend_options={
        "batch_size": length, 
        "cutoff_dim": 10 
    })
    result = eng.run(prog, args={
        "alice_p": alice_bits[2][start:finish].astype(np.float32),
        "bob_p": bob_bits[2][start:finish].astype(np.float32),
        "alice_A":alice_bits[3][start:finish].astype(np.float32),
        "bob_A":bob_bits[3][start:finish].astype(np.float32),
    })
    print(result.samples)
    measures = np.empty((length, 2))
    for i in range(length):
        measures[i][0]=int(result.samples[i][0][0])
        measures[i][1]=int(result.samples[i][0][1])
    print(measures)
    return measures

n = 1000
alphas = np.full((n),math.sqrt(0.422))
nulls = np.full((n),math.sqrt(0))
psis = np.full((n),0)

alice = [0,0,psis,alphas]
bob = [0,0,psis,nulls]
measures = run_trial(alice,bob,0,n)

tf_utils.make_heatmap(measures,"figures/heatmap.png")

