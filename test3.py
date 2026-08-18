import strawberryfields as sf
from strawberryfields.ops import Coherent,Vacuum,Fock
from strawberryfields import ops
import numpy as np
import src.Twinfield_QKD.tf_utils as tf_utils
import math

mu1 = 0.422
alpha1 = np.sqrt(mu1)
mu2 = 10
alpha2 = np.sqrt(mu2)

loss_2 = math.exp(-100/22)

prog = sf.Program(2)

with prog.context as q:
    Coherent(alpha1,0   ) | q[0]
    Coherent(alpha1,0) | q[1]
    #ops.LossChannel(loss_2) | q[1]
    ops.BSgate(math.pi/4, math.pi/2) | (q[0], q[1])


prog2 = sf.Program(2)
amplitude_A = prog.params("alice_p")
amplitude_B = prog.params("bob_p")

with prog2.context as p:
    #Coherent(alpha1, 0) | p[0]
    Coherent(amplitude_A,0) | p[0]
    #Vacuum() | p[1]
    Coherent(amplitude_B,0) | p[1]

    #ops.LossChannel(loss_2) | p[0]
    ops.BSgate(math.pi/4, math.pi/2) | (p[0], p[1])


eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
eng2 = sf.Engine("tf", backend_options={"cutoff_dim": 5})


print(math.sin(math.pi/2))


alphas = np.full((10),0.422)

result = eng.run(prog)
result2 = eng.run(prog, args={
        "alice_p": alphas[0:9].astype(np.float32),
        "bob_p": alphas[0:9].astype(np.float32),
    })
state = result.state
state2 = result2.state
probs = state.all_fock_probs()
probs2 = state2.all_fock_probs()

#print(probs)
tf_utils.make_heatmap_array(np.array(probs), "figures/heatmap3_phase.png",4)
tf_utils.make_heatmap_array(np.array(probs2), "figures/heatmap4_nophase.png",4)
