import numpy as np
import random
from tqdm import tqdm
import sys
from os import path
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("/home/felix/QKD_felix/src/Twinfield_QKD")

import tf_utils
import twinfield_communication_tensorflow

def aopp_pz_mu(savepath,savepath2,generate = False):
    twin =twinfield_communication_tensorflow.Twinfield()
    
    if generate:
        length = 10000
        for i in range(10):
            for j in range(10):
                twin.p_z = 1 - i* 0.1
                twin.mu3 = j*0.1
                loadpath = "saves/pz_mu2/pz" + str(i) + "_mu" + str(j) +  ".npz"
                twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    signal_lengths = np.zeros((10,10))
    error_rate = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            loadpath = "saves/pz_mu2/pz" + str(i) + "_mu" + str(j) +  ".npz"
            twin.tf_communicate_load_settings(loadpath)
            
            twin.tf_communicat_load(loadpath)
            signal_lengths[i][j] = twin._signal_length * twin._error_rate
            error_rate[i][j] = twin._error_rate


    sns.heatmap(error_rate, annot=True, cmap="coolwarm")

    plt.title("error rate")
    plt.xlabel("mu")
    plt.ylabel("pz")
    plt.savefig(savepath, dpi=300)

    sns.heatmap(signal_lengths, annot=False, cmap="coolwarm")
    
    plt.title("signal lengths")
    plt.xlabel("mu")
    plt.ylabel("pz")
    plt.savefig(savepath2, dpi=300)


aopp_pz_mu("figures/my_graph/pzmu_errors2.png","figures/my_graph/pzmu_length2.png",generate=True)


 