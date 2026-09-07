import numpy as np
import random
from tqdm import tqdm
import sys
from os import path
import matplotlib.pyplot as plt

sys.path.append("/home/felix/QKD_felix/src/Twinfield_QKD")
sys.path.append("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python")

import tf_utils
import twinfield_communication_tensorflow
import aopp
import ldpc


def pz_mu(savepath,savepath2,generate = False):
    
    
    if generate:
        length = 100000
        for i in range(10):
            twin = twinfield_communication_tensorflow.Twinfield()
            twin.p_x = 0
            twin.p_z = 1 - i* 0.1
            for j in range(10):
                twin.mu3 = j*0.1
                loadpath = "saves/pz_mu/pz" + str(i) + "_mu" + str(j) +  ".npz"
                twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    signal_lengths = np.zeros((10,9))
    error_rate = np.zeros((10,9))
    twin =twinfield_communication_tensorflow.Twinfield()
    for i in range(10):
        for j in range(1,10):
            loadpath = "saves/pz_mu/pz" + str(i) + "_mu" + str(j) +  ".npz"
            twin.tf_communicate_load_settings(loadpath)
            
            twin.tf_communicat_load(loadpath)
            signal_lengths[i][j-1] = twin._signal_length 
            error_rate[i][j-1] = twin._error_rate


    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_rate, cmap="coolwarm", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Error Rate")
    ax.set_title("Error Rate")
    ax.set_xlabel("mu")
    ax.set_ylabel("pz")
    mu_vals = [round(j*0.1, 1) for j in range(1,10)]
    pz_vals = [round(i*0.1, 1) for i in range(10)]

    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(pz_vals)))
    ax.set_yticklabels(pz_vals)

    threshold = (error_rate.max() + error_rate.min()) / 2
    for i in range(error_rate.shape[0]):
        for j in range(error_rate.shape[1]):
            val = error_rate[i, j]
            color = "white" if val > threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(signal_lengths, cmap="coolwarm", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signal Length")
    ax.set_title("Signal Lengths")
    ax.set_xlabel("mu")
    ax.set_ylabel("pz")

    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(pz_vals)))
    ax.set_yticklabels(pz_vals)

    threshold = (signal_lengths.max() + signal_lengths.min()) / 2
    for i in range(signal_lengths.shape[0]):
        for j in range(signal_lengths.shape[1]):
            val = signal_lengths[i, j]
            color = "white" if val > threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(savepath2, dpi=300, bbox_inches="tight")
    plt.close()


def pz_mu_smallWindow(savepath,savepath2,generate = False, enable_aopp = False, enable_ldpc = False):

    loss_db_per_km = 0.3
    distance = 100  # km
    total_loss_db = loss_db_per_km * distance  # 30 dB

    transmittance = 10 ** (-total_loss_db / 10)

    if generate:
        length = 100000
        for i in range(9,10):
            twin = twinfield_communication_tensorflow.Twinfield()
            twin.loss_1 = transmittance
            twin.loss_2 = transmittance
            twin.p_x = 0
            twin.p_z = 1 - i* 0.05
            for j in range(1,10):
                twin.mu3 = j*0.05
                loadpath = "saves/pz_mu2/pz" + str(i) + "_mu" + str(j) +  ".npz"
                twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    signal_lengths = np.zeros((10,9))
    error_rate = np.zeros((10,9))
    key_rate = np.zeros((10,9))

    twin =twinfield_communication_tensorflow.Twinfield()
    for i in range(10):
        for j in range(1,10):
            loadpath = "saves/pz_mu2/pz" + str(i) + "_mu" + str(j) +  ".npz"
           
            twin.tf_communicate_load_settings(loadpath)
            alice_key, bob_key = twin.tf_communicat_load(loadpath)
            
            disclosed_info = 0
            if enable_aopp:
                alice_key, bob_key = aopp.aopp(alice_key, bob_key)
            if enable_ldpc:
                if len(alice_key) > 0:
                    if np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int))) > 0:
                        alice_key, bob_key,disclosed_info = ldpc.ldpc(alice_key, bob_key, np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int))))
                
            signal_lengths[i][j-1] = len(alice_key)- disclosed_info
            if len(alice_key) > 0:
                error_rate[i][j-1] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
            key_rate[i][j-1] = (len(alice_key)- disclosed_info)/twin._N_pulses
    np.savez("/home/felix/QKD_felix/saves/graph data/pzmu_ldpc.npz",
        first=error_rate,
        second=signal_lengths,
        third=key_rate,)


    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_rate, cmap="autumn_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Error Rate")
    ax.set_title("Error Rate")
    ax.set_xlabel("mu")
    ax.set_ylabel("pz")
    mu_vals = [round(j*0.05, 2) for j in range(1,10)]
    pz_vals = [round(i*0.05, 2) for i in range(10)]

    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(pz_vals)))
    ax.set_yticklabels(pz_vals)

    threshold = (error_rate.max() + error_rate.min()) / 2
    for i in range(error_rate.shape[0]):
        for j in range(error_rate.shape[1]):
            val = error_rate[i, j]
            color = "white" if val > threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(signal_lengths, cmap="autumn", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signal Length")
    ax.set_title("Signal Lengths")
    ax.set_xlabel("mu")
    ax.set_ylabel("pz")

    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(pz_vals)))
    ax.set_yticklabels(pz_vals)

    threshold = (signal_lengths.max() + signal_lengths.min()) / 2
    for i in range(signal_lengths.shape[0]):
        for j in range(signal_lengths.shape[1]):
            val = signal_lengths[i, j]
            color = "white" if val > threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(savepath2, dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(key_rate, cmap="autumn_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Key Rate")
    ax.set_title("Key Rate")
    ax.set_xlabel("mu")
    ax.set_ylabel("pz")
    mu_vals = [round(j*0.05, 2) for j in range(1,10)]
    pz_vals = [round(i*0.05, 2) for i in range(10)]

    ax.set_xticks(np.arange(len(mu_vals)))
    ax.set_xticklabels(mu_vals)
    ax.set_yticks(np.arange(len(pz_vals)))
    ax.set_yticklabels(pz_vals)

    threshold = (key_rate.max() + key_rate.min()) / 2
    for i in range(key_rate.shape[0]):
        for j in range(key_rate.shape[1]):
            val = key_rate[i, j]
            color = "white" if val > threshold else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig("figures/my_graph/pzmu_ldpc_key_rate.png", dpi=300, bbox_inches="tight")
    plt.close()




#pz_mu("figures/my_graph/pzmu_errors2.png","figures/my_graph/pzmu_length2.png",generate=False)
#pz_mu_smallWindow("figures/my_graph/pzmu_errors.png","figures/my_graph/pzmu_length.png",generate=False)
#pz_mu_smallWindow("figures/my_graph/pzmu_aopp_errors.png","figures/my_graph/pzmu_aopp_length.png",generate=False,enable_aopp=True)
pz_mu_smallWindow("figures/my_graph/pzmu_ldpc_errors.png","figures/my_graph/pzmu_ldpc_length.png",generate=False,enable_ldpc=True)
#pz_mu_smallWindow("figures/my_graph/pzmu_aopp_ldpc_errors.png","figures/my_graph/pzmu_aopp_ldpc_length.png",generate=False,enable_ldpc=True,enable_aopp=True)
 