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
import math


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

def graph_compare_LDPC(generate = False):
    twin =twinfield_communication_tensorflow.Twinfield()
    
    if generate:
        length = 1000000
        for i in range(0,3):
            loss_db_per_km = 0.2
            distance = i*50  # km
            total_loss_db = loss_db_per_km * distance  # 30 dB
            twin.eta1 = 0.3
            twin.eta2 = 0.3
            transmittance = 10 ** (-total_loss_db / 10)
            twin.loss_1 = transmittance
            twin.loss_2 = transmittance
            twin.p_x = 0
            loadpath = "saves/distance/distance__" + str(i) + ".npz"
            twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    
    signal_lengths = np.zeros((3,8))
    error_rate = np.zeros((3,8))
    key_rate = np.zeros((3,8))
    plob = np.zeros(8)
    x = []
    y = []
    comm_iters = np.zeros((2,8))
    for i in range(0,1):
        loadpath = "saves/distance/distance_" + str(i) + ".npz"
        twin.tf_communicate_load_settings(loadpath)
        x.append(i*100)
        y.append(i)

        loss_db_per_km = 0.2
        distance = i*50  # km
        total_loss_db = loss_db_per_km * distance  # 30 dB
        twin.eta1 = 0.3
        twin.eta2 = 0.3
        transmittance = 10 ** (-total_loss_db / 10)
    
        if transmittance < 1:
            loss_db_per_km = 0.2
            distance = i*100  # km
            total_loss_db = loss_db_per_km * distance  # 30 dB
            twin.eta1 = 0.3
            twin.eta2 = 0.3
            transmittance = 10 ** (-total_loss_db / 10)
            plob[i] = - math.log2(1-transmittance)
        else:
            plob[i] = 1

        alice_key, bob_key = twin.tf_communicat_load(loadpath)
        
        disclosed_info = 0
        signal_lengths[0][i] = len(alice_key)- disclosed_info
        if len(alice_key) > 0:
            error_rate[0][i] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
        key_rate[0][i] = (len(alice_key)- disclosed_info)/twin._N_pulses
        #print("-----------------------aopp------------------------------------")
        #alice_key, bob_key = aopp.aopp(alice_key, bob_key)
        #signal_lengths[1][i] = len(alice_key)- disclosed_info
        #if len(alice_key) > 0:
        #    error_rate[1][i] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
        #key_rate[1][i] = (len(alice_key)- disclosed_info)/twin._N_pulses
        print("-----------------------ldpc------------------------------------")
        alice_key_1 = alice_key.copy()
        bob_key_1 = bob_key.copy()
        if np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int))) > 0:
            alice_key, bob_key,disclosed_info,comm_iters[0][i] = ldpc.ldpc(alice_key_1, bob_key_1, np.mean(np.abs(alice_key_1.astype(int) - bob_key_1.astype(int))),twin.pd0,twinfield=True)
            alice_key, bob_key,disclosed_info,comm_iters[1][i] = ldpc.ldpc(alice_key_1, bob_key_1, np.mean(np.abs(alice_key_1.astype(int) - bob_key_1.astype(int))),twin.pd0,twinfield=False)
        signal_lengths[2][i] = len(alice_key)- disclosed_info
        if len(alice_key) > 0:
            error_rate[2][i] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
        key_rate[2][i] = (len(alice_key)- disclosed_info)/twin._N_pulses
    print("comm iters: ", comm_iters)

    np.savez("/home/felix/QKD_felix/saves/graph data/ldpc_test.npz",
            first=error_rate,
            second=signal_lengths,
            third=key_rate,)


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(key_rate[0], label="pre reconciliation")
    ax.plot(key_rate[1], label="aopp")
    ax.plot(key_rate[2], label="aopp + ldpc")
    ax.plot(plob, label="PLOB")
    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label("Key Rate")
    plt.title("Key Rate")
    plt.xlabel("distance in km")
    plt.ylabel("key rate")
    plt.yscale('log')

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/my_graph/ldpc_1.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(comm_iters[0], label="new")
    ax.plot(comm_iters[1], label="old")

    #cbar = fig.colorbar(im, ax=ax)
    #cbar.set_label("Key Rate")
    plt.title("com iters")
    plt.xlabel("distance in km")
    plt.ylabel("key rate")
    plt.yscale('log')

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/my_graph/ldpc_2.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    #pz_mu("figures/my_graph/pzmu_errors2.png","figures/my_graph/pzmu_length2.png",generate=False)
    #pz_mu_smallWindow("figures/my_graph/pzmu_errors.png","figures/my_graph/pzmu_length.png",generate=False)
    #pz_mu_smallWindow("figures/my_graph/pzmu_aopp_errors.png","figures/my_graph/pzmu_aopp_length.png",generate=False,enable_aopp=True)
    #pz_mu_smallWindow("figures/my_graph/pzmu_ldpc_errors.png","figures/my_graph/pzmu_ldpc_length.png",generate=False,enable_ldpc=True)
    graph_compare_LDPC(generate=False)
    #pz_mu_smallWindow("figures/my_graph/pzmu_aopp_ldpc_errors.png","figures/my_graph/pzmu_aopp_ldpc_length.png",generate=False,enable_ldpc=True,enable_aopp=True)
 