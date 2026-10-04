import math
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/felix/QKD_felix/src/Twinfield_QKD")
sys.path.append("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python")
sys.path.append("/home/felix/QKD_felix/src/reconciliation")

import tf_utils
import twinfield_communication_tensorflow
import aopp
import ldpc

def simulate_physics(path,pd = math.pow(10,-5) , eta  =0.8, ea = 0.15,distance = 0,length = 1000000):
    twin =twinfield_communication_tensorflow.Twinfield()

    loss_db_per_km = 0.2 # km
    total_loss_db = loss_db_per_km * distance  # 30 dB
    transmittance = 10 ** (-total_loss_db / 10)

    twin.pd0 = pd
    twin.pd1 = pd
    twin.eta1 = eta
    twin.eta2 = eta
    twin.phase_shift_average = ea
    twin.loss_1 = transmittance
    twin.loss_2 = transmittance

    twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,path)

def simulate_comm(path):
    twin =twinfield_communication_tensorflow.Twinfield()
    twin.tf_communicate_load_settings(path)

    alice_key, bob_key = twin.tf_communicat_load(path)
            
    
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