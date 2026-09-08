import twinfield_communication_tensorflow
import tf_utils
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



def graph_phase_slices(savepath,loadpath = "saves/only_decoy_temp.npz",generate = False):
    print("if you use this part of the program make sure you have a high decoy window")
    twin =twinfield_communication_tensorflow.Twinfield()
    length = 1000000
    if generate:
        twin.p_x = 1
        twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,"saves/only_decoy_temp.npz")
        loadpath = "saves/only_decoy_temp.npz"
    twin.tf_communicate_load_settings(loadpath)
    decoy_lengths = []
    emu1 = []
    emu2 = []
    exmu1 = []
    exmu2 = []
    x = [4]
    y = []
    for i in range(2,8):
        twin.n_phaseSlice = math.pow(2,i)
        x.append(math.pow(2,i))
        y.append(int(i))
        twin.tf_communicat_load(loadpath)
        decoy_lengths.append(twin._decoy_length)
        emu1.append(twin._emu1)
        emu2.append(twin._emu2)
        exmu1.append(twin._ex1)
        exmu2.append(twin._ex2)


    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(y, decoy_lengths, color='red', marker='o', linestyle='-', linewidth=2, label='number of decoy bits', zorder=3)
    ax1.set_xlabel('phase slices', fontsize=12) # Shared X label
    ax1.set_ylabel('number of decoy bits', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  

    ax2.bar(y, emu1, width=0.6, color='skyblue', alpha=0.5, label='error rate', zorder=1)
    ax2.set_ylabel('error rate', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(x)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)


def graph_losses(savepath,generate = False):
    twin =twinfield_communication_tensorflow.Twinfield()
    
    if generate:
        length = 10000000
        for i in range(5,8):
            loss_db_per_km = 0.2
            distance = i*50  # km
            total_loss_db = loss_db_per_km * distance  # 30 dB
            twin.eta1 = 0.3
            twin.eta2 = 0.3
            transmittance = 10 ** (-total_loss_db / 10)
            twin.loss_1 = transmittance
            twin.loss_2 = transmittance
            twin.p_x = 0
            loadpath = "saves/distance/distance_" + str(i) + ".npz"
            twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    
    signal_lengths = np.zeros((3,8))
    error_rate = np.zeros((3,8))
    key_rate = np.zeros((3,8))
    plob = np.zeros(8)
    x = []
    y = []
    for i in range(8):
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
        print("-----------------------aopp------------------------------------")
        alice_key, bob_key = aopp.aopp(alice_key, bob_key)
        signal_lengths[1][i] = len(alice_key)- disclosed_info
        if len(alice_key) > 0:
            error_rate[1][i] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
        key_rate[1][i] = (len(alice_key)- disclosed_info)/twin._N_pulses
        print("-----------------------ldpc------------------------------------")

        if np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int))) > 0:
            alice_key, bob_key,disclosed_info = ldpc.ldpc(alice_key, bob_key, np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int))),twin.pd0)
        signal_lengths[2][i] = len(alice_key)- disclosed_info
        if len(alice_key) > 0:
            error_rate[2][i] = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))
        key_rate[2][i] = (len(alice_key)- disclosed_info)/twin._N_pulses
 
    np.savez("/home/felix/QKD_felix/saves/graph data/plob.npz",
            first=error_rate,
            second=signal_lengths,
            third=key_rate,)

    #fig, ax1 = plt.subplots(figsize=(8, 5))

    #ax1.plot(y, error_rate, color='red', marker='o', linestyle='-', linewidth=2, label='error rate', zorder=3)
    #ax1.set_xlabel('Distance in km', fontsize=12) # Shared X label
    #ax1.set_ylabel('number of decoy bits', color='red', fontsize=12)
    #ax1.tick_params(axis='y', labelcolor='red')

    #ax2 = ax1.twinx()  

    #ax2.bar(y, signal_lengths, width=0.6, color='skyblue', alpha=0.5, label='key size', zorder=1)
    #ax2.set_ylabel('error rate', color='blue', fontsize=12)
    #ax2.tick_params(axis='y', labelcolor='blue')
    #ax1.set_xticklabels(x)

    #ax1.set_zorder(ax2.get_zorder() + 1)
    #ax1.patch.set_visible(False)

    #plt.tight_layout()
    #plt.savefig(savepath, dpi=300)

    #plt.close()

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
    plt.savefig("figures/my_graph/key_rate_distance.png", dpi=300, bbox_inches="tight")
    plt.close()



def graph_eta(savepath,generate = False):

    twin =twinfield_communication_tensorflow.Twinfield()
    
    if generate:
        length = 100000
        for i in range(4):

            twin.eta1 = 1-i*0.05
            twin.eta2 = 1-i*0.05
            twin.p_x = 0
            loadpath = "saves/eta2/eta" + str(i) + ".npz"
            twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    
    signal_lengths = []
    error_rate = []
    x = []
    y = []
    for i in range(10):
        loadpath = "saves/eta/eta" + str(i) + ".npz"
        twin.tf_communicate_load_settings(loadpath)
        x.append(1-i*0.02)
        y.append(i)
        twin.tf_communicat_load(loadpath)
        signal_lengths.append(twin._signal_length)
        if twin._error_rate == None:
            error_rate.append(1)
        else:
            error_rate.append(twin._error_rate)
 


    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(y, error_rate, color='red', marker='o', linestyle='-', linewidth=2, label='error rate', zorder=3)
    ax1.set_xlabel('eta', fontsize=12) # Shared X label
    ax1.set_ylabel('error rate', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  

    ax2.bar(y, signal_lengths, width=0.6, color='skyblue', alpha=0.5, label='key size', zorder=1)
    ax2.set_ylabel('signal length', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(x)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)

def graph_psi(savepath,generate = False):
    twin =twinfield_communication_tensorflow.Twinfield()
    
    if generate:
        length = 100000
        for i in range(11):
            twin.phase_shift_average = 0.05 * i
            twin.p_x = 1
            loadpath = "saves/psi/psi_" + str(i) + ".npz"
            twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,loadpath)
            
    
    signal_lengths = []
    error_rate = []
    x = []
    y = []
    for i in range(11):
        loadpath = "saves/psi/psi_" + str(i) + ".npz"
        twin.tf_communicate_load_settings(loadpath)
        x.append(round(i*0.05,2))
        y.append(i)
        twin.tf_communicat_load(loadpath)
        signal_lengths.append(twin._signal_length)
        if twin._error_rate == None:
            error_rate.append(1)
        else:
            error_rate.append(twin._emu1)
 


    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(y, error_rate, color='red', marker='o', linestyle='-', linewidth=2, label='error rate', zorder=3)
    ax1.set_xlabel('psi', fontsize=12) # Shared X label
    ax1.set_ylabel('error rate', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')

    #ax2 = ax1.twinx()  

    #ax2.bar(y, signal_lengths, width=0.6, color='skyblue', alpha=0.5, label='key size', zorder=1)
    #ax2.set_ylabel('error rate', color='blue', fontsize=12)
    #ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(x)

    #ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)

def just_signal(generate= False):
    twin =twinfield_communication_tensorflow.Twinfield()
    length = 100000
    if generate:
        twin.p_x = 0
        twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,"saves/only_signal_temp.npz")
    twin.tf_communicat_load("saves/only_signal_temp.npz")
    twin.tf_communicat_load("saves/only_signal_temp.npz")

def both(generate= False):
    twin =twinfield_communication_tensorflow.Twinfield()
    length = 100000
    if generate:
        twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,"saves/both.npz")
    twin.n_percentage = 1
    twin.tf_communicat_load("saves/both.npz")
    twin.tf_communicat_load("saves/both.npz")

if __name__ == '__main__':
    #loadpath = "saves/only_decoy_temp.npz"
    #graph_phase_slices("figures/my_graph/phase_slices.png",generate=False)
    graph_losses("figures/my_graph/errorrate per distance2.png",generate=False)
    #graph_eta("figures/my_graph/errorrate per eta.png",generate=False)
    #just_signal(generate=True)
    #both(generate=False)
    #graph_psi("figures/my_graph/errorrate per psi.png",generate=False)
    #twin =twinfield_communication_tensorflow.Twinfield()
    #twin.tf_communicat_load(loadpath)