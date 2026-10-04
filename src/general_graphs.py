import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

sys.path.append("/home/felix/QKD_felix/src/Twinfield_QKD")
sys.path.append("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python")
sys.path.append("/home/felix/QKD_felix/src/reconciliation")

import tf_utils
import twinfield_communication_tensorflow
import aopp
import ldpc
import hashing

def simulate_physics(path,pd = math.pow(10,-5) , eta  =0.8, ea = 0.15,distance = 0,length = 100):
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

    print("hey i was doing something")

def simulate_comm(path,aopp_enabled = False,ldpc_enabled = False, hash_enabled = False):
    twin =twinfield_communication_tensorflow.Twinfield()
    twin.tf_communicate_load_settings(path)

    alice_key, bob_key = twin.tf_communicat_load(path,aopp_enabled = aopp_enabled)

    comm_iters = 0
    error_rate = 0

    if ldpc_enabled:
        alice_key, bob_key,disclosed_info,comm_iters = ldpc.ldpc(alice_key_1, bob_key_1, twin._ez,twin.pd0,twinfield=True)
    if hash_enabled:
        alice_key, bob_key = hashing.priv_ampl(alice_key, bob_key, twin._ex1)
    signal_length = len(alice_key)
    if len(alice_key) > 0:
        error_rate = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))        
    key_rate = (len(alice_key))/twin._N_pulses
    return signal_length, error_rate , key_rate , comm_iters

def plot_general(key_rate, error_rate,labels,image_savepath):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    y = np.arange(0,len(key_rate))
    ax1.plot(y, key_rate, color='red', marker='o', linestyle='-', linewidth=2, label='bit rate', zorder=3)
    ax1.set_xlabel('phase slices', fontsize=12) # Shared X label
    ax1.set_ylabel('bit rate', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')

    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()  
    ax2.bar(y, error_rate, width=0.6, color='skyblue', alpha=0.5, label='error rate', zorder=1)
    ax2.set_ylabel('error rate', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Something something legend
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.savefig(image_savepath, dpi=300)

def main():
    parser = argparse.ArgumentParser(description="Run Twinfield QKD simulation and graph generation.")
    parser.add_argument("--savefile", type=str, required=True, help="Directory to save job outputs.")
    parser.add_argument("--pd", type=float, default=math.pow(10, -5), help="Dark count probability.")
    parser.add_argument("--eta", type=float, default=0.8, help="Efficiency.")
    parser.add_argument("--ea", type=float, default=0.15, help="Phase shift average.")
    parser.add_argument("--distance", type=float, default=0.0, help="Distance in km.")

    args = parser.parse_args()
    print(args)
    simulate_physics(args.savefile, args.pd, args.eta, args.ea, args.distance)

if __name__ == "__main__":
    main()

#key_rate = [0.1,0.2,0.3,0.1,0.4,0.6,0.7,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
#error_rate = [0.5,0.42,0.43,0.41,0.4,0.46,0.47,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]
#labels = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12,0.13,0.14,0.15]
#plot_general(key_rate,error_rate,labels, "test.png")