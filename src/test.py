import math
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import csv

cw_path = os.getcwd()
sys.path.append(cw_path + "/src/Twinfield_QKD")
sys.path.append(cw_path + "/src/reconciliation/QKD_LDPC_python")
sys.path.append(cw_path + "/src/reconciliation")

import tf_utils
import twinfield_communication_tensorflow
import aopp
import ldpc2
import hashing


path = os.getcwd() + "/saves/pd2"
length = 1000000
def do():
    twin =twinfield_communication_tensorflow.Twinfield()
    #twin.p_x = 1
    twin.tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),length,path)

def simulate_comm(path,aopp_enabled = False,ldpc_enabled = False, hash_enabled = False, enable_twinfield=True):
    twin =twinfield_communication_tensorflow.Twinfield()
    twin.tf_communicate_load_settings(path)

    alice_key, bob_key = twin.tf_communicat_load(path,aopp_enabled = aopp_enabled)

    comm_iters = 0
    error_rate = 0

    print(dir(ldpc2))
    if ldpc_enabled:
        alice_key, bob_key,disclosed_info,comm_iters = ldpc2.ldpc1(alice_key, bob_key, qber_est = twin._esignal,qber_est_0 =twin.pd0,twinfield = enable_twinfield)
    if hash_enabled:
        alice_key, bob_key = hashing.priv_ampl(alice_key, bob_key, twin._ephi,twin._n1)
    signal_length = len(alice_key)
    if len(alice_key) > 0:
        error_rate = np.mean(np.abs(alice_key.astype(int) - bob_key.astype(int)))        
    key_rate = (len(alice_key))/twin._N_pulses
    return signal_length, error_rate , key_rate , comm_iters

if __name__ == '__main__':
    #do()
    path = path + ".npz"
    simulate_comm(path,aopp_enabled=True,ldpc_enabled = True, hash_enabled = True, enable_twinfield=True)

'''
def delta_2(Y, xi):
    def objective_function(delta):
        if delta <= 0.0 or delta >= 1.0:
            return np.nan
        return Y * (-delta - (1.0 - delta) * np.log(1.0 - delta)) - np.log(xi / 2.0)
    fa = objective_function(1e-12)
    fb = objective_function(1.0 - 1e-12)
    if fa * fb > 0: #handle if there is no result
        return  0.0 
    delta_2_prime, info = brentq(objective_function, 1e-12, 1.0 - 1e-12, full_output=True)#finds nullstellen
    print(delta_2_prime)
    return delta_2_prime

def delta_1(Y, xi):

    def objective_function(delta):
        if delta <= 0.0:
            return np.nan
        return Y * (delta - (1.0 + delta) * np.log(1.0 + delta)) - np.log(xi / 2.0)
    fa = objective_function(1e-12)
    fb = objective_function(100)
    if fa * fb > 0: #handle if there is no result
        return  0.0 
    delta_1_prime, info = brentq(objective_function, 1e-12, 100.0, full_output=True)
    return delta_1_prime

def phi_U(Y,xi):
    delta_1_val = delta_1(Y,xi)
    return (1+ delta_1_val) * Y

def phi_L(Y,xi):
    delta_2_val = delta_2(Y, xi)
    return (1.0-delta_2_val) * Y'''