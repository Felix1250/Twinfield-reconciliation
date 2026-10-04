import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
from os import path
import math


def determine_codes_and_length(key):
    length_key = len(key)
    n_1 = math.ceil(length_key/1944.0)
    n_2 = math.ceil(length_key/4000.0)
    if n_1*1944 < n_2 * 4000 :
        a = length_key/n_1
        return a, codes_from_file("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python/codes_1944.txt") 
    a = length_key/n_2
    return a,n_1, codes_from_file("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python/codes_4000.txt") 


def calc_params(k,n,R_range):

    R= k/n
    R_min = min(abs(x - R) for x in R_range)
    s = n-k
    p = -(k-s)/R_min - s+n
    return R_min,s,p


def ldpc(correct_key,faulty_key,qber_est = 0.21,qber_est_0 = 0.0,twinfield=True, target_f = 1.2):
    #this loades a code from teh ieee codes with a length of 1944
    f_start = 1.0 # starting efficiency
    discl_k = 1

    #codes = codes_from_file("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python/codes_1944.txt")  
    #n = 1944
    n,codes = determine_codes_and_length(correct_key)

    # Computing the range of rates for given codes
    R_range = []
    for code in codes:
        R_range.append(code[0])
    # calculate how many bits have to be punctured or added
    #R, s_n, p_n = ec.choose_sp(qber_est, f_start, R_range, n)
    R, s_n, p_n = calc_params(len(correct_key), n, R_range)

    print(f"R range is: {np.sort(R_range)}")

    # select the code to use from the available codes
    R, s_n, p_n = ec.choose_sp(qber_est, f_start, R_range, n)
    print(f"R range is: {np.sort(R_range)}")
    code_params = codes[(R, n)]
    s_y_joins = code_params['s_y_joins']
    y_s_joins = code_params['y_s_joins']
    punct_list = code_params['punct_list']

    discl_n = int(round(n*(0.0280-0.02*R)*discl_k))
    disclosed_info = 0

    # length = n - shortened - punctured bits
    payload_per_block = n - s_n - p_n

    # Chunk the key into blocks of payload_per_block
    faulty_blocks = [faulty_key[i:i + payload_per_block] for i in range(0, len(faulty_key), payload_per_block)]
    correct_blocks = [correct_key[i:i + payload_per_block] for i in range(0, len(correct_key), payload_per_block)]

    error_map = []
    com_iters_counter = 0
    for faulty_chunk, correct_chunk in zip(faulty_blocks, correct_blocks):
        # If the last chunk is smaller, pad it with 0s
        if len(faulty_chunk) < payload_per_block:
            pad_len = payload_per_block - len(faulty_chunk)
            faulty_chunk = np.pad(faulty_chunk, (0, pad_len), 'constant')
            correct_chunk = np.pad(correct_chunk, (0, pad_len), 'constant')
        
        add_info, com_iters, dec_chunk, ver_check = ec.perform_ec(
            faulty_chunk, correct_chunk, s_y_joins, y_s_joins, qber_est,qber_est_0, s_n, p_n, 
            punct_list=punct_list, discl_n=discl_n, show=1,twinfield=twinfield
        )

        disclosed_info += len(s_y_joins) - s_n +add_info
        com_iters_counter += com_iters
        
        # Trim padding if last chunk was padded
        if len(faulty_chunk) > len(error_map_chunk := dec_chunk[:len(faulty_chunk) - pad_len if 'pad_len' in locals() else len(dec_chunk)]):
            error_map.append(error_map_chunk)
        else:
            error_map.append(dec_chunk)

    error_map = np.concatenate(error_map)
    faulty_key = np.asarray(faulty_key, dtype=np.uint8)
    error_map = np.asarray(error_map, dtype=np.uint8)
    corrected_key= np.bitwise_xor(faulty_key,error_map)
    remaining_errors = np.abs(corrected_key.astype(int) - correct_key.astype(int))
    print("Total remaining errors: " + str(np.sum(remaining_errors)))
    print("remaining errorrate: " + str(np.mean(remaining_errors)))
    return correct_key,corrected_key, disclosed_info, com_iters_counter

#if __name__ == '__main__':
    #R_range = [0.5,2/3,3/4,1,4/5,70/71]
    #qber_est = 0.0000003
    #R, s_n, p_n = ec.choose_sp(qber_est, 1, R_range, 1944)
    #print(R)
