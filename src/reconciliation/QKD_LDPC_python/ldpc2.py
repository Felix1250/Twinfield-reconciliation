import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
import math




def determine_codes_and_length(key):
    length_key = len(key)
    n_1 =length_key % 1944.0 #math.modf(length_key,1944.0)
    n_2 =length_key % 4000.0 #math.modf(length_key,4000.0)
    cw_path = os.getcwd()

    if n_1 < n_2 :
        return 1944, codes_from_file(cw_path + "/src/reconciliation/QKD_LDPC_python/codes_1944.txt") 
    return 4000, codes_from_file(cw_path + "/src/reconciliation/QKD_LDPC_python/codes_4000.txt") 


def shannon(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return  -p * math.log2(p) - (1-p) * math.log2(1-p)

def calc_params(k,n,R_range,qber):
    s,p =0
    Rate = 1 - shannon(qber)
    R_min = min(abs(x - Rate) for x in R_range)
    if Rate < R_min:
        s = k- Rate * n / (1 - Rate)
    else:
        p = n - k/Rate
    return R_min,s,p


def ldpc1(correct_key,faulty_key,qber_est = 0.21,qber_est_0 = 0.0,twinfield=True):
    if qber_est == 0:
        qber_est = 0.0000001
    if len(correct_key) == 0:
        return correct_key,faulty_key, 0, 0
    #this loades a code from teh ieee codes with a length of 1944
    f_start = 1.0 # starting efficiency
    discl_k = 1

    #codes = codes_from_file("/home/felix/QKD_felix/src/reconciliation/QKD_LDPC_python/codes_1944.txt")  
    #n = 1944
    n,codes = determine_codes_and_length(correct_key)

    # Computing the range of rates for given codes
    R_range = []
    m_range = []
    for code in codes:
        R_range.append(codes[(code[0], n)]['R'])
        m_range.append(codes[(code[0], n)]['syndrome_len'])
    s_n = p_n =0
    #select which code fits the required rate best
    Rate = 1 - shannon(qber_est)
    index = min(enumerate(R_range), key=lambda pair: abs(pair[1] - Rate))[0]
    R_min = R_range[index]
    k = n-m_range[index]

    #calculate how many shortened / punctured bits 
    syndrome_length = m_range[index]
    if Rate < R_min:
        s_n = math.floor(k- Rate * n / (1 - Rate))
    else:
        p_n = math.floor(n - k/Rate)
    R = R_min
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

    corrected_chunks = []
    correct_chunks = []
    com_iters_counter = 0
    for faulty_chunk, correct_chunk in zip(faulty_blocks, correct_blocks):
        # If the last chunk is smaller, more shortened bits
        if len(faulty_chunk) < payload_per_block:
            p_n = 0
            s_n = n-len(faulty_chunk)
        add_info, com_iters, dec_chunk, ver_check = ec.perform_ec(
            faulty_chunk, correct_chunk, s_y_joins, y_s_joins, qber_est,qber_est_0, s_n, p_n, 
            punct_list=punct_list, discl_n=discl_n, show=1,twinfield=twinfield
        )
        this_iteration_disclosed =  math.ceil((syndrome_length +add_info ) * len(faulty_chunk)/payload_per_block)
        disclosed_info += this_iteration_disclosed
        com_iters_counter += com_iters

        #remove so much bits as were disclosed
        arr2 = np.arange(0,len(correct_key))
        np.random.shuffle(arr2) 
        arr = arr2[len(correct_key)-this_iteration_disclosed:]

        error_map = np.asarray(dec_chunk[arr], dtype=np.uint8)
        faulty_chunk = np.asarray(faulty_chunk[arr], dtype=np.uint8)
        correct_chunks.append(np.asarray(correct_chunk[arr], dtype=np.uint8))
        corrected_chunks.append(np.bitwise_xor(faulty_chunk,error_map))

    corrected_key = np.concatenate(corrected_chunks)
    correct_key = np.concatenate(correct_chunks)
    remaining_errors = np.abs(corrected_key.astype(int) - correct_key.astype(int))
    print("Total remaining errors: " + str(np.sum(remaining_errors)))
    print("remaining errorrate: " + str(np.mean(remaining_errors)))

    return correct_key,corrected_key, disclosed_info, com_iters_counter




#if __name__ == '__main__':
    #R_range = [0.5,2/3,3/4,1,4/5,70/71]
    #qber_est = 0.0000003
    #R, s_n, p_n = ec.choose_sp(qber_est, 1, R_range, 1944)
    #print(R)
