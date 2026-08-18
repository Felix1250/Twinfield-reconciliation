import error_correction_lib as ec
import numpy as np
from file_utils import codes_from_file
from os import path

'''

hello this is old


'''

# Choose of the codes pool:
#codes = codes_from_file('codes_4000.txt'); n = 4000
codes = codes_from_file('codes_1944.txt')
n = 1944

# Computing the range of rates for given codes
R_range = []
for code in codes:
    R_range.append(code[0])
print(f"R range is: {np.sort(R_range)}")

fname = 'output.txt'  # file name for the output
f_start = 1.0  # initial efficiency of decoding
qber_start = 0.2
qber_end = 0.3
qber_step = 0.01  # range of QBERs
n_tries = 2  # number of keys proccessed for each QBER value

if path.exists(fname):
    file_output = open(fname, 'a')
else:
    file_output = open(fname, 'w')
    file_output.write(
        "code_n, n_tries, qber, f_mean, com_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER \n")

for qber in np.arange(qber_start, qber_end, qber_step):
    # Here we test the error correction for a given QBER
    # Output: f_mean= mean efficiency of decoding, com_iters_mean= mean number of communication iterations, 
    # R= rate of the code, s_n= number of shortened bits, p_n= number of punctured bits, p_n_max= maximal number of punctured bits, 
    # k_n= n-s_n-p_n, discl_n= number of disclosed bits, FER= frame error rate
    f_mean, com_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER = ec.test_ec(
        qber, R_range, codes, n, n_tries, f_start=f_start, show=1, discl_k=1)
    # print(f"qber: {qber}, f_mean: {f_mean}, com_iters_mean: {com_iters_mean}, R: {R}, s_n: {s_n}, p_n: {p_n}, p_n_max: {p_n_max}, k_n: {k_n}, discl_n: {discl_n}, FER: {FER}")
    file_output.write('%d,%d,%8.4f,%14.8f,%14.8f,%14.8f,%10d,%10d,%10d,%14d,%10d,%14.8f\n' % (n, n_tries,
                                                                                              qber, f_mean, com_iters_mean, R, s_n, p_n, p_n_max, k_n, discl_n, FER))
file_output.close()
