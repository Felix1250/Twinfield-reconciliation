import numpy as np
import random
from tqdm import tqdm
import sys
from os import path

sys.path.append("/home/felix/QKD_felix/src/Twinfield_QKD")

import tf_utils

def __pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def make_odd_parity_pairs(key):
    arr1 = np.arange(len(key))

    np.random.shuffle(arr1)
    arr2 = arr1.tolist()
    pairs= []
    while len(arr2) != 0:
        first = __pop_random(arr2)
        second = -1
        for i in range(len(arr2)):
            if key[arr2[i]] != key[first]:
                second = arr2[i]
        if len(arr2) == 0:
            break
        if second == -1:
            second = arr2[0]
        pairs.append([first,second])
    return pairs




def aopp(alice_key,bob_key):
    orig_len = len(bob_key)     
    pairs = make_odd_parity_pairs(bob_key)
    #bit mask that if = 1 keeps the bit
    drop_bits = np.zeros(len(bob_key))
    for i in tqdm(range(len(pairs)),"odd parity paring"):
        parity_alice = (alice_key[pairs[i][0]] + alice_key[pairs[i][1]]) % 2
        parity_bob = (bob_key[pairs[i][0]] + bob_key[pairs[i][1]]) % 2
        #stupid ass debug
        #if bob_key[pairs[i][0]] != alice_key[pairs[i][0]] and bob_key[pairs[i][1]] != alice_key[pairs[i][1]]:
        #    print("i am an error")
        #    print("parity alice: " + str(parity_alice))
        #    print("parity bob: " + str(parity_bob))

        if parity_bob == parity_alice:
            select_bit = random.randint(0,1)
            drop_bits[pairs[i][select_bit]] = 1
    aopp_length = np.count_nonzero(drop_bits)
    alice_key_aopp = np.zeros(aopp_length)
    bob_key_aopp=  np.zeros(aopp_length)
    counter = 0
    for i in range(len(drop_bits)):
        if drop_bits[i] == 1:
            alice_key_aopp[counter]= alice_key[i]
            bob_key_aopp[counter]= bob_key[i]
            counter +=1
    print("---------------------------------------")
    print("length after aopp: " + str(aopp_length))
    if len(orig_len) != 0:
        print("percentage of original key: " + str(aopp_length/len(orig_len)))
    tf_utils.print_error_rate(alice_key_aopp,bob_key_aopp)
    return alice_key_aopp,bob_key_aopp