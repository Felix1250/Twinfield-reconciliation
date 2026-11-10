'''
This file implements AOPP

'''




import os
import numpy as np
import random
from tqdm import tqdm
import sys

cw_path = os.getcwd()
sys.path.append(cw_path + "/src/reconciliation")
import tf_utils


'''
helper function
'''
def __pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)
'''
old implementation, was too slow
'''
def make_odd_parity_pairs_old(key):
    indices = list(range(len(key)))
    random.shuffle(indices)
    
    pairs = []
    while len(indices) >= 2:
        first = indices.pop(random.randrange(len(indices)))
        for i, idx in enumerate(indices):
            if key[idx] != key[first]:
                match_idx = i
                break
        
        # If an opposite key value exists, pop it; otherwise pop the first available element
        if match_idx is not None:
            second = indices.pop(match_idx)
        else:
            second = indices.pop(0)
            
        pairs.append([first, second])
    return np.array(pairs)

'''
new implementation of making odd parities
'''
def make_odd_parity_pairs(key):
    key = np.asarray(key)

    zeros = np.where(key == 0)[0]
    ones = np.where(key == 1)[0]
    
    np.random.shuffle(zeros)
    np.random.shuffle(ones)
    
    num_odd_pairs = min(len(zeros), len(ones))

    pairs = list(zip(zeros[:num_odd_pairs], ones[:num_odd_pairs]))

    # apparently the rest of the bits get dropped. didnt know that at first
    #remaining = np.concatenate([zeros[num_odd_pairs:], ones[num_odd_pairs:]])
    #np.random.shuffle(remaining)
    #
    #for i in range(0, len(remaining) - 1, 2):
    #    pairs.append((remaining[i], remaining[i + 1]))
        
    return np.array(pairs),len(zeros), len(ones)

'''
AOPP implementation
'''
def aopp(alice_key,bob_key):
    print("aopp start")
    orig_len = len(bob_key)     
    pairs,N0,N1 = make_odd_parity_pairs(bob_key) #pair the bits, N0 / N1 are needed for the decoy window
    #bit mask that if = 1 keeps the bit
    drop_bits = np.zeros(len(bob_key))
    print("aopp start 2")
    for i in tqdm(range(len(pairs)),"odd parity paring"):
        parity_alice = (alice_key[pairs[i][0]] + alice_key[pairs[i][1]]) % 2 #calculate parity for alice
        parity_bob = (bob_key[pairs[i][0]] + bob_key[pairs[i][1]]) % 2 #calculate parity for bob
        #stupid ass debug
        #if bob_key[pairs[i][0]] != alice_key[pairs[i][0]] and bob_key[pairs[i][1]] != alice_key[pairs[i][1]]:
        #    print("i am an error")
        #    print("parity alice: " + str(parity_alice))
        #    print("parity bob: " + str(parity_bob))

        if parity_bob == parity_alice: #check their parities
            select_bit = random.randint(0,1)
            drop_bits[pairs[i][select_bit]] = 1 #if this bit is one it is kept otherwise not
    aopp_length = np.count_nonzero(drop_bits)
    alice_key_aopp = np.zeros(aopp_length)
    bob_key_aopp=  np.zeros(aopp_length)
    counter = 0
    for i in range(len(drop_bits)): #drop all the designated bits and concatenate
        if drop_bits[i] == 1:
            alice_key_aopp[counter]= alice_key[i]
            bob_key_aopp[counter]= bob_key[i]
            counter +=1
    print("---------------------------------------")
    print("length after aopp: " + str(aopp_length))
    if orig_len != 0:
        print("percentage of original key: " + str(aopp_length/orig_len))
    tf_utils.print_error_rate(alice_key_aopp,bob_key_aopp) 
    return alice_key_aopp,bob_key_aopp, N0,N1 