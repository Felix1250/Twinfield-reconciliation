import random
import numpy as np
import math
import strawberryfields as sf
from strawberryfields import ops
from tqdm import tqdm
import datetime


import tf_utils
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


pd0 = math.pow(10,-50) #darkcount of each detector individually
pd1 = math.pow(10,-50)

eta1 = 0.95 #detection efficiency of the detectors
eta2 = 0.95 

mu1 = 0.1 #  intesity of decoy states
mu2 = 0.298
mu3 = 0.422 # intensity of the 1 in  X window

p_1 = 0.846 # probability of weak decoy state 
p_2 = 0.076 # probability of strong decoy state

p_x = 1 - 0.735 #probability of decoy state
p_z = 1 - 0.269 # probability of not sending in signal base

loss_1 = math.exp(-100/22) #losses for both distances, as the distances do not have to be equal
loss_2 = math.exp(-100/22)

phase_shift_average = 0.01


n_phaseSlice = 16 #number of phase slices
s_phaseSlice = 2 *math.pi /n_phaseSlice # size of a phase slice

batch_size = 1000 # defines the size for each batch of single photon events

#variables for export
error_rate = 0


def generate_random_phaseShift(length):
    psi_AB = np.random.normal(loc=0, scale=phase_shift_average, size=length)
    for i in range(length):
        psi_AB[i] = psi_AB[i] % (math.pi *2)
    
    return psi_AB

def generate_random_Qbits(seed, length):
    random.seed(seed)



    #generate random qubits
    bit_vector = np.zeros(length)
    phase_vector = np.zeros(length)
    amplitude_vector = np.zeros(length)
    window_vector = np.zeros(length)


    # 1 = X window = signal window 
    # 0 = Z window = decoy window
    for i in range(0,length):
        if random.random() < p_x:
            window_vector[i] = 0
            j = random.random()                
            if j < p_1:
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu1)
            elif j < p_1 + p_2:
                bit_vector[i] = 2
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu2)
            else :
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
        else:
            window_vector[i] = 1

            if random.random() < p_z:
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
            else: 
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = np.sqrt(mu3)
        
    
    qbits = [window_vector,bit_vector,phase_vector, amplitude_vector]
    return qbits



def run_trial(alice_bits,bob_bits,start,finish):
    length = finish-start
    prog = sf.Program(2)

    phase_A = prog.params("alice_p")
    phase_B = prog.params("bob_p")
    amplitude_A = prog.params("alice_A")
    amplitude_B = prog.params("bob_A")

    theta = np.float32(np.pi / 4)
    phi = np.float32(np.pi / 2)

    with prog.context as q:
        # State preparation

        ops.Coherent(amplitude_A, phase_A) | q[0]
        ops.Coherent(amplitude_B, phase_B) | q[1]


        # Channel loss
        if loss_1 > 0:
            ops.LossChannel(loss_1) | q[0]
            ops.LossChannel(loss_2) | q[1]

        # Interference
        ops.BSgate(theta,phi) | (q[0], q[1]) 

        # Detection
        ops.MeasureFock() | q

    #eng = sf.Engine("fock", backend_options={"cutoff_dim": 10}) # use fock to access single Photons
    eng = sf.Engine("tf", backend_options={
        "batch_size": length, 
        "cutoff_dim": 10 
    })
    result = eng.run(prog, args={
        "alice_p": alice_bits[2][start:finish].astype(np.float32),
        "bob_p": bob_bits[2][start:finish].astype(np.float32),
        "alice_A":alice_bits[3][start:finish].astype(np.float32),
        "bob_A":bob_bits[3][start:finish].astype(np.float32),
    })

    measures = np.empty((length, 2))
    for i in range(length):
        measures[i][0]=int(result.samples[i][0][0])
        measures[i][1]=int(result.samples[i][0][1])
    for j in range(length):
        for i in range(int(measures[j][0])):
            if random.random() > eta1:
                measures[j][0] -= 1
        for i in range(int(measures[j][1])):
            if random.random() > eta2:
                measures[j][1] -= 1

        # dark counts. is an additional photon
        if random.random() < pd0:
            measures[j][0] += 1
        if random.random() < pd1:
            measures[j][1] += 1
    return measures


def calc_ez(alice_bits,bob_bits,signal_bits,n,counter_signal_window):
    if n > len(signal_bits):
        raise ValueError("too many error test bits for key length")
    if n == 0:
        return signal_bits, 0
    arr2 = signal_bits.copy()
    np.random.shuffle(arr2)
    arr2 = arr2[:n]
    arr = signal_bits[n:]
    counter_NN = 0
    counter_SN = 0
    counter_SS = 0
    for i in range(len(arr2)):
        if alice_bits[1][arr2[i][0]] == 1 and bob_bits[1][arr2[i][0]] == 1:
            counter_SS += 1
            alice_bits[2][arr2[i][0]] - bob_bits[2][arr2[i][0]] < math.pi/2
        elif alice_bits[1][arr2[i][0]] == 0 and bob_bits[1][arr2[i][0]] == 0:
            counter_NN += 1     
        else:
            counter_SN += 1
    print("---------------------------------------")
    print("NN in signal window: " + str(counter_NN))
    print("NS/SN in signal window: " + str(counter_SN/counter_signal_window))
    print("SS in signal window: " + str(counter_SS/counter_signal_window))
    Ez = (counter_NN + counter_SS) / n
    print("Error rate of subset: " + str(Ez))
    print("---------------------------------------")
    return arr , Ez


def calc_ex(alice_bits,bob_bits,decoy_bits):
    if len(decoy_bits) == 0:
        print("no decoy bits")
        return
    s0 = 0
    smu1 = 0
    smu2 = 0
    emu1 = 0
    emu2 = 0
    for i in range(len(decoy_bits)):
        x = decoy_bits[i][0]
        if alice_bits[1][x] == 0 and bob_bits[1][x] == 0:
            s0 +=1
        elif alice_bits[1][x] == 1 and bob_bits[1][x] == 1:
            smu1 +=1
            if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) < 0 and decoy_bits[i][1] == 0:
                emu1 +=1
            if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) > 0 and decoy_bits[i][1] == 1:
                emu1 +=1
        elif alice_bits[1][x] == 2 and bob_bits[1][x] == 2:
            smu2 +=1
            if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) < 0 and decoy_bits[i][1] == 0:
                emu2 +=1
            if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) > 0 and decoy_bits[i][1] == 1:
                emu2 +=1
    emu1 = emu1/smu1
    emu2 = emu2/smu2
    s0 = s0/len(decoy_bits)
    smu1 = smu1/len(decoy_bits)
    smu2 = smu2/len(decoy_bits)
    
    p2mu1 = 2 * mu1 * mu1 * math.pow(math.e,-2*mu1)
    p2mu2 = 2 * mu2 * mu2 * math.pow(math.e,-2*mu2)
    p1mu1 = 2 * mu1 * math.pow(math.e,-2*mu1)
    p1mu2 = 2 * mu2 * math.pow(math.e,-2*mu2)
    p0mu1 = math.pow(math.e,-2*mu1)
    p0mu2 = math.pow(math.e,-2*mu2)
    s1 = 0
    if p2mu2*p1mu1 - p2mu1*p1mu2 != 0:
        s1 = (p2mu2 * (smu1-p0mu1 * s0) - p2mu1 * (smu2 - p0mu2*s0)) / (p2mu2*p1mu1 - p2mu1*p1mu2)
    if s1 != 0:
        ex1 =( emu1 * smu1 - math.pow(math.e , -2 * mu1)*s0/2)/(math.pow(math.e, -2 * mu1)* 2 * mu1 * s1)
        ex2 =( emu2 * smu2 - math.pow(math.e , -2 * mu2)*s0/2)/(math.pow(math.e, -2 * mu2)* 2 * mu2 * s1)
        print("---------------------------------------")
        print("emu1: " + str(emu1))
        print("emu2: " + str(emu2))
        print("exmu1: " + str(ex1))
        print("exmu2: " + str(ex2))
        print("---------------------------------------")
    else:
        print("---------------------------------------")
        print("warning! not enough decoy window data.")
        print("---------------------------------------")
    




def aftercomm(alice_bits,bob_bits,measures,length,psi_AB,cls):
    counter_signal_window = 0 # counts how many signal windows  exist
    signal_bits = []
    decoy_bits = []
    
    #determine weather the events are effective
    for i in tqdm(range(length),"effective events"):
        # still have to figure out what to do with the phase
        if alice_bits[0][i] == 0 and bob_bits[0][i] == 0: # check whether both use the Z basis
            if alice_bits[1][i] == bob_bits[1][i]:
                if abs(math.cos(alice_bits[2][i]-bob_bits[2][i] + psi_AB[i])) <= abs(s_phaseSlice):
                    if measures[i][0] == 0 and measures[i][0] != measures[i][1]: #only if detector 2 clicks
                        decoy_bits.append((i,1))
                    elif measures[i][1] == 0 and measures[i][0] != measures[i][1]: #only if detector 1 clicks
                        decoy_bits.append((i,0))
        elif alice_bits[0][i] == 1 and bob_bits[0][i] == 1: #check whether both use the x basis
            counter_signal_window += 1
            #check phase slices
            #if math.floor(alice_bits[2][i]/s_phaseSlice) == math.floor(bob_bits[2][i]/s_phaseSlice): if you actually slice the thing in the number of slices
            #if abs(alice_bits[2][i] - bob_bits[2][i] - psi_AB[i]) < s_phaseSlice or abs(alice_bits[2][i] - bob_bits[2][i] - psi_AB[i] - math.pi) < s_phaseSlice:
                # chekc if only one detector clicked
            if measures[i][0] == 0 and measures[i][0] != measures[i][1]:
                signal_bits.append((i,1))
            elif measures[i][1] == 0 and measures[i][0] != measures[i][1]:
                signal_bits.append((i,0))
    print("---------------------------------------")
    print("signal bits " + str(len(signal_bits)))
    print("decoy bits: " + str(len(decoy_bits)))


    # do analysis on the signal window
    n = min(1000, len(signal_bits)*0.05)
    if n< 100:
        print("Warning, very few signal bits!")
    signal_bits = self.calc_ez(alice_bits,bob_bits,signal_bits,n,counter_signal_window)
    if len(decoy_bits) > 0:
        cls.calc_ex(alice_bits,bob_bits,decoy_bits)
    else:
        print("there are no decoy bits lol!")

    #generate keys
    alice_key = np.zeros(len(signal_bits))
    bob_key = np.zeros(len(signal_bits))
    for i in range(len(alice_key)):
        if alice_bits[1][signal_bits[i][0]] == 1:
            alice_key[i] = 1
        if bob_bits[1][signal_bits[i][0]] == 0:
            bob_key[i] = 1
    tf_utils.print_error_rate(alice_key,bob_key)
    tf_utils.make_heatmap(measures,"figures/heatmap.png")

    #take some random bits and do some error analysis


    # Active odd parity paring 
    pairs = tf_utils.map_pairs(len(signal_bits))
    #bit mask that if = 1 keeps the bit
    drop_bits = np.zeros(len(signal_bits))
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
    if len(signal_bits) != 0:
        print("percentage of original key: " + str(aopp_length/len(signal_bits)))
    tf_utils.print_error_rate(alice_key_aopp,bob_key_aopp)
    

def tf_communicate(alice_seed, bob_seed,length,path):
    alice_bits = generate_random_Qbits(alice_seed,length)
    bob_bits = generate_random_Qbits(bob_seed,length)
    psi_AB = generate_random_phaseShift(length)
    print("generated all the random numbers")
    measures = np.empty((0, 2))
    for i in tqdm(range(math.ceil(length/batch_size)),"twinflied"): 
        temp =run_trial(alice_bits,bob_bits,i*batch_size,min((i+1)*batch_size,length))
        measures = np.concatenate((measures, temp), axis=0)
    
    print(len(measures))
    sent_photons =sum(alice_bits[3]) + sum(bob_bits[3])
    print("sent photons: " + str(sent_photons) )
    rec_photons = tf_utils.make_heatmap(measures,"figures/heatmap3.png")
    print(rec_photons/sent_photons)
    np.savez(path,
            first=alice_bits,
            second=bob_bits,
            third=measures,
            fourth=length,
            fifth= psi_AB)
    #data = np.load("output.npz")
    #aftercomm(data["first"],data["second"],data["third"],data["fourth"])

def tf_communicat_load(path="output_long.npz"):
    data = np.load(path)
    aftercomm(data["first"],data["second"],data["third"],data["fourth"],data["fifth"])

#tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),1000000,"decoy_signal_loss_2.npz")

tf_communicat_load("decoy_signal_loss_2.npz")
#print(generate_random_phaseShift(100))
#tf_communicat_load("only_decoy.npz")
#run_trial(0.1,0.1,0,0,0)

#noise(generate_random_Qbits(generate_seed,10,0.5,0.5))