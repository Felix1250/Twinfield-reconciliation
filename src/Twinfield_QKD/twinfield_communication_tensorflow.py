import random
import numpy as np
import math
import strawberryfields as sf
from strawberryfields import ops
import multiprocessing as mp
from tqdm import tqdm
import datetime
import gc


import tf_utils
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# --- Worker function MUST be defined at top-level for multiprocessing spawn/fork ---
def _run_trial_worker(
    queue,
    length,
    loss_1,
    loss_2,
    eta1,
    eta2,
    pd0,
    pd1,
    alice_p,
    bob_p,
    alice_A,
    bob_A,
):

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
        ops.BSgate(theta, phi) | (q[0], q[1])

        # Detection
        ops.MeasureFock() | q

    eng = sf.Engine(
        "tf", backend_options={"batch_size": length, "cutoff_dim": 8}
    )
    result = eng.run(
        prog,
        args={
            "alice_p": alice_p,
            "bob_p": bob_p,
            "alice_A": alice_A,
            "bob_A": bob_A,
        },
    )

    measures = np.empty((length, 2))
    for i in range(length):
        measures[i][0] = int(result.samples[i][0][0])
        measures[i][1] = int(result.samples[i][0][1])

    for j in range(length):
        for i in range(int(measures[j][0])):
            if random.random() > eta1:
                measures[j][0] -= 1
        for i in range(int(measures[j][1])):
            if random.random() > eta2:
                measures[j][1] -= 1

        # Dark counts
        if random.random() < pd0:
            measures[j][0] += 1
        if random.random() < pd1:
            measures[j][1] += 1

    del eng
    del prog
    tf.keras.backend.clear_session()

    # Put the resulting matrix into the multiprocessing queue
    queue.put(measures)


class Twinfield:
    #variables for generating
    pd0 = 0#math.pow(10,-50) #darkcount of each detector individually
    pd1 = 0#math.pow(10,-50)

    eta1 = 1#0.95 #detection efficiency of the detectors
    eta2 = 1#0.95 

    mu1 = 0.1 #  intesity of decoy states
    mu2 = 0.298
    mu3 = 0.422 # intensity of the 1 in  X window

    p_1 = 0.846 # probability of weak decoy state 
    p_2 = 0.076 # probability of strong decoy state

    p_x = 1 - 0.735 #probability of decoy state
    p_z = 1 - 0.269 # probability of not sending in signal base

    loss_1 = 0#math.exp(-100/22) #losses for both distances, as the distances do not have to be equal
    loss_2 = 0#math.exp(-100/22)

    phase_shift_average = 0.01

    # variables for aftercomm

    n_phaseSlice = 16 #number of phase slices
    s_phaseSlice = 1 /n_phaseSlice

    batch_size = 16000 # defines the size for each batch of single photon events
    n_percentage = 0.05

    #variables for export
    _error_rate = 0
    _emu1 = 0
    _emu2 = 0
    _emu3 = 0
    _ex1 = 0
    _ex2 = 0
    _decoy_length = 0
    
    _error_number = 0
    _signal_length = 0

    _N_pulses = 0
    

    def generate_random_phaseShift(cls,length):
        psi_AB = np.random.normal(loc=0, scale=cls.phase_shift_average, size=length)
        for i in range(length):
            psi_AB[i] = psi_AB[i] % (math.pi *2)
        
        return psi_AB

    def generate_random_Qbits(cls,seed, length):
        random.seed(seed)



        #generate random qubits
        bit_vector = np.zeros(length)
        phase_vector = np.zeros(length)
        amplitude_vector = np.zeros(length)
        window_vector = np.zeros(length)


        # 1 = X window = signal window 
        # 0 = Z window = decoy window
        for i in range(0,length):
            if random.random() < cls.p_x:
                window_vector[i] = 0
                j = random.random()                
                if j < cls.p_1:
                    bit_vector[i] = 1
                    phase_vector[i] = random.uniform(0,2*math.pi)
                    amplitude_vector[i] = np.sqrt(cls.mu1)
                elif j < cls.p_1 + cls.p_2:
                    bit_vector[i] = 2
                    phase_vector[i] = random.uniform(0,2*math.pi)
                    amplitude_vector[i] = np.sqrt(cls.mu2)
                else :
                    bit_vector[i] = 0
                    phase_vector[i] = 0
                    amplitude_vector[i] = 0
            else:
                window_vector[i] = 1

                if random.random() < cls.p_z:
                    bit_vector[i] = 0
                    phase_vector[i] = 0
                    amplitude_vector[i] = 0
                else: 
                    bit_vector[i] = 1
                    phase_vector[i] = random.uniform(0,2*math.pi)
                    amplitude_vector[i] = np.sqrt(cls.mu3)
            
        
        qbits = [window_vector,bit_vector,phase_vector, amplitude_vector]
        return qbits

    def run_trial(self, alice_bits, bob_bits, start, finish):
        length = finish - start

        # Extract slicing parameters cast to float32
        alice_p = alice_bits[2][start:finish].astype(np.float32)
        bob_p = bob_bits[2][start:finish].astype(np.float32)
        alice_A = alice_bits[3][start:finish].astype(np.float32)
        bob_A = bob_bits[3][start:finish].astype(np.float32)

        # Create inter-process queue to retrieve trial data
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        # Run backend code in an isolated process
        p = ctx.Process(
            target=_run_trial_worker,
            args=(
                queue,
                length,
                self.loss_1,
                self.loss_2,
                self.eta1,
                self.eta2,
                self.pd0,
                self.pd1,
                alice_p,
                bob_p,
                alice_A,
                bob_A,
            ),
        )
        p.start()

        # Get results and ensure process termination
        measures = queue.get()
        p.join()

        return measures


    def calc_ez(cls,alice_bits,bob_bits,signal_bits,n,counter_signal_window):
        if n > len(signal_bits):
            print("too many error test bits for key length")
            return signal_bits, 0
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
        return arr,Ez

    def calc_ex(cls,alice_bits,bob_bits,decoy_bits):
        if len(decoy_bits) == 0:
            print("no decoy bits")
            return
        s0 = 0
        smu1 = 0
        smu2 = 0
        max = 0
        for i in range(len(decoy_bits)):
            
            x = decoy_bits[i][0]
            if alice_bits[2][x]- bob_bits[2][x] > max:
                max = alice_bits[2][x]- bob_bits[2][x] 
            if alice_bits[1][x] == 0 and bob_bits[1][x] == 0:
                s0 +=1
            elif alice_bits[1][x] == 1 and bob_bits[1][x] == 1:
                smu1 +=1
                if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) < 0 and decoy_bits[i][1] == 0:
                    cls._emu1 +=1
                if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) > 0 and decoy_bits[i][1] == 1:
                    cls._emu1 +=1
            elif alice_bits[1][x] == 2 and bob_bits[1][x] == 2:
                smu2 +=1
                if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) < 0 and decoy_bits[i][1] == 0:
                    cls._emu2 +=1
                if tf_utils.pos_neg_diff(alice_bits[2][x], bob_bits[2][x]) > 0 and decoy_bits[i][1] == 1:
                    cls._emu2 +=1
        if smu1 != 0:
            cls._emu1 = cls._emu1/smu1
        if smu2 != 0:
            cls._emu2 = cls._emu2/smu2
        s0 = s0/len(decoy_bits)
        smu1 = smu1/len(decoy_bits)
        smu2 = smu2/len(decoy_bits)
        
        p2mu1 = 2 * cls.mu1 * cls.mu1 * math.pow(math.e,-2*cls.mu1)
        p2mu2 = 2 * cls.mu2 * cls.mu2 * math.pow(math.e,-2*cls.mu2)
        p1mu1 = 2 * cls.mu1 * math.pow(math.e,-2*cls.mu1)
        p1mu2 = 2 * cls.mu2 * math.pow(math.e,-2*cls.mu2)
        p0mu1 = math.pow(math.e,-2*cls.mu1)
        p0mu2 = math.pow(math.e,-2*cls.mu2)
        s1 = 0
        if p2mu2*p1mu1 - p2mu1*p1mu2 != 0:
            s1 = (p2mu2 * (smu1-p0mu1 * s0) - p2mu1 * (smu2 - p0mu2*s0)) / (p2mu2*p1mu1 - p2mu1*p1mu2)
        if s1 != 0:
            cls._ex1 =( cls._emu1 * smu1 - math.pow(math.e , -2 * cls.mu1)*s0/2)/(math.pow(math.e, -2 * cls.mu1)* 2 * cls.mu1 * s1)
            cls._ex2 =( cls._emu2 * smu2 - math.pow(math.e , -2 * cls.mu2)*s0/2)/(math.pow(math.e, -2 * cls.mu2)* 2 * cls.mu2 * s1)
            print("---------------------------------------")
            print("_emu1: " + str(cls._emu1))
            print("_emu2: " + str(cls._emu2))
            print("exmu1: " + str(cls._ex1))
            print("exmu2: " + str(cls._ex2))
            print("max: " + str(max/math.pi))
            print("---------------------------------------")
        else:
            print("---------------------------------------")
            print("warning! not enough decoy window data.")
            print("---------------------------------------")
        
    def aftercomm(cls,alice_bits,bob_bits,measures,length,psi_AB):
        cls.s_phaseSlice = 1* math.pi /cls.n_phaseSlice
        print("s phase",str(cls.s_phaseSlice))
        print("n phase",str(cls.n_phaseSlice))
        counter_signal_window = 0 # counts how many signal windows  exist
        signal_bits = []
        decoy_bits = []

        cls._N_pulses = len(alice_bits[0])
        #determine weather the events are effective
        for i in tqdm(range(length),"effective events"):
            # still have to figure out what to do with the phase
            if alice_bits[0][i] == 0 and bob_bits[0][i] == 0: # check whether both use the Z basis
                if alice_bits[1][i] == bob_bits[1][i]:
                    if 1-abs(math.sin(alice_bits[2][i]-bob_bits[2][i] + psi_AB[i])) <= abs(cls.s_phaseSlice): 
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
        cls._signal_length = len(signal_bits)
        cls._decoy_length = len(decoy_bits)
        print("signal bits " + str(cls._signal_length))
        print("decoy bits: " + str(cls._decoy_length))
        


        # do analysis on the signal window
        n = max(1000, math.floor(len(signal_bits)*cls.n_percentage))
        signal_bits, ez = cls.calc_ez(alice_bits,bob_bits,signal_bits,n,counter_signal_window)
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
        cls._error_rate ,cls._error_number= tf_utils.print_error_rate(alice_key,bob_key)
        print("error rate" + str(cls._error_rate))
        tf_utils.make_heatmap(measures,"figures/heatmap.png")    

        return alice_key, bob_key
        
    def tf_communicate(cls,alice_seed, bob_seed,length,path = "temp.npz"):
        alice_bits = cls.generate_random_Qbits(alice_seed,length)
        bob_bits = cls.generate_random_Qbits(bob_seed,length)
        psi_AB = cls.generate_random_phaseShift(length)
        #asprint("generated all the random numbers")
        measures = np.empty((length, 2))
        for i in tqdm(range(math.ceil(length/cls.batch_size)),"twinflied"): 
            temp =cls.run_trial(alice_bits,bob_bits,i*cls.batch_size,min((i+1)*cls.batch_size,length))
            #measures = np.concatenate((measures, temp), axis=0)
            measures[i*cls.batch_size:min((i+1)*cls.batch_size, length)] = temp

            if i % 100 == 0: #garbage collection every 100 batches to avoid memory issues
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                gc.collect()
        
        sent_photons =sum(alice_bits[3]) + sum(bob_bits[3])
        rec_photons = tf_utils.make_heatmap(measures,"figures/heatmap3.png")
        #print(rec_photons/sent_photons)
        settings = [cls.pd0 ,cls.pd1, cls.eta1, cls.eta2, cls.mu1, cls.mu2, cls.mu3, cls.p_1, cls.p_2, cls.p_x, cls.p_z, cls.loss_1, cls.loss_2, cls.phase_shift_average]
        np.savez(path,
                first=alice_bits,
                second=bob_bits,
                third=measures,
                fourth=length,
                fifth= psi_AB,
                sixth= settings)
        #data = np.load("output.npz")
        #aftercomm(data["first"],data["second"],data["third"],data["fourth"])

    def tf_communicat_load(cls,path="temp.npz"):
        data = np.load(path)
        return cls.aftercomm(data["first"],data["second"],data["third"],data["fourth"],data["fifth"])
    
    def tf_communicate_load_settings(cls, path = "temp.npz"):
        data = np.load(path)
        cls.pd0 =data["sixth"][0]
        cls.pd1=data["sixth"][1]
        cls.eta1=data["sixth"][2]
        cls.eta2=data["sixth"][3]
        cls.mu1=data["sixth"][4]
        cls.mu2=data["sixth"][5]
        cls.mu3=data["sixth"][6]
        cls.p_1=data["sixth"][7]
        cls.p_2=data["sixth"][8]
        cls.p_x=data["sixth"][9]
        cls.p_z=data["sixth"][10]
        cls.loss_1=data["sixth"][11]
        cls.loss_2=data["sixth"][12]
        cls.phase_shift_average=data["sixth"][13]

#tf_communicate(tf_utils.generate_seed(),tf_utils.generate_seed(),1000000,"decoy_signal_loss_2.npz")
#twin = Twinfield()
#twin.tf_communicat_load("decoy_signal_loss.npz")
#print(generate_random_phaseShift(100))
#tf_communicat_load("only_decoy.npz")
#run_trial(0.1,0.1,0,0,0)

#noise(generate_random_Qbits(generate_seed,10,0.5,0.5))