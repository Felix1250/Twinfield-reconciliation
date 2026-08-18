import random
import numpy as np
import math
import strawberryfields as sf
def generate_seed():
    return random.getrandbits(32)

def generate_random_Qbits(seed, length, px, pz):
    random.seed(seed)

    mu1 = 1
    mu2 = 2
    mu3 = 1 # intensity of the 1 in  X window

    #generate random qubits
    bit_vector = np.zeros(length)
    phase_vector = np.zeros(length)
    amplitude_vector = np.zeros(length)
    window_vector = np.zeros(length)


    # 0 = X window = signal window 
    # 1 = Z window = decoy window
    for i in range(0,length):
        if random.random() < px:
            window_vector[i] = "0"
            j = random.randint(0,2)
            if j == 0:
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
            elif j == 1:
                bit_vector[i] = 0
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = mu1
            else:
                bit_vector[i] = 0
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = mu2
        else:
            window_vector[i] = "1"
            if random.random() < pz:
                bit_vector[i] = 0
                phase_vector[i] = 0
                amplitude_vector[i] = 0
            else: 
                bit_vector[i] = 1
                phase_vector[i] = random.uniform(0,2*math.pi)
                amplitude_vector[i] = mu3

        bit_vector[i] = random.randint(0,1)
        phase_vector[i] = random.uniform(0,2*math.pi)
        amplitude_vector[i] = random.randint(0,1)
        
    
    qbits = [window_vector,bit_vector,phase_vector, amplitude_vector]
    return qbits

def beamsplitter_events(n,m,nu,gamma):
    p = math.exp(-nu) *(math.pow((nu*gamma),n))/math.factorial(int(n)) * (math.pow(nu*(1-gamma),m))/math.factorial(int(m))
    return p

def beamsplitter_probability(x1,x2,r, theta):
    nu = x1 +x2
    gamma = (x1 * (1-r) + x2*r + 2 * math.sqrt(r*(1-r)*x1*x2)*math.cos(theta))/nu
    #print(nu)
    #print(gamma)
    #print(theta)
    #print(math.factorial(1))
    #print(math.factorial(0))
    #print(math.factorial(2))
    print("-----------")
    #print(beamsplitter_events(0,0,nu,gamma))
    #print(beamsplitter_events(1,0,nu,gamma))
    #print(beamsplitter_events(0,1,nu,gamma))
    print(beamsplitter_events(1,1,nu,gamma))
    print(beamsplitter_events(0,2,nu,gamma))
    print(beamsplitter_events(2,0,nu,gamma))
    #print(beamsplitter_events(2,1,nu,gamma))
    #return p

beamsplitter_probability(1,1,0.5,math.pi/2)
beamsplitter_probability(1,1,0.5,math.pi)
beamsplitter_probability(1,1,0.5,0)


def noise(qbits):
    eta_1 = 0.9  #looks like a n
    eta_2 = 0.9 # different values for each detector, might come in handy later
    Pd_1 = 10^-10 # dark count rate
    Pd_2 = 10^-10   
    r = 0.5 #reflectivity of beamsplitter
    #beamsplitter_probability(1,1,1,0)

    #calculate the probability of the number of photons on each side after the beamsplitter happening



    return qbits

def tf_communicate(alice_seed, bob_seed,length):
    alice_bits = generate_random_Qbits(alice_seed,length)
    bob_bits = generate_random_Qbits(bob_seed,length)




#noise(generate_random_Qbits(generate_seed,10,0.5,0.5))