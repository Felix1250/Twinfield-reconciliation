import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib
import matplotlib as mpl
import matplotlib.patches as patches  
from numba import njit
import math

@njit
def generate_seed():
    return random.getrandbits(32)

def __pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def pos_neg_diff(a,b):
    return math.sin(a-b)#(a - b+ math.pi) % (2*math.pi) - math.pi


def map_pairs(length):
    if length %2 == 1:
        length -=1
    lst = []
    for i in range(length):
        lst.append(i)
    pairs = []
    while lst:
        rand1 = __pop_random(lst)
        rand2 = __pop_random(lst)
        pair = rand1, rand2
        pairs.append(pair)
    return pairs

def print_error_rate(array1,array2,signal_bits = []):
    if len(array1) == 0:
        return 0,0
    counter = 0
    for i in range(len(array1)):
        if array1[i] != array2[i]:
            counter +=1
    print("faulty bits: " + str(counter))
    print("error rate: " + str(counter/len(array1)))
    return counter/len(array1), counter



def make_heatmap(array,path):
    count_photons = 0
    for i in range(len(array)):
        count_photons+= array[i][0] + array[i][1]
    print("measured photons: " + str(count_photons))



    heatmap = np.zeros((6,6))
    for i in range(len(array)):
        if array[i][0] <= 5 and array[i][1] <= 5 :

            heatmap[int(array[i][0])][int(array[i][1])] += 1




    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)

    # Show all ticks and label them with the respective list entries
    #ax.set_xticks(range(len(farmers)), labels=farmers,
    #            rotation=45, ha="right", rotation_mode="anchor")
    #ax.set_yticks(range(len(vegetables)), labels=vegetables)

    # Loop over data dimensions and create text annotations.
    for i in range(len(heatmap[0])):
        for j in range(len(heatmap)):
            text = ax.text(j, i, heatmap[i, j],
                        ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)


    ax.set_title("detecor clicks")
    plt.xlabel("Photons on side D0")
    plt.ylabel("Photons on side D1")
    fig.tight_layout()
    plt.savefig(path)
    plt.close()

    return count_photons
    #plt.show()

def make_heatmap_array(array,path,n):
    array = array[0:n,0:n]
    fig, ax = plt.subplots()
    im = ax.imshow(array)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ax.text(
                j, i,
                f"{array[i, j]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    
    rect = patches.Rectangle(
        xy=(0.5, -0.485),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=3,             # Cover 4 columns (1, 2, 3, 4)
        height=0.985,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='lime',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='lime',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect)
    rect2 = patches.Rectangle(
        xy=(-0.485, 0.5),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=0.985,             # Cover 4 columns (1, 2, 3, 4)
        height=3,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='lime',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='lime',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect2)
    rect3 = patches.Rectangle(
        xy=(0.515, 0.515),      # Start at x=0.5 (left edge of col 1), y=-0.5 (top edge of row 0)
        width=2.985,             # Cover 4 columns (1, 2, 3, 4)
        height=2.985,            # Cover 1 row (0)
        linewidth=3,         # Thickness of the border
        edgecolor='red',    # Bright green color ('g' works too, but 'lime' pops more on heatmaps)
        facecolor='red',     # Keep the inside transparent
        alpha=0.2
    )
    ax.add_patch(rect3)

    #fig.colorbar(im, ax=ax)

    ax.set_title("detecor clicks")
    plt.xlabel("Photons on side D0")
    plt.ylabel("Photons on side D1")
    fig.tight_layout()
    plt.savefig(path)

def factorial(n1):
  c1 = 1
  for i1 in range(n1):
    c1 = c1 * (n1 - i1)
  return c1


def possoin_graph(mu = 2.1):
    a= []
    labels = [0,1,2,3,4,5,6,7,"...","∞"]
    x_positions = np.arange(len(labels))
    
    for i in range(10):
        a.append( math.pow(mu,i) / factorial(i) *math.pow(math.e,-mu))
    fig, ax = plt.subplots()
    for spine in ["top", "right"]: #, "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    #ax.spines["left"].set_position("zero")
    #ax.spines["bottom"].set_position("zero")
    plt.plot(a, label="Sine Wave", color="blue")
    ax.set_xticks(x_positions,labels)
    ax.annotate('', xy=(1, 0), xycoords='axes fraction', xytext=(0, 0), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2, shrinkA=0, shrinkB=0))

# 2. Up arrow for Y-axis (at the end of the left spine)
    ax.annotate('', xy=(0, 1), xycoords='axes fraction', xytext=(0, 0), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2, shrinkA=0, shrinkB=0))
    plt.xlabel("Photon count")
    plt.ylabel("probability")  
    plt.rcParams.update({'font.size': 16})
    ax.set_title("Probability of getting n Photons when \u03bc = 2.1")
    #ax.set_yticks([])
    plt.show()

#possoin_graph()

def plob():
    a = []
    for i in range(10):
        loss_db_per_km = 0.2
        distance = i*50  # km
        total_loss_db = loss_db_per_km * distance
        a.append(−log(1-total_loss_db))

    plt.plot(a,label="plob")
    plt.yscale('log')
    plt.show()

plob()


def shannon_entr(x):
    return -x * math.log2(x) - (1-x) * math.log2(1-x)

def secure_key_rate(N_bits,n_1,n_t,eph,f,E_t):
    '''
    N_bits is the number of pulses sent
    n_1 is the lower bound number of effective events (after AOPP if it is applied)
    nt is the actual amount
    shannon_entr is the shannon entropy
    eph is the phase flip rate
    E_t is the bit lflip rate
    '''
    R_tail = 1/N_bits *(2*math.log2(2/e_cor) + 4 * math.log2(1/(math.sqrt(2) *e_pa * e_hat)) + 2 * math.log2(n_vy + n_yv))

    R = 1/N_bits*(n_1*(1-shannon_entr(eph)) - f * n_t * shannon_entr(E_t)) - R_tail
    return R
