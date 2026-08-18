import numpy as np
import random

def __pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def make_odd_parity_pairs(key):
    arr1 = np.arange(len(key))
    #print(arr1)
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
    #print(pairs)
    return pairs

a= np.random.choice([0,1],100)
#print(a)
b = (make_odd_parity_pairs(a))
for i in range(len(b)):
    print("/" + str(a[b[i][0]]) + " " + str(a[b[i][1]]))
