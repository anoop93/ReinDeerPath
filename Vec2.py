import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import operator
import pickle
import os
import string

output = pd.read_csv('submission_ver_i5_opt_2.csv')
data = pd.read_csv('data/cities.csv', index_col='CityId')

numerals = string.digits + string.ascii_letters

# data['xplusy'] = data['X'] + data['Y']

def is_prime(n):
    if n == 1:
        return False
    i = 2
    while i*i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

numbers = range(data.shape[0])
prime_list = {i:is_prime(i) for i in numbers}

def getstep(idx, window_size):
    stepNumbers = int(window_size/10)+1
    step = 10 - idx%10
    step_list = []
    for i in range(stepNumbers):
        if step < window_size and step !=0:
            step_list.append(step)
        step += 10
    return step_list

# def getFastDistanceL1(ids, window_size, idx):
#     data_dict = {i:data['xplusy'].loc[i] for i in ids[0]}
#     f = lambda i: data_dict[i]
#     non_prime_distance = 0.04880884817 ## sqrt(1.1)
#     mat1 = np.vectorize(f)(ids)
#     mat2 = mat1[:,1:]
#     mat1 = mat1[:,:-1]
#     #distance = np.linalg.norm(vector1.values - vector2.values, axis=1)
#     distance = abs(mat1 - mat2)
#     primeStep = getstep(idx)
#     if primeStep !=0 and primeStep < window_size:
#         f1 = lambda i: int(is_prime(i))
#         prime_col = np.vectorize(f1)(ids[:,primeStep])
#         distance[:,primeStep-1] = distance[:,primeStep-1] + prime_col*distance[:,primeStep-1]*non_prime_distance
# #         distance[primeStep-1] *= 1.1
#     distance = np.sum(distance, axis=1)
#     return distance

# def getFastDistance(ids, window_size, idx):
#     data_dict = {i: tuple(data.loc[i].values) for i in ids[0]}
#     local_prime_list = [i for i in ids[0] if prime_list[i]]
#     f = lambda i: data_dict[i]
#     non_prime_distance = 0.1 ## sqrt(1.1)
#     mat = np.array(np.vectorize(f)(ids))
#
#     mat1 = mat[:,:,1:]
#     mat = mat[:,:,:-1]
#
#     distance = np.linalg.norm(mat - mat1, axis=0)
#     primeStep = getstep(idx, window_size)
#     if primeStep !=0 and primeStep < window_size:
#         f1 = lambda i: int(i in local_prime_list)
#         prime_col = np.vectorize(f1)(ids[:,primeStep])
#         distance[:,primeStep-1] = distance[:,primeStep-1] + prime_col*distance[:,primeStep-1]*non_prime_distance
#     distance = np.sum(distance, axis=1)
#     return distance

# def getCityIdsPermutation(per, ids, window_size, idx):
#     f = lambda i: ids[i]
#     new_ids = np.vectorize(f)(per)
#     distance = getFastDistance(new_ids, window_size, idx)
#     return new_ids, distance

def getDistanceValue(r,c,ids):
    r_id = ids[r]
    c_id = ids[c]
    r_cord = data.loc[r_id].values
    c_cord = data.loc[c_id].values
    distance = np.linalg.norm(r_cord - c_cord)
    return distance


def getDistanceMatrix(ids):
    ids_len = len(ids)
    matrix = np.zeros(shape=(ids_len, ids_len))
    for r in range(ids_len):
        for c in range(r + 1, ids_len):
            matrix[r][c] = getDistanceValue(r,c,ids)
    matrix = matrix + np.transpose(matrix)

    for r in range(ids_len):
        matrix[r][r] = 99999

    return matrix

#
# def baseN(num, b, numerals):
#     return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

def getPermuatationSortedByDistance(ids, disMatrix, minDistaceIndex, nearest, window_size, primeSteps):
    first = 0
    last = window_size - 1
    permutations = []
    permutations_distance = []
    disMat = disMatrix[:-1,:-1]
    primeIdsIndex = [1 if prime_list[i] else 1.1 for i in ids][:-1]
    disColumn = disMatrix[-1]
    diagonalVector = [disMatrix[i][i+1] for i in range(window_size-1)]
    baseDistance = sum([d*1.1 if i+1 in primeSteps and not prime_list[ids[i+1]] else d for i,d in enumerate(diagonalVector)])
    for i in range(nearest): #for min distance index
        mdi = minDistaceIndex[i]
        seq = [first]
        row = 0
        distance = 0
        for j in range(window_size-2): #for min distance matrix
            factor = 1
            if j+1 in primeSteps:
                sortedIndex = np.argsort(disMat[row]*primeIdsIndex)
                factor = 1.1
            else:
                sortedIndex = np.argsort(disMat[row])
            sortedIndex = [idx for idx in sortedIndex if idx not in seq]
            sortedIndex = sortedIndex[mdi[j]-1]
            distance += disMat[row][sortedIndex]*factor
            row = sortedIndex
            seq.append(sortedIndex)
        if window_size-1 in primeSteps and not prime_list[ids[last]]:
            distance += 1.1*disColumn[row]
        else:
            distance += disColumn[row]
        seq.append(last)
        permutations.append(seq)
        permutations_distance.append(distance)
    return permutations, permutations_distance, baseDistance

minDistaceIndex = None

def getMinDistanceIndex(window_size, nearest):
    global minDistaceIndex
    r = 7
    minDistaceIndex = np.array(list(range(r)))
    dic = {i: minDistaceIndex[:-(i + 1)] for i in minDistaceIndex}
    for i in range(0, len(dic) - 1):
        minDistaceIndex = minDistaceIndex.reshape((np.prod(minDistaceIndex.shape), 1)) * 10 + dic[i]
    minDistaceIndex = minDistaceIndex[:nearest]
    base = window_size - 2
    minDistaceIndex = [str(i[0]) for i in minDistaceIndex]
    minDistaceIndex = [list(map(int, list(i))) if len(i) == base else list(map(int, list('0' * (base - len(i)) + i)))
                       for i in minDistaceIndex]
    minDistaceIndex = np.array(minDistaceIndex) + 1


def getPermuations(ids, window_size, nearest, primeSteps):
    disMatrix = getDistanceMatrix(ids)
    permutations, distance, baseDistance = getPermuatationSortedByDistance(ids, disMatrix, minDistaceIndex, nearest, window_size, primeSteps)
    minDistIndex = np.argmin(distance)
    minDistancePerumutation = permutations[minDistIndex]
    if distance[minDistIndex] > baseDistance:
        minDistancePerumutation = list(range(window_size))
    return minDistancePerumutation


def arrangeCityIds(ids, window_size, nearest, primeSteps):
    # window = list(range(window_size))
    if len(ids) == window_size:
        # per = np.insert(list(itertools.permutations(window[1:-1])), 0, values=0, axis=1)
        per = getPermuations(ids, window_size, nearest, primeSteps)
        per = [ids[i] for i in per]
        # seq, distance =
        # per = np.insert(per, lastIndex, lastIndex, axis=1)
        # new_ids, distance = getCityIdsPermutation(per, ids, window_size, idx)
        # min_distance_index = np.argmin(distance)
        return per
    return ids

def primeArrange():
    i = 0
    # global new_output
    # exists = os.path.isfile('new_output.pickle')
    # if exists:
    #     with open('new_output.pickle', 'rb') as f:
    #         new_output = pickle.load(f)
    #     with open('iteration.pickle', 'rb') as f:
    #         i = pickle.load(f)

    data_len = len(new_output)
    # save_interval = 200
    window_size = 35
    #lower_index = int(window_size/2)
    end_index = data_len - window_size
    pbar = tqdm(total=data_len)
    nearest = 1500

    getMinDistanceIndex(window_size, nearest)
    # if i!=0:
    #     pbar.update(i)
    while i <= end_index:
        #if (i+1)%10 == 0 and i+2 < data_len:
        # if i <= end_index:
        ids = new_output[i:i+window_size]
        #isPrime, primeIdx = oneSidePrimeCheck(ids)
        #start = time.time()
        # prime = False
        # for j in ids[1:]:
        #     if prime_list[j]:
        #         prime = True
        #         break
        # if not prime:
        #     i += 1
        #     pbar.update(1)
        #     continue
        # output = list(arrangeCityIds(ids, window_size, i))
        # new_output[i:i+window_size] = output
        primeSteps = getstep(i, window_size)
        # if not operator.eq(list(ids), list(arrangeCityIds(ids, window_size, nearest, primeSteps))):
        #     print('abc')
        outputVal = list(arrangeCityIds(ids, window_size, nearest, primeSteps))

        new_output[i:i + window_size] = outputVal
        #print('arrangeCityIds: ', time.time() - start)
        if operator.eq(list(ids), outputVal):
            k = 20
            i += k
            pbar.update(k)
        else:
            i += 1
            pbar.update(1)
        # if i>save_interval:
        #     save_interval +=200
        #     with open('new_output.pickle', 'wb') as f:
        #         pickle.dump(new_output, f)
        #     with open('iteration.pickle', 'wb') as f:
        #         pickle.dump(i, f)

new_output = output.values.reshape(-1)
version = 3
while True:
    previous_output = new_output
    primeArrange()
    pd.DataFrame({'Path': new_output}).to_csv('submission_ver_i5_opt_'+str(version)+'.csv', index=False)
    # count = 0
    # for i, cityId in enumerate(new_output):
    #     if cityId != previous_output[i]:
    #         count += 1
    # version += 1
    # if count == 0:
    break