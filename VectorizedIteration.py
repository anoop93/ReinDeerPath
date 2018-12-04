import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import operator
import pickle
import os

output = pd.read_csv('submission_ver_i5.csv')
data = pd.read_csv('data/cities.csv', index_col='CityId')

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
    step = 10 - idx%10
    if window_size>10 and step ==0:
        step = 10
    return step

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

def getFastDistance(ids, window_size, idx):
    data_dict = {i: tuple(data.loc[i].values) for i in ids[0]}
    local_prime_list = [i for i in ids[0] if prime_list[i]]
    f = lambda i: data_dict[i]
    non_prime_distance = 0.1 ## sqrt(1.1)
    mat = np.array(np.vectorize(f)(ids))

    mat1 = mat[:,:,1:]
    mat = mat[:,:,:-1]

    distance = np.linalg.norm(mat - mat1, axis=0)
    primeStep = getstep(idx, window_size)
    if primeStep !=0 and primeStep < window_size:
        f1 = lambda i: int(i in local_prime_list)
        prime_col = np.vectorize(f1)(ids[:,primeStep])
        distance[:,primeStep-1] = distance[:,primeStep-1] + prime_col*distance[:,primeStep-1]*non_prime_distance
    distance = np.sum(distance, axis=1)
    return distance

def getCityIdsPermutation(per, ids, window_size, idx):
    f = lambda i: ids[i]
    new_ids = np.vectorize(f)(per)
    distance = getFastDistance(new_ids, window_size, idx)
    return new_ids, distance


def arrangeCityIds(ids, window_size, idx):
    window = list(range(window_size))
    if len(ids) == window_size:
        lastIndex = window_size - 1
        per = np.insert(list(itertools.permutations(window[1:-1])), 0, values=0, axis=1)
        per = np.insert(per, lastIndex, lastIndex, axis=1)
        new_ids, distance = getCityIdsPermutation(per, ids, window_size, idx)
        min_distance_index = np.argmin(distance)
        return new_ids[min_distance_index]
    return ids

def primeArrange():
    i = 0
    global new_output
    exists = os.path.isfile('new_output.pickle')
    if exists:
        with open('new_output.pickle', 'rb') as f:
            new_output = pickle.load(f)
        with open('iteration.pickle', 'rb') as f:
            i = pickle.load(f)

    data_len = len(new_output)
    save_interval = 200
    window_size = 30
    #lower_index = int(window_size/2)
    end_index = data_len - window_size
    pbar = tqdm(total=data_len)
    if i!=0:
        pbar.update(i)
    while i <= end_index:
        #if (i+1)%10 == 0 and i+2 < data_len:
        # if i <= end_index:
        ids = new_output[i:i+window_size]
        #isPrime, primeIdx = oneSidePrimeCheck(ids)
        #start = time.time()
        prime = False
        for j in ids[1:]:
            if prime_list[j]:
                prime = True
                break
        if not prime:
            i += 1
            pbar.update(1)
            continue
        # output = list(arrangeCityIds(ids, window_size, i))
        # new_output[i:i+window_size] = output
        new_output[i:i + window_size] = list(arrangeCityIds(ids, window_size, i))
        #print('arrangeCityIds: ', time.time() - start)
        # if operator.eq(ids, output):
        #     i += 5
        #     pbar.update(5)
        # else:
        i += 1
        pbar.update(1)
        if i>save_interval:
            save_interval +=200
            with open('new_output.pickle', 'wb') as f:
                pickle.dump(new_output, f)
            with open('iteration.pickle', 'wb') as f:
                pickle.dump(i, f)

new_output = [i[0] for i in output.values]
version = 6
while True:
    previous_output = new_output
    primeArrange()
    pd.DataFrame({'Path': new_output}).to_csv('submission_ver_'+str(version)+'.csv', index=False)
    # count = 0
    # for i, cityId in enumerate(new_output):
    #     if cityId != previous_output[i]:
    #         count += 1
    # version += 1
    # if count == 0:
    break