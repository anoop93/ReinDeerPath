import pandas as pd
import numpy as np
from progressbar import ProgressBar
import time
import math
import itertools
# import ipdb
from operator import eq

data = pd.read_csv('data/cities.csv', index_col='CityId')
output = pd.read_csv('submission.csv')

bar = ProgressBar()

def is_prime(n):
    if n == 1:
        return False
    i = 2
    while i*i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

def getDistance():
    total_distance = 0
    from_index = None
    step = 1
    for i in bar(output.values):
        i = i[0]
        if from_index == None:
            from_index = i
            print(from_index)
        else:
            to_index = i
            distance = np.linalg.norm(np.array(data.loc[from_index]) - np.array(data.loc[to_index]))

            if step%10 == 0 and not is_prime(to_index):
                distance = distance * 1.1
            total_distance += distance
            from_index = to_index
        step += 1
    return total_distance

def get_score(output):
    plist = [i for i in range(len(output)) if is_prime(i)]
    cities = pd.read_csv('data/cities.csv')
    all_ids = cities['CityId'].values
    all_x = cities['X'].values
    all_y = cities['Y'].values

    arr = dict()
    for i, id in enumerate(all_ids):
        arr[id] = (all_x[i], all_y[i])

    score = 0.0
    s = output
    for i in bar(range(0, len(s)-1)):
        p1 = arr[s[i]]
        p2 = arr[s[i+1]]
        stepSize = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        if ((i + 1) % 10 == 0) and (s[i] not in plist):
            stepSize *= 1.1
        # print(stepSize)
        score += stepSize
    return score

def getstep(window_size, idx):
    indices = [i+idx for i in range(window_size)]
    step = 10 - idx%10
    return step

def getNewDistance(ids, window_size, idx):
    if len(ids) == window_size:
        total_distance = 0
        from_index = None
        primeStep = getstep(window_size, idx)
        step = 1
        for i in ids:
            if from_index == None:
                from_index = i
            else:
                to_index = i
                distance = np.linalg.norm(np.array(data.loc[from_index]) - np.array(data.loc[to_index]))

                if step == primeStep and not is_prime(to_index):
                    distance = distance * 1.1
                total_distance += distance
                from_index = to_index
                step += 1
        return total_distance
    return None

def getFastDistance(ids, window_size, idx):
    vector1 = data.loc[ids]
    vector2 = vector1.shift(-1)[:-1]
    vector1 = vector1[:-1]
    distance = np.linalg.norm(vector1.values - vector2.values, axis=1)
    primeStep = getstep(window_size, idx)
    if primeStep !=0 and primeStep < window_size and is_prime(ids[primeStep]):
        distance[primeStep-1] *= 1.1
    distance = np.sum(distance)
    return distance

distance_measure_time = 0
def getCityIdsPermutation(p, ids, window_size, idx):
    #global distance_measure_time
    new_ids = []
    for i in p:
        new_ids.append(ids[i])
    #start = time.time()
    distance = getFastDistance(new_ids, window_size, idx)
    #distance_measure_time += time.time() - start
    return new_ids, distance

def arrangeCityIds(ids, window_size, idx):
    arrangeIds = []
    listCombination = []
    window = list(range(window_size))
    if len(ids) == window_size:
        #start = time.time()
        permutations = [[0]+list(i)+[window_size-1] for i in list(itertools.permutations(window[1:-1]))]
        for p in permutations:
            new_ids, distance = getCityIdsPermutation(p, ids, window_size, idx)
            listCombination.append((new_ids, distance))
        listCombination = sorted(listCombination,key=lambda d: d[1])
        #print('getCityIdsPermutation: ', time.time() - start)
        #print('DistanceMeasureTime: ', distance_measure_time)
        return listCombination[0][0]
    return ids

def primeArrange():
    data_len = len(new_output)
    i = 400
    window_size = 7
    #lower_index = int(window_size/2)
    end_index = data_len - window_size
    for id in bar(new_output):
        #if (i+1)%10 == 0 and i+2 < data_len:
        if i <= end_index:
            ids = new_output[i:i+window_size]
            #isPrime, primeIdx = oneSidePrimeCheck(ids)
            #start = time.time()
            if i==405:
                print(ids)
            new_output[i:i+window_size] = arrangeCityIds(ids, window_size, i)
            #print('arrangeCityIds: ', time.time() - start)
        i += 1

new_output = [i[0] for i in output.values]
version = 1
while True:
    previous_output = new_output
    primeArrange()
    #pd.DataFrame({'Path': new_output}).to_csv('submission_ver_'+str(version)+'.csv', index=False)
    count = 0
    for i, cityId in enumerate(new_output):
        if cityId != previous_output[i]:
            count += 1
    if count == 0:
        break