from pyChris import christo
import pandas as pd
import numpy as np


data = pd.read_csv('data/cities.csv')

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
    # matrix = matrix + np.transpose(matrix)
    #
    # for r in range(ids_len):
    #     matrix[r][r] = 99999

    return matrix

disMatrix = getDistanceMatrix(data.CityId.values)
print(disMatrix)