import numpy as np
from tqdm import tqdm
from itertools import product
import requests

def compute_vector_combinations(vectors0, vectors1, Nvectors=2, Lbound=100):
    # Compute all possible vector combinations for Nvectors vectors
    # between first Lbound vectors0 and first Lbound vectors1
    #vectors = np.asarray(vectors)  # Ensure it's a NumPy array at the start
    #L = min(len(vectors), Lbound)  # Keep L within bounds
    L0 = min(len(vectors0), Lbound)  # Keep L within bounds
    L1 = min(len(vectors1), Lbound)  # Keep L within bounds
    result_vectors = []
    indices = []
    
    if Nvectors == 2:
        # Compute all possible (+ or -) combinations between 2 vectors
        for j in tqdm(range(L0)):
            for k in range(j+1, L1):  # Avoid duplicate pairs
                seen_combinations = set()
                for signs in product([-1, 1], repeat=2):  # Generate 2^2 = 4 sign combinations
                    vec_combination = signs[0] * vectors0[j] + signs[1] * vectors1[k]
                    key = tuple(np.round(vec_combination, decimals=8))  # Avoid precision errors
                    if key not in seen_combinations:
                        seen_combinations.add(key)
                        result_vectors.append(vec_combination)
                        indices.append((j, k, *['+' if s == 1 else '-' for s in signs]))
    


    # Directly return NumPy array instead of converting after
    return np.vstack(result_vectors), indices

"""    elif Nvectors == 3:
        # Compute all possible (+ or -) combinations between 3 vectors
        for i in tqdm(range(L)):
            for j in range(i+1, L):
                for k in range(j+1, L):
                    seen_combinations = set()
                    for signs in product([-1, 1], repeat=3):  # Generate 2^3 = 8 sign combinations
                        vec_combination = signs[0] * vectors[i] + signs[1] * vectors[j] + signs[2] * vectors[k]
                        key = tuple(np.round(vec_combination, decimals=8))  # Avoid precision errors
                        if key not in seen_combinations:
                            seen_combinations.add(key)
                            result_vectors.append(vec_combination)
                            indices.append((i, j, k, *['+' if s == 1 else '-' for s in signs]))"""

def get_description(url):
    # Get more description for the given RNA
    description = ""
    try:
        response = requests.get(url=url) #, params=params)
        if response.status_code == 200:
            page = response.json()
            description = page['results'][0]['accession']['description']
    except:
        description = ''
    return description