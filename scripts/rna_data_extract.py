# Prepare global dataset
# from RNA central database
# Add corresponding embeddings from RNAfm model
# for multiple runs
# Output: general_rnafm_0_2000.csv, ...
# with start=0 end=2000, ...

import requests
import sys
import os
from tqdm import tqdm
import time

sys.path.append(os.path.abspath('../'))
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Define the RNAcentral API URL
BASE_URL = "https://rnacentral.org/api/v1/"
path_file = "../data/processed/vit1/rna_central/rna_sequences.csv"

from multimolecule import RnaTokenizer, RnaFmModel

# Get start and end web pages for extraction
try:
    # Get user input
    start = float(input("Enter start value: "))
    end = float(input("Enter end value: "))

    # Validate condition: end must be greater than start
    if end <= start:
        print("Error: 'end' must be greater than 'start'. Please try again.")
        sys.exit(1)

except ValueError:
    print("Error: Invalid input. Please enter numerical values.")
    sys.exit(1)

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
model_rnafm = RnaFmModel.from_pretrained("multimolecule/rnafm")

# get embedding from sequence using rnafm model
def emb_from_seq(sequence): 
    input = tokenizer(sequence, return_tensors="pt")
    output = model_rnafm(**input)
    token_embeddings = output.last_hidden_state  # Shape: (1, sequence_length, hidden_size)
    # sequence_embedding = output.pooler_output
    sequence_embedding = token_embeddings[:, 0, :]
    embedding_rnafm = sequence_embedding
    return embedding_rnafm

def df_rnafm_init():
    df_rnafm = pd.DataFrame(columns = [
                                "rna_type",
                                "sequence", 
                                "embedding", 
                                "count_distinct_organisms",
                                "rnacentral_id",
                                "md5",
                                "xrefs",
                                "publications",
                                "description",
                                "distinct_databases",
                                "is_active"
                                ])
    return df_rnafm



start, end = int(start), int(end)
path = "../data/raw/data_rna_central/general_dataframes"
os.makedirs(path, exist_ok=True)
path_file = path + "/general_rnafm.csv"

# file name to save
path_file2 = path_file[:-4]+"_"+str(start)+"_"+str(end)+".csv"

if os.path.exists(path_file2):
    # continue with previous version of dataframe
    df_rnafm = pd.read_pickle(path_file2)
    print("Found: ", path_file2.split("/")[-1])
    start = start + int(len(df_rnafm)/50 )
    print("Starting from start = ", start, "...")
else: 
    # or create new dataframe
    df_rnafm = df_rnafm_init()

# url of API
# https://rnacentral.org/api/v1/rna/?page=8000&page_size=100
url2 = "https://rnacentral.org/api/v1/rna/?page={}&page_size=50"

# for Npage in tqdm(np.arange(7945, 20000000)): 
for Npage in tqdm(np.arange(start, end)):
    # time.sleep(1.1)
    url = url2.format(Npage)

    # Send the request to the RNAcentral API
    response = requests.get(url=url) #, params=params)
    # print(response.status_code)
    if response.status_code == 200:
        try:
            page = response.json()
            for el in page['results']:
                sequence = el["sequence"]
                if len(sequence)<=1000:
                    embedding_rnafm = emb_from_seq(sequence)
                    embedding_rnafm = embedding_rnafm.unsqueeze(0).cpu().detach()
                    embedding_rnafm = embedding_rnafm.numpy().flatten()
                else:
                    embedding_rnafm = np.nan
                df_rnafm.loc[len(df_rnafm)] = [
                                            el["rna_type"], 
                                            sequence, 
                                            embedding_rnafm,
                                            el["count_distinct_organisms"],
                                            el["rnacentral_id"],
                                            el["md5"],
                                            el["xrefs"],
                                            el["publications"],
                                            el["description"],
                                            el["distinct_databases"],
                                            el["is_active"]]
        except:
            print("bad url:", url)
    else:
        print("\n"+url)
    if (Npage+1)%100==0:
        df_rnafm.to_pickle(path_file2)
df_rnafm.to_pickle(path_file2)
