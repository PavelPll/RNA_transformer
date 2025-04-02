# Prepare the dataset of RNA with sequences of interest
# Here for peptidyl transferase center
# Add corresponding embeddings from RNAfm model

import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))
import torch.nn.functional as F
import numpy as np
import pandas as pd

from models.RNA_transformer.RNAsequences import RNAsequences 


from multimolecule import RnaTokenizer, RnaFmModel

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


df_rnafm = df_rnafm_init()

# ADD specific sequences
sequences = ['']
sequence_names = ["empty", "PTC", "SymR_A", "SymR_P", "SymR_PA", 
                  "PTC2", "PTC3", "PTC4", "PTC5",] 
#                  "PTC_inv", "SymR_A_inv", "SymR_P_inv"]
rna_sequences = RNAsequences()
sequences.append(rna_sequences.get_PTCsequence())
sequences.append(rna_sequences.get_SymR_Asequence())
sequences.append(rna_sequences.get_SymR_Psequence())
sequences.append(rna_sequences.get_SymR_Psequence()+rna_sequences.get_SymR_Asequence())
sequences.append(rna_sequences.get_PTC2sequence())
sequences.append(rna_sequences.get_PTC3sequence())
sequences.append(rna_sequences.get_PTC4sequence())
sequences.append(rna_sequences.get_PTC5sequence())
#sequences.append(rna_sequences.get_PTCsequence()[::-1])
#sequences.append(rna_sequences.get_SymR_Asequence()[::-1])
#sequences.append(rna_sequences.get_SymR_Psequence()[::-1])



print("N of start sequences", len(sequences))
print(sequence_names, len(sequence_names))



for i,sequence in tqdm(enumerate(sequences)):
    embedding_rnafm = emb_from_seq(sequence)
    embedding_rnafm = embedding_rnafm.unsqueeze(0).cpu().detach()
    embedding_rnafm = embedding_rnafm.numpy().flatten()
    df_rnafm.loc[len(df_rnafm)] = [
                                #   "start", 
                                   sequence_names[i],
                                   sequence,
                                   embedding_rnafm,
                                   len(sequence), sequence] + [np.nan]*6

path_file = "../data/raw/data_rna_central/general_dataframes/general_rnafm_unique.csv"
print(df_rnafm.tail(20))
df_rnafm.to_pickle(path_file)





