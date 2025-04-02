# Prepare a synthetic dataset of RNA with symmetric sequence
# abab and abba, where abba has the same structure as SymR region of ribosome
# each random sequence was generated using rna_data_extract_random.py
# with the presence of "N" nucleotide
# Add corresponding embeddings from RNAfm model

import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))
# import torch.nn.functional as F
import numpy as np
import pandas as pd

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





# Read random sequences
path = "../data/raw/data_rna_central/general_dataframes/"
df_rnafm0 = pd.read_pickle(path+"general_rnafm_random_8nucl_full.csv") #[0:50]
df_rnafm0 = df_rnafm0[(df_rnafm0['sequence'].str.len() > 0)&(df_rnafm0['sequence'].str.len() <=7)]
# df_rnafm0 = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
df_rnafm0 = df_rnafm0.reset_index(drop=True)

# df_rnafm = df_rnafm_init()

print("Generating sequences...")
sequence_names = []
sequences = []
# SEARCH for couples
for i in tqdm(range(len(df_rnafm0))):
    seq1 = df_rnafm0["sequence"].iloc[i]
    seq1_short = seq1.replace("N", "")
    for j in range(i+1, len(df_rnafm0)):
        seq2 = df_rnafm0["sequence"].iloc[j]
        seq2_short = seq2.replace("N", "")
        if seq1_short==seq2_short:
            #print("similar:", seq1, seq2)
            #df_rnafm.at[len(df_rnafm), "sequence"] = seq1 + "" + seq2
            sequences.append(seq1 + seq2)
            sequence_names.append("symmetric_abab_withN")
            sequences.append(seq1 + seq2[::-1])
            sequence_names.append("symmetric_abba_withN")
#print(df_rnafm.tail(5))
#df_rnafm = df_rnafm.iloc[::-1].reset_index(drop=True)

# ADD specific sequences
#sequences = df_rnafm.sequence.tolist()
#print("seq length:", len(sequences))



print("N of start sequences", len(sequences))
#sequence_names = ["random"] * len(sequences)
# print(sequence_names, len(sequence_names))

df_rnafm = df_rnafm_init()

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

path_file = "../data/raw/data_rna_central/general_dataframes/general_rnafm_symmetric_7nucl_full_withN.csv"
print(df_rnafm.tail(20))
df_rnafm.to_pickle(path_file)





