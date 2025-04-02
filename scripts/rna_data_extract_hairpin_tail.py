# Prepare a synthetic dataset of RNA with a hairpin with tails.
# Add corresponding embeddings from RNAfm model


import sys
import os
from tqdm import tqdm
import itertools
import random

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


# Define empty dataframe
df_rnafm = df_rnafm_init()

# ADD specific sequences
sequences = ['']

# Generate random sequences
# Define the four RNA nucleotides
#nucleotides = ['A', 'C', 'G', 'U', 'N']
nucleotides = ['A', 'C', 'G', 'U',] # remove N

# Generate all possible sequences of length L
#sequences1 = itertools.product(nucleotides, repeat=1)
#sequences2 = itertools.product(nucleotides, repeat=2)
#sequences3 = itertools.product(nucleotides, repeat=3)
sequences4 = itertools.product(nucleotides, repeat=4)
sequences5 = itertools.product(nucleotides, repeat=5)
sequences6 = itertools.product(nucleotides, repeat=6)
sequences7 = itertools.product(nucleotides, repeat=7)
sequences8 = itertools.product(nucleotides, repeat=8)
#sequences9 = itertools.product(nucleotides, repeat=9)

#sequences = ['']
#sequences = sequences + [''.join(seq) for seq in sequences1]
#sequences = sequences + [''.join(seq) for seq in sequences2] 
#sequences = sequences + [''.join(seq) for seq in sequences3]  
sequences = sequences + [''.join(seq) for seq in sequences4] 
sequences = sequences + [''.join(seq) for seq in sequences5] 
sequences = sequences + [''.join(seq) for seq in sequences6] 
sequences = sequences + [''.join(seq) for seq in sequences7]
sequences = sequences + [''.join(seq) for seq in sequences8]
#sequences = sequences + [''.join(seq) for seq in sequences9]
print("N of start sequences", len(sequences))
# sequence_names = ["hairpin_tail"] * len(sequences)
# print(sequence_names, len(sequence_names))

print("Generating hairpin sequences")
sequences_hairpin_tail = []
for seq in tqdm(sequences):
    seq2 = ""
    for s in seq:
        if s=="A":
            seq2 += "U"
        elif s=="U":
            seq2 += "A"
        elif s=="C":
            seq2 += "G"
        elif s=="G":
            seq2 += "C"
    for i in range(2):
        # construct RNA loop
        random_number = random.randint(2, 8)
        loop = ''.join(random.choice('ACGT') for _ in range(random_number))
        # construct tail1
        random_number = random.randint(0, 4)
        tail1 = ''.join(random.choice('ACGT') for _ in range(random_number))
        # construct tail2
        random_number = random.randint(0, 4)
        tail2 = ''.join(random.choice('ACGT') for _ in range(random_number))

        sequences_hairpin_tail.append(tail1+seq + loop + seq2[::-1]+tail2) 
sequences = sequences_hairpin_tail
print("N of hairpin_tail sequences", len(sequences))
print("Example of sequence:", sequences[10000])
sequence_names = ["hairpin_tail"] * len(sequences)
# assert 0==1



# Add embeddings
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

path_file = "../data/raw/data_rna_central/general_dataframes/general_rnafm_hairpin_tail_8nucl_full.csv"
print(df_rnafm.tail(20))
df_rnafm.to_pickle(path_file)





