# extract the first embedding of an autoencoder as a dataframe
# for the current model and for rnafm for comparison

import os
import torch
from models.RNA_transformer.vocabulary import Vocabulary
import pandas as pd
from multimolecule import RnaTokenizer, RnaFmModel
from torch.nn import CosineSimilarity
import warnings
import logging


def comparison(model, validation_ds, device, num_examples=5, path_file="."):

    # Suppress warnings
    warnings.filterwarnings('ignore')
    # Suppress warnings from transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)

    vocab = Vocabulary()
    # vocab.get_inverted_rna_types()
    tokenizer_tgt_inv = vocab.get_inverted_tokenizer_tgt()
    cos = CosineSimilarity(dim=1, eps=1e-08)

    # save extracted data to
    # path_file_my = "../data/processed/RNA_transformer/inference/embeddings_my.csv"
    # path_file_rnafm = "../data/processed/RNA_transformer/inference/embeddings_rnafm.csv"
    os.makedirs(path_file, exist_ok=True)
    path_file_my = path_file + "/embeddings_my.csv"
    path_file_rnafm = path_file + "/embeddings_rnafm.csv"

    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model_rnafm = RnaFmModel.from_pretrained("multimolecule/rnafm")
    df_my = pd.DataFrame(columns=["rna_type", "norm", "cosinus", "embedding", "length", "sequence"])
    df_rnafm = pd.DataFrame(columns=["rna_type", "norm", "cosinus", "embedding", "length", "sequence"])
    for i,batch in enumerate(validation_ds):
        
        # get input val data 
        encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
        encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
        label_expec = batch["class"].to(device)
        label = vocab.get_inverted_rna_types()[torch.argmax(label_expec, dim=1)[0].cpu().item()]

        # get rna sequence
        sequence = "".join([tokenizer_tgt_inv[el] for el in encoder_input.tolist()[0]])

        # get embedding from this model
        encoder_output = model.encode(encoder_input, encoder_mask)
        #embedding_my = encoder_output[:, -1, :]
        embedding_my = encoder_output[:, 0, :]

        # get embedding from rnafm
        input = tokenizer(sequence, return_tensors="pt")
        output = model_rnafm(**input)
        token_embeddings = output.last_hidden_state  # Shape: (1, sequence_length, hidden_size)
        # sequence_embedding = output.pooler_output
        sequence_embedding = token_embeddings[:, 0, :]
        embedding_rnafm = sequence_embedding

        # save data to dataframe in loop
        if i==0:
            for batch in validation_ds:
                embedding_my0 = embedding_my
                embedding_rnafm0 = embedding_rnafm

        if i>0:
            cosinus_my = cos(embedding_my0, embedding_my).float()
            norm_my = torch.norm(embedding_my, dim=1).float()
            df_my.loc[len(df_my)] = [label, 
                                     norm_my.unsqueeze(0).cpu().detach().numpy()[0][0],
                                     cosinus_my.unsqueeze(0).cpu().detach().numpy()[0][0],
                                     embedding_my.unsqueeze(0).cpu().detach().numpy().flatten(),
                                     len(sequence), sequence]
            cosinus_rnafm = cos(embedding_rnafm0, embedding_rnafm).float()
            norm_rnafm = torch.norm(embedding_rnafm, dim=1).float()
            df_rnafm.loc[len(df_rnafm)] = [label,  
                                     norm_rnafm.unsqueeze(0).cpu().detach().numpy()[0][0],
                                     cosinus_rnafm.unsqueeze(0).cpu().detach().numpy()[0][0],
                                     embedding_rnafm.unsqueeze(0).cpu().detach().numpy().flatten(),
                                     len(sequence), sequence]
            
            if (i)%2==0:
                df_my.to_pickle(path_file_my)
                df_rnafm.to_pickle(path_file_rnafm)

            if i>=num_examples:
                print("Each dataframe contains", num_examples, "lines")
                break
            

    return [df_my, df_rnafm]
