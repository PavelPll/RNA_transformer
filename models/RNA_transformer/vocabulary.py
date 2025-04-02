import torch

class Vocabulary:
    def __init__(self):
        # Predefined dictionary
        self.rna_types = {'misc_RNA':0, 'tRNA':1, 'rRNA':2, 'sRNA':3, 'pre_miRNA':4, 'ncRNA':5, 'other':6,
        'RNase_P_RNA':7, 'circRNA':8, 'SRP_RNA':9, 'tmRNA':10, 'lncRNA':11, 'hammerhead_ribozyme':12,
        'miRNA':13, 'snRNA':14, 'antisense_RNA':15, 'snoRNA':16, 'telomerase_RNA':17, 'scaRNA':18, 
        'ribozyme':19, 'RNase_MRP_RNA':20}
        # Invert the dictionary (keys become values and vice versa)
        self.inverted_rna_types = {v: k for k, v in self.rna_types.items()}


        self.tokenizer_src = {"O":4, "U":6, "G":7, "C":8, "A":9, "X":10, "N":11, "Y":12, "R":13, "S":14, "M":15, 
                              "K":16, "W":17, "V":18, "D":19, "B":20, "H":21}
        self.tokenizer_tgt = self.tokenizer_src
        
        tokenizer_src_inv = {v: k for k, v in self.tokenizer_src.items()}
        tokenizer_src_inv[1], tokenizer_src_inv[2] = "", "" 
        tokenizer_src_inv[3] = ""
        self.tokenizer_src_inv = tokenizer_src_inv
        self.tokenizer_tgt_inv = self.tokenizer_src_inv

    # Method to get the original dictionary (rna_types)
    def get_rna_types(self):
        return self.rna_types
    # Method to get the inverted dictionary
    def get_inverted_rna_types(self):
        return self.inverted_rna_types
    
    # Method to get tokenizer
    def get_tokenizer_src(self):
        return self.tokenizer_src
    def get_tokenizer_tgt(self):
        return self.tokenizer_tgt
    # Method to get the inverted tokenizer
    def get_inverted_tokenizer_src(self):
        return self.tokenizer_src_inv
    def get_inverted_tokenizer_tgt(self):
        return self.tokenizer_tgt_inv
    
def custom_embedding(d_model: int, vocab_size: int):
    # in order to start training from non-random embeddings

    ### add properties IUPAC nucleotide code
    fixed_values = torch.zeros(vocab_size, d_model)
    # Additional 8 physical-chemical properties for the first 4 tokens: A, G, C, U
    G = torch.tensor([1, 5, 5, 5, -183.9, -2498.2, 151, -13])
    A = torch.tensor([0, 5, 5, 5, 96.9, -2779.0, 135, -9])
    C = torch.tensor([1, 3, 4, 5, -221, -2067, 111, -13])
    U = torch.tensor([2, 2, 4, 4, -424.4, -1721.3, 112, -9])
    Av = (A + G + C + U) / 4
    G = G.div(Av)
    A = A.div(Av)
    C = C.div(Av)
    U = U.div(Av)
    G = G - G.mean()
    A = A - A.mean()
    C = C - C.mean()
    U = U - U.mean()
    # print("custom emb max:", G.max(), A.max(), C.max(), U.max())
    # print("custom emb min", G.min(), A.min(), C.min(), U.min())
    # "U":6, "G":7, "C":8, "A":9
    Nfeat = len(A)
    d_blocks = d_model // Nfeat
    # "X":10, "N":11, "Y":12, "R":13, "S":14, "M":15, "K":16, "W":17, "V":18, "D":19, "B":20, "H":21
    fixed_values[6] = U.repeat(d_blocks)[:d_model]
    fixed_values[7] = G.repeat(d_blocks)[:d_model]
    fixed_values[8] = C.repeat(d_blocks)[:d_model]
    fixed_values[9] = A.repeat(d_blocks)[:d_model]

    fixed_values[10] = ((G+A+C+U)/4).repeat(d_blocks)[:d_model]
    fixed_values[11] = ((G+A+C+U)/4).repeat(d_blocks)[:d_model]
    fixed_values[12] = ((C+U)/2).repeat(d_blocks)[:d_model]
    fixed_values[13] = ((G+A)/2).repeat(d_blocks)[:d_model]
    fixed_values[14] = ((G+C)/2).repeat(d_blocks)[:d_model]
    fixed_values[15] = ((A+C)/2).repeat(d_blocks)[:d_model]
    fixed_values[16] = ((G+U)/2).repeat(d_blocks)[:d_model]
    fixed_values[17] = ((A+U)/2).repeat(d_blocks)[:d_model]
    fixed_values[18] = ((G+A+C)/3).repeat(d_blocks)[:d_model]
    fixed_values[19] = ((G+A+U)/3).repeat(d_blocks)[:d_model]
    fixed_values[20] = ((G+C+U)/3).repeat(d_blocks)[:d_model]
    fixed_values[21] = ((A+C+U)/3).repeat(d_blocks)[:d_model]

    return fixed_values