import torch
import random
from torch.utils.data import DataLoader, Dataset
from models.RNA_transformer.vocabulary import Vocabulary

class BilingualDataset(Dataset):

    #def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang=1, tgt_lang=1, seq_len=100):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, 
                 seq_len=100, apply_masking=True):
        super().__init__()
        self.seq_len = seq_len # "seq_len" from config

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_src_inv = {v: k for k, v in self.tokenizer_src.items()}
        self.tokenizer_tgt = tokenizer_tgt

        # self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.sos_token = torch.tensor([1], dtype=torch.int64)
        # self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([2], dtype=torch.int64)
        # self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        self.pad_token = torch.tensor([3], dtype=torch.int64)
        self.mask_token = torch.tensor([4], dtype=torch.int64)
        # self.free_token = torch.tensor([5], dtype=torch.int64)
        self.apply_masking = apply_masking

    def __len__(self):
        return len(self.ds)
    
    def replace_char_at_position(self, sequence, position, new_char):
        # Ensure the position is valid and new_char is a single character
        if not isinstance(sequence, str):
            raise ValueError("The input sequence must be a string.")
        if not isinstance(new_char, str) or len(new_char) != 1:
            raise ValueError("The new character must be a single character string.")
        # If position is out of range, handle it
        if position < 0:
            raise IndexError("Position cannot be negative.")
        if position >= len(sequence):
            # If position is greater than or equal to the length, pad with characters and set the value
            sequence = sequence + " " * (position - len(sequence))  # Add spaces up to the desired position
            sequence = sequence[:position] + new_char + sequence[position+1:]  # Insert the new character
        else:
            # If the position is valid, replace the character
            sequence = sequence[:position] + new_char + sequence[position+1:]
        return sequence
    
    def mask_modification(self, sequence):
        # add mask tokens to input sequence
        N_nucl_to_change = int(15 * len(sequence) / 100)
        for i in range(N_nucl_to_change):
            random_pos = random.randint(0, len(sequence)-1)
            rnd = random.uniform(0, 1)
            if rnd<=0.8:
                nucl = "O" # mask token    
            elif (rnd>0.8) and (rnd<=0.9):
                nucl = self.tokenizer_src_inv[random.randint(6, 21)]
            else:
                nucl = sequence[random_pos]
            # enc_input_tokens[random_pos] = nucl
            sequence = self.replace_char_at_position(sequence, random_pos, nucl)
        return sequence

    def __getitem__(self, idx):
        src_target_pair = self.ds.iloc[idx]
        # src_text = src_target_pair['translation'][self.src_lang]
        # tgt_text = src_target_pair['translation'][self.tgt_lang]
        src_text = src_target_pair["sequence"]
        if self.apply_masking:
            src_text = self.mask_modification(src_text)
        tgt_text = src_target_pair["sequence"]

        # Transform the text into tokens
        # enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        enc_input_tokens = [self.tokenizer_src[el] for el in src_text]
        # dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        dec_input_tokens = [self.tokenizer_src[el] for el in tgt_text]

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # one more

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        # dec to enc
        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                # torch.tensor(enc_input_tokens, dtype=torch.int64),
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # enc to dec
        label_bert = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token, the last N is absent
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        if len(src_text)<80:
            cl = torch.tensor([1, 0], dtype=torch.float64)
        else:
            cl = torch.tensor([0, 1], dtype=torch.float64)
        vocab = Vocabulary()
        #print("vocab", vocab.get_rna_types())
        rna_type = src_target_pair["rna_type"]
        L = len(vocab.get_rna_types())
        cl = torch.zeros((L))
        cl[vocab.get_rna_types()[rna_type]] = torch.tensor([1])
        

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "label_bert": label_bert,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
            "class": cl, 
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

