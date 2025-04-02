# Validation

import torch
import torchmetrics
from models.RNA_transformer.dataset import causal_mask
from models.RNA_transformer.vocabulary import Vocabulary
from models.RNA_transformer.config import get_config
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    #sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    #eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    sos_idx = 1 #torch.tensor([1], dtype=torch.int64)
    eos_idx = 2 #torch.tensor([2], dtype=torch.int64)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # get next token
        prob = model.project(out[:, -1])
        #print("lol out", out.shape, out[:, -1].shape, prob.shape)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    # classification
    #cls_token_rep = encoder_output[:, -1, :]
    cls_token_rep = encoder_output[:, 0, :]
    #print("cls_token_rep:", cls_token_rep.shape)
    proj_output_cl = model.project_cl(cls_token_rep) # logits 
    output = F.softmax(proj_output_cl, dim=1)
    #print("decoder_input.squeeze", decoder_input.shape)
    #print("decoder_input.squeeze(0)", decoder_input.squeeze(0).shape)
    #print("decoder_input.squeeze(0):", decoder_input.squeeze(0))
    return decoder_input.squeeze(0), output

def bert_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # Precompute the encoder output and reuse it for every step
    sos_idx = 1 #torch.tensor([1], dtype=torch.int64)
    config = get_config()
    VOCAB_SRC_LENGTH = config["vocab_src_length"]
    encoder_output = model.encode(source, source_mask)
    prob = model.project_bert(encoder_output)
    #output = F.softmax(proj_output_cl, dim=1)
    #out = F.softmax(out, dim=0)
    #print("encoder_output", encoder_output.shape) # torch.Size([1, 206, 160])
    #print("proj_output_bert (prob)", prob.shape) # torch.Size([1, 206, 128])
    _, next_word = torch.max(prob, dim=2) # max values, indices [1, 206]
    #print(_.shape, next_word.shape)
    #print("output bert", next_word.squeeze(0))

    #decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    #print("decoder_input.shape", decoder_input.shape)
    #decoder_input = torch.cat(
    #        [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
    #    )
    #print(out)
    out = next_word.squeeze(0)
    return out


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, 
                   num_examples=10, verbose=False):
    config = get_config()
    autoencoder_vanilla = config["autoencoder_vanilla"]
    autoencoder_bert = config["autoencoder_bert"]
    classification = config["classification"] 
    vocab = Vocabulary()
    tokenizer_tgt_inv = vocab.get_inverted_tokenizer_tgt()

    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    label_expected = []
    label_predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds: # size of validation_ds is 1
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
            label_expec = batch["class"].to(device)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"
            
            model_out, output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            #print(source_text)
            #print(target_text)
            l_out = model_out.detach().cpu().numpy()
            #if autoencoder_bert:
            if autoencoder_vanilla:
                model_out_text = [tokenizer_tgt_inv[el] for el in l_out]
                model_out_text = "".join(model_out_text )
                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)
            if autoencoder_bert:
                bert_out = bert_decode(model, encoder_input, 
                                    encoder_mask, 
                                    tokenizer_src, 
                                    tokenizer_tgt, max_len, device)
                l_out = bert_out.detach().cpu().numpy()
                model_out_text = [tokenizer_tgt_inv[el] for el in l_out]
                model_out_text = "".join(model_out_text )
                delta = len(target_text)-len(model_out_text)
                if delta<=0:
                    print("target_text", target_text)
                    print("bert output", model_out_text)
                    #print("bert output", l_out)
                expected.append(target_text)
                predicted.append(model_out_text)
                #prob = model.project(out[:, -1])
                #proj_output_bert = model.project_bert(decoder_output)
            # for classification
            label_expected.append([torch.argmax(label_expec, dim=1)])
            label_predicted.append( [torch.argmax(output, dim=1)] )

            if count == num_examples:
                break
    
    # Flatten the lists of tensors into lists of values
    true_labels = [label[0].item() for label in label_expected]
    predicted_labels = [pred[0].item() for pred in label_predicted]
    # Compute the F1 score with different averaging methods
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')

    # Step 2: Compute the confusion matrix
    if verbose and classification:
        print("f1_micro", f1_micro)
        print("f1_macro", f1_macro)
        cm = confusion_matrix(true_labels, predicted_labels)
        vocab = Vocabulary()
        labels = list(vocab.get_rna_types().keys())  # ['label1', 'label2'] inverted_rna_types
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

    if writer:
        # Evaluate the character error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        if autoencoder_vanilla or autoencoder_bert:
            output = "Character error rate (CER) for {} sequences: {}".format(num_examples, cer)
            print(output)
            #print("Character error rate (CER):", cer)
            writer.add_scalar('validation cer', cer, global_step)
        if classification:
            print("f1_micro", f1_micro)
            print("f1_macro", f1_macro)
            writer.add_scalar('validation f1_micro:', f1_micro, global_step)
            writer.add_scalar('validation f1_macro:', f1_macro, global_step)
        writer.flush()