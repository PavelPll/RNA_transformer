from pathlib import Path

def get_config():
    return {
        "batch_size": 64, # 256,
        "num_epochs": 3,
        "lr": 10**-4,
        "seq_len": 206,
        "RNA_seq_len_max": 205, # at least one padding token
        "d_model": 160, #240, #160, #160, # 64, #24, #32, #16, # 64,
        "vocab_src_length": 128,
        "vocab_tgt_length": 128,
        "datasource": '../data/processed/RNA_transformer',
        "model_folder": "weights",
        "inference_folder": "inference",
        "model_basename": "tmodel_",
        "experiment_name": "runs/RNA_transformer",
        "autoencoder_bert": False,
        "autoencoder_vanilla": True,
        "classification": False,
        "trainable_classification_weight": False,
        "custom_emb_percentage": 1.0,
        "apply_masking": True
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}//{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}//{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])