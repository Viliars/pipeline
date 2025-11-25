from glob import glob
from tqdm import tqdm
import os
import torch
from vinp.feature import (
    load_wav,
    save_wav,
    norm_amplitude,
)
from collections import defaultdict
import toml
from typing import Dict
import json
from vinp.utils import initialize_module
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.toml")

config_path = Path(config_path).expanduser().absolute()

config = toml.load(config_path.as_posix())

def average_checkpoints(checkpoints):
    param_sums = defaultdict(lambda: 0)
    num_checkpoints = len(checkpoints)
    for ckpt in checkpoints:
        if "use_ema" in ckpt and ckpt["use_ema"]:
            state_dict = ckpt["model_ema"]
        else:
            state_dict = ckpt["model"]

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            param_sums[new_key] += value.float()

    averaged_state_dict = {}
    for key, sum_value in param_sums.items():
        averaged_state_dict[key] = sum_value / num_checkpoints

    return averaged_state_dict

acoustic_config = config["acoustic"]
model_config = config["model"]
EM_algo = config["EM_algo"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TF = initialize_module(acoustic_config["path"], acoustic_config["args"])
sr = TF.sr

mymodel = initialize_module(model_config["path"], model_config["args"])

model_path = "models/vinp.ckpt"

checkpoints = [torch.load(model_path, map_location="cpu")]
averaged_state_dict = average_checkpoints(checkpoints)
mymodel.load_state_dict(averaged_state_dict, strict=True)

mymodel.eval()
mymodel.to(device)

rkem = initialize_module(EM_algo["path"], EM_algo["args"])

def process_vinp(audio_path: str, save_path: str):
    input_wav = load_wav(audio_path, sr)
    input_wav, scale = norm_amplitude(input_wav)

    spch_est, _, _, _ = rkem.process(input_wav, mymodel, TF, device)

    spch_est *= scale

    save_wav(spch_est / spch_est.abs().max(), save_path, sr)
