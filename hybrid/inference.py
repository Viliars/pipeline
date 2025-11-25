import os
import numpy as np
from .generator import TSCNet
from .utils import *
import torchaudio
import soundfile as sf
import torch

import warnings
warnings.filterwarnings("ignore")

def load_model(model_path: str):
    C = 36
    num_features = 201
    model = TSCNet(num_channel=C, num_features=num_features).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))

    return model

model_path = "models/small.ckpt"
model = load_model(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@torch.no_grad()
def enhance_one_track(audio_path: str, save_path: str):
    hop=100
    n_fft=400
    cut_len=16000*16

    noisy, sr = torchaudio.load(audio_path)
    noisy = noisy.to(device)
    assert sr == 16000

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)

    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)

    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True, return_complex=False
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)

    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_spec_uncompress = torch.view_as_complex(est_spec_uncompress)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(device),
        onesided=True,
        return_complex=False
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu().numpy()

    # сохраняем
    sf.write(save_path, est_audio, sr)

    return save_path

def process_hybrid(audio_path: str, save_path: str):
    return enhance_one_track(audio_path, save_path)
