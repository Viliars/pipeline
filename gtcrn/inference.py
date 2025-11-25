import os
import torch
import soundfile as sf
import numpy as np
from .gtcrn import GTCRN

# ==== параметры ====
N_FFT = 512
HOP = 256
WIN = 512

def load_model(checkpoint_path, device="cpu"):
    model = GTCRN().to(device).eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    return model

def ensure_16k(y, sr):
    if sr == 16000:
        return y.astype(np.float32)
    # простая линейная ресемплинг без внешних библиотек
    n_to = int(round(len(y) * 16000 / sr))
    t_from = np.linspace(0, len(y) / sr, num=len(y), endpoint=False)
    t_to   = np.linspace(0, len(y) / sr, num=n_to, endpoint=False)
    return np.interp(t_to, t_from, y).astype(np.float32)

def denoise_file(model, in_path, out_path, device="cpu"):
    mix, fs = sf.read(in_path, dtype="float32")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)
    mix = ensure_16k(mix, fs)
    fs = 16000

    x = torch.from_numpy(mix).to(device)
    window = torch.hann_window(WIN, device=device, dtype=x.dtype).pow(0.5)

    # STFT -> [F,T] complex
    spec_c = torch.stft(x, N_FFT, HOP, WIN, window=window, return_complex=True)
    spec_ri = torch.view_as_real(spec_c).unsqueeze(0)  # [1,F,T,2]

    with torch.no_grad():
        out_ri = model(spec_ri)[0]  # [F,T,2]

    # convert back to complex
    out_c = torch.view_as_complex(out_ri.contiguous())

    # ISTFT
    enh = torch.istft(out_c, N_FFT, HOP, WIN, window=window)
    enh = enh.cpu().numpy().astype(np.float32)

    # Clip/clean NaN
    if not np.isfinite(enh).all():
        enh = np.nan_to_num(enh, nan=0.0, posinf=0.0, neginf=0.0)
    enh = np.clip(enh, -1.0, 1.0)

    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, enh, fs)

    # простая разница для контроля
    diff = np.abs(enh - mix[: len(enh)])
    mae = float(np.mean(diff))
    linf = float(np.max(diff))
    return mae, linf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("models/gtcrn.ckpt", device)

def process_grcrn(audio_path: str, save_path: str):
    denoise_file(model, audio_path, save_path, device)
