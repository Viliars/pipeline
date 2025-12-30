import os
import sys
import math
from pathlib import Path
from typing import Optional, Tuple, List

import torch


def add_sys_path_if_needed(path: Path):
    s = str(path.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def load_gtcrn_streaming(
    gtcrn_ckpt_path: Path,
    device: str,
    gtcrn_root: Optional[Path] = None,
):
    """
    Loads GTCRN offline model + creates StreamGTCRN and converts weights.
    Expects repo-style imports:
      - from gtcrn import GTCRN
      - from modules.convert import convert_to_stream
      - StreamGTCRN class importable from one of common module names
    """
    import torch

    dev = torch.device(device)

    if gtcrn_root is not None:
        add_sys_path_if_needed(gtcrn_root)

    # Imports (repo-dependent)
    from gtcrn import GTCRN
    from stream.modules.convert import convert_to_stream

    StreamGTCRN = None
    for mod_name in ["stream_gtcrn", "gtcrn_stream", "streaming_gtcrn", "gtcrn_streaming"]:
        try:
            mod = __import__(mod_name, fromlist=["StreamGTCRN"])
            StreamGTCRN = getattr(mod, "StreamGTCRN", None)
            if StreamGTCRN is not None:
                break
        except Exception:
            pass
    if StreamGTCRN is None:
        raise ImportError(
            "Cannot import StreamGTCRN. Put your StreamGTCRN class into one of: "
            "stream_gtcrn.py / gtcrn_stream.py / gtcrn_streaming.py and ensure it's on PYTHONPATH "
            "(or pass --gtcrn_root)."
        )

    offline = GTCRN().to(dev).eval()
    ckpt = torch.load(str(gtcrn_ckpt_path), map_location=dev)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    offline.load_state_dict(sd, strict=True)

    stream = StreamGTCRN().to(dev).eval()
    convert_to_stream(stream, offline)

    return stream

@torch.no_grad()
def gtcrn_enhance_streaming_time_domain(
    wav,                 # torch [T]
    stream_model,
    device: str,
    sr: int = 16000,
    n_fft: int = 512,
    hop: int = 256,      # GTCRN default (16ms @16k)
):
    """
    TRUE streaming:
      - frame = last n_fft samples (causal)
      - rFFT -> (1, 257, 1, 2)
      - StreamGTCRN forward with caches
      - iRFFT -> time frame
      - OLA with sqrt-Hann and weight normalization
    """
    import torch

    dev = torch.device(device)
    x = wav.to(dev).view(-1)
    T = x.numel()

    win = torch.hann_window(n_fft, device=dev, periodic=True).sqrt()
    win2 = win * win

    # caches per your StreamGTCRN implementation (these dims are from the common GTCRN stream code you pasted)
    conv_cache = torch.zeros(2, 1, 16, 16, 33, device=dev)
    tra_cache = torch.zeros(2, 3, 1, 1, 16, device=dev)
    inter_cache = torch.zeros(2, 1, 33, 16, device=dev)

    # pad to whole frames
    n_frames = int(math.ceil(max(1, (T - n_fft) / hop) + 1))
    total_len = (n_frames - 1) * hop + n_fft
    pad = total_len - T
    if pad > 0:
        x = torch.cat([x, torch.zeros(pad, device=dev)], dim=0)

    y = torch.zeros_like(x)
    wsum = torch.zeros_like(x)

    for i in range(n_frames):
        s = i * hop
        frame = x[s:s + n_fft]
        frame_w = frame * win

        spec = torch.fft.rfft(frame_w, n=n_fft)  # [257] complex
        spec_in = torch.stack([spec.real, spec.imag], dim=-1)  # [257,2]
        spec_in = spec_in.unsqueeze(0).unsqueeze(2)  # [1,257,1,2]

        spec_out, conv_cache, tra_cache, inter_cache = stream_model(spec_in, conv_cache, tra_cache, inter_cache)

        so = spec_out[0, :, 0, 0] + 1j * spec_out[0, :, 0, 1]
        frame_out = torch.fft.irfft(so, n=n_fft)  # [512]
        frame_out = frame_out * win

        y[s:s + n_fft] += frame_out
        wsum[s:s + n_fft] += win2

    y = y[:T] / (wsum[:T] + 1e-8)
    return y
