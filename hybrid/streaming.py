from .generator import TSCNet
from .utils import *
import torchaudio
import torch
import math
import torch
import torchaudio
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

EPS = 1e-8

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

def update_rms_ema(rms_state, chunk, alpha=0.95):
    power = (chunk ** 2).mean(dim=-1)
    if rms_state is None:
        rms_state = power
    else:
        rms_state = alpha * rms_state + (1.0 - alpha) * power

    c = 1.0 / torch.sqrt(rms_state + EPS)
    return rms_state, c


def run_model_on_normed_chunk(
    noisy_chunk_norm: torch.Tensor,
    model,
    n_fft: int = 400,
    hop: int = 100,
):
    device = noisy_chunk_norm.device
    length = noisy_chunk_norm.size(-1)

    frame_num = int(math.ceil(length / hop))
    padded_len = frame_num * hop
    padding_len = padded_len - length
    if padding_len > 0:
        noisy_chunk_norm = torch.cat(
            [
                noisy_chunk_norm,
                torch.zeros(noisy_chunk_norm.size(0), padding_len, device=device),
            ],
            dim=-1,
        )

    window = torch.hamming_window(n_fft, device=device)

    noisy_spec = torch.stft(
        noisy_chunk_norm,
        n_fft,
        hop_length=hop,
        window=window,
        onesided=True,
        return_complex=False,
    )

    noisy_spec_c = power_compress(noisy_spec).permute(0, 1, 3, 2)

    est_real, est_imag = model(noisy_spec_c)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_spec_uncompress = torch.view_as_complex(est_spec_uncompress)

    est_audio_norm = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop_length=hop,
        window=window,
        onesided=True,
        return_complex=False,
    )

    est_audio_norm = est_audio_norm[..., :length]
    return est_audio_norm


def enhance_stream_waveform_chunks_sliding_norm(
    input_path: str,
    output_path: str,
    sr: int = 16000,
    n_fft: int = 400,
    hop: int = 100,
    chunk_ms: float = 2000.0,
    emit_ms: float = 1850.0,
    rms_alpha: float = 0.95,
    max_gain: float = 10.0
):
    noisy, sr_loaded = torchaudio.load(input_path)
    assert sr_loaded == sr, f"Expected {sr}, got {sr_loaded}"

    device = next(model.parameters()).device
    noisy = noisy.to(device)
    C, T = noisy.shape

    chunk_size = int(chunk_ms * sr / 1000.0)
    emit_size = int(emit_ms * sr / 1000.0)
    overlap = chunk_size - emit_size

    N = overlap
    n = torch.arange(N, device=device, dtype=noisy.dtype)
    fade_out = 0.5 * (1 + torch.cos(math.pi * n / N))
    fade_in = 1.0 - fade_out

    out = torch.zeros((C, T + chunk_size), device=device)

    rms_norm_state = None
    rms_in_state = None
    rms_out_state = None

    start = 0
    first = True

    while start < T:
        end = min(start + chunk_size, T)
        chunk = noisy[:, start:end]
        L_cur = chunk.size(-1)

        if L_cur < chunk_size:
            pad_len = chunk_size - L_cur
            reflect_src = chunk[:, max(0, L_cur - pad_len):L_cur]
            if reflect_src.size(-1) < pad_len:
                repeat_times = math.ceil(pad_len / reflect_src.size(-1))
                reflect_src = reflect_src.repeat(1, repeat_times)[..., :pad_len]
            pad = torch.flip(reflect_src, dims=[-1])
            chunk = torch.cat([chunk, pad], dim=-1)

        rms_in_state, _ = update_rms_ema(
            rms_in_state, chunk[:, :L_cur], alpha=rms_alpha
        )

        rms_norm_state, c_norm = update_rms_ema(
            rms_norm_state, chunk[:, :L_cur], alpha=rms_alpha
        )
        chunk_norm = chunk * c_norm.unsqueeze(-1)

        enh_chunk_norm = run_model_on_normed_chunk(
            chunk_norm, model, n_fft=n_fft, hop=hop
        )
        enh_chunk = enh_chunk_norm / c_norm.unsqueeze(-1)

        rms_out_state, _ = update_rms_ema(
            rms_out_state, enh_chunk[:, :L_cur], alpha=rms_alpha
        )

        gain = torch.sqrt((rms_in_state + EPS) / (rms_out_state + EPS))
        gain = torch.clamp(gain, 1.0 / max_gain, max_gain)

        chunk_in_power = (chunk[:, :L_cur] ** 2).mean(dim=-1)
        silence_thresh = 1e-4
        one = torch.ones_like(gain, device=device)
        is_silence = chunk_in_power < silence_thresh
        gain = torch.where(is_silence, torch.minimum(gain, one), gain)

        enh_chunk = enh_chunk * gain.unsqueeze(-1)

        if first:
            out[:, start:start + chunk_size] = enh_chunk
            first = False
        else:
            ovl_start = start
            ovl_end = start + overlap

            prev_part = out[:, ovl_start:ovl_end]
            curr_part = enh_chunk[:, :overlap]

            mixed = prev_part * fade_out.unsqueeze(0) + curr_part * fade_in.unsqueeze(0)
            out[:, ovl_start:ovl_end] = mixed

            out[:, ovl_end:ovl_start + chunk_size] = enh_chunk[:, overlap:]

        start += emit_size

    est_audio = out[:, :T]
    est_audio = torch.flatten(est_audio).detach().cpu().numpy()

    sf.write(output_path, est_audio, 16000)

    return output_path


def streaming_process_hybrid(audio_path: str, save_path: str, chunk_ms=2000.0, emit_ms=1850.0):
    return enhance_stream_waveform_chunks_sliding_norm(
        audio_path, save_path, chunk_ms=chunk_ms, emit_ms=emit_ms
    )
