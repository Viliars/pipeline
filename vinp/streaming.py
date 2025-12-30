import math
import torch
import vinp.feature as vf
from vinp.vinp_streaming import load_vinp_from_config
from tqdm import tqdm

EPS = 1e-8


@torch.no_grad()
def _vem_stream_core(
    wav: torch.Tensor,
    model,
    TF,
    rkem,
    sr: int,
    context_ms: float,
    delay_ms: float,
    hop_ms: float,
    device: str = "cuda",
    mix_offline: bool = False,
    mix_alpha: float = 0.5,
) -> torch.Tensor:
    device = torch.device(device)
    wav = wav.to(device)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    assert wav.dim() == 1, f"wav must be 1D, got {wav.shape}"

    T = wav.shape[0]

    C = int(sr * context_ms / 1000.0)
    D = int(sr * delay_ms / 1000.0)
    H = int(sr * hop_ms / 1000.0)

    if C <= 0 or H <= 0 or D < 0:
        raise ValueError("Bad (context_ms, hop_ms, delay_ms)")


    spch_full, _, _, _ = rkem.process(wav, model, TF, device)
    if isinstance(spch_full, torch.Tensor):
        spch_full = spch_full.to(device).view(-1)
    else:
        spch_full = torch.from_numpy(spch_full).to(device).view(-1)

    # Базовый выход — оффлайновая оценка
    out = spch_full.clone()
    written = torch.zeros(T, dtype=torch.bool, device=device)

    # Стриминговый цикл
    t_in = 0
    max_steps = int(math.ceil((T + D) / H)) + 5

    for _ in tqdm(range(max_steps)):
        t_in += H
        if t_in <= 0:
            continue

        t_end = min(t_in, T)

        # Прогрев: пока не накопили context_ms, не гоняем VEM
        if t_end < C:
            continue

        base = t_end - C
        window = wav[base:t_end]

        # VINP+VEM на окне
        spch_win, _, _, _ = rkem.process(window, model, TF, device)
        if isinstance(spch_win, torch.Tensor):
            spch_win = spch_win.to(device).view(-1)
        else:
            spch_win = torch.from_numpy(spch_win).to(device).view(-1)

        matured_end = t_end - D
        matured_start = matured_end - H

        g_start = max(matured_start, 0)
        g_end = min(matured_end, T)

        if g_end <= g_start:
            continue

        # Локальные индексы в окне
        l_start = g_start - base
        l_end = l_start + (g_end - g_start)

        # На всякий случай обрежем, чтобы не вылезти за окно
        if l_start < 0:
            g_start -= l_start
            l_start = 0
        if l_end > spch_win.shape[0]:
            diff = l_end - spch_win.shape[0]
            g_end -= diff
            l_end = spch_win.shape[0]

        if g_end <= g_start:
            continue

        seg_stream = spch_win[l_start:l_end]
        seg_offline = spch_full[g_start:g_end]

        if mix_offline:
            # мягкое сглаживание: комбинируем стриминг + оффлайн
            out[g_start:g_end] = mix_alpha * seg_stream + (1.0 - mix_alpha) * seg_offline
        else:
            # жёстко используем стриминговую оценку на этом участке
            out[g_start:g_end] = seg_stream

        written[g_start:g_end] = True

        # Если уже покрыли почти весь сигнал и t_in ушёл далеко — можно остановиться
        if t_in > T + D and written.all():
            break

    return out


@torch.no_grad()
def process_vinp_stream_variant1(
    audio_path: str,
    save_path: str,
    model,
    TF,
    rkem,
    sr: int,
    context_ms: float = 2000.0,
    delay_ms: float = 200.0,
    hop_ms: float = 20.0,
    device: str = "cuda"
):
    wav = vf.load_wav(audio_path, sr)
    est = _vem_stream_core(
        wav=wav,
        model=model,
        TF=TF,
        rkem=rkem,
        sr=sr,
        context_ms=context_ms,
        delay_ms=delay_ms,
        hop_ms=hop_ms,
        device=device,
        mix_offline=False
    )
    est = est / (est.abs().max() + EPS)
    vf.save_wav(est, save_path, sr)
    return save_path


device = "cuda" if torch.cuda.is_available() else "cpu"

config_path = "vinp/config.toml"
ckpt_path   = "models/vinp.ckpt"

# грузим модель один раз
model, TF, rkem, sr = load_vinp_from_config(
    config_path=config_path,
    checkpoint_path=ckpt_path,
    device=device
)

def streaming_process_vinp(audio_path: str, save_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process_vinp_stream_variant1(
        audio_path=audio_path,
        save_path=save_path,
        model=model,
        TF=TF,
        rkem=rkem,
        sr=sr,
        context_ms=2000.0,
        delay_ms=200.0,
        hop_ms=20.0,
        device=device
    )
