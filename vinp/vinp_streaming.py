import torch
import toml
from pathlib import Path
from collections import defaultdict

from .feature import load_wav, save_wav, norm_amplitude
from .utils import initialize_module

EPS = 1e-8


def average_checkpoints(checkpoints):
    param_sums = defaultdict(lambda: 0)
    num_checkpoints = len(checkpoints)
    for ckpt in checkpoints:
        # Веса могут лежать либо в ckpt["model"], либо в ckpt["model_ema"].
        if "use_ema" in ckpt and ckpt["use_ema"] and "model_ema" in ckpt:
            state_dict = ckpt["model_ema"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            # Фоллбек: если это "голый" state_dict
            state_dict = ckpt

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            param_sums[new_key] += value.float()

    averaged_state_dict = {}
    for key, sum_value in param_sums.items():
        averaged_state_dict[key] = sum_value / num_checkpoints

    return averaged_state_dict


def load_vinp_from_config(
    config_path: str,
    checkpoint_path: str,
    device: str = "cpu",
):
    device = torch.device(device)

    config_path = Path(config_path).expanduser().absolute()
    cfg = toml.load(config_path.as_posix())

    acoustic_cfg = cfg["acoustic"]
    model_cfg = cfg["model"]
    em_cfg = cfg["EM_algo"]

    # Трансформы (STFT/ISTFT + препроцесс/постпроцесс)
    TF = initialize_module(acoustic_cfg["path"], acoustic_cfg.get("args", {}))
    # В оригинальном inference SR берут из TF.sr
    sr = getattr(TF, "sr", 16000)

    # Модель VINP
    model = initialize_module(model_cfg["path"], model_cfg.get("args", {}))

    # Загрузка весов как в inference.py (через average_checkpoints)
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    ckpt = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    averaged_state_dict = average_checkpoints([ckpt])
    model.load_state_dict(averaged_state_dict, strict=True)

    model.to(device)
    model.eval()

    # EM-алгоритм (VEM)
    rkem = initialize_module(em_cfg["path"], em_cfg.get("args", {}))

    return model, TF, rkem, sr


@torch.no_grad()
def vinp_denoise_chunk(
    chunk_norm: torch.Tensor,
    model,
    TF,
    rkem,
    device: torch.device,
) -> torch.Tensor:
    if chunk_norm.dim() == 2:
        chunk_norm = chunk_norm.squeeze(0)
    if chunk_norm.dim() != 1:
        raise ValueError(f"chunk_norm должен быть 1D, а не {chunk_norm.shape}")

    spch_est, _, _, _ = rkem.process(chunk_norm, model, TF, device)

    # rkem.process возвращает 1D тензор на CPU
    if isinstance(spch_est, torch.Tensor):
        return spch_est.to(device)
    else:
        # на всякий случай, если где-то numpy
        return torch.from_numpy(spch_est).to(device)


@torch.no_grad()
def enhance_stream_vinp_waveform(
    wav_norm: torch.Tensor,
    model,
    TF,
    rkem,
    sr: int,
    window_ms: float = 200.0,
    hop_ms: float = 20.0,
    device: str = "cpu",
) -> torch.Tensor:
    device = torch.device(device)
    wav_norm = wav_norm.to(device)

    if wav_norm.dim() == 2:
        wav_norm = wav_norm.squeeze(0)
    if wav_norm.dim() != 1:
        raise ValueError(f"wav_norm должен быть 1D, а не {wav_norm.shape}")

    T = wav_norm.shape[0]

    win_size = int(sr * window_ms / 1000.0)
    hop_size = int(sr * hop_ms / 1000.0)

    if win_size <= 0:
        raise ValueError("window_ms слишком маленькое")
    if hop_size <= 0:
        hop_size = win_size
    if hop_size > win_size:
        raise ValueError("hop_ms не может быть больше window_ms")

    # Окно Ханна для плавного OLA
    win = torch.hann_window(win_size, device=device)

    # Буферы для overlap-add
    out = torch.zeros(T + win_size, device=device)
    weight = torch.zeros(T + win_size, device=device)

    pos = 0
    while pos < T:
        end = pos + win_size

        # Берём кусок входа с паддингом в конце
        chunk = torch.zeros(win_size, device=device)
        valid = max(0, min(win_size, T - pos))
        if valid > 0:
            chunk[:valid] = wav_norm[pos:pos + valid]

        # Прогон через VINP+VEM на одном окне
        enh = vinp_denoise_chunk(chunk, model, TF, rkem, device)

        # На всякий случай обрезаем/паддим к win_size
        if enh.shape[0] < win_size:
            tmp = torch.zeros(win_size, device=device)
            tmp[:enh.shape[0]] = enh
            enh = tmp
        elif enh.shape[0] > win_size:
            enh = enh[:win_size]

        # Окно Ханна для плавного склеивания
        enh_win = enh * win

        out[pos:pos + win_size] += enh_win
        weight[pos:pos + win_size] += win

        pos += hop_size

    est = out[:T] / (weight[:T] + EPS)
    return est.contiguous()


@torch.no_grad()
def process_vinp_streaming(
    audio_path: str,
    save_path: str,
    model,
    TF,
    rkem,
    sr: int,
    window_ms: float = 200.0,
    hop_ms: float = 20.0,
    device: str = "cpu",
):
    # 1) Загрузка
    wav = load_wav(audio_path, sr)  # torch.Tensor

    # 2) Глобальная нормализация по всему сигналу
    wav_norm, scale = norm_amplitude(wav)

    # 3) Стриминговая обработка
    est_norm = enhance_stream_vinp_waveform(
        wav_norm,
        model=model,
        TF=TF,
        rkem=rkem,
        sr=sr,
        window_ms=window_ms,
        hop_ms=hop_ms,
        device=device,
    )

    # 4) Обратная нормализация
    est = est_norm * scale

    # Нормализуем под [-1, 1] для сохранения
    est = est / (est.abs().max() + EPS)

    # 5) Сохранение
    save_wav(est, save_path, sr)

    return save_path
