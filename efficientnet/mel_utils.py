import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm
    from matplotlib import colors as mpl_colors
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def compute_mel_spectrogram(
    wav_path: str,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 100,
    win_length: int = 400,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    center: bool = True,
    pad_mode: str = "reflect",
    to_db: bool = True,
    ref: float = 1.0,
    duration_sec: Optional[float] = 3.0,
) -> np.ndarray:
    """
    Строит mel-спектрограмму по WAV файлу с параметрами максимально близкими к
    используемым в проекте CMGAN (sr=16k, n_fft=400, hop=100, win=400).

    По умолчанию берутся только первые duration_sec секунд аудио (3.0 сек). Чтобы
    использовать весь файл, передайте duration_sec=None.

    Args:
        wav_path: Путь к аудио файлу.
        sr: Частота дискретизации для загрузки/ресемплинга.
        n_fft: Размер FFT (длина окна для STFT паддингом).
        hop_length: Шаг окна (семпл).
        win_length: Длина окна (семпл).
        n_mels: Количество мел-фильтров.
        fmin: Минимальная частота мел-шкалы.
        fmax: Максимальная частота мел-шкалы (по умолчанию sr/2, если None).
        power: Степень спектра (1.0 — амплитуда, 2.0 — мощность).
        center: Центрировать окна в STFT.
        pad_mode: Режим паддинга для STFT.
        to_db: Конвертировать ли mel мощность в dB шкалу.
        ref: Опорный уровень для преобразования в dB (librosa.power_to_db).
        duration_sec: Длительность используемого сегмента в секундах (по умолчанию 3.0).
                     Если None, используется весь файл.

    Returns:
        np.ndarray: mel-спектрограмма формы [n_mels, num_frames]. Если to_db=True,
        возвращается в dB, иначе — в единицах мощности/амплитуды согласно power.
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {wav_path}")

    # Загрузка аудио с ресемплингом до sr
    audio, _ = librosa.load(str(path), sr=sr)

    # Используем только первые duration_sec секунд, если задано
    if duration_sec is not None and duration_sec > 0:
        max_len = int(sr * duration_sec)
        if len(audio) > max_len:
            audio = audio[:max_len]

    # Mel-спектрограмма через librosa.feature.melspectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=(fmax if fmax is not None else sr / 2),
        power=power,
        center=center,
        pad_mode=pad_mode,
        window="hann",
    )

    if to_db:
        mel_db = librosa.power_to_db(mel, ref=ref)
        return mel_db

    return mel


def save_mel_image(
    mel: np.ndarray,
    out_path: str,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Сохраняет mel-спектрограмму в RGB картинку с использованием цветовой карты.

    Args:
        mel: Массив mel-спектрограммы [n_mels, num_frames] (в dB или мощности).
        out_path: Куда сохранить изображение (например, "/tmp/mel.png").
        cmap: Цветовая карта matplotlib (например, "magma", "inferno", "viridis").
        vmin, vmax: Границы для отображения. Если None и normalize=True, берутся по перцентилям.
        normalize: Если True и границы не заданы, используем перцентили (1, 99) для контраста.
        target_size: Итоговое разрешение (width, height) для ресайза. Если None — без ресайза.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib не установлен. Установите его: pip install matplotlib")

    mel_to_plot = np.asarray(mel)

    # Нормализация/границы отображения
    local_vmin = vmin
    local_vmax = vmax
    if normalize:
        if local_vmin is None:
            local_vmin = float(np.percentile(mel_to_plot, 1))
        if local_vmax is None:
            local_vmax = float(np.percentile(mel_to_plot, 99))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Если нужен точный контроль разрешения, используем colormap -> RGB и PIL
    if target_size is not None and _HAS_PIL:
        norm = mpl_colors.Normalize(vmin=local_vmin, vmax=local_vmax, clip=True)
        colormap = mpl_cm.get_cmap(cmap)
        rgba = colormap(norm(mel_to_plot))  # [H, W, 4] float in 0..1
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)  # drop alpha
        img = Image.fromarray(rgb)
        img = img.resize((int(target_size[0]), int(target_size[1])), resample=Image.BILINEAR)
        img.save(str(out))
        return

    # Фолбэк: сохраняем напрямую через imsave (разрешение контролируется не пиксельно)
    plt.imsave(
        str(out),
        mel_to_plot,
        cmap=cmap,
        vmin=local_vmin,
        vmax=local_vmax,
        origin='lower',
        format=out.suffix.replace('.', '') or 'png',
    )


def compute_and_save_mel_image(
    wav_path: str,
    out_image_path: str,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 100,
    win_length: int = 400,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    center: bool = True,
    pad_mode: str = "reflect",
    to_db: bool = True,
    ref: float = 1.0,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    normalize: bool = True,
    duration_sec: Optional[float] = 3.0,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Вычисляет mel-спектрограмму из WAV (по умолчанию первые 3 сек) и сохраняет её в RGB картинку.

    Args:
        target_size: Итоговое разрешение (width, height) для ресайза выходной картинки.
    """
    mel = compute_mel_spectrogram(
        wav_path=wav_path,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power,
        center=center,
        pad_mode=pad_mode,
        to_db=to_db,
        ref=ref,
        duration_sec=duration_sec,
    )
    save_mel_image(
        mel,
        out_image_path,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        normalize=normalize,
        target_size=target_size,
    )
    return mel
