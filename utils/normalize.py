import os
import soundfile as sf
import pyloudnorm as pyln

def measure_loudness(audio_data, sample_rate):
    """Measure the loudness of an audio signal in LKFS."""
    meter = pyln.Meter(sample_rate)  # Create loudness meter
    loudness = meter.integrated_loudness(audio_data)
    return loudness

def normalize_loudness(audio_data, sample_rate, target_loudness):
    """Normalize the loudness of an audio signal to the target loudness in LKFS."""
    meter = pyln.Meter(sample_rate)  # Create loudness meter
    loudness = meter.integrated_loudness(audio_data)

    if loudness < -100:
        return audio_data, 0.0

    gain = target_loudness - loudness
    normalized_audio = pyln.normalize.loudness(audio_data, loudness, target_loudness)
    return normalized_audio, gain

sr = 16000


def normalize_audio(input_wav: str, result_wav: str):
    orig_audio, sr = sf.read(input_wav)

    orig_loudness = measure_loudness(orig_audio, sr)

    pred_audio, _ = sf.read(result_wav)
    pred_loudness = measure_loudness(pred_audio, sr)

    normalized_pred, gain = normalize_loudness(pred_audio, sr, orig_loudness)

    sf.write(result_wav, normalized_pred, sr)
