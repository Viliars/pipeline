from efficientnet import predict_class
from hybrid import process_hybrid
from vinp import process_vinp
from gtcrn import process_grcrn
from utils import normalize_audio

def process_with_pipeline(wav_file, result_path):
    print(f"Process {wav_file} -> ", end="", flush=True)

    _, class_name = predict_class(wav_file)

    print(f"Predicted Class: {class_name} -> ", end="", flush=True)

    if class_name == "Reverb":
        print(f"VINP processing -> ", end="", flush=True)
        process_vinp(wav_file, result_path)
    else:
        print(f"Hybrid processing -> ", end="", flush=True)
        process_hybrid(wav_file, result_path)
        print(f"GTCRN processing -> ", end="", flush=True)
        process_grcrn(result_path, result_path)
        normalize_audio(wav_file, result_path)

    print(f"Result saved in {result_path}", flush=True)


def process_only_hybrid(wav_file, result_path):
    print(f"Process {wav_file} -> ", end="", flush=True)

    print(f"Hybrid processing -> ", end="", flush=True)
    process_hybrid(wav_file, result_path)
    normalize_audio(wav_file, result_path)

    print(f"Result saved in {result_path}", flush=True)


def main(input_dir, output_dir, simple):
    if simple:
        process_wav = process_only_hybrid
    else:
        process_wav = process_with_pipeline

    os.makedirs(output_dir, exist_ok=True)

    wavs = sorted([f for f in os.listdir(input_dir) if f.endswith(".wav")])
    print(f"Found {len(wavs)} wav files in {input_dir}")

    for fname in wavs:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        process_wav(in_path, out_path)

    print(f"\n✅ Done! Enhanced files saved to: {output_dir}")


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline #5 inference on dir")
    parser.add_argument("--input_dir", type=str, required=True, help="Input wav file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output wav path")
    parser.add_argument("--simple", action="store_true", help="Use only Hybrid 3 model")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.simple)
