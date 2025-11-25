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


def main(input, output, simple):
    if simple:
        process_only_hybrid(input, output)
    else:
        process_with_pipeline(input, output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline #5 inference on 1 wav file")
    parser.add_argument("--input", type=str, required=True, help="Input wav file")
    parser.add_argument("--output", type=str, required=True, help="Output wav path")
    parser.add_argument("--simple", action="store_true", help="Use only Hybrid 3 model")
    args = parser.parse_args()
    main(args.input, args.output, args.simple)
