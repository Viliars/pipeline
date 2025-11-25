# Pipeline P5

### Install with new env
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run pipeline P5 on one .wav file
```bash
python process_wav.py --input test.wav --output result.wav
```

### Run pipeline P5 on Dir with wavs
```bash
python process_dir.py --input_dir noisy/ --output output/
```

### Run only Hybrid 3 on one .wav file
```bash
python process_wav.py --input test.wav --output result.wav --simple
```

### Run only Hybrid 3 on Dir with wavs
```bash
python process_dir.py --input_dir noisy/ --output output/ --simple
```


# Docker

### build & run docker
```bash
docker build . -t pipeline
docker run --gpus all -v ./dataset/:/app/dataset/ --rm -it pipeline bash
```

### pull & run docker
```bash
docker run --gpus all -v ./dataset/:/app/dataset/ --rm -it sdapi/pipeline:latest bash
```

### Run pipeline P5 in docker
```bash
python3 process_dir.py --input_dir dataset/input/ --output dataset/output/
```