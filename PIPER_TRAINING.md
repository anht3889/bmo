# Training a BMO voice with Piper TTS

This guide goes from your `wav_output/` (from `bmo_voice_fetch.py`) to a Piper ONNX model you can use with `piper`.

**Target environment:** Ubuntu; GPU examples below for **RTX 6000 Ada (48GB)** or **NVIDIA GH200 (Grace Hopper, 96GB)**. GH200 nodes are **aarch64** (ARM); that’s supported—see [GH200 / aarch64](#2a-gh200-aarch64) below.

## 1. Prepare the dataset

Convert the dialogue CSV and WAVs into Piper’s expected layout (e.g. `piper_dataset/wav/` + `metadata.csv`, 22050 Hz mono):

```bash
# With audio normalization (recommended: 22050 Hz mono)
pip install pydub   # optional; for resample/mono conversion
python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset

# Without normalization (if your WAVs are already 22050 Hz mono)
python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset --no-normalize
```

You should see:

- `piper_dataset/wav/` – one WAV per line (mono, 22050 Hz if you didn’t use `--no-normalize`)
- `piper_dataset/metadata.csv` – pipe-delimited `id|text`, no header

**Note:** Piper recommends 1,300–1,500+ phrases for fine-tuning. With ~150 samples you can still try fine-tuning; more data will improve quality.

---

## 2. Install Piper training environment (Ubuntu)

Install system deps and espeak-ng:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev espeak-ng build-essential
```

Clone Piper and set up the Python env:

```bash
git clone https://github.com/rhasspy/piper.git
cd piper/src/python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
```

Build the alignment extension:

```bash
./build_monotonic_align.sh
```

Ensure NVIDIA drivers and CUDA are installed so PyTorch can use the GPU (`nvidia-smi` should work).

### 2a. GH200 / aarch64

GH200 servers use an **aarch64** (ARM64) Grace CPU. Piper training is fine on aarch64: the `monotonic_align` extension and espeak-ng build from source.

**PyTorch + CUDA on aarch64:** On bare-metal aarch64, PyTorch from pip sometimes doesn’t see CUDA correctly. The reliable approach on GH200 is to use an **NVIDIA NGC PyTorch container** (published for aarch64 for Grace):

```bash
# Example: run a shell in the NGC PyTorch image (use a tag that supports your driver)
docker run -it --gpus all -v /path/to/bmo:/workspace/bmo nvcr.io/nvidia/pytorch:24.10-py3 bash
# Inside the container:
apt-get update && apt-get install -y espeak-ng
cd /workspace/bmo
git clone https://github.com/rhasspy/piper.git
cd piper/src/python
pip install --upgrade pip wheel setuptools && pip install -e .
./build_monotonic_align.sh
# Then run preprocess and piper_train as in sections 3–4, using /workspace/bmo paths
```

If your cluster provides a preconfigured env (module load, conda, or venv) with PyTorch and CUDA working on aarch64, you can use that and follow the normal steps above; just ensure `python3-dev`, `espeak-ng`, and `build-essential` are installed so the Piper extension builds.

---

## 3. Preprocess the dataset

From `piper/src/python` (with the venv active):

```bash
python3 -m piper_train.preprocess \
  --language en-us \
  --input-dir /path/to/bmo/piper_dataset \
  --output-dir /path/to/bmo/piper_training \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050
```

Replace `/path/to/bmo/` with the path to your `bmo` project. This produces:

- `piper_training/config.json`
- `piper_training/dataset.jsonl`
- Normalized audio and spectrogram files used during training

---

## 4. Fine-tune from a checkpoint (recommended)

Training from scratch needs a lot of data; fine-tuning from an existing English checkpoint works better with ~150 clips.

1. **Download a checkpoint** (e.g. medium-quality, English US):
   - [Piper checkpoints on Hugging Face](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US)
   - Example: [lessac, medium](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/lessac/medium) – download the `.ckpt` file.

2. **Run training** (from `piper/src/python`, venv active):

**RTX 6000 Ada (48GB):**

```bash
python3 -m piper_train \
  --dataset-dir /path/to/bmo/piper_training \
  --accelerator gpu \
  --devices 1 \
  --batch-size 32 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 1000 \
  --resume_from_checkpoint /path/to/lessac/epoch=2164-step=1355540.ckpt \
  --checkpoint-epochs 50 \
  --precision 32
```

Add `--quality high` for a larger, better-sounding model (48GB is enough).

**NVIDIA GH200 (Grace Hopper, 96GB HBM3):** same command with a larger batch and high quality:

```bash
python3 -m piper_train \
  --dataset-dir /path/to/bmo/piper_training \
  --accelerator gpu \
  --devices 1 \
  --batch-size 64 \
  --quality high \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 1000 \
  --resume_from_checkpoint /path/to/lessac/epoch=2164-step=1355540.ckpt \
  --checkpoint-epochs 50 \
  --precision 32
```

- If you hit OOM, lower `--batch-size` (e.g. 32 or 16).
- Checkpoints are written under `piper_training/lightning_logs/version_0/checkpoints/`.

---

## 5. Export to ONNX

After training, export the best checkpoint to ONNX and copy the config:

```bash
python3 -m piper_train.export_onnx \
  /path/to/piper_training/lightning_logs/version_0/checkpoints/your-best.ckpt \
  /path/to/bmo/beemo.onnx

cp /path/to/piper_training/config.json /path/to/bmo/beemo.onnx.json
```

---

## 6. Run Piper with your model

```bash
echo "Hello! It's me, BMO!" | piper -m beemo.onnx --output_file out.wav
# Or use the Piper Python API / other Piper clients with beemo.onnx + beemo.onnx.json
```

---

## References

- [Piper training guide](https://tderflinger.github.io/piper-docs/guides/training/)
- [Piper GitHub](https://github.com/rhasspy/piper)
- [Piper checkpoints (Hugging Face)](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
