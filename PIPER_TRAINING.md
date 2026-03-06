# Training a BMO voice with Piper TTS (piper1-gpl)

This guide goes from your `wav_output/` (from `bmo_voice_fetch.py`) to a Piper ONNX model you can use with `piper`. It follows the [OHF-Voice/piper1-gpl training docs](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md).

**Target environment:** Ubuntu; GPU examples below for **RTX 6000 Ada (48GB)** or **NVIDIA GH200 (Grace Hopper, 96GB)**. GH200 nodes are **aarch64** (ARM); that’s supported—see [GH200 / aarch64](#2a-gh200-aarch64) below.

## 1. Prepare the dataset

Convert the dialogue CSV and WAVs into Piper’s expected layout: `piper_dataset/wav/` plus a pipe-delimited `metadata.csv` (no header). Piper expects the first column to be the **audio file name** and the second the **text** for that utterance (used with espeak-ng for phonemization).

```bash
# With audio normalization (recommended: 22050 Hz mono)
pip install pydub   # optional; for resample/mono conversion
python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset

# Without normalization (if your WAVs are already 22050 Hz mono)
python prepare_piper_dataset.py wav_output/dialogue.csv -o piper_dataset --no-normalize
```

You should see:

- `piper_dataset/wav/` – one WAV per utterance (mono, 22050 Hz if you didn’t use `--no-normalize`)
- `piper_dataset/metadata.csv` – pipe-delimited `filename.wav|text`, no header

**Note:** Piper recommends 1,300–1,500+ phrases for fine-tuning. With ~150 samples you can still try fine-tuning; more data will improve quality.

---

## 2. Install Piper training environment (piper1-gpl)

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-dev espeak-ng
```

Clone **piper1-gpl** and set up the Python env:

```bash
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[train]'
```

Build the Cython extension:

```bash
./build_monotonic_align.sh
```

If you are developing from the repo and need a dev build:

```bash
python3 setup.py build_ext --inplace
```

Ensure NVIDIA drivers and CUDA are installed so PyTorch can use the GPU (`nvidia-smi` should work).

### 2a. GH200 / aarch64

GH200 servers use an **aarch64** (ARM64) Grace CPU. Piper training is fine on aarch64: the `monotonic_align` extension and espeak-ng build from source.

**PyTorch + CUDA on aarch64:** On bare-metal aarch64, PyTorch from pip sometimes doesn’t see CUDA correctly. The reliable approach on GH200 is to use an **NVIDIA NGC PyTorch container** (published for aarch64 for Grace):

```bash
# Example: run a shell in the NGC PyTorch image (use a tag that supports your driver)
docker run -it --gpus all -v /path/to/bmo:/workspace/bmo nvcr.io/nvidia/pytorch:24.10-py3 bash
# Inside the container:
apt-get update && apt-get install -y espeak-ng build-essential cmake ninja-build
cd /workspace/bmo
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
pip install -e '.[train]'
./build_monotonic_align.sh
# Then run training as in sections 3–4, using /workspace/bmo paths
```

If your cluster provides a preconfigured env (module load, conda, or venv) with PyTorch and CUDA working on aarch64, you can use that and follow the normal steps above; just ensure `python3-dev`, `espeak-ng`, `build-essential`, `cmake`, and `ninja-build` are installed.

---

## 3. Train (no separate preprocess)

Piper1-gpl trains directly from the CSV and audio directory; there is no separate preprocess step. Phonemization is done at training time with espeak-ng.

**Download a checkpoint** (recommended for fine-tuning with limited data). Only **medium** quality checkpoints are supported without tweaking other settings.

- [Piper checkpoints on Hugging Face](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
- Example: [en_US lessac, medium](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/lessac/medium) – download the `.ckpt` file.

From the `piper1-gpl` directory (with the venv active), run:

```bash
python3 -m piper.train fit \
  --data.voice_name "bmo" \
  --data.csv_path /path/to/bmo/piper_dataset/metadata.csv \
  --data.audio_dir /path/to/bmo/piper_dataset/wav/ \
  --model.sample_rate 22050 \
  --data.espeak_voice "en-us" \
  --data.cache_dir /path/to/bmo/piper_dataset/cache \
  --data.config_path /path/to/bmo/piper_dataset/config.json \
  --data.batch_size 32 \
  --ckpt_path /path/to/lessac/epoch=2164-step=1355540.ckpt
```

Replace `/path/to/bmo/` with your actual path. The **config** for the voice is written during training to `--data.config_path`; you will use it next to export and for the final `.onnx.json`.

**GPU / batch size:**

- **RTX 6000 Ada (48GB):** `--data.batch_size 32` is usually fine; you can try `--data.batch_size 64` if you have headroom.
- **NVIDIA GH200 (96GB):** e.g. `--data.batch_size 64` or higher.
- If you hit OOM, lower `--data.batch_size` (e.g. 16 or 8).

Run `python3 -m piper.train fit --help` for more options (e.g. `max_epochs`, checkpoint frequency, precision).

Checkpoints are written under the Lightning default (e.g. `lightning_logs/version_0/checkpoints/`). Use the best one for export.

---

## 4. Export to ONNX

Export the best checkpoint to ONNX:

```bash
python3 -m piper.train.export_onnx \
  --checkpoint /path/to/lightning_logs/version_0/checkpoints/your-best.ckpt \
  --output-file /path/to/bmo/beemo.onnx
```

To match other Piper voices, name the model and config as a pair:

- `en_US-bmo-medium.onnx` (or `beemo.onnx` if you prefer)
- `en_US-bmo-medium.onnx.json` (or `beemo.onnx.json`)

The JSON is the file that was written to `--data.config_path` **during training**. Copy/rename it to sit next to the ONNX file:

```bash
cp /path/to/bmo/piper_dataset/config.json /path/to/bmo/beemo.onnx.json
```

---

## 5. Run Piper with your model

```bash
echo "Hello! It's me, BMO!" | piper -m beemo.onnx --output_file out.wav
# Or use the Piper Python API / other Piper clients with beemo.onnx + beemo.onnx.json
```

---

## References

- [piper1-gpl TRAINING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md) – official training guide
- [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl)
- [Piper checkpoints (Hugging Face)](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
