# Error Preservation in ASR for Language Learners

Bachelor's thesis (TFG) project: evaluating whether automatic speech recognition systems preserve or auto-correct grammatical errors in L2 English learner speech, using the [Speak & Improve Corpus 2025](https://arxiv.org/abs/2412.11986).

**Repository:** https://github.com/FRojas1/TFG  
**Thesis (PDF):** [memoria.pdf](memoria.pdf)

## Contents

| Path | Description |
|------|-------------|
| `memoria.pdf` | Full thesis document |
| `doc/main-en.typ` | Typst source of the thesis |
| `doc/figures/` | Figures embedded in the thesis |
| `src/` | Transcription, evaluation, fine-tuning, and analysis scripts |
| `configs/` | Model configuration YAML files |
| `results/` | Pre-computed evaluation JSONs (WER + GEP) for published models |
| `checkpoints/` | DPO training logs (weights not included; see below) |

## Dataset

The **Speak & Improve Corpus 2025** is not included in this repository (license / size). Obtain it from the official distribution:

1. Register and download from the [Speak & Improve Corpus 2025](https://researchdatasets.cambridge.org/datasets/speak-and-improve-corpus-2025) page.
2. Extract the archive and set the corpus root, e.g.:

```bash
export SANDI_CORPUS=/path/to/sandi-corpus-2025
```

The pipeline expects the standard layout under `data/` and `reference-materials/` as described in the corpus `README.txt`.

## Requirements

- Python 3.10+
- CUDA GPU recommended (12 GB+ for Whisper fine-tuning)
- [Typst](https://typst.app/) (optional, to recompile `memoria.pdf`)

```bash
# PyTorch with CUDA (adjust CUDA version if needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Optional: NeMo for Parakeet / Canary models
# pip install nemo_toolkit[asr]
```

## Reproducing published results

### 1. Transcribe (baseline models)

```bash
python src/transcribe.py --config configs/whisper_small.yaml --split eval
python src/transcribe.py --config configs/whisper_medium.yaml --split eval
python src/transcribe.py --config configs/wav2vec2_large.yaml --split eval
python src/transcribe.py --config configs/parakeet_ctc.yaml --split eval
python src/transcribe.py --config configs/parakeet_tdt_ctc_110m.yaml --split eval
python src/transcribe.py --config configs/canary_180m_flash.yaml --split eval
python src/transcribe.py --config configs/moonshine_base.yaml --split eval
```

### 2. Evaluate (WER + GEP)

```bash
python src/evaluate.py --results results/whisper-small-en/eval.json
# Repeat for each model output directory; produces *-eval.json summaries.
```

Pre-computed `*-eval.json` files under `results/` match the numbers reported in `memoria.pdf`.

### 3. Regenerate thesis figures

```bash
python doc/generate_pipeline_flowchart.py
python doc/generate_figures.py
python doc/generate_proficiency_figures.py
python doc/generate_lightweight_figures.py
python doc/generate_parakeet_comparison_figures.py
python doc/generate_finetune_dpo_figures.py
```

### 4. DPO fine-tuning (optional; requires GPU + restored preference pool)

```bash
python src/build_finetune_pool.py
python src/build_gep_targets.py
python src/restore_errors_llm.py   # requires LLM API credentials
python src/finetune_dpo.py
python src/transcribe_lora_fair.py
python src/evaluate.py --results results/whisper-small-dpo-3epochs/eval-fair.json
```

Training curves use `checkpoints/whisper-small-dpo-3epochs/train_log.json` (included). Adapter weights are **not** shipped; re-run `finetune_dpo.py` to obtain them.

### 5. Recompile thesis PDF

```bash
cd doc
typst compile main-en.typ ../memoria.pdf
```

## Citation

If you use this code or methodology, please cite the Speak & Improve corpus and myself, Felipe Rojas as indicated in `memoria.pdf`.

## License

Code in this repository is provided for academic reproducibility. The Speak & Improve dataset remains subject to its own terms of use.
