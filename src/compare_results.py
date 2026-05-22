"""Compare WER+GEP across all evaluated models."""
import json, sys
from pathlib import Path

evals = [
    ("Whisper Small (baseline)", "results/whisper-small-en/eval-test-eval.json"),
    ("LoRA v2 (standard)", "results/whisper-small-en-lora-v2/eval-test-eval.json"),
    ("LoRA v2 + EOS penalty", "results/whisper-small-en-lora-v2-eos/eval-test-eval.json"),
    ("LoRA GEP-masked", "results/whisper-small-en-lora-gep/eval-test-eval.json"),
    ("Whisper Medium (baseline)", "results/whisper-medium-en/eval-test-eval.json"),
    ("Wav2Vec2 (baseline)", "results/wav2vec2-large-960h/eval-test-eval.json"),
    ("Parakeet CTC (baseline)", "results/parakeet-ctc-1.1b/eval-test-eval.json"),
    ("Full FT GEP (decoder)", "results/whisper-small-en-full-gep/eval-test-eval.json"),
]

print(f"{'Model':<30} {'WER':>8} {'GEP%':>8} {'Corr%':>8} {'EPR%':>8}")
print("-" * 70)

for name, path in evals:
    if not Path(path).exists():
        print(f"{name:<30} {'(pending)':>8}")
        continue
    d = json.load(open(path))
    wer = d.get("wer", {}).get("overall_wer") or d.get("overall_wer", 0)

    g = d.get("gep", {}).get("summary", {})
    if g:
        t = sum(v["total"] for v in g.values())
        p = sum(v["preserved"] for v in g.values())
        c = sum(v["corrected"] for v in g.values())
        gep = p / t * 100 if t else 0
        corr = c / t * 100 if t else 0
    else:
        gep = corr = 0

    epr_data = d.get("epr", {}).get("summary", {})
    if epr_data:
        epr_total = sum(v.get("total", 0) for v in epr_data.values())
        epr_pres = sum(v.get("preserved", 0) for v in epr_data.values())
        epr = epr_pres / epr_total * 100 if epr_total else 0
    else:
        epr = 0

    print(f"{name:<30} {wer*100:>7.2f}% {gep:>7.2f}% {corr:>7.2f}% {epr:>7.2f}%")
