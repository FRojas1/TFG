"""
Data loaders for the Speak & Improve Corpus 2025.

Handles three data formats:
  - TSV file lists  (file_id -> audio path)
  - STM references  (file_id -> time-aligned verbatim transcript)
  - JSON annotations (file_id -> per-word marks: disfluency, pronunciation, partial)
"""

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Utterance:
    file_id: str
    audio_path: str
    reference: str | None = None
    segments: list = field(default_factory=list)
    annotations: dict | None = None


class SANDiDataset:
    """Speak & Improve corpus loader.

    Parameters
    ----------
    flist_path : path to a TSV file list  (file_id \\t relative_audio_path)
    data_root  : corpus root directory (audio paths in the TSV are relative to this)
    stm_path   : optional STM file with verbatim reference transcripts
    annotations_path : optional JSON file with per-word annotation marks
    """

    def __init__(self, flist_path, data_root, stm_path=None, annotations_path=None):
        self.data_root = Path(data_root)
        self.utterances: dict[str, Utterance] = {}
        self._load_flist(flist_path)
        if stm_path:
            self._load_stm(stm_path)
        if annotations_path:
            self._load_annotations(annotations_path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_flist(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_id, rel_audio = line.split("\t", 1)
                self.utterances[file_id] = Utterance(
                    file_id=file_id,
                    audio_path=str(self.data_root / rel_audio),
                )

    def _load_stm(self, path):
        """Parse NIST STM file.

        Format per line:
            file_id channel speaker start end <metadata> text
        """
        segments: dict[str, list] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(";;") or not line.strip():
                    continue
                parts = line.strip().split(None, 6)
                if len(parts) < 7:
                    continue
                file_id = parts[0]
                start, end = float(parts[3]), float(parts[4])
                text = parts[6]
                if text == "IGNORE_TIME_SEGMENT_IN_SCORING":
                    continue
                segments.setdefault(file_id, []).append(
                    {"start": start, "end": end, "text": text}
                )

        for file_id, segs in segments.items():
            if file_id in self.utterances:
                self.utterances[file_id].reference = " ".join(
                    s["text"] for s in segs
                )
                self.utterances[file_id].segments = segs

    def _load_annotations(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("files", []):
            file_id = entry["File-id"]
            if file_id in self.utterances:
                self.utterances[file_id].annotations = entry

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_transcript_info(transcript_items):
        """Build plain text and per-word annotations from the JSON transcript list.

        Returns dict with keys:
            text  : str — all spoken words joined by spaces (uppercase, as in corpus)
            words : list[str]
            marks : list[dict] — only words that carry marks (disfluency, pronunciation, partial)
            tags  : list[dict] — structural tags (hesitation, speech-unit-*)
        """
        words = []
        marks = []
        tags = []
        word_idx = 0

        for item in transcript_items:
            if "word" in item:
                word = item["word"]
                words.append(word)
                if "marks" in item:
                    marks.append(
                        {"word_idx": word_idx, "word": word, "marks": item["marks"]}
                    )
                word_idx += 1
            elif "tag" in item:
                tags.append({"after_word_idx": word_idx - 1, "tag": item["tag"]})

        return {
            "text": " ".join(words),
            "words": words,
            "marks": marks,
            "tags": tags,
        }

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances.values())

    def __getitem__(self, file_id):
        return self.utterances[file_id]
