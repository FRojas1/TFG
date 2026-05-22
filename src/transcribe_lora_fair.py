"""Backward-compatible alias for ``transcribe_lora.py``.

The "fair" suffix is no longer meaningful: ``transcribe_lora.py`` now uses
the HuggingFace pipeline with long-form decoding, so this file just
re-exports its ``main`` for any scripts that still call ``transcribe_lora_fair.py``.

Prefer ``transcribe_lora.py`` for new code.
"""

from transcribe_lora import main


if __name__ == "__main__":
    main()
