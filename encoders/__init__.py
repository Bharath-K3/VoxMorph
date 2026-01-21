"""
Encoders Package
---------------
Provides a modular interface for patching ChatterboxTTS with various 
speaker verification encoders (ECAPA-TDNN, HuBERT, Wav2Vec2).
"""

# Import specific encoder patchers
from .ecapa import apply_patch as apply_ecapa_patch
from .hubert import apply_patch as apply_hubert_patch
from .wav2vec2 import apply_patch as apply_wav2vec2_patch

# Registry mapping configuration names to patch functions
# This allows easy extension if you want to add new encoders later.
PATCHERS = {
    "ecapa": apply_ecapa_patch,
    "hubert": apply_hubert_patch,
    "wav2vec2": apply_wav2vec2_patch,
}

__all__ = ["PATCHERS"]