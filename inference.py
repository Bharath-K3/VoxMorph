"""
VoxMorph: Scalable Zero-Shot Voice Identity Morphing via Disentangled Embeddings.
ICASSP Accepted Paper Implementation - Command Line Interface.

This module provides a professional CLI for the VoxMorph framework, enabling
batch processing and headless synthesis operations.

Usage:
    python inference.py --alpha 0.5 --text "Hello World"
    python inference.py --source_a "path/to/a.wav" --source_b "path/to/b.wav"

"""

import argparse
import sys
import os
import gc
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Chatterbox TTS dependencies
try:
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from chatterbox.models.t3.modules.cond_enc import T3Cond
except ImportError as e:
    sys.exit(f"[ERROR] Critical dependency missing: {e}. Please install chatterbox-tts.")

# ==============================================================================
# 1. Configuration & Logging Setup
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VoxMorph-CLI")

class VoxConfig:
    """Configuration parameters for the VoxMorph inference engine."""
    SAMPLE_RATE = 24000
    DOT_THRESHOLD = 0.9995
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Default Paths
    DEFAULT_ASSET_DIR = Path("Assets")
    DEFAULT_A = DEFAULT_ASSET_DIR / "Audio1.flac"
    DEFAULT_B = DEFAULT_ASSET_DIR / "Audio2.flac"
    DEFAULT_TEXT = "The concept of morphing implies a smooth and seamless transition."
    DEFAULT_OUTPUT_DIR = Path("Outputs")

# ==============================================================================
# 2. Mathematical Utilities
# ==============================================================================

class MathUtils:
    """Static utilities for geometric operations on the hypersphere."""
    
    @staticmethod
    def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """
        Performs Spherical Linear Interpolation (Slerp) between two vectors.
        Preserves the geometric structure of the speaker embedding manifold.
        """
        try:
            v0_t = torch.from_numpy(v0)
            v1_t = torch.from_numpy(v1)

            if v0_t.numel() == 0 or v1_t.numel() == 0:
                raise ValueError("Input vectors for Slerp are empty.")

            v0_t = v0_t / torch.norm(v0_t)
            v1_t = v1_t / torch.norm(v1_t)

            dot = torch.sum(v0_t * v1_t)
            dot = torch.clamp(dot, -1.0, 1.0)

            if torch.abs(dot) > VoxConfig.DOT_THRESHOLD:
                return ((1 - t) * v0 + t * v1)

            theta_0 = torch.acos(torch.abs(dot))
            sin_theta_0 = torch.sin(theta_0)

            if sin_theta_0 < 1e-8:
                return ((1 - t) * v0 + t * v1)

            theta_t = theta_0 * t
            sin_theta_t = torch.sin(theta_t)
            
            s0 = torch.cos(theta_t) - dot * sin_theta_t / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            
            result = s0 * v0_t + s1 * v1_t
            return result.numpy()

        except Exception:
            return ((1 - t) * v0 + t * v1)

# ==============================================================================
# 3. Inference Engine
# ==============================================================================

class VoxMorphEngine:
    """
    Core engine for handling model loading, embedding extraction, 
    interpolation, and audio synthesis.
    """
    
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model is None:
            logger.info(f"Initializing ChatterboxTTS on {VoxConfig.DEVICE}...")
            self.model = ChatterboxTTS.from_pretrained(device=VoxConfig.DEVICE)
            logger.info("Model loaded successfully.")

    def _cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _extract_embeddings(self, audio_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        self.model.prepare_conditionals(str(audio_path))
        conds = Conditionals(self.model.conds.t3, self.model.conds.gen)
        
        emb_t3 = conds.t3.speaker_emb.detach().squeeze(0).cpu().numpy()
        emb_gen = conds.gen['embedding'].detach().squeeze(0).cpu().numpy()
        
        return emb_t3, emb_gen

    def synthesize(self, 
                   path_a: Path, 
                   path_b: Path, 
                   text: str, 
                   alpha: float) -> Tuple[int, np.ndarray]:
        """
        Executes the full zero-shot morphing pipeline.
        """
        self._cleanup_memory()

        try:
            # Stage 1: Extraction
            logger.info("Profiling Source Speaker A...")
            t3_a, gen_a = self._extract_embeddings(path_a)
            # Template for reconstruction
            conds_template = Conditionals(self.model.conds.t3, self.model.conds.gen)

            logger.info("Profiling Target Speaker B...")
            t3_b, gen_b = self._extract_embeddings(path_b)

            # Stage 2: Interpolation
            logger.info(f"Interpolating Embeddings (Alpha={alpha:.2f})...")

            if alpha == 0.0:
                final_t3_emb = torch.from_numpy(t3_a).unsqueeze(0)
                final_gen_emb = torch.from_numpy(gen_a).unsqueeze(0)
            elif alpha == 1.0:
                final_t3_emb = torch.from_numpy(t3_b).unsqueeze(0)
                final_gen_emb = torch.from_numpy(gen_b).unsqueeze(0)
            else:
                morphed_t3 = MathUtils.slerp(t3_a, t3_b, alpha)
                final_t3_emb = torch.from_numpy(morphed_t3).unsqueeze(0)

                morphed_gen = MathUtils.slerp(gen_a, gen_b, alpha)
                final_gen_emb = torch.from_numpy(morphed_gen).unsqueeze(0)

            final_t3_cond = T3Cond(
                speaker_emb=final_t3_emb,
                cond_prompt_speech_tokens=conds_template.t3.cond_prompt_speech_tokens,
                emotion_adv=conds_template.t3.emotion_adv
            )
            
            final_gen_cond = conds_template.gen.copy()
            final_gen_cond['embedding'] = final_gen_emb
            
            final_conds = Conditionals(final_t3_cond, final_gen_cond)

            # Stage 3: Synthesis
            logger.info("Synthesizing Morphed Waveform...")
            self.model.conds = final_conds.to(self.model.device)
            
            # Using tqdm for synthesis progress if possible, otherwise blocking
            wav_tensor = self.model.generate(text)
            
            wav_cpu = wav_tensor.cpu().squeeze().numpy()
            self._cleanup_memory()
            
            return VoxConfig.SAMPLE_RATE, wav_cpu

        except Exception as e:
            self._cleanup_memory()
            logger.error(f"Synthesis failed: {e}")
            raise e

# ==============================================================================
# 4. Command Line Interface Handler
# ==============================================================================

def validate_inputs(args):
    """Ensures input files exist or defaults are valid."""
    path_a = Path(args.source_a) if args.source_a else VoxConfig.DEFAULT_A
    path_b = Path(args.source_b) if args.source_b else VoxConfig.DEFAULT_B
    
    if not path_a.exists():
        logger.error(f"Source A not found at: {path_a}")
        logger.info(f"Please create the directory '{VoxConfig.DEFAULT_ASSET_DIR}' and add 'Audio1.flac', or provide a path.")
        sys.exit(1)
        
    if not path_b.exists():
        logger.error(f"Source B not found at: {path_b}")
        sys.exit(1)
        
    return path_a, path_b

def main():
    parser = argparse.ArgumentParser(
        description="VoxMorph: Zero-Shot Voice Identity Morphing CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input Arguments
    parser.add_argument("--source_a", type=str, help="Path to Source Speaker A audio file.")
    parser.add_argument("--source_b", type=str, help="Path to Source Speaker B audio file.")
    parser.add_argument("--text", type=str, default=VoxConfig.DEFAULT_TEXT, help="Text to synthesize.")
    
    # Morphing Arguments
    parser.add_argument("--alpha", type=float, default=0.5, help="Morphing Factor (0.0 to 1.0).")
    
    # Output Arguments
    parser.add_argument("--output_dir", type=str, default=str(VoxConfig.DEFAULT_OUTPUT_DIR), help="Directory to save results.")
    parser.add_argument("--filename", type=str, help="Custom output filename (optional).")

    args = parser.parse_args()

    # 1. Validation
    path_a, path_b = validate_inputs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialization
    print("=" * 60)
    print("  VOXMORPH CLI INFERENCE  ")
    print("=" * 60)
    
    engine = VoxMorphEngine()

    # 3. Execution
    start_time = time.time()
    try:
        sample_rate, waveform = engine.synthesize(
            path_a=path_a,
            path_b=path_b,
            text=args.text,
            alpha=args.alpha
        )

        # 4. Saving Output
        if args.filename:
            out_name = args.filename
            if not out_name.endswith(".wav"): out_name += ".wav"
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_name = f"morph_alpha{args.alpha:.2f}_{timestamp}.wav"
        
        output_path = output_dir / out_name
        
        sf.write(output_path, waveform, sample_rate)
        
        elapsed = time.time() - start_time
        print("-" * 60)
        logger.info(f"Success! Audio saved to: {output_path}")
        logger.info(f"Total processing time: {elapsed:.2f} seconds")
        print("=" * 60)

    except Exception as e:
        logger.error("Inference process terminated due to an error.")
        sys.exit(1)

if __name__ == "__main__":
    main()