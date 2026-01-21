"""
VoxMorph Utilities
------------------
Contains mathematical primitives for embedding interpolation and 
robust audio signal processing for directory consolidation.

Methods supported:
1. Linear Averaging (Baseline)
2. Lerp (Linear Interpolation)
3. Slerp (Spherical Linear Interpolation)
"""

import os
import torch
import numpy as np
import torchaudio
from pathlib import Path
from typing import Union

class MathUtils:
    """Geometric operations for vector interpolation."""

    @staticmethod
    def linear_averaging(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
        """
        Simple Linear Averaging.
        Formula: (v0 + v1) / 2
        Note: This ignores alpha and produces the static midpoint.
        """
        return (v0 + v1) / 2.0

    @staticmethod
    def lerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """
        Linear Interpolation (Weighted).
        Formula: (1 - t) * v0 + t * v1
        """
        return (1.0 - t) * v0 + t * v1

    @staticmethod
    def slerp(v0: np.ndarray, v1: np.ndarray, t: float, dot_threshold: float = 0.9995) -> np.ndarray:
        """
        Spherical Linear Interpolation (Weighted).
        Preserves the geometric magnitude on the hypersphere.
        """
        try:
            v0_t = torch.from_numpy(v0)
            v1_t = torch.from_numpy(v1)

            # Validation
            if v0_t.numel() == 0 or v1_t.numel() == 0:
                raise ValueError("Input vectors for Slerp are empty.")

            # Normalize vectors to unit sphere for angle calculation
            v0_norm = v0_t / torch.norm(v0_t)
            v1_norm = v1_t / torch.norm(v1_t)

            # Calculate dot product (cosine of angle)
            dot = torch.sum(v0_norm * v1_norm)
            dot = torch.clamp(dot, -1.0, 1.0)

            # If vectors are parallel (close to 1 or -1), fallback to Lerp
            if torch.abs(dot) > dot_threshold:
                return MathUtils.lerp(v0, v1, t)

            # Calculate angular distance
            theta_0 = torch.acos(torch.abs(dot))
            sin_theta_0 = torch.sin(theta_0)

            if sin_theta_0 < 1e-8:
                return MathUtils.lerp(v0, v1, t)

            # Slerp Formula
            theta_t = theta_0 * t
            sin_theta_t = torch.sin(theta_t)
            
            s0 = torch.cos(theta_t) - dot * sin_theta_t / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            
            result = s0 * v0_t + s1 * v1_t
            return result.numpy()

        except Exception:
            # Safe fallback
            return MathUtils.lerp(v0, v1, t)

class AudioUtils:
    """Audio processing for handling single files or directories of clips."""

    @staticmethod
    def load_and_prep(path: Path, target_sr: int = 24000) -> torch.Tensor:
        """Loads audio, converts to mono, and resamples."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        return wav

    @staticmethod
    def consolidate_input(input_path: Union[str, Path], temp_dir: Path, target_sr: int = 24000) -> Path:
        """
        Analyzes input. 
        - If file: returns path. 
        - If directory: concatenates all audio files into a temporary file.
        
        Note: We default to target_sr=24000 because Chatterbox's internal S3Gen 
        requires 24kHz audio. The external encoder patches will downsample to 16kHz 
        internally for their respective models.
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            return input_path
        
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path not found: {input_path}")
            
        # Collect all audio files
        extensions = ['*.wav', '*.flac', '*.mp3', '*.ogg']
        audio_files = []
        for ext in extensions:
            audio_files.extend(list(input_path.glob(ext)))
            
        if not audio_files:
            raise ValueError(f"No audio files found in directory: {input_path}")
            
        # Sort by size (descending) to prioritize substantive clips
        audio_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        
        # Concatenate (Limit to top 15 clips to avoid OOM while ensuring diverse profile)
        combined_audio = []
        for f in audio_files[:15]:
            try:
                wav = AudioUtils.load_and_prep(f, target_sr)
                combined_audio.append(wav)
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
                
        if not combined_audio:
            raise ValueError("Failed to load any valid audio clips.")

        final_tensor = torch.cat(combined_audio, dim=1)
        
        # Save temp file
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_out = temp_dir / f"consolidated_{input_path.name}.wav"
        
        torchaudio.save(temp_out, final_tensor, target_sr)
        
        return temp_out