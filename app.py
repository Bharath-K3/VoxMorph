"""
VoxMorph: Scalable Zero-Shot Voice Identity Morphing via Disentangled Embeddings.
ICASSP Accepted Paper Implementation.

This module provides a Gradio-based interface for the VoxMorph framework.
It enables the interpolation of vocal characteristics (prosody and timbre) 
between two source speakers using Spherical Linear Interpolation (Slerp).

"""

import os
import sys
import gc
import torch
import torchaudio
import numpy as np
import gradio as gr
from typing import Tuple, Optional

# Chatterbox TTS dependencies
from chatterbox.tts import ChatterboxTTS, Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond

# ==============================================================================
# 1. Configuration & Utilities
# ==============================================================================

class VoxConfig:
    """Configuration parameters for the VoxMorph inference engine."""
    SAMPLE_RATE = 24000
    DOT_THRESHOLD = 0.9995  # Threshold for Slerp linearity check
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_TEXT = (
        "The concept of morphing implies a smooth and seamless transition "
        "between two distinct states, blending identity and style."
    )

class MathUtils:
    """Static utilities for geometric operations on the hypersphere."""
    
    @staticmethod
    def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """
        Performs Spherical Linear Interpolation (Slerp) between two vectors.
        
        Equation (1) from the VoxMorph paper. preserving the geometric 
        structure of the speaker embedding manifold.

        Args:
            v0 (np.ndarray): Source vector A.
            v1 (np.ndarray): Target vector B.
            t (float): Interpolation factor alpha [0.0, 1.0].

        Returns:
            np.ndarray: The interpolated vector.
        """
        try:
            # Convert to tensors for robust calculation
            v0_t = torch.from_numpy(v0)
            v1_t = torch.from_numpy(v1)

            # Validation
            if v0_t.numel() == 0 or v1_t.numel() == 0:
                raise ValueError("Input vectors for Slerp are empty.")

            # Normalize vectors to unit sphere
            v0_t = v0_t / torch.norm(v0_t)
            v1_t = v1_t / torch.norm(v1_t)

            # Calculate dot product (cosine of angle)
            dot = torch.sum(v0_t * v1_t)
            dot = torch.clamp(dot, -1.0, 1.0)

            # If vectors are close (parallel), use linear interpolation
            if torch.abs(dot) > VoxConfig.DOT_THRESHOLD:
                v_lerp = torch.lerp(v0_t, v1_t, t)
                norm = torch.norm(v_lerp)
                return (v_lerp / norm).numpy() if norm > 1e-8 else v0_t.numpy()

            # Calculate angular distance
            theta_0 = torch.acos(torch.abs(dot))
            sin_theta_0 = torch.sin(theta_0)

            if sin_theta_0 < 1e-8:
                return torch.lerp(v0_t, v1_t, t).numpy()

            # Slerp Formula
            theta_t = theta_0 * t
            sin_theta_t = torch.sin(theta_t)
            
            s0 = torch.cos(theta_t) - dot * sin_theta_t / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            
            result = s0 * v0_t + s1 * v1_t
            return result.numpy()

        except Exception:
            # Fallback to linear interpolation in case of numerical instability
            return ((1 - t) * v0 + t * v1)


# ==============================================================================
# 2. Inference Engine
# ==============================================================================

class VoxMorphEngine:
    """
    Core engine for handling model loading, embedding extraction, 
    interpolation, and audio synthesis.
    """
    
    def __init__(self):
        """Initializes the TTS model and sets up the execution environment."""
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the ChatterboxTTS model onto the configured device."""
        if self.model is None:
            print(f"[VoxMorph] Initializing model on {VoxConfig.DEVICE}...", flush=True)
            self.model = ChatterboxTTS.from_pretrained(device=VoxConfig.DEVICE)
            print("[VoxMorph] Model loaded successfully.")

    def _cleanup_memory(self):
        """Forces garbage collection and clears CUDA cache to prevent OOM errors."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _extract_embeddings(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Profiles a speaker from an audio file and extracts disentangled embeddings.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Timbre Embedding, Prosody/Gen Embedding)
        """
        self.model.prepare_conditionals(audio_path)
        
        # Access internal conditional structures
        conds = Conditionals(self.model.conds.t3, self.model.conds.gen)
        
        # Detach and convert to numpy for safe manipulation
        # Squeeze(0) removes batch dimension for interpolation
        emb_t3 = conds.t3.speaker_emb.detach().squeeze(0).cpu().numpy()
        emb_gen = conds.gen['embedding'].detach().squeeze(0).cpu().numpy()
        
        return emb_t3, emb_gen

    def synthesize(self, 
                   path_a: str, 
                   path_b: str, 
                   text: str, 
                   alpha: float, 
                   progress=gr.Progress()) -> Tuple[int, np.ndarray]:
        """
        Executes the full zero-shot morphing pipeline.

        1. Extracts disentangled embeddings from Source A and Source B.
        2. Interpolates prosody and timbre independently using Slerp.
        3. Synthesizes the final waveform using the fused embeddings.

        Args:
            path_a (str): Filepath for Speaker A.
            path_b (str): Filepath for Speaker B.
            text (str): The text content to be synthesized.
            alpha (float): Morphing factor [0.0 - 1.0].

        Returns:
            Tuple[int, np.ndarray]: Sample rate and generated waveform data.
        """
        self._cleanup_memory()

        if not path_a or not path_b:
            raise ValueError("Both audio sources must be provided.")

        try:
            # --- Stage 1: Extraction ---
            progress(0.1, desc="Profiling Source Speaker A...")
            t3_a, gen_a = self._extract_embeddings(path_a)
            # Store structural templates from A (tokens, emotion_adv) for reconstruction
            conds_template = Conditionals(self.model.conds.t3, self.model.conds.gen)

            progress(0.3, desc="Profiling Target Speaker B...")
            t3_b, gen_b = self._extract_embeddings(path_b)

            # --- Stage 2: Interpolation ---
            progress(0.5, desc=f"Interpolating Embeddings (Alpha: {alpha:.2f})...")

            if alpha == 0.0:
                final_t3_emb = torch.from_numpy(t3_a).unsqueeze(0)
                final_gen_emb = torch.from_numpy(gen_a).unsqueeze(0)
            elif alpha == 1.0:
                final_t3_emb = torch.from_numpy(t3_b).unsqueeze(0)
                final_gen_emb = torch.from_numpy(gen_b).unsqueeze(0)
            else:
                # Independent Slerp for Timbre (Identity)
                morphed_t3 = MathUtils.slerp(t3_a, t3_b, alpha)
                final_t3_emb = torch.from_numpy(morphed_t3).unsqueeze(0)

                # Independent Slerp for Prosody (Style)
                morphed_gen = MathUtils.slerp(gen_a, gen_b, alpha)
                final_gen_emb = torch.from_numpy(morphed_gen).unsqueeze(0)

            # Reconstruct the Conditional objects required by Chatterbox
            # Note: We utilize the auxiliary tokens/emotion data from Speaker A as the base structure
            final_t3_cond = T3Cond(
                speaker_emb=final_t3_emb,
                cond_prompt_speech_tokens=conds_template.t3.cond_prompt_speech_tokens,
                emotion_adv=conds_template.t3.emotion_adv
            )
            
            final_gen_cond = conds_template.gen.copy()
            final_gen_cond['embedding'] = final_gen_emb
            
            final_conds = Conditionals(final_t3_cond, final_gen_cond)

            # --- Stage 3: Synthesis ---
            progress(0.8, desc="Synthesizing Morphed Waveform...")
            
            # Load fused conditionals into the model
            self.model.conds = final_conds.to(self.model.device)
            
            # Generate audio
            wav_tensor = self.model.generate(text)
            
            # Post-processing for Gradio
            wav_cpu = wav_tensor.cpu().squeeze().numpy()
            
            self._cleanup_memory()
            return VoxConfig.SAMPLE_RATE, wav_cpu

        except Exception as e:
            self._cleanup_memory()
            raise RuntimeError(f"Morphing process failed: {str(e)}")


# ==============================================================================
# 3. User Interface Construction
# ==============================================================================

def create_interface():
    """Builds and launches the Gradio user interface."""
    
    # Initialize Engine
    engine = VoxMorphEngine()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate"), title="VoxMorph") as app:
        
        # Header Section
        gr.Markdown(
            """
            # 🗣️ VoxMorph: Scalable Zero-Shot Voice Identity Morphing
            **University of North Texas | ICASSP Accepted Paper**
            
            This interface implements the **VoxMorph framework**, allowing for high-fidelity voice morphing 
            via disentangled prosody and timbre embeddings.
            """
        )
        
        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 1. Source Input")
                
                with gr.Group():
                    audio_input_a = gr.Audio(
                        label="Source Identity A",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    audio_input_b = gr.Audio(
                        label="Target Identity B",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                
                text_input = gr.Textbox(
                    label="Linguistic Content",
                    value=VoxConfig.DEFAULT_TEXT,
                    lines=4,
                    placeholder="Enter the text to be synthesized..."
                )

            # Right Column: Controls & Output
            with gr.Column(scale=1):
                gr.Markdown("### 2. Morphing Controls")
                
                alpha_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Interpolation Factor (α)",
                    info="0.0 = Source A | 1.0 = Source B | 0.5 = Equal Fusion"
                )
                
                process_btn = gr.Button(
                    "Perform Zero-Shot Morphing", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("### 3. Acoustic Output")
                audio_output = gr.Audio(
                    label="Synthesized Morph", 
                    interactive=False,
                    type="numpy"
                )

        # Footer / Technical Details
        with gr.Accordion("Technical Architecture", open=False):
            gr.Markdown(
                """
                **Methodology:**
                The framework operates by disentangling vocal characteristics into **Prosody (Style)** and **Timbre (Identity)** embeddings.
                These embeddings are projected onto a hypersphere and interpolated using **Spherical Linear Interpolation (Slerp)** 
                to ensure geometric consistency. The fused embeddings condition an autoregressive language model and a 
                Conditional Flow Matching (CFM) network to synthesize the final waveform.
                """
            )

        # Event Binding
        process_btn.click(
            fn=engine.synthesize,
            inputs=[audio_input_a, audio_input_b, text_input, alpha_slider],
            outputs=[audio_output]
        )

    return app

if __name__ == "__main__":
    # Launch application
    # share=True creates a public link (optional)
    demo = create_interface()
    demo.queue().launch(share=False)