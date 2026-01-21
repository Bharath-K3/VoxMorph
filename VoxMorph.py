"""
VoxMorph: Scalable Zero-Shot Voice Identity Morphing
Main Inference Script

- Supports Directory Input (Multi-Clip Consolidation)
- Supports Dynamic Encoder Switching (Default, ECAPA, Wav2Vec2, HuBERT)
- Supports 3 Interpolation Modes: SLERP, LERP, LINEAR (Averaging)
"""

import argparse
import yaml
import torch
import shutil
import logging
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

# Local Imports
from utils import MathUtils, AudioUtils

# Modular Encoder Imports
# We import the PATCHERS dictionary from the encoders package
try:
    from encoders import PATCHERS
except ImportError:
    raise ImportError("Could not import encoders package. Ensure 'encoders/' directory exists.")

# Chatterbox Imports
try:
    from chatterbox.tts import ChatterboxTTS, Conditionals
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.s3gen import S3GEN_SR
except ImportError:
    raise ImportError("ChatterboxTTS not installed. Please check requirements.")

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VoxMorph: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Main Processor Class
# ==============================================================================

class VoxMorphProcessor:
    def __init__(self, config_path="config.yaml", encoder_type=None):
        self.cfg = self._load_config(config_path)
        
        # OVERRIDE: CLI encoder argument takes precedence over config
        if encoder_type:
            self.cfg['model']['encoder_type'] = encoder_type
            
        self.device = self.cfg['system']['device']
        self.temp_dir = Path(self.cfg['system']['temp_dir'])
        self.target_sr = self.cfg['system']['sample_rate']
        
        encoder_name = self.cfg['model']['encoder_type'].upper()
        logger.info(f"Initializing Engine on {self.device} at {self.target_sr}Hz with Encoder: {encoder_name}...")
        
        # Load Model
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        
        # Apply Encoder Patches via Modular Registry
        enc_type = self.cfg['model']['encoder_type'].lower()
        if enc_type in PATCHERS:
            # Retrieve the patch function from the encoders directory
            patch_function = PATCHERS[enc_type]
            # Apply it to the model instance
            patch_function(self.model, self.device)
        else:
            logger.info(f"Using Default Encoder (No Patch Applied).")

    def _load_config(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def cleanup(self):
        """Clean up CUDA memory and temporary directories."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if self.temp_dir.exists():
            try: shutil.rmtree(self.temp_dir)
            except Exception: pass

    def get_embeddings(self, input_path):
        """
        Profiles speaker. Handles directory consolidation.
        
        Returns:
            t3_emb: Timbre embedding (numpy)
            gen_emb: Generator embedding (numpy)
            template_conds: Full conditionals object
        """
        # 1. Consolidate (Pass target_sr to ensure consistency)
        # Note: We use 24kHz (target_sr) because Chatterbox S3Gen requires it.
        # The encoder patches will downsample to 16kHz internally if needed.
        clean_path = AudioUtils.consolidate_input(input_path, self.temp_dir, target_sr=self.target_sr)
        
        # 2. Profile
        exaggeration = self.cfg['model'].get('exaggeration', 0.5)
        self.model.prepare_conditionals(str(clean_path), exaggeration=exaggeration)
        
        # 3. Detach & Extract
        # Detaching removes gradients, CPU moves it off GPU to RAM, numpy makes it python-ready
        emb_t3 = self.model.conds.t3.speaker_emb.detach().squeeze(0).cpu().numpy()
        emb_gen = self.model.conds.gen['embedding'].detach().squeeze(0).cpu().numpy()
        
        # 4. Store Template
        # We store the reference directly. Since prepare_conditionals creates a NEW 
        # object on next call, we don't need copy.deepcopy (which fails on graph tensors).
        template_conds = self.model.conds
        
        return emb_t3, emb_gen, template_conds

    def generate(self, source_a, source_b, text, alpha) -> Dict[str, np.ndarray]:
        """
        Main generation pipeline: Profile -> Interpolate -> Synthesize
        """
        try:
            # --- STAGE 1: Profiling ---
            logger.info("Profiling Source A...")
            t3_a, gen_a, template_a = self.get_embeddings(source_a)
            
            # Explicit Cache Cleanup to prevent OOM
            torch.cuda.empty_cache()
            
            logger.info("Profiling Source B...")
            t3_b, gen_b, template_b = self.get_embeddings(source_b)
            
            torch.cuda.empty_cache()

            # --- STAGE 2: Interpolation ---
            method = self.cfg['morphing']['method'].lower()
            threshold = self.cfg['morphing']['dot_threshold']
            
            logger.info(f"Interpolating via {method.upper()} (Alpha: {alpha})")
            
            # Logic for 3 Interpolation Modes
            if method == 'slerp':
                # Spherical Linear Interpolation
                morphed_t3 = MathUtils.slerp(t3_a, t3_b, alpha, threshold)
                morphed_gen = MathUtils.slerp(gen_a, gen_b, alpha, threshold)
            elif method == 'lerp':
                # Linear Interpolation (Weighted)
                morphed_t3 = MathUtils.lerp(t3_a, t3_b, alpha)
                morphed_gen = MathUtils.lerp(gen_a, gen_b, alpha)
            elif method == 'linear':
                # Linear Averaging (Baseline): (A+B)/2
                logger.warning("Using Linear Averaging. Alpha value is ignored.")
                morphed_t3 = MathUtils.linear_averaging(t3_a, t3_b)
                morphed_gen = MathUtils.linear_averaging(gen_a, gen_b)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            # Tensor conversion
            # Explicitly move to device to avoid CPU/CUDA mismatch errors
            t3_tensor = torch.from_numpy(morphed_t3).unsqueeze(0).to(self.device)
            gen_tensor = torch.from_numpy(morphed_gen).unsqueeze(0).to(self.device)

            results = {}

            # --- STAGE 3: Synthesis ---
            
            # 3.1 Clone A (Reference)
            logger.info("Synthesizing Source A Clone...")
            # We must use .to() here because template_a might have been moved if device changed 
            # (though normally it stays). Ideally Conditionals handles .to()
            if hasattr(template_a, 'to'):
                self.model.conds = template_a.to(self.device)
            else:
                self.model.conds = template_a
            results['clone_a'] = self.model.generate(text).cpu().squeeze().numpy()
            
            # Synchronize to ensure GPU finishes before resetting state
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            torch.cuda.empty_cache()
            
            # 3.2 Clone B (Reference)
            logger.info("Synthesizing Source B Clone...")
            if hasattr(template_b, 'to'):
                self.model.conds = template_b.to(self.device)
            else:
                self.model.conds = template_b
            results['clone_b'] = self.model.generate(text).cpu().squeeze().numpy()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            torch.cuda.empty_cache()

            # 3.3 Morph
            logger.info("Synthesizing Morph...")
            # Note: t3_tensor is (1, 256). Chatterbox T3Cond often expects (1, 1, 256) 
            # for sequence length, but (1, 256) is usually broadcasted correctly.
            # We ensure shape (1, 1, 256) explicitly to match batched expectations.
            if t3_tensor.dim() == 2:
                t3_tensor = t3_tensor.unsqueeze(1)
                
            final_t3 = T3Cond(
                speaker_emb=t3_tensor,
                cond_prompt_speech_tokens=template_a.t3.cond_prompt_speech_tokens,
                emotion_adv=template_a.t3.emotion_adv
            )
            final_gen = template_a.gen.copy()
            final_gen['embedding'] = gen_tensor
            
            morph_conds = Conditionals(final_t3, final_gen)
            self.model.conds = morph_conds.to(self.device)
            results['morph'] = self.model.generate(text).cpu().squeeze().numpy()
            
            return results

        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="VoxMorph Advanced Inference")
    parser.add_argument("--source_a", required=True, help="Path to file OR directory for Speaker A")
    parser.add_argument("--source_b", required=True, help="Path to file OR directory for Speaker B")
    parser.add_argument("--alpha", type=float, default=0.5, help="Morphing Factor")
    parser.add_argument("--text", type=str, help="Text to speak")
    parser.add_argument("--output_dir", default="Results", help="Directory to save outputs")
    parser.add_argument("--encoder", type=str, default=None, choices=["default", "ecapa", "wav2vec2", "hubert"], help="Override encoder type")
    
    args = parser.parse_args()
    
    # Initialize Processor (Selects encoder based on config or CLI arg)
    engine = VoxMorphProcessor(encoder_type=args.encoder)
    
    # Determine Text
    text = args.text if args.text else engine.cfg['inference']['default_text']
    
    # Run Generation
    outputs = engine.generate(args.source_a, args.source_b, text, args.alpha)
    
    # Save Results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sr = engine.cfg['system']['sample_rate']
    
    sf.write(out_dir / f"{timestamp}_source_A.wav", outputs['clone_a'], sr)
    sf.write(out_dir / f"{timestamp}_source_B.wav", outputs['clone_b'], sr)
    sf.write(out_dir / f"{timestamp}_morph.wav", outputs['morph'], sr)
    
    logger.info(f"Success. Files saved to {out_dir}")

if __name__ == "__main__":
    main()