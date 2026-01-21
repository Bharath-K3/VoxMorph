"""
ECAPA-TDNN Encoder Patch
------------------------
Patches ChatterboxTTS to use SpeechBrain's ECAPA-TDNN speaker verification 
model for timbre encoding.

Input: 16kHz audio (Downsampled internally)
Output: 256-dim embedding (Interpolated from 192-dim)
"""

import torch
import types
import librosa

# Chatterbox Imports
from chatterbox.tts import Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond

def apply_patch(model_instance, device: str):
    """
    Applies ECAPA-TDNN patch to ChatterboxTTS model instance.
    
    Args:
        model_instance: The ChatterboxTTS model to patch.
        device: The device (cuda/cpu) to load the encoder on.
    """
    try:
        from speechbrain.inference import EncoderClassifier
        print("[INFO] Loading ECAPA-TDNN Encoder (192->256 Interpolation)...")
        
        # Load ECAPA-TDNN from SpeechBrain
        ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        model_instance.external_encoder = ecapa_classifier
        
    except ImportError:
        print("[ERROR] SpeechBrain not found. Install with: pip install speechbrain")
        raise

    def ecapa_prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """
        Replaces prepare_conditionals to use ECAPA for speaker embeddings.
        """
        # Import Chatterbox constants locally for clarity
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.s3gen import S3GEN_SR

        # =================================================================
        # 1. Standard Reference Loading (For Prosody/Gen)
        # =================================================================
        # Chatterbox S3Gen requires 24kHz reference audio.
        s3gen_ref_wav, _ = librosa.load(str(wav_fpath), sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # Trim to max context length for the generator
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        
        # Embed reference for prosody generator (S3Gen)
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # =================================================================
        # 2. Speech Prompt Tokens (Essential for Quality)
        # =================================================================
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # =================================================================
        # 3. ECAPA-TDNN Timbre Extraction (16kHz)
        # =================================================================
        # ECAPA requires 16kHz input.
        waveform_16k, _ = librosa.load(str(wav_fpath), sr=16000)
        waveform_tensor = torch.tensor(waveform_16k).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # ECAPA returns a (Batch, 192) embedding.
            # We detach to make it a leaf node for further manipulation.
            emb = self.external_encoder.encode_batch(waveform_tensor).squeeze(1).detach()
            
        # =================================================================
        # 4. Dimension Matching (192 -> 256)
        # =================================================================
        # Using Linear Interpolation to expand dimensions smoothly.
        # We ensure `emb` is strictly 1D (192) before reshaping for 3D interpolation.
        if emb.ndim > 1:
            # If batch dim exists (e.g. shape is 1, 192), flatten it to ensure 1D.
            emb = emb.flatten()
        
        # Reshape to 3D (Batch=1, Channel=1, Length=192) for interpolation
        # This satisfies torch.nn.functional.interpolate requirements.
        emb_3d = emb.unsqueeze(0).unsqueeze(1)
        
        # Interpolate to 256 length (Result: Batch=1, Channel=1, Length=256)
        expanded_3d = torch.nn.functional.interpolate(emb_3d, size=256, mode='linear', align_corners=True)
        
        # Squeeze back to 1D (256)
        emb = expanded_3d.squeeze(0).squeeze(0)
        
        # =================================================================
        # 5. Construct T3 Conditionals
        # =================================================================
        # Ensure shape matches Chatterbox expectations: (Batch=1, Time=1, Dim=256)
        t3_cond = T3Cond(
            speaker_emb=emb.unsqueeze(1), 
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, device=self.device),
        )
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    # Apply Monkey Patch
    model_instance.prepare_conditionals = types.MethodType(ecapa_prepare_conditionals, model_instance)
    print("[INFO] ECAPA-TDNN Encoder Patch Applied.")