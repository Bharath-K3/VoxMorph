"""
HuBERT Encoder Patch
------------------
Patches ChatterboxTTS to use Facebook's HuBERT speaker verification 

Input: 16kHz audio (Downsampled internally)
Output: 256-dim embedding (Truncated from 768-dim)
"""

import torch
import types
import librosa

# Chatterbox Imports
from chatterbox.tts import Conditionals
from chatterbox.models.t3.modules.cond_enc import T3Cond

def apply_patch(model_instance, device: str):
    """
    Applies the HuBERT patch to the ChatterboxTTS model instance.
    """
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        print("[INFO] Loading HuBERT-Base Encoder (768->256 Truncate)...")
        
        hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device).eval()
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        
        model_instance.external_encoder_model = hubert_model
        model_instance.external_encoder_processor = hubert_feature_extractor
        
    except ImportError:
        print("[ERROR] Transformers not found. Install with: pip install transformers")
        raise

    def hubert_prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        from chatterbox.models.s3tokenizer import S3_SR
        from chatterbox.models.s3gen import S3GEN_SR

        # 1. Standard Reference Loading (24kHz)
        s3gen_ref_wav, _ = librosa.load(str(wav_fpath), sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # 2. Speech Prompt Tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        
        # 3. HuBERT Embedding Extraction (16kHz)
        waveform_16k, _ = librosa.load(str(wav_fpath), sr=16000)
        inputs = model_instance.external_encoder_processor(waveform_16k, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
        
        with torch.no_grad():
            if attention_mask is not None:
                outputs = model_instance.external_encoder_model(input_values, attention_mask=attention_mask)
            else:
                outputs = model_instance.external_encoder_model(input_values)
            
            # Mean pooling: (Batch, Seq, 768) -> (Batch, 768)
            emb = outputs.last_hidden_state.mean(dim=1)
        
        # 4. Dimension Matching (768 -> 256) - Truncation
        emb = emb[..., :256]
        
        # 5. Construct T3 Conditionals
        t3_cond = T3Cond(
            speaker_emb=emb.unsqueeze(1), # (1, 1, 256)
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, device=device),
        )
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    model_instance.prepare_conditionals = types.MethodType(hubert_prepare_conditionals, model_instance)
    print("[INFO] HuBERT Encoder Patch Applied.")