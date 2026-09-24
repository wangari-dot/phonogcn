"""
translation.py — mBART Gloss-to-Text Translation Head
======================================================
Fine-tuned mBART (facebook/mbart-large-cc25) sequence-to-sequence
translation head that converts CTC-decoded gloss sequences into
fluent natural-language sentences.

Input:  gloss token ids   — output of the CTC recogniser
Output: text token ids    — Swahili or English translation

Training (§3.2.5):
    - Encoder: 2 fine-tuned layers (rest frozen)
    - Decoder: 2 fine-tuned layers
    - Cross-attention between fused encoder output and mBART decoder
    - Target language: "sw_KE" (Swahili) or "en_XX" (English)

The translation component is optional — the system can be evaluated
on recognition only (gloss-level WER) without instantiating this module.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Lazy import: transformers is a heavy dependency and may not always be needed.
try:
    from transformers import MBartForConditionalGeneration, MBart50Tokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning(
        "transformers not installed or import failed.  "
        "TranslationHead will not be available.  "
        "Install with: pip install transformers"
    )


class TranslationHead(nn.Module):
    """
    Gloss-to-text translation using a partially fine-tuned mBART model.

    Args:
        model_name:      HuggingFace model identifier
                         (default: 'facebook/mbart-large-cc25').
        target_language: mBART language token for the output language.
                         Use 'sw_KE' for Swahili, 'en_XX' for English.
        n_encoder_layers: Number of mBART encoder layers to fine-tune
                          (remaining layers are frozen).
        n_decoder_layers: Number of mBART decoder layers to fine-tune.
        fused_dim:        Dimension of the fused encoder output (used to
                          project into mBART's model dimension if different).
        freeze_embeddings: Whether to freeze mBART token embeddings.
    """

    def __init__(self,
                 model_name: str = "facebook/mbart-large-cc25",
                 target_language: str = "sw_KE",
                 n_encoder_layers: int = 2,
                 n_decoder_layers: int = 2,
                 fused_dim: int = 512,
                 freeze_embeddings: bool = True):
        super().__init__()

        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers library required for TranslationHead.  "
                "pip install transformers"
            )

        self.target_language = target_language

        logger.info(f"Loading mBART model: {model_name}")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = MBart50Tokenizer.from_pretrained(
            model_name, src_lang="en_XX", tgt_lang=target_language
        )

        mbart_dim = self.model.config.d_model  # 1024 for mbart-large

        # Project fused embeddings into mBART model dimension if needed
        if fused_dim != mbart_dim:
            self.input_proj = nn.Linear(fused_dim, mbart_dim)
        else:
            self.input_proj = nn.Identity()

        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze top N encoder layers
        encoder_layers = self.model.model.encoder.layers
        for layer in encoder_layers[-n_encoder_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze top N decoder layers
        decoder_layers = self.model.model.decoder.layers
        for layer in decoder_layers[-n_decoder_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze output projection (lm_head)
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        if not freeze_embeddings:
            for param in self.model.model.shared.parameters():
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"mBART fine-tune: {n_trainable:,} / {n_total:,} params trainable "
            f"({100*n_trainable/n_total:.1f}%)"
        )

    def forward(self,
                fused_encoder_output: torch.Tensor,
                attention_mask: torch.Tensor,
                decoder_input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute translation logits (training mode).

        Args:
            fused_encoder_output: (B, T, fused_dim) — output from FusionTransformer
            attention_mask:       (B, T) — 1 = valid frame, 0 = padded
            decoder_input_ids:    (B, L) — target gloss/text token ids (teacher forcing)
            labels:               (B, L) — target ids shifted right (for loss computation)

        Returns:
            logits: (B, L, vocab_size)
            loss:   scalar cross-entropy loss (if labels provided, else None)
        """
        # Project to mBART dimension
        enc_out = self.input_proj(fused_encoder_output)  # (B, T, mbart_dim)

        # Wrap in HuggingFace BaseModelOutput-compatible structure
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_out)

        output = self.model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        logits = output.logits         # (B, L, vocab_size)
        loss   = output.loss           # scalar or None

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 fused_encoder_output: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_new_tokens: int = 50,
                 num_beams: int = 4,
                 ) -> List[str]:
        """
        Autoregressively generate translations (inference mode).

        Args:
            fused_encoder_output: (B, T, fused_dim)
            attention_mask:       (B, T)
            max_new_tokens:       Maximum output token length.
            num_beams:            Beam search width.

        Returns:
            translations: List of decoded strings, one per batch element.
        """
        enc_out = self.input_proj(fused_encoder_output)

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=enc_out)

        tgt_lang_id = self.tokenizer.lang_code_to_id[self.target_language]

        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        translations = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return translations

    def tokenize_targets(self, sentences: List[str]) -> dict:
        """
        Tokenize a list of target-language sentences for training.

        Returns a dict with input_ids and attention_mask tensors.
        """
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
