"""
PhoneGCN Model
==============
Multi-stream encoder–decoder architecture for continuous KSL recognition
and gloss-to-text translation.

Streams:
    hand    — EfficientNet-Lite hand-crop CNN
    body    — ST-GCN + Phonological Adjacency Matrix (PAM)
    face    — I3D variant for facial expression

Fusion:
    Cross-modal transformer (2 layers, 4 heads, d_model=512)

Translation:
    Fine-tuned mBART (facebook/mbart-large-cc25) gloss-to-text decoder
"""

from .phonogcn   import PhoneGCN
from .hand_stream  import HandStream
from .body_stream  import BodyStream, PAM
from .face_stream  import FaceStream
from .fusion       import FusionTransformer
from .translation  import TranslationHead

__all__ = [
    "PhoneGCN",
    "HandStream",
    "BodyStream",
    "PAM",
    "FaceStream",
    "FusionTransformer",
    "TranslationHead",
]
