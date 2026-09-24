"""
PhoneGCN Data Collection Pipeline
==================================
Semi-automated pipeline for constructing KSL sign language datasets
from broadcast media, following the methodology of PhoneGCN (2025).

Stages:
    1. broadcast_alignment  — keyword spotting on ASR-aligned broadcasts
    2. clustering           — K-means pseudo-labelling
    3. expert_validation    — negative-selection review interface
    4. augmentation         — geometric, temporal, and skeletal augmentation
"""
