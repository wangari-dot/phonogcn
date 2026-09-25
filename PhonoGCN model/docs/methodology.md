# 3. Methodology

This section describes how the KSL-Daily corpus was assembled from broadcast
television, how labels and features were derived from it, and how PhonoGCN was
trained and evaluated on the result. Figure 1 summarises the pipeline. Items
marked **[AUTHOR TO CONFIRM]** record facts that the project files do not
establish and that must be supplied before submission.

```
Broadcast episodes (8)  ->  Interpreter-inset crop + caption-aligned segmentation  ->  2,086 clips
   -> transcript correction, advert removal, signer attribution          ->  1,995 programme clips
   -> gloss drafting from transcript, restriction to 607-gloss vocabulary ->  1,836 labelled clips
   -> MediaPipe Holistic at 10 fps, hand-detection gate (>= 0.60)        ->    999 clips
   -> signer-independent partition                                       ->  843 / 112 / 44
```
*Figure 1. Corpus construction pipeline, with the number of clips surviving each stage.*

## 3.1 Source material

Video was drawn from the official YouTube channel of Citizen TV Kenya, whose
news and current-affairs programmes carry a Kenyan Sign Language (KSL)
interpreter in a picture-in-picture inset. Of 47 candidate episodes catalogued,
eight were processed for this study (Table 1). They span five programmes and
three formats (news bulletin, talk show and news analysis), were broadcast
between January and October 2025, and total 6.6 hours of source video.

**Table 1. Source episodes.**

| Episode | Programme | Format | Broadcast date | Duration (min) | Clips |
|---|---|---|---|---|---|
| ep026 | Citizen Nipashe (Swahili) | Bulletin | 2025-09-25 | 29.0 | 159 |
| ep031 | Citizen Tonight | Bulletin | 2025-10-02 | 65.1 | 362 |
| ep034 | Sunday Live | Bulletin | 2025-03-02 | 55.6 | 261 |
| ep035 | Sunday Live | Bulletin | 2025-04-20 | 57.0 | 329 |
| ep036 | Sunday Live | Bulletin | 2025-06-22 | 32.8 | 161 |
| ep037 | Sunday Live | Bulletin | 2025-07-20 | 53.8 | 329 |
| ep040 | JKLive | Talk show | 2025-01-29 | 50.8 | 235 |
| ep046 | The Explainer | News analysis | 2025-02-04 | 52.0 | 250 |
| **Total** | | | | **396.1** | **2,086** |

Raw broadcast video is not redistributed. Use of the material for research
analysis relies on [AUTHOR TO CONFIRM: legal basis / permission from the
broadcaster, and ethics approval reference].

## 3.2 Interpreter-inset extraction and segmentation

**Inset cropping.** The interpreter inset was located manually on sampled
frames of each episode and verified with contact sheets. On the 1280×720
source frame it occupied the same region, a 191×148-pixel box with its
top-left corner at (1015, 417), in all eight episodes and all five programmes.
Each clip was therefore cropped from this fixed region and encoded as H.264 at
190×148 pixels (width rounded to an even value for the encoder) at the source
frame rate (25 or 30 fps). The pre-show introduction of each episode (97–306 s)
was skipped.

**Caption-aligned segmentation.** Clip boundaries were proposed from the timing
of the episode's caption track rather than from the signing itself. Each
caption cue defined one candidate clip, shifted by a fixed lag of 2.5 s to
account for the interpreter trailing the spoken audio; this lag was spot-checked
by visual inspection of test cuts in each episode. Cues longer than 15 s were
split on time into shorter clips. Segmentation yielded 2,086 clips of 1.5–15.8 s
(median 13.0 s), 6.6 hours in total. Because boundaries follow the speech
rather than the signing, most clips are sentence fragments: only 22 % were
judged to contain a complete sentence, and 124 force-split clips duplicate a
caption that describes only one of the two halves.

## 3.3 Transcript processing and signer attribution

**Transcripts.** The English translation of each clip was taken from the
broadcast's original-language caption track for seven episodes and from a human
translation for the Swahili bulletin (ep026). Automatic-captioning errors in
proper names and domain terms were corrected with 195 curated substitution
rules (419 substitutions in 236 clips). Rules that could collide with ordinary
words or Swahili syllables, and names whose spelling could not be confirmed,
were deliberately left uncorrected. The uncorrected text is retained alongside
every edit, so all corrections are reversible.

**Advertising.** Advertisement blocks were detected automatically and checked at
their boundaries by hand. Ninety-one clips (77 advertisements and 14 sponsored
gambling segments presented as news items) were excluded, leaving 1,995
programme clips.

**Signer attribution.** Signer identity was assigned per episode from the
interpreter's name as announced on air. Episodes whose interpreter was not
announced received an anonymous identifier. This produced four identities:
signer_YN (ep031, ep034, ep037, ep040; 57 % of clips), signer_JS (ep035, ep046;
28 %), signer_UNK01 (ep026) and signer_UNK02 (ep036; 8 % each). An earlier
attribution by visual description had assigned eight identities to these four
people; it was discarded because it would have placed the same interpreter on
both sides of a train/test split. The announcement is made at the start of each
broadcast, so any change of interpreter during a programme would not be
captured. This was not verified.

## 3.4 Gloss vocabulary

Gloss drafts (Section 3.5) were compiled into a frequency dictionary of 3,603
candidate gloss types, each classified as lexical (2,265), fingerspelled (758),
compound (303), numeric (246), question (23) or reduplicated (8). Two KSL
experts reviewed candidate entries. The released vocabulary contains 607
glosses. Of these, 606 appear in the dictionary: 598 lexical, 4 fingerspelled
(e.g. FS-HIV, FS-ARV), 3 compound (e.g. TEAR-GAS) and 1 numeric (YEAR-2023).
All 606 occur at least seven times in the drafts. The remaining gloss, TRUE,
is absent from the dictionary.

**[AUTHOR TO CONFIRM: the rule by which the 607 were selected.]** The review
spreadsheet (`expert_validated_dictionary_ksl.xlsx`) records a decision from at
least one expert for only 218 of the 3,603 entries. It records none for 574 of
the 606 selected glosses it contains. Seventeen selected glosses are marked
"No" by the first expert, five carry the note "Invalid", and none of the 57
entries confirmed by both experts is in the vocabulary. Unless the review was recorded elsewhere, the vocabulary should be
described as frequency-selected from the drafts rather than expert-validated.

Gloss identifiers were assigned deterministically: the list is de-duplicated,
sorted alphabetically and numbered 0–606. The same script writes the model's
vocabulary file and the ELAN controlled vocabulary for annotators, so the labels
annotators can choose and the classes the model can predict cannot drift apart.
The class count is read from the vocabulary file at training time, and a
mismatch with the model configuration aborts the run.

## 3.5 Clip labels

No clip has yet been glossed by a KSL signer from the video. The labels used in
this study are **pseudo-labels** derived from the transcripts, as follows.

1. For each clip, a KSL gloss sequence was drafted from the corrected transcript
   by [AUTHOR TO CONFIRM: method or tool that produced `gloss_draft`], following
   the project's glossing conventions (upper case, KSL sign order, `FS-` for
   fingerspelling). The draft was not checked against the signing.
2. The draft was restricted to the 607-gloss vocabulary by deleting
   out-of-vocabulary tokens, keeping the rest in draft order. Clips with no
   remaining token were discarded.

Of the 1,995 programme clips, 1,836 retain at least one vocabulary gloss. They
carry 10,622 gloss tokens; 13,444 out-of-vocabulary draft tokens were removed.
Consequently most of the signing in a typical clip has no label. The label
sequence is a subsequence of the signs performed, not a transcription of them.
Every label carries the provenance tag `draft_in_vocab_tokens` and a
pseudo-label weight of δ = 0.4 (Section 3.8).

A second, smaller label set records the example clip that the vocabulary
spreadsheet cites for each gloss. It covers 334 clips, with 1–10 glosses per
clip. It was used for corpus analysis but not for training: each gloss occurs
in exactly one of these clips, so no split of them can place a test gloss in
the training set.

## 3.6 Pose and appearance extraction

**Temporal resampling.** All clips were resampled to 10 fps before feature
extraction by keeping the first frame of every 100-ms interval. This serves
three purposes. It removes the mixture of 25 and 30 fps sources. It lets the
model's input window of 160 frames cover the longest clip (15.8 s, 158 frames),
so that the model sees the whole span that its label describes. And it reduces
extraction time and storage by about two-thirds.

**Landmarks and crops.** Each retained frame was processed once with MediaPipe
Holistic (v0.10.9, detection and tracking confidence 0.5). From each detection
three outputs were derived:

- **Skeleton**, a (T, 32, 3) array of normalised image coordinates (x, y) and
  relative depth (z). Joints 0–5 are pose landmarks 11–16 (shoulders, elbows
  and wrists, in MediaPipe order: left before right). Joints 6–26 are all 21
  right-hand landmarks. Joints 27–31 summarise the left hand (wrist and the
  thumb, index, middle and little-finger tips). The layout favours the right
  hand, assumed dominant for these interpreters.
- **Hand crops**, (T, 3, 112, 112) uint8 RGB per hand, cut from a square box
  around the hand landmarks padded by 40 % of the box size.
- **Face crop**, (T, 3, 64, 64) uint8 RGB, padded by 25 %.

Undetected landmarks and crops were zero-filled.

**Quality gate.** A clip was retained only if a right hand was detected in at
least 60 % of its frames, the threshold the acquisition protocol treats as the
limit of a usable inset. 999 of the 1,836 labelled clips (54 %) passed.
Detection varied strongly by interpreter: the median right-hand detection rate
was 0.78 for signer_YN, 0.71 for signer_UNK01 and 0.69 for signer_UNK02, but
0.41 for signer_JS. For signer_JS both hands were detected poorly (right 0.42,
left 0.50 on a sample), which rules out left-hand dominance as the cause, so
most of that signer's clips were removed.

## 3.7 Partitioning

Partitions are signer-independent: every clip of a given interpreter falls in
the same partition. With only four interpreters, a random clip-level split would
place the same person in training and test and reward memorisation of the
signer rather than recognition of signs. signer_YN and signer_UNK02 form the
training set, signer_UNK01 the validation set and signer_JS the test set
(Table 2).

**Table 2. Partitions after the quality gate.**

| Partition | Signers | Clips | Gloss tokens | Distinct glosses | Tokens whose gloss occurs in training |
|---|---|---|---|---|---|
| Train | signer_YN (737), signer_UNK02 (106) | 843 | 5,166 | 586 | — |
| Validation | signer_UNK01 | 112 | 676 | 281 | 99.9 % |
| Test | signer_JS | 44 | 232 | 174 | 91.8 % |

Coverage remains far below the project's floor of 25 clips per gloss. 434
glosses occur five or more times in training, and 21 of the 607 never occur in
training. The test partition is small, because the quality gate removed most of
the test signer's clips, and test estimates are correspondingly imprecise.

## 3.8 Model

PhonoGCN is a three-stream encoder with a transformer fusion layer and a
connectionist temporal classification (CTC) output. It has 14,550,325
trainable parameters, all randomly initialised.

**Hand stream.** Each hand crop is encoded per frame by a compact
EfficientNet-Lite0-style CNN (256-d output). The two hand embeddings are
concatenated and projected to a 256-d hand embedding.

**Body stream.** The skeleton passes through a two-layer spatial–temporal graph
convolutional network (ST-GCN). Each layer aggregates over joints, applies a
temporal convolution of kernel 9, and uses batch normalisation, dropout
(p = 0.2) and a residual connection. A learned pooling over joints then gives a
256-d frame embedding. Spatial aggregation uses the **Phonological Adjacency
Matrix (PAM)**:

  A_eff = A_phys + λ · σ(A_ling + M),

where A_phys is the normalised physical skeleton graph and A_ling a fixed graph
of phonologically motivated edges. These edges link the two wrists (bimanual
coordination), each wrist to its shoulder (signing-space location), and
corresponding joints of the two hands (symmetric two-handed signs), following
Mweri (2018). M is a learnable edge-weight matrix initialised to zero, λ a
learnable scalar initialised to 0.5, and σ the logistic function.
**[AUTHOR TO CONFIRM / FIX before submission: the joint indices of A_phys and
A_ling do not fully match the extraction layout in Section 3.6. Three of the
right hand's finger chains join landmarks from different fingers, four
right-hand joints have no edges, and the "symmetric" edges pair right-hand
joints with left-hand fingertips. All training runs so far used these
graphs.]**

**Face stream.** The face crop sequence is encoded by a small I3D-style 3-D CNN
(a 3-D convolutional stem followed by two factorised Inception blocks), giving
256-d frame embeddings intended to capture non-manual features.

**Fusion and output.** Each stream is projected to 512 dimensions and given a
learned modality embedding and sinusoidal positional encoding. The three
streams are then concatenated and projected back to 512 dimensions. A two-layer
pre-LayerNorm transformer encoder (4 heads, feed-forward width 2,048) attends
over time, with padded frames masked. A linear CTC head outputs 608 classes:
the 607 glosses plus a blank symbol at index 607. The model also has auxiliary
heads for handshape (38 classes; Mweri 2018) and signing-space location
(12 classes). They were not trained here because the corpus has no frame-level
labels for them.

## 3.9 Training

Only the CTC term of the multi-task objective was optimised: the handshape,
location and contrastive terms were given zero weight, since the corpus lacks
the labels and paired views they need. The per-clip CTC negative log-likelihood
is weighted by the clip's pseudo-label weight δ and averaged over the batch with
weights normalised to sum to one. Because every clip in this corpus has the same
weight (δ = 0.4), the weighting has no effect in these experiments. It is
retained so that video-verified clips (δ = 1.0) can be up-weighted once they
exist.

Optimisation used AdamW (learning rate 1 × 10⁻⁴, weight decay 1 × 10⁻⁴), with
the PAM parameters at one-tenth of the base rate for stability. The learning
rate followed cosine annealing to zero over the full run. Gradients were
clipped to a norm of 1.0. Training ran for 20 epochs with a batch size of 8 and
seed 42. Clips were zero-padded to the longest in each batch, with a padding
mask passed to the transformer and true lengths passed to the CTC loss. No data
augmentation, pretraining or transfer from other sign languages was used. The
reference PhonoGCN recipe pretrains the encoder on WLASL-2000; that checkpoint
was not available. Training ran on a single NVIDIA Tesla T4 (15 GB) in Google
Colab under PyTorch 2.x; a batch size of 16 exceeded GPU memory.

## 3.10 Evaluation

**Metric.** Recognition is scored by gloss-level word error rate (WER). CTC
outputs are decoded by greedy best-path decoding: take the most probable class
per frame, collapse consecutive repeats, and delete blanks. WER is the
Levenshtein distance between decoded and reference sequences, summed over the
partition and divided by the total reference length. The checkpoint with the
lowest validation WER is evaluated once on the test partition; ties go to the
later epoch. WER is computed against the pseudo-labels of Section 3.5, which
cover only the vocabulary glosses in each clip. It therefore measures agreement
with transcript-derived labels, not verified recognition accuracy.

**Diagnostics.** Two further measurements accompany WER.

1. *Blank rate.* An empty hypothesis scores a WER of exactly 1.0 through
   deletions alone. We therefore report the share of frames whose most probable
   class is the blank, the mean blank probability, and the largest
   non-blank probability, on both training and validation clips. This
   separates a model that emits nothing ("blank collapse") from one that emits
   wrong glosses.
2. *Capacity check.* The model is trained on 32 training clips and evaluated on
   the same 32 clips (150 epochs, learning rate 1 × 10⁻³). A model and
   training loop that work should be able to memorise so small a set. Failure
   to do so points to a defect in the model, loss or data pipeline rather than
   to insufficient data.

**Corpus analysis.** The 334 clips cited as gloss examples in the vocabulary
spreadsheet and the remaining 1,752 clips were compared on duration, frame rate, and ffmpeg `signalstats` measures of
brightness, contrast, saturation and motion (mean absolute inter-frame luma
difference). Differences were summarised by medians, interquartile ranges and
the common-language effect size P(kept > excluded).

## 3.11 Reproducibility

Every stage is scripted in `dataset_toolkit/`. The stages are: vocabulary
construction (`build_vocab.py`), label generation
(`06c_annotations_from_drafts.py`), landmark and crop extraction
(`03b_extract_crops.py --target_fps 10 --min_hand_rate 0.6`), manifest and
partition construction (`04_build_manifests.py`), and coverage reporting
(`05_coverage_report.py`). Training uses `train.py` with the command-line
overrides `train.epochs=20 train.n_seeds=1 train.batch_size=8
data.n_frames_max=160`. The training configuration supports repetition over
multiple seeds, with results reported as mean ± standard deviation; the runs
described here used a single seed. A provenance check at the start of every
training job writes a warning to the log when training clips are synthetic, or
when there are fewer than five training clips per gloss.
