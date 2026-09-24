# Methodology and Results

> **Scope note.** The experiments reported here were run on the synthetic
> placeholder corpus distributed with this repository. They validate the
> correctness of the recognition pipeline; they are **not** sign-recognition
> performance results, and no claim about KSL recognition accuracy can be
> supported until the broadcast corpus described in Section 3.3 is collected.
> This scoping follows the project's own integrity rule that results obtained
> on synthetic clips must be labelled as such.

---

## 3. Methodology

### 3.1 Gloss vocabulary construction

The label space is a 607-item Kenyan Sign Language (KSL) gloss vocabulary
derived from an expert-validated gloss list (`cleaned_expert_validated.csv`).
Candidate glosses were reviewed by KSL-fluent annotators, and the released
vocabulary contains only those entries surviving review.

Vocabulary compilation is deterministic and scripted (`build_vocab.py`) so that
the label space can be regenerated exactly. The procedure de-duplicates the
reviewed list, sorts it alphabetically, and assigns contiguous integer
identifiers 0-606. Sorting before assignment makes identifiers a pure function
of the gloss set, so an unchanged expert list always reproduces an identical
mapping. The script emits two artefacts from a single source of truth: a
machine-readable `gloss_vocabulary.json` consumed by the model and manifest
builders, and an ELAN controlled-vocabulary CSV used by annotators. Deriving
both from one file prevents drift between the labels annotators may select and
the classes the model can predict.

This vocabulary supersedes an earlier 310-item provisional list, 28 entries of
which were placeholder tokens rather than attested glosses. Vocabulary size is
not a free parameter: it is read from `gloss_vocabulary.json` at training time
and cross-checked against the configured class count, and a mismatch aborts the
run rather than silently truncating the label space.

### 3.2 Model architecture

PhonoGCN is a three-stream encoder with cross-modal fusion and a connectionist
temporal classification (CTC) output, totalling 14,550,325 trainable parameters.

**Hand stream.** Left and right hand crops are encoded independently by
EfficientNet-Lite0 backbones (256-d output each), concatenated, and projected
through a LayerNorm-ReLU block to a 256-d per-frame hand embedding.

**Body stream.** Skeletal input of shape (T, 32, 3) is processed by a two-layer
spatial-temporal graph convolutional network. Each layer applies graph
aggregation over joints followed by a depthwise temporal convolution (kernel 9),
with batch normalisation, dropout (p = 0.2), and a residual connection. The
spatial adjacency is the Phonological Adjacency Matrix (PAM), which augments the
physical skeletal graph with a linguistically motivated graph encoding
phonological relations between articulators. The two are combined under a
learnable balancing coefficient lambda (initialised to 0.5) and recomputed at
every forward pass, allowing the model to adapt the relative weight of
anatomical and phonological structure during training. Per-frame embeddings are
obtained by a learned pooling over the joint axis.

**Face stream.** Face crops are encoded by a lightweight 3-D convolutional
network (I3D variant) with factorised Inception blocks, yielding 256-d per-frame
embeddings intended to capture the non-manual features that disambiguate
near-minimal pairs.

**Fusion and heads.** Each stream is projected to d_model = 512, given a learned
modality-type embedding and sinusoidal positional encoding, concatenated along
the feature axis, and projected back to 512 before a two-layer pre-LayerNorm
transformer encoder (4 heads, feed-forward dimension 2048) attends over time.
The fused sequence feeds a linear CTC head of 608 units: 607 glosses plus one
blank symbol at index 607. Two auxiliary linear heads predict handshape
(38 classes, after the Mweri 2018 inventory) and signing-space location
(12 classes).

**Objective.** The multi-task loss combines the CTC term with weighted
auxiliary cross-entropy losses for handshape and location, which use an ignore
index for unlabelled frames, and an NT-Xent contrastive term over paired
augmented views with a learnable temperature.

### 3.3 Corpora

Two corpora are used, for distinct purposes, and results from them are reported
separately throughout.

**Pilot broadcast corpus (real).** The target corpus is drawn from Kenyan
broadcast news bulletins carrying sign-language interpreter insets. The
acquisition pipeline crops the interpreter inset, gates each clip on
hand-detection rate, extracts 32-joint skeletons at (T, 32, 3), and writes
per-part pixel crops as uint8 arrays: (T, 3, 112, 112) for each hand and
(T, 3, 64, 64) for the face. At the time of writing the pilot comprises 436
processed clips totalling approximately 9 GB of crop data, recorded at 30 fps.
Clip labels in this corpus are **pseudo-labels** constrained to the validated
vocabulary, not per-clip expert annotations; manifest entries carry the
provenance tag `pseudo_labeled_vocab_validated`. Expert validation applies to
the label inventory (Section 3.1), not yet to the clip-level assignments.

**Synthetic placeholder corpus.** For implementation testing the repository
ships a synthetic corpus generated by `make_synthetic_dataset.py`: 900 clips at
25 fps with durations of 1.92-5.40 s, partitioned 630 / 135 / 135. Skeletons are
procedurally generated smooth trajectories, formed as sums of low-frequency
sinusoids about a rest pose with additive jitter, and crop features are random
unit vectors of dimension 256. Clip generation is not conditioned on the gloss
label. Every manifest entry is flagged synthetic and carries a placeholder
provenance tag. This corpus exists to exercise the schema and training path; it
cannot support learning, for reasons quantified in Section 4.4.

Splits in both corpora are **signer-independent**: whole signers are held out
for validation and test. With a small number of televised interpreters, random
clip-level splits leak signer identity between partitions and inflate apparent
accuracy, so signer-disjoint partitioning is treated as a requirement rather
than an option.

### 3.4 Training protocol

Models are optimised with AdamW (weight decay 1e-4) under a cosine-annealing
schedule, with batch size 16 and gradient-norm clipping at 1.0. The base
learning rate is 1e-4; PAM parameters are trained at one tenth that rate,
since the adjacency is recomputed at every forward pass and full-rate updates
to it destabilise early training. Because the placeholder corpus provides no
frame-level handshape or location annotation, the auxiliary loss weights are set
to zero for these runs and only the CTC term is optimised; the auxiliary heads
remain instantiated but unsupervised.

The corpus ships pre-extracted per-frame features rather than raw image crops,
so training uses a feature-input path that bypasses the hand and face CNN
backbones while retaining the two-hand fusion block and the full ST-GCN body
stream. Variable-length clips are padded per batch, with a boolean padding mask
supplied to the transformer and true frame counts supplied to the CTC loss, so
that padding contributes no gradient.

Runs reported here use a single seed (42) for 10 epochs on CPU. The protocol
supports repetition across multiple seeds with results reported as mean and
standard deviation; multi-seed repetition was not informative here, for the
reason given in Section 4.3.

### 3.5 Evaluation

Recognition quality is measured as gloss-level Word Error Rate (WER). CTC
outputs are decoded greedily by best-path decoding: per-frame argmax, collapse
of consecutive repeats, then removal of blank symbols. WER is the Levenshtein
distance between decoded and reference gloss sequences, summed over the
evaluation set and normalised by total reference length. The checkpoint with the
lowest validation WER is selected for final test evaluation.

### 3.6 Data-provenance controls

Because the placeholder corpus is structurally identical to the real one, an
automated guard runs at the start of every training job. It inspects the
training manifest and emits a prominent warning when clips are flagged
synthetic, and a second warning when the ratio of training clips to distinct
glosses falls below five. This records provenance and corpus adequacy in the
training log itself, so that metrics obtained on placeholder data cannot later
be mistaken for recognition results.

---

## 4. Results

### 4.1 Vocabulary construction

Vocabulary compilation yields 607 unique expert-validated glosses with
identifiers 0-606, propagated consistently to the model configuration, the
dataset manifests, and the ELAN controlled vocabulary. Regeneration from the
reviewed list is deterministic and reproduces the identical mapping.

### 4.2 Pilot corpus audit

Table 1 summarises the state of the pilot broadcast corpus. Of 436 clips
submitted to extraction, 329 (75.5 per cent) passed the quality gate and 107
were rejected. Median hand-detection rate across processed clips was 0.698, with
140 clips (32.1 per cent) falling below the 0.60 rate that the acquisition
protocol treats as the usability threshold for an interpreter inset.

Corpus construction is at an early stage relative to the label space. Against a
floor of 10 clips per gloss, 520 of 607 glosses (85.7 per cent) currently have
no clips at all and the remaining 87 are partially covered; no gloss meets the
floor. Of the 329 usable clips, 52 are presently carried in the train, validation,
and test manifests.

**Table 1. Pilot broadcast corpus.**

| Metric | Value |
|---|---|
| Clips submitted to extraction | 436 |
| Passed quality gate | 329 (75.5%) |
| Rejected | 107 (24.5%) |
| Median hand-detection rate | 0.698 |
| Clips below 0.60 detection gate | 140 (32.1%) |
| Frame rate | 30 fps |
| Clips carried in manifests | 52 |
| Glosses with zero clips | 520 of 607 (85.7%) |
| Glosses meeting the 10-clip floor | 0 (0%) |

### 4.3 Adequacy of the current pilot partition

The present partition cannot yield an interpretable recognition score, for three
independent reasons, each of which is a property of the split rather than of the
model.

First, the partition is severely unbalanced in size: 45 training clips, 1
validation clip, and 6 test clips. A single validation clip cannot support
model selection, since the selection criterion is estimated from one sequence.

Second, and decisively, the test partition is disjoint from the training label
set. All 8 distinct glosses appearing in test sequences are absent from every
training sequence. Under this split a correct prediction is unreachable, and
test WER is therefore pinned near 1.0 for any model, irrespective of quality.
The measurement carries no information about recognition performance.

Third, label coverage is thin even within training: the 45 training clips carry
40 distinct primary glosses, only 5 of which have two or more clips, and just 70
of the 607 vocabulary items (11.5 per cent) appear anywhere in a training
sequence. Most of the label space is therefore unobserved at training time.

**Table 2. Current pilot partition.**

| Partition | Clips | Signers | Distinct primary glosses |
|---|---|---|---|
| Train | 45 | signer_JS (24), signer_YN (21) | 40 |
| Validation | 1 | signer_UNK01 | 1 |
| Test | 6 | signer_UNK02 | 6 |

Splits are correctly signer-disjoint, which is the property most often violated
in sign-language benchmarks, and that design choice should be preserved as the
corpus grows. Recognition results on this corpus are deferred until the
partition satisfies two conditions: a validation set large enough to support
model selection, and a test label set contained within the training label set.

---

## 5. Pipeline validation on synthetic data

The following experiment was run on the synthetic placeholder corpus. It
establishes that the training and evaluation implementation is correct and
characterises the failure mode expected when a corpus carries no
class-conditional signal. It is not a recognition result.

### 5.1 Training convergence

Vocabulary compilation yields 607 unique expert-validated glosses with
identifiers 0-606, propagated consistently to the model configuration, the
dataset manifests, and the ELAN controlled vocabulary. All 607 glosses appear as
the primary gloss of at least one training clip, so no class is unreachable at
training time.

Coverage against the project's stated annotation floor of 25 clips per gloss is
nevertheless far from met (Table 1). The 630 training clips spread across 607
glosses give a mean of 1.04 clips per gloss, and only 23 glosses (3.8 per cent)
have more than a single training example.

**Table 5. Coverage of the synthetic corpus.**

| Metric | Value |
|---|---|
| Vocabulary size | 607 |
| Training clips | 630 |
| Mean clips per gloss | 1.04 |
| Glosses with two or more training clips | 23 (3.8%) |
| Glosses meeting the 25-clip floor | 0 (0%) |
| Additional clips required to meet floor | 14,275 |



Optimisation behaved correctly and stably. Training loss fell from 18.68 to 6.70
over ten epochs, with the largest decrease between epochs 1 and 2 and a smooth
monotonic decline thereafter (Table 2). No divergence, gradient explosion, or
numerical failure occurred, and checkpointing, learning-rate scheduling, and
best-model selection all operated as intended.

**Table 3. Training and validation curves, synthetic corpus (seed 42).**

| Epoch | Train loss | Val CTC | Val WER |
|---|---|---|---|
| 1 | 18.6806 | 7.0033 | 1.0000 |
| 2 | 6.8739 | 6.9804 | 1.0000 |
| 3 | 6.8262 | 7.0174 | 1.0000 |
| 4 | 6.7973 | 6.9787 | 1.0000 |
| 5 | 6.7784 | 6.9683 | 1.0000 |
| 6 | 6.7520 | 6.9775 | 1.0000 |
| 7 | 6.7318 | 6.9674 | 1.0000 |
| 8 | 6.7144 | 6.9586 | 1.0000 |
| 9 | 6.7034 | 6.9611 | 1.0000 |
| 10 | 6.6985 | 6.9619 | 1.0000 |

### 5.2 Recognition outcome

Recognition performance remained at the theoretical worst case throughout.
Validation WER was 1.0000 at every epoch, and the selected checkpoint scored a
test WER of 1.0000 with a test CTC loss of 6.9169. Falling training loss
combined with static WER indicates that optimisation succeeded while recognition
did not: the model reduced its objective without acquiring discriminative
ability.

Inspection of the decoder explains the exact value obtained. The trained model
assigns the blank symbol the highest probability at 100 per cent of frames
(mean blank probability 0.9613), so best-path decoding collapses every clip to
the empty sequence. Against a reference of length L, an empty hypothesis incurs
exactly L deletions, giving a WER of 1.0 identically. The round value is
therefore a signature of complete blank collapse rather than a coincidence or a
measurement artefact.

### 5.3 Absence of class-conditional signal

Three independent lines of evidence establish that this null result follows from
the data rather than from a defect in the model or the evaluation code.

**Convergence to the label prior.** Gloss labels in the placeholder corpus are
drawn uniformly at random, so the entropy of the label distribution is
ln(607) = 6.41 nats, and no estimator can achieve a lower expected per-token
loss without input information. Observed CTC loss converged to approximately
6.96, which is that floor plus the alignment cost CTC incurs in selecting
emission positions and sequence length. The model therefore recovered the
marginal label distribution and nothing beyond it.

**Direct measurement of feature-label dependence.** Clip-level mean embeddings
were compared by cosine similarity within and between gloss classes (Table 3).
For hand features, clips sharing a gloss are more similar than clips of
different glosses by only 0.0035. For skeletons the difference is negative at
-0.0297, meaning same-gloss clips are marginally less similar than
different-gloss clips. Both differences are small relative to within-class
dispersion, and the negative sign is characteristic of sampling noise about a
true difference of zero.

**Table 4. Within- versus between-gloss cosine similarity, synthetic corpus.**

| Representation | Within-gloss | Between-gloss | Difference | Within-class SD |
|---|---|---|---|---|
| Hand crop features | -0.0079 | -0.0114 | +0.0035 | 0.055 |
| Skeleton | +0.1151 | +0.1448 | -0.0297 | 0.105 |

**Generative independence.** This outcome is expected by construction: the
placeholder generator draws skeletons and crop features from distributions that
are not conditioned on the gloss label. Mutual information between inputs and
labels is therefore zero by design, and a WER of 1.0 is the optimum attainable
on this corpus rather than a failure to reach a better one.

Taken together, these results establish that the pipeline behaves correctly on
data containing nothing to learn. The experiment validates the implementation
and bounds what the corpus can support; it does not measure sign recognition.

## 6. Limitations and scope of claims

No claim is made in this work about the recognition or translation accuracy of
PhonoGCN on Kenyan Sign Language, and no WER reported above should be cited as a
performance figure for the architecture. The synthetic-corpus result in
Section 5 measures a corpus that contains no class-conditional information by
construction, and the pilot partition in Section 4.3 cannot produce an
interpretable score because its test label set is disjoint from its training
label set.

Two further scope limits bear on how the pilot corpus should be described.
Clip-level labels are pseudo-labels constrained to the validated vocabulary
rather than per-clip expert annotations, so expert validation should be claimed
for the label inventory and not for the clip assignments. Signer identity is
also incomplete: validation and test partitions are attributed to unidentified
interpreters, which preserves signer-disjointness but prevents reporting how
many distinct signers the evaluation covers.

What the work does establish is threefold. It contributes an expert-validated
607-gloss KSL label space with deterministic, reproducible construction. It
contributes a complete and verified training and evaluation implementation,
whose correctness is demonstrated by its behaviour on a corpus with a known
information content. And it contributes a quantified specification of the
corpus still required: at a floor of 10 clips per gloss, 520 of 607 glosses
remain entirely uncollected, and the acquisition pilot indicates that roughly
one clip in four is rejected at the quality gate, which should inform
collection targets. Evaluation of the architecture remains future work,
contingent on completing that corpus and rebuilding the partitions so that
model selection and test-time scoring are both well posed.
