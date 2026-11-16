# Leveraging Architectural Diversity for DNA Sequence Classification

Code for the benchmark of four deep learning paradigms for DNA sequence classification from raw nucleotides:

| Paradigm | Architecture | Key components |
| :--- | :--- | :--- |
| Hybrid CNN–Transformer | **GenExplorer** (proposed) | Multi-scale convolutional `PatternBranch` (max pooling) and `FrequencyBranch` (average pooling), a parallel 4-layer `TransformerBranch`, late MLP fusion |
| Multi-branch CNN ensemble | **FullE** | Five convolutional branches (average, max and $L_p$ power-average pooling) and five merger networks, combined by sum rule (10 networks) |
| Convolutional Transformer | **Nucleic Transformer** | Convolutional $k$-mer aggregation ($k=2..6$), 6-layer Transformer encoder, GeM pooling |
| Genomic foundation model | **DNABERT2-CAMP** | Pre-trained DNABERT-2 (117M) fine-tuned with a CAMP module (4-layer CNN cascade + multi-head attention) |

**HelixPR** (helix-aware periodic representation + CNN) is included as a literature baseline, and the
**Komlós–Hadamard** encoding can be used as input of GenExplorer.

---

## Benchmarks

| Task (`--task`) | Name | Length | Description |
| :--- | :--- | :---: | :--- |
| `viral` | VM | 300 bp | Viral metagenomic reads (ViraMiner), ≈2% positives |
| `viral` (test only) | SIM | 300 bp | VM test set with 1%, 5% and 10% substitution noise, 5 realizations per level |
| `ecoli` | ECO | 81 bp | *E. coli* promoters, balanced |
| `enhancer` | ENH | 200 bp | Human enhancers, balanced |
| `anellovirus` | OOD | 300 bp | *Anelloviridae* excluded from training/validation and used only for testing |

**Splits.** VM has its own training/validation/test files. ECO, ENH and OOD provide only training and test
files: the training file is shuffled with the run seed and 20% is held out for validation, so each seed uses a
different training/validation partition. The VM, SIM and OOD test sets are fixed.

**Leakage-free ECO and ENH.** The original ECO and ENH test sets share sequences with the training files (all
466 ECO test negatives and 100 of the 200 ENH test enhancers are exact duplicates of training sequences). The
paper therefore uses rebuilt splits, one per seed (2020, 1969, 1933), released in `datasets_leakfree/`:
training and test sequences were pooled, an all-against-all blastn search grouped similar sequences
(identity ≥ 80% over ≥ 80% of the shorter sequence) into clusters, and whole clusters were assigned to the
test set until it reached the original size and class composition (ECO: 461 + 466; ENH: 200 + 200). A final
blastn search of each training file against its test set returns no hit. Training, validation and
test sets are released separately for each seed. See [Leakage-free splits](#leakage-free-eco-and-enh-splits).

### Data

The archives in `datasets/` contain all benchmark files. Extract them in place:

```bash
cd datasets
for z in *.zip; do unzip -o "$z"; done
cd ..
```

The loaders search `datasets/` (or the folder passed with `--data_folder` / `$DATA_DIR`) recursively, so the
extracted folder names (`DNA_data/`, `SIM/`, `ecoli/`, `enhancer/`, `unseen_anell/`) can be kept as they are.

The `ecoli/` and `enhancer/` archives contain the **original** files. The ECO and ENH results of the paper use
the leakage-free splits in `datasets_leakfree/` (see below).

### Leakage-free ECO and ENH splits

```
datasets_leakfree/seed_<seed>/ECO_trainval.txt  5,247 sequences (2,853 promoters, 2,394 non-promoters)
datasets_leakfree/seed_<seed>/ECO_train.txt     4,198 sequences   (80% of ECO_trainval.txt)
datasets_leakfree/seed_<seed>/ECO_val.txt       1,049 sequences   (20% of ECO_trainval.txt)
datasets_leakfree/seed_<seed>/ECO_test.txt        927 sequences (461 promoters, 466 non-promoters)
datasets_leakfree/seed_<seed>/ENH_trainval.txt  2,868 sequences (1,384 enhancers, 1,484 non-enhancers)
datasets_leakfree/seed_<seed>/ENH_train.txt     2,295 sequences   (80% of ENH_trainval.txt)
datasets_leakfree/seed_<seed>/ENH_val.txt         573 sequences   (20% of ENH_trainval.txt)
datasets_leakfree/seed_<seed>/ENH_test.txt        400 sequences (200 enhancers, 200 non-enhancers)
```

Tab-separated text with a `sequence  label` header; the sizes are the same for the three seeds, the content of
the splits is different. `*_train.txt`, `*_val.txt` and `*_test.txt` are exactly the training, validation and
test sets used in the paper for that seed, and they are disjoint. `*_trainval.txt` is the training file of the
resplit: the training scripts read it and hold out its first 20% after
`DataFrame.sample(frac=1, random_state=seed)`, which reproduces `*_val.txt`. The order of the sequences in
`*_val.txt` and `*_test.txt` is the order of the saved logits.

Convert them to the layout read by the training scripts, then point `DATA_DIR` to the folder of the run seed:

```bash
python scripts/data/prepare_leakfree.py            # -> datasets_leakfree_csv/seed_<seed>/{ecoli,enhancer}/
for SEED in 2020 1969 1933; do
  DATA_DIR=datasets_leakfree_csv/seed_$SEED SEED=$SEED TASKS="ecoli enhancer" ARCH=genexplorer \
      bash scripts/run_benchmark.sh
done
```

The splits can be regenerated from the original archives with `scripts/data/resplit.py` (requires BLAST+):

```bash
python scripts/data/resplit.py --data datasets --out datasets_leakfree_csv --no_priority --threads 8
```

---

## Training configurations

Each architecture is trained under four configurations, each derived from the native protocol of one
architecture (`--config A|B|C|D` = C1–C4):

| Config | Origin | Loss | Augmentation | Training length |
| :--- | :--- | :--- | :--- | :--- |
| A (C1) | GenExplorer | Focal loss ($\gamma=2$, label smoothing 0.05) | Reverse complement ($p=0.5$, not on ECO) + shift ±5 bp | Early stopping on validation AUROC |
| B (C2) | FullE | BCE | none | Fixed number of epochs |
| C (C3) | Nucleic Transformer | Cross-entropy (2 classes) | Online point mutation ($\min(40, \lfloor 0.13L \rfloor)$ positions) | Fixed number of epochs |
| D (C4) | DNABERT-2 | Cross-entropy (2 classes) | none | Early stopping, gradient clipping 0.8 |

Optimizers, learning rates, schedules, epochs and patience differ per architecture and are set in the
training scripts (they are listed in the appendix of the paper). The focal-loss $\alpha$ is 0.75, except for the
Nucleic Transformer, which sets it from the class frequencies of the training set (clipped to [0.5, 0.9]).

---

## Installation

```bash
pip install -r requirements.txt
```

DNABERT2-CAMP needs `transformers` and, for the DNABERT-2 flash-attention kernels, `triton`. The pre-trained
weights are downloaded from Hugging Face (`zhihan1996/DNABERT-2-117M`) unless `DNABERT2_MODEL_PATH` points
to a local copy.

---

## Repository layout

```
models/        GenExplorer, GenExplorer ablation, FullE, Nucleic Transformer, DNABERT2-CAMP, HelixPR
utils/         dataset splits and encodings (data_utils.py), pooling layers, HelixPR / Komlós encodings
pipelines/     training scripts (one per architecture) + GE-T ablation + classical k-mer baselines
evaluation/    evaluate.py (validation/test logits), summarize_auroc.py (AUROC tables)
scripts/       benchmark and ablation launchers, Slurm template; data/: leakage-free splits (resplit.py, prepare_leakfree.py)
datasets/      benchmark archives (original files)
datasets_leakfree/  leakage-free ECO and ENH splits used in the paper (one folder per seed)
```

---

## Usage

### 1. Single model: train, then evaluate

```bash
# Train GenExplorer on ECO under C1
python pipelines/train_genexplorer.py --task ecoli --config A --seed 2020 \
    --data_folder ./datasets --checkpoint_dir ./checkpoints_seed_2020

# Validation + test logits (for viral: VM test and the 15 SIM sets)
python evaluation/evaluate.py --arch genexplorer --task ecoli --config A --seed 2020 \
    --checkpoint ./checkpoints_seed_2020/best_genexplorer_ecoli_A.pt \
    --data_folder ./datasets \
    --logits_dir ./global_logits_2020 --val_logits_dir ./global_logits_val_2020
```

Training scripts: `train_genexplorer.py`, `train_fulle.py`, `train_nucleic_transformer.py`,
`train_dnabert2_camp.py`, `train_helixpr.py`. `--arch` values for `evaluate.py`: `genexplorer`, `fulle`,
`nucleictransformer`, `dnabert2camp`, `helixpr`.

Use the same `--seed` for training and evaluation: on ECO, ENH and OOD it also defines the validation split.

### 2. Full grid for one architecture and one seed

```bash
ARCH=genexplorer SEED=2020 bash scripts/run_benchmark.sh
ARCH=helixpr     SEED=2020 bash scripts/run_benchmark.sh      # HelixPR: config A only
ARCH=fulle SEED=1969 TASKS="ecoli enhancer" CONFIGS="A B" bash scripts/run_benchmark.sh
```

The launcher trains and evaluates every (task, configuration) pair, skips pairs whose logits already exist and
restarts interrupted trainings. Other variables: `DATA_DIR`, `OUT_DIR`, `PYTHON`, `DNABERT2_PYTHON`
(e.g. a container wrapper for DNABERT2-CAMP), `EXTRA_TRAIN_ARGS`.

On a Slurm cluster, adapt the header and the variables of `scripts/slurm/run_benchmark.slurm`, then:

```bash
mkdir -p logs
sbatch --export=ALL,ARCH=genexplorer,SEED=2020 scripts/slurm/run_benchmark.slurm
```

The paper uses seeds 2020 (main experiments), 1969 and 1933.

### 3. Ablations and classical baselines

```bash
bash scripts/run_ablations.sh c1          # C1 on ECO/ENH: with RC, without RC, focal alpha = 0.5
bash scripts/run_ablations.sh get         # GE-T: GenExplorer without the TransformerBranch
bash scripts/run_ablations.sh classical   # k-mer frequencies + logistic regression / random forest
```

Reverse complement and focal $\alpha$ can also be set directly on the training scripts:
`--rc auto|on|off` (default `auto`: off on ECO) and `--focal_alpha` (C1).

### 4. AUROC tables

```bash
python evaluation/summarize_auroc.py --seed 2020 \
    --logits_dir global_logits_2020 --val_logits_dir global_logits_val_2020
```

---

## Output files

```
checkpoints_seed_<seed>/best_<arch>_<task>_<cfg>.pt
global_logits_<seed>/<arch>_cfg<cfg>_<task>_test_<split>.txt
global_logits_val_<seed>/<arch>_cfg<cfg>_<task>_val.txt        (viral: _val_VM.txt)
```

with `<split>` = `VM`, `ecoli`, `enhancer`, `anellovirus` or `sim_noise_<n>_seed_<s>_fullset_test`.
Each logit file has one score per line, in the order of the corresponding CSV file. For configurations
A and B the score is the logit of the positive class; for C and D it is $z_1 - z_0$, so that a sigmoid gives
the same probability as the softmax.

Ensembles, normalizations and statistical analyses are computed from these files.

---

## Reproducibility

Seeds fix the data partition, the initialization and the data order. GPU training is not bit-wise
deterministic (cuDNN kernels, floating-point non-associativity), so repeated runs with the same seed can
differ slightly; the paper reports mean ± standard deviation over three seeds.
