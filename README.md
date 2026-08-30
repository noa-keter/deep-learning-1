# Resolution Bias and Cross-Generator Detection of AI-Generated Images

Deep Learning, Tel Aviv University, 2026b. Noa Keter and Ido Josephsberg.

Detectors of AI-generated images score almost perfectly on the generator they trained on and
degrade sharply on unseen generators. In GenImage-style benchmarks each generator emits one
fixed image size while real photographs vary, so image dimensions alone identify both the
class and the generator. On this data a rule with **zero parameters** ("predict synthetic if
the image is square") reaches **0.977** accuracy on every source/target pair - higher than the
cross-generator accuracy of every one of our four trained detectors.

Removing that cue changes what the detector sees. We hold the architecture, dataset and
protocol fixed and vary only how image size is equalized, across four strategies; the ranking
of generators by difficulty turns out not to be stable under that choice.

## Result

One *arm* is the whole experiment run under one equalization strategy. Each arm
covers 7 source generators x 7 target generators = 49 cells, every cell balanced
at n = 1,000 and seed-averaged over two seeds.

| Strategy | Diagonal (in-domain) | Off-diagonal (transfer) | Gap |
|----------|----------------------|-------------------------|-----|
| `center_crop` | 0.908 | 0.655 | 0.254 |
| `random_crop` | 0.892 | 0.645 | 0.247 |
| `rescale` | 0.902 | **0.563** | 0.338 |
| `pad` | 0.978 | 0.828 | 0.150 |
| square rule (0 parameters) | 0.977 | **0.977** | 0.000 |
| size-lookup rule (0 parameters) | 1.000 | 0.619 | 0.381 |

Rank agreement between strategies, Spearman rho with exact permutation p-values (n = 7, all
5,040 orderings enumerated):

| Pair | rho (p) | Pair | rho (p) |
|------|---------|------|---------|
| `center_crop` / `random_crop` | **+0.964 (0.003)** | `center_crop` / `pad` | -0.286 (0.556) |
| `center_crop` / `rescale` | +0.000 (1.000) | `random_crop` / `pad` | -0.321 (0.498) |
| `random_crop` / `rescale` | -0.179 (0.713) | `rescale` / `pad` | +0.643 (0.139) |

The two strategies that do not resample agree almost perfectly. Rescaling produces an
unrelated ordering, and ADM moves from first place to last.

`pad` posts the best transfer, and that number is an artifact: every generator emits square
images and only real photographs have variable aspect ratio, so padding turns the size cue
into a border that marks the real class exactly. A border-only detector ceilings at 0.977,
and `pad`'s measured diagonal is 0.978.

## Install

```bash
pip install -r requirements.txt
```

`torch` is imported lazily inside the functions that need it, so the analysis paths run on a
machine with no GPU and no torch installed.

## Reproducing the results

Two paths. Everything that touches the dataset runs from the Colab notebooks in
`notebooks/`; the analysis runs from committed JSON on any laptop.

### From Colab

The notebooks mount Drive, check the runtime, then call the modules in `src/`. All the logic
lives in `src/` - the notebooks only drive it, and each one prints the exact
`python -m src.<module>` command it is about to run, so any step can be lifted out and run
directly against a local cache. Run them in order.

| Notebook | Step | Modules | GPU |
|----------|------|---------|-----|
| `00_build_cache.ipynb` | download Tiny-GenImage, write the four equalization caches | `src.data`, `src.pilot` | no |
| `01_run_matrix.ipynb` | the 56-run transfer matrix, resumable | `src.train` | yes |
| `02_train_val_gap.ipynb` | train vs validation accuracy, re-scored from checkpoints | `src.trainval` | yes |
| `03_rotation_sensitivity.ipynb` | D4 orientation gate | `src.rotation` | yes |
| `04_cues.ipynb` | radial spectra and input-gradient attribution | `src.analyze` | yes |
| `05_ensemble.ipynb` | per-cell test logits, then the off-GPU arm combination | `src.logits`, `src.ensemble` | yes |

Notebook 00 is 8.4 GB of download and decode with no training in it, so a CPU runtime is the
right one there. The other five want a T4. Notebook 01 resumes on `if metrics.json exists:
skip`, so a disconnect costs at most the run that was in flight.

### On a laptop, from the committed files

These three read the committed JSON and nothing else:

```bash
python -m src.analyze matrices --baseline-dir results/baseline --out-dir results/figures
python -m src.analyze ranking --out-dir results/figures
python -m src.ensemble --logits results/figures/logits_ido.npz
```

`analyze matrices` prints the six transfer matrices with their diagonal, off-diagonal and gap;
`analyze ranking` prints both rank tables with Spearman rho and exact permutation p-values.
Each rewrites its JSON byte-identically to the committed copy. `src.ensemble` reproduces both
arms cell by cell against the recorded matrix before combining anything.

The defaults already point at `results/runs`; passing `--results-dir results` breaks them.
`--baseline-dir` is what adds the two zero-parameter rules as extra panels.

### The two commands with no notebook

Both need the cache built by notebook 00.

```bash
# zero-parameter baselines: per-row metadata only, never pixels, so no GPU
python -m src.baseline --cache /path/to/cache --out results/baseline

# epoch-budget control: three sources at 80 epochs instead of 40
python -m src.train --cache-dir /path/to/cache --strategy center_crop --source VQDM \
  --seed 0 --epochs 80 --results-dir results/experiments/ctrl80
```

## Layout

```
src/data.py        dataset download, decode, the four equalization transforms, cache build
src/model.py       CompactCNN, 1,173,473 parameters, trained from scratch
src/train.py       training loop, checkpoint selection, per-run metrics
src/baseline.py    square rule and size-lookup rule, zero parameters
src/analyze.py     transfer matrices, ranking, Spearman, attribution, spectra
src/trainval.py    train/val gap diagnostic, re-scored from checkpoints
src/logits.py      dumps per-cell test logits so arms can be combined off-GPU
src/ensemble.py    off-GPU combination of arms from dumped logits
src/rotation.py    D4 orientation ablation; supplies constants to logits and ensemble
src/pilot.py       cache-build pilot, used to verify shards before the full build
notebooks/         Colab drivers; all logic lives in src/
results/runs/         56 metrics.json, committed
results/baseline/     14 metrics.json, committed
results/experiments/  epoch-budget control (ctrl80)
results/figures/      figures and analysis JSON
```

## Data

[Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), a 35,000-image
subset of GenImage (Zhu et al., 2023), CC BY-NC-SA 4.0. 17,500 real ImageNet photographs and
2,500 images from each of seven generators: ADM, BigGAN, GLIDE, Midjourney, SD 1.5, VQDM and
Wukong.

The shipped `validation` split is used as the **test** set; our validation set is carved from
10% of `train`. Per run: 3,600 train / 400 validation / 4,000 test. Model selection uses the
internal validation set only, so no reported number ever influenced training.

## Notes on reproducibility

- Seeds are fixed and the cache build is deterministic, so each reported cell reproduces from
  a clean clone.
- The analysis is reproducible without a GPU: re-running `analyze matrices` and
  `analyze ranking` regenerates `transfer_matrices.json` and `ranking.json` byte-identical to
  the committed files.
- `*.pt` and `cache/` are gitignored. Every `metrics.json` is committed, which is what lets
  the analysis run without the weights.
- Reported cells come from the official test split and never influenced model selection.
- The train/val diagnostic checks itself: recomputed validation accuracy must reproduce the
  value recorded during training, and it does so exactly across all 28 runs.
