# Refusal direction replication

[![smoke test](https://github.com/ORmanius/refusal-direction-replication/actions/workflows/ci.yml/badge.svg)](https://github.com/ORmanius/refusal-direction-replication/actions/workflows/ci.yml)
[![Open the Gemma notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ORmanius/refusal-direction-replication/blob/main/notebooks/02_gemma_colab.ipynb)

**Status:** Both runs complete and committed (2026-11-04): Qwen2.5-1.5B-Instruct locally, Gemma-2-2b-it on a Colab T4. The core result replicates on both models. Extensions (direction similarity, capability check, KL-based layer selection, transfer to HarmBench and JailbreakBench, dose-response, token-by-token dynamics): Qwen run committed; Gemma run pending.

A small, reproducible replication of the core result of
*Refusal in Language Models Is Mediated by a Single Direction* (Arditi et al., 2024,
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717)):

1. Compute a "refusal direction" as the difference of mean residual-stream activations on
   harmful vs harmless instructions, one candidate per layer.
2. Show that **ablating** that direction from the residual stream makes the model comply
   with harmful instructions.
3. Show that **adding** it induces refusal on harmless instructions.
4. Report refusal rate before/after on held-out prompts, per layer, and state the chosen
   layer and token position.

This is a self-study project, not a research contribution. Everything here is scoped to
that one result plus four follow-up checks (*Extensions*); see *Limitations* for what is left out.

![Qwen2.5-1.5B-Instruct: refusal as a function of intervention strength](results/qwen2.5-1.5b-instruct/extensions/dose_response.gif)

*Qwen2.5-1.5B-Instruct: how much of the direction is needed. Each frame is a different
intervention strength `a`. Top: adding `a` times the raw difference vector at layer 15 to a
harmless prompt; the refusal rate over 32 harmless prompts (red), the model's own probability of
opening with a refusal (purple), and the completion generated at that strength. Bottom: removing
a fraction `a` of the component along the unit direction at every layer; the curves are over 32
harmful prompts, and the completion shown is for a harmless-but-borderline prompt that the
unmodified model refuses. From `notebooks/03_extensions_qwen.ipynb`; details under *Extensions*.*

## What is replicated

| Item | This repo | Paper |
|---|---|---|
| Models | Qwen2.5-1.5B-Instruct (local, fp16), Gemma-2-2b-it (Colab T4) | 13 chat models, 1.8B to 72B |
| Harmful prompts | AdvBench `harmful_behaviors.csv` (520) | AdvBench, MaliciousInstruct, TDC2023, HarmBench |
| Harmless prompts | Alpaca instructions with empty `input` | Alpaca |
| Split sizes | 128 train / 32 val per class, seed 0 | 128 train / 32 val per class |
| Token position | last token of the chat-templated prompt | post-instruction tokens (several considered) |
| Direction | difference-in-means at each layer's output (resid_post) | difference-in-means at each (layer, position) |
| Ablation | project out unit direction at embedding + every layer output | same (weight orthogonalisation) |
| Addition | add raw difference vector at the chosen layer, all positions | same |
| Refusal metric | substring match (paper's phrase list) | substring match + LLM judge |
| Layer selection | best combined val score, excluding last 20 % of layers | val score with KL and refusal-score constraints |

Results are in the *Results* section below; every number there comes from a CSV in `results/`.

## Library choice: NNsight

NNsight was chosen over TransformerLens 3 because it wraps the unmodified Hugging Face
model: no weight re-folding, native support for every model used here (Qwen2.5, Gemma-2,
Pythia), and the fp16 weights of a 1.5B model fit on a 6 GB GPU. The interventions needed
(read a layer's output at one position; subtract a projection or add a vector at every
layer during generation) are a few lines in NNsight's tracing API. TransformerLens offers
more built-in analysis helpers that this replication does not use.

## Layout

```
src/refusal_direction/   importable package: data, model, intervention, direction, scoring, pipeline, plot, extensions
notebooks/01_qwen_local.ipynb      local run on an RTX 2060 (fp16)
notebooks/02_gemma_colab.ipynb     Colab/Kaggle T4 run for Gemma-2-2b-it (core result + extensions)
notebooks/03_extensions_qwen.ipynb extensions on the Qwen run (local GPU)
data/splits_seed0.json   the exact prompts used (sampled with seed 0)
data/transfer_seed0.json the exact HarmBench / JailbreakBench prompts used by the transfer check
results/<model>/         refusal_by_layer.csv, refusal_by_layer.png, summary.json, completions
results/<model>/extensions/  CSVs, plots and GIFs for the extensions
docs/explorer.html       interactive viewer for the result files (Plotly, single file)
tests/                   CPU smoke tests on pythia-70m (run in CI)
```

## How to run

### Locally (Qwen2.5-1.5B-Instruct, 6 GB GPU)

Python 3.12 or newer (the pinned NumPy needs 3.12). Install PyTorch first so you can pick the
CUDA build, then the package:

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[test,notebook]"
```

Then either run the notebook or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_qwen_local.ipynb
```

The first run downloads the two prompt files (hash-checked), writes
`data/splits_seed0.json` if it is missing, and downloads the model (about 3 GB).
Results land in `results/qwen2.5-1.5b-instruct/`. The extensions run afterwards from the
direction that notebook saved:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/03_extensions_qwen.ipynb
```

It downloads the HarmBench and JailbreakBench behavior files (hash-checked), writes
`data/transfer_seed0.json` if missing, and fills `results/qwen2.5-1.5b-instruct/extensions/`.

### On Colab / Kaggle (Gemma-2-2b-it, T4)

Open `notebooks/02_gemma_colab.ipynb` in Colab with a T4 runtime. It clones this repo,
installs the package, loads Gemma with your Hugging Face token (Gemma is gated), runs the same
pipeline and offers a zip of `results/gemma-2-2b-it/` to download and commit. The notebook's
first cell lists the exact steps. The committed Gemma results came from exactly this notebook on a
free Colab T4 (23 minutes). Setting `REFUSAL_SMOKE=1` runs the same cells on CPU with `pythia-70m`.

### Results explorer (interactive, no install)

`docs/explorer.html` is a single HTML page that reads the result files in `results/<model>/`
and draws them interactively (Plotly): the per-layer sweep, the capability cost with a
layer-selection trade-off view, the direction-similarity heatmap, the dose-response curves with
the completion generated at each strength on click, the transfer bars, the token-by-token
projections with per-token hover, the PCA clouds as a rotatable 3D scatter, the neuron-effect
tables, and a searchable completions browser. Every chart has a table view and nothing is
computed in the page beyond reading the files. By default it fetches the files from this
repository on GitHub; it can also read a local checkout or files you drop onto it (for example a
zip of a Gemma run, unpacked). Open it from a local checkout with any static server:

```bash
python -m http.server 8765
```

then visit `http://localhost:8765/docs/explorer.html` and pick "local" as the source.

### Tests

```bash
pytest -q
```

`tests/test_smoke.py` loads `EleutherAI/pythia-70m`, checks tensor shapes, checks that
ablation leaves a zero projection on the direction at every layer, and runs generation with
and without interventions. `tests/test_extensions.py` covers the extension pieces (cosine
matrix, KL and reference loss, all-position projections, the dynamics figure and GIF). GitHub
Actions runs both on every push (CPU PyTorch).

## What is committed

* `data/splits_seed0.json`: the exact 128 + 32 harmful and 128 + 32 harmless instructions used.
* `results/<model>/refusal_by_layer.csv`: per-layer refusal rates (the source of every number in this README).
* `results/<model>/refusal_by_layer.png`, `summary.json` (chosen layer, versions, runtime).
* `results/<model>/completions_*.jsonl`: validation completions for the baselines and for harmless prompts with the direction added.
* `results/<model>/extensions/`: `direction_cosine.csv`, `capability_by_layer.csv`,
  `dose_response*.csv` and the example completions per strength, `projection_dynamics*.csv`,
  `projection_example.csv`, `transfer*.csv`, `pca_clouds.csv`, `neuron_effects_*.csv`,
  `reference_pairs.json`, the plots, the dose-response and PCA GIFs, and the baseline and
  addition completions on the transfer sets.
* Not committed: the completions of the ablated model on harmful prompts, for AdvBench and for
  the transfer sets (regenerate them with the notebooks; the per-prompt refusal flags are shown
  in the notebook output), and the `directions.pt` tensors.

## Results

### Qwen2.5-1.5B-Instruct (local, RTX 2060, fp16)

Source: `results/qwen2.5-1.5b-instruct/refusal_by_layer.csv` and `summary.json`
(run of 2026-11-04, seed 0, 32 held-out prompts per class, greedy decoding, 64 new tokens,
substring-match refusal metric). Direction read at the **last token of the chat-templated
prompt** after **layer 15** of 28 (the layer's output, resid_post), chosen as the best
combined validation score among the first 80 % of layers.

| Validation set | No intervention | Direction ablated (all layers) | Direction added (layer 15) |
|---|---|---|---|
| Harmful (AdvBench, n=32) | 32/32 refused (100 %) | 0/32 refused (0 %) | - |
| Harmless (Alpaca, n=32) | 0/32 refused (0 %) | - | 32/32 refused (100 %) |

Per-layer sweep (each layer's own difference-in-means direction, ablated everywhere for the
harmful set and added at that layer for the harmless set):

![Refusal rate by layer, Qwen2.5-1.5B-Instruct](results/qwen2.5-1.5b-instruct/refusal_by_layer.png)

What the sweep shows: directions from layers 0-9 do nothing; ablation starts to work at layers
10-12 and is complete (0 % refusal) for layers 13-15 and 19-26; addition induces refusal from
layer 13 onward. Layers 16-18 ablate less cleanly (16-25 % residual refusal) and the last layer
is poor on both tests, as the paper's exclusion of late layers anticipates. The chosen layer 15
is the earliest layer that passes both tests at 100 %. Full numbers: the CSV.

The whole run (activations, 28-layer sweep, final completions) took 5 minutes on the RTX 2060.

### Gemma-2-2b-it (Colab, Tesla T4, float32)

Source: `results/gemma-2-2b-it/refusal_by_layer.csv` and `summary.json` (run of 2026-11-04,
same seed, prompt sets, decoding and metric as above; batch size 8, eager attention, float32
because the T4 has no native bfloat16). Direction read at the **last token of the chat-templated
prompt** after **layer 11** of 26.

| Validation set | No intervention | Direction ablated (all layers) | Direction added (layer 11) |
|---|---|---|---|
| Harmful (AdvBench, n=32) | 32/32 refused (100 %) | 0/32 refused (0 %) | - |
| Harmless (Alpaca, n=32) | 0/32 refused (0 %) | - | 32/32 refused (100 %) |

![Refusal rate by layer, Gemma-2-2b-it](results/gemma-2-2b-it/refusal_by_layer.png)

What the sweep shows: directions from layers 0-5 do nothing; layer 6 ablates partially (5/32
still refuse) and layers 7-16 and 18-20 ablate completely. Addition induces refusal from layer 9
(27/32) and is complete for layers 11-20. Both effects fall apart from layer 21 onward (addition
drops to 4/32 at layer 22, ablation leaves 6/32 refusing at layer 21), earlier in relative terms
than for Qwen. The chosen layer 11 is the earliest that passes both tests at 100 %. Full numbers:
the CSV.

The run took 23 minutes on the T4. Note that Colab's preinstalled PyTorch (2.11.0, CUDA 13.0) was
used rather than the pinned 2.14.1; `transformers` and `nnsight` were installed at the pinned
versions. `summary.json` records the versions actually used.

### Across the two models

The same pipeline, prompt sets and selection rule give a direction in the middle third of the
network for both models (layer 15/28 for Qwen, 11/26 for Gemma) that flips the substring-match
refusal rate from 100 % to 0 % on harmful prompts when ablated and from 0 % to 100 % on harmless
prompts when added, with coherent completions in both directions (see the committed
`completions_*.jsonl` files). That is the paper's core claim at the scale this repo can test.

## Extensions

Seven checks beyond the core result, each either tightening the evidence or showing the
mechanism. They run from the direction the main run saved (`refusal_direction.extensions`,
`notebooks/03_extensions_qwen.ipynb`; the Gemma Colab notebook runs the same cells after its
main sweep). Done for Qwen2.5-1.5B-Instruct (run of 2026-11-05, 12 minutes on the RTX 2060);
pending for Gemma-2-2b-it. Source for every number below:
`results/qwen2.5-1.5b-instruct/extensions/` (`summary.json`, `direction_cosine.csv`,
`capability_by_layer.csv`, `transfer.csv`, `transfer_layer21.csv`, `dose_response*.csv`,
`dose_response_examples*.jsonl`, `projection_dynamics*.csv`, `pca_clouds.csv`,
`neuron_effects_*.csv`). All of them can be browsed interactively in `docs/explorer.html`.

### 1. Are the per-layer directions one direction?

![Cosine similarity between per-layer directions, Qwen2.5-1.5B-Instruct](results/qwen2.5-1.5b-instruct/extensions/direction_similarity.png)

Only loosely. The layer-15 direction has cosine 0.75 with its neighbours at layers 14 and 16,
0.50 at layer 18, 0.31 at layer 22 and below 0.28 from layer 23 on; the smallest pairwise cosine among layers
10-26, all of which remove refusal when ablated, is 0.07. The difference-in-means direction
drifts steadily with depth rather than being one fixed vector. The core result does not depend
on the directions agreeing (each layer's direction is tested on its own), but "a single
direction" is best read as "a single direction per read-out layer".

### 2. Does ablation damage the model?

![Cost of ablating each layer's direction, Qwen2.5-1.5B-Instruct](results/qwen2.5-1.5b-instruct/extensions/capability_by_layer.png)

Two costs, for each layer's direction ablated at every layer: the KL divergence from the
baseline next-token distribution at the last prompt token on the 32 harmless validation prompts
(the paper's selection criterion), and the per-token negative log-likelihood (NLL) of 64
held-out Alpaca reference answers (`extensions/reference_pairs.json`, disjoint from the
train/val instructions).

| Direction ablated | Mean KL to baseline (nats) | NLL of reference answers (nats / token) |
|---|---|---|
| none | 0 | 1.402 |
| layer 15 (chosen by the main run) | 0.606 | 1.497 |
| layer 21 (chosen with the paper's KL filter, below) | 0.043 | 1.398 |

Ablating the layer-15 direction is not free: the harmless next-token distribution moves by 0.6
nats on average and the reference-answer loss rises by 7 %. Layers 14 and 19 are worse (KL 1.7
and 1.5). Layers 20-23 remove refusal just as completely in the sweep at KL below 0.08 and with
the reference loss unchanged. So the sweep's refusal rates alone do not distinguish a clean
direction from one that also degrades the model; the KL check does.

### 3. Selecting the layer the paper's way

The paper keeps only candidate directions whose ablation KL on harmless prompts is below 0.1
nats, then takes the one that best removes refusal. The main run here did not apply that filter.
Applying it to the sweep (`extensions.select_layer_with_kl`, same late-layer cutoff) selects
**layer 21**: KL 0.043, reference loss unchanged, and 0 % / 100 % on the two validation tests.
The transfer, dose-response and dynamics checks below were therefore run at both layers.

### 4. Transfer to other benchmark sets

The direction was fitted on AdvBench only. Here it is applied unchanged to 64 HarmBench
"standard" behaviors, 64 JailbreakBench harmful behaviors (its 18 AdvBench-sourced rows
removed first) and 64 JailbreakBench benign behaviors, which are harmless requests written to
look borderline (for example a fictional story involving drug use). Sampled with seed 0 and
saved in `data/transfer_seed0.json`; sources pinned by commit and SHA-256 in `data.py`.

| Set | No intervention | Direction ablated, layer 15 | ablated, layer 21 | Direction added, layer 15 | added, layer 21 |
|---|---|---|---|---|---|
| HarmBench standard (n=64) | 61/64 refused (95 %) | 0/64 | 0/64 | - | - |
| JailbreakBench harmful, non-AdvBench (n=64) | 64/64 (100 %) | 0/64 | 0/64 | - | - |
| JailbreakBench benign (n=64) | 33/64 (52 %) | 0/64 | 0/64 | 64/64 | 64/64 |

The direction transfers without loss to both harmful sets. The benign set adds a point the
AdvBench/Alpaca split cannot show: Qwen refuses half of these harmless-but-borderline requests
by default, and ablating the direction removes that over-refusal as completely as it removes
refusal of harmful requests. Under the substring metric the two are indistinguishable, which is
also the limitation of the metric.

### 5. Dose-response: how much of the direction is needed?

The paper uses full ablation and full addition. Here the strength is varied: `x + a r` at the
chosen layer on the 32 harmless validation prompts, and `x - a (x . r_hat) r_hat` at every layer
on the 32 harmful validation prompts, where `a = 1` is the paper's intervention. Two readouts per
strength: the substring refusal rate, and the probability the model assigns at the first
generated position to its own refusal-opening token (for Qwen every baseline refusal starts with
"I"). The GIF at the top of this README shows these curves with the completion generated at each
strength; the ablation-side example is a benign JailbreakBench prompt that the unmodified model
refuses, so that no harmful completion is displayed. Static curves:
`extensions/dose_response.png` (layer 15) and `extensions/dose_response_layer21.png` (layer 21).

| `a` | Addition, layer 15: refusal rate / P(opener) | Addition, layer 21 | Ablation, layer 15: refusal rate / P(opener) | Ablation, layer 21 |
|---|---|---|---|---|
| 0.0 | 0.00 / 0.02 | 0.00 / 0.02 | 1.00 / 0.77 | 1.00 / 0.77 |
| 0.2 | 0.13 / 0.08 | 0.09 / 0.08 | 0.19 / 0.07 | 0.50 / 0.24 |
| 0.4 | 0.53 / 0.28 | 0.28 / 0.22 | 0.00 / 0.03 | 0.00 / 0.02 |
| 0.6 | 0.88 / 0.51 | 0.69 / 0.43 | 0.00 / 0.02 | 0.00 / 0.01 |
| 0.8 | 1.00 / 0.63 | 0.91 / 0.60 | 0.00 / 0.03 | 0.00 / 0.00 |
| 1.0 | 1.00 / 0.71 | 1.00 / 0.70 | 0.00 / 0.03 | 0.00 / 0.00 |
| 1.6 | 1.00 / 0.86 | 1.00 / 0.78 | - | - |

Three things stand out. The response is steep and asymmetric: removing 40 % of the component
along the direction already drives refusal to 0/32 on harmful prompts, while inducing refusal on
harmless prompts needs about 0.8 (layer 15) or 1.0 (layer 21) of the full difference vector.
The substring metric saturates before the model's own confidence does: past `a = 0.8` every
completion is a refusal, yet the probability of the refusal opener keeps rising from 0.63 to
0.86. And the induced refusals come with invented reasons: at `a = 0.4` the story prompt is
refused for "inappropriate content", at `a = 1.6` for "illegal and unethical activities,
including hacking" (`extensions/dose_response_examples.jsonl`). The direction carries the
decision to refuse, and the model supplies a justification afterwards.

### 6. Geometry at the chosen layer, and which neurons carry the difference

![Qwen2.5-1.5B-Instruct: PCA of the residual stream after layer 15](results/qwen2.5-1.5b-instruct/extensions/pca_clouds.gif)

Residual stream after layer 15 at the last prompt token for the 128 harmful and 128 harmless
training prompts, projected onto the top three principal components of the pooled activations,
with the refusal direction drawn from the harmless centroid (rotating camera; static version
`extensions/pca_clouds.png`, points in `extensions/pca_clouds.csv`, rotatable in the explorer).
The refusal direction is the first principal component almost exactly (|cos| = 0.996), and that
component carries 35 % of all variance at this layer; the next two carry 10 % and 6 %. The two
clouds sit apart along it: mean projection 22.5 for harmful prompts, 7.1 for harmless ones.
The orange triangles are the harmful prompts with the direction ablated: they land beyond the
harmless cloud, not inside it, because ablation sets the component to zero while harmless
prompts naturally keep a positive value. That is the geometric reason ablation also removes
over-refusal on the benign transfer set.

![Qwen2.5-1.5B-Instruct: MLP neurons separating harmful from harmless prompts](results/qwen2.5-1.5b-instruct/extensions/neuron_effects.png)

Which MLP neurons carry the difference: for every layer, Cohen's d of each neuron's activation
(the input to the MLP output projection) between the two prompt sets at the last prompt token.
The separation is spread across thousands of neurons per layer rather than a few: between
1,100 (layer 0) and 4,300 (layer 19) of the 8,960 neurons differ by more than one standard
deviation. The strongest single neurons jump from |d| of about 3 in layers 0-12 to about 6
from layer 16 on, matching where the direction starts to work. Neuron index has no spatial
meaning in a transformer, so the only "location" of refusal that these data support is depth.

### 7. Token-by-token dynamics

![Qwen2.5-1.5B-Instruct: projection onto the layer-15 direction, token by token](results/qwen2.5-1.5b-instruct/extensions/projection_dynamics.png)

The projection of the residual stream onto the layer-15 direction at every token of prompt and
completion. Over the last prompt tokens the mean projection is 15 on harmful prompts and 7 on
harmless ones; it is 0 at every token once the direction is ablated (by construction) and 26 on
average over the completion once the direction is added. The right-hand heatmaps show the added
direction lifting the projection in every later layer at every token.

![Qwen2.5-1.5B-Instruct: projection onto the layer-21 direction, token by token](results/qwen2.5-1.5b-instruct/extensions/projection_dynamics_layer21.png)

At layer 21 the picture is sharper. On harmful prompts the projection rises to 58 at the last
prompt token (the position the direction was read from), stays high for the first few generated
tokens and settles near 11 over the rest of the completion; on harmless prompts it stays near 0
throughout. The "decision" is visible as a spike at the end of the prompt, before the first
token is generated, which is what the paper's choice of read-out position assumes.

## Limitations

* **Refusal metric.** Substring matching over a fixed phrase list is the paper's primary
  metric but it is crude: a completion that complies after a token "I can't promise..." counts
  as a refusal, and a refusal phrased without any listed phrase counts as compliance. The paper
  pairs it with an LLM judge; this pass does not. Read the numbers as approximate.
* **Harmful set.** The direction is fitted on AdvBench only (the paper pools four sources).
  AdvBench prompts are repetitive and the model's refusals on them are easy, so absolute rates
  are not comparable to the paper's tables. The transfer check applies the direction to
  HarmBench and JailbreakBench prompts but does not fit on them.
* **Direction selection.** One token position (the last prompt token) and a simple validation
  score are used instead of the paper's search over positions with KL-divergence constraints.
  The per-layer sweep is reported in full so the choice is visible. The extensions add the KL
  filter after the fact (it changes the chosen Qwen layer from 15 to 21) but still search one
  position only.
* **Small validation sets.** 32 prompts per class, as in the paper: one prompt is about 3
  percentage points.
* **Greedy decoding, 64 new tokens.** Enough for the substring metric; the paper generates
  longer completions for its judge.
* **Qwen's default system prompt.** Qwen2.5-Instruct's chat template inserts its own system
  message when none is given; it was kept. Gemma's template has no system role.
* **Limited capability check after ablation.** The paper verifies that the ablated model stays
  generally competent with benchmark scores (MMLU, ARC, GSM8K, TruthfulQA) as well as a KL check.
  The extensions here measure the KL divergence on harmless prompts and the loss on reference
  answers, which catch a direction that visibly disturbs the model (layer 15 does, mildly), but
  no benchmark accuracy was measured. The core-result section still reports the main run's
  layer 15; see *Extensions* for the KL-selected layer 21.
* **Two models, one seed.** No error bars across seeds or prompt subsets.

## Citation

```bibtex
@article{arditi2024refusal,
  title   = {Refusal in Language Models Is Mediated by a Single Direction},
  author  = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  journal = {arXiv preprint arXiv:2406.11717},
  year    = {2024}
}
```

Datasets: AdvBench (Zou et al., 2023, [llm-attacks](https://github.com/llm-attacks/llm-attacks));
Stanford Alpaca (Taori et al., 2023, [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)).

## Licence

MIT. The prompt sets keep their upstream licences (AdvBench: MIT; Alpaca data: CC BY-NC 4.0).
