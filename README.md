# Refusal direction replication

**Status:** repo scaffolded; Qwen2.5-1.5B local run pending; Gemma-2-2b-it Colab run pending.

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
that one result; see *Limitations* for what is left out.

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

Results: pending.

## Library choice: NNsight

NNsight was chosen over TransformerLens 3 because it wraps the unmodified Hugging Face
model: no weight re-folding, native support for every model used here (Qwen2.5, Gemma-2,
Pythia), and the fp16 weights of a 1.5B model fit on a 6 GB GPU. The interventions needed
(read a layer's output at one position; subtract a projection or add a vector at every
layer during generation) are a few lines in NNsight's tracing API. TransformerLens offers
more built-in analysis helpers that this replication does not use.

## Layout

```
src/refusal_direction/   importable package: data, model, intervention, direction, scoring, pipeline, plot
notebooks/01_qwen_local.ipynb    local run on an RTX 2060 (fp16)
notebooks/02_gemma_colab.ipynb   Colab/Kaggle T4 run for Gemma-2-2b-it
data/splits_seed0.json   the exact prompts used (sampled with seed 0)
results/<model>/         refusal_by_layer.csv, refusal_by_layer.png, summary.json, completions
tests/test_smoke.py      CPU smoke test on pythia-70m (runs in CI)
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
Results land in `results/qwen2.5-1.5b-instruct/`.

### On Colab / Kaggle (Gemma-2-2b-it, T4)

Open `notebooks/02_gemma_colab.ipynb` in Colab with a T4 runtime. It clones this repo,
installs the package, loads Gemma with your Hugging Face token (Gemma is gated), runs the same
pipeline and offers a zip of `results/gemma-2-2b-it/` to download and commit. The notebook's
first cell lists the exact steps. The Colab cells were tested on CPU with `pythia-70m` by setting
`REFUSAL_SMOKE=1`; the Gemma run itself has to happen on the T4.

### Tests

```bash
pytest -q
```

`tests/test_smoke.py` loads `EleutherAI/pythia-70m`, checks tensor shapes, checks that
ablation leaves a zero projection on the direction at every layer, and runs generation with
and without interventions. GitHub Actions runs it on every push (CPU PyTorch).

## What is committed

* `data/splits_seed0.json`: the exact 128 + 32 harmful and 128 + 32 harmless instructions used.
* `results/<model>/refusal_by_layer.csv`: per-layer refusal rates (the source of every number in this README).
* `results/<model>/refusal_by_layer.png`, `summary.json` (chosen layer, versions, runtime).
* `results/<model>/completions_*.jsonl`: validation completions for the baselines and for harmless prompts with the direction added.
* Not committed: the completions of the ablated model on harmful prompts (regenerate them with
  the notebook; the per-prompt refusal flags are shown in the notebook output), and the
  `directions.pt` tensors.

## Results

Pending.

## Limitations

* **Refusal metric.** Substring matching over a fixed phrase list is the paper's primary
  metric but it is crude: a completion that complies after a token "I can't promise..." counts
  as a refusal, and a refusal phrased without any listed phrase counts as compliance. The paper
  pairs it with an LLM judge; this pass does not. Read the numbers as approximate.
* **Harmful set.** AdvBench only (the paper pools four sources). AdvBench prompts are
  repetitive and the model's refusals on them are easy, so absolute rates are not comparable to
  the paper's tables.
* **Direction selection.** One token position (the last prompt token) and a simple validation
  score are used instead of the paper's search over positions with KL-divergence constraints.
  The per-layer sweep is reported in full so the choice is visible.
* **Small validation sets.** 32 prompts per class, as in the paper: one prompt is about 3
  percentage points.
* **Greedy decoding, 64 new tokens.** Enough for the substring metric; the paper generates
  longer completions for its judge.
* **Qwen's default system prompt.** Qwen2.5-Instruct's chat template inserts its own system
  message when none is given; it was kept. Gemma's template has no system role.
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
