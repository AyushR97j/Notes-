# Brief for the Deep Learning book (handed to the DL writer)

You are writing **`Deep Learning.pdf`** — a Napkin-style (Evan Chen, *An Infinitely Large Napkin*)
LaTeX study book that takes an Indian campus-placement candidate (IIT/NIT/IIIT on-campus) from the
perceptron → CNN/RNN → Transformers → LLMs. A companion book, *Classical Machine Learning*, is being
written in parallel by someone else in `chapters/ml/`; you own everything DL. Work autonomously:
never ask questions, make the reasonable choice, record it in `progress/dl.md`, keep going.

**Acceptance test:** a reader who works through the book can solve every DL question Indian campus
placements ask in online assessments (OAs) and technical interviews, and can derive and implement the
core algorithms from a blank file.

**Author name** everywhere (title page, running head, PDF metadata, any mention): **Philospher**
(exactly this spelling).

## 0. Infrastructure that already exists (read these first)
- `latex/preamble.tex` — shared template adapted from Napkin (GPL notice in header). Do **not** edit it
  except for genuine bugs (log any edit in `progress/dl.md`). Put DL-only macros in `latex/dl-macros.tex`.
- `ml.tex` — the ML main file. Create `dl.tex` the same way (`\bookid{dl}`, `\booktitle{Deep Learning}`,
  `\bookshort{Deep Learning}`, `\input{latex/dl-macros}`, `\addbibresource{latex/refs-dl.bib}`,
  `\input{gen/dl/all}`, chapters `\include{chapters/dl/NN-name}`, hints/solutions appendices via
  `\printanswers`, bibliography).
- `test.tex` + `chapters/test/t01.tex` + `code/test/t01_demo.py` — a working example of every
  environment and of the code→book pipeline. Copy its patterns.
- `code/common.py` — `out = setup(__file__)` (seeds numpy/random/torch; import torch BEFORE calling
  setup so torch gets seeded); `out.val(key, value, nd)` → `\gv{<script>/<key>}` in LaTeX;
  `out.dat(name, {col: array})` → `gen/dl/data/<script>-<name>.dat` for pgfplots;
  `out.text(name, str)` → `gen/dl/out/<script>-<name>.txt` shown with `\outputfile{...}`;
  `out.tex(key, latex_fragment)` for generated tables; `out.check(label, ours, ref, atol, rtol)` =
  `assert np.allclose` + prints both numbers + logs to `gen/dl/checks.log` (build fails on mismatch).
  Code regions marked `# [[name]]` … `# [[/name]]` are extracted to `gen/dl/snip/<script>-<name>.py`
  and shown with `\codefile[title]{gen/dl/snip/...}`.
- `build.sh dl` runs every `code/dl/*.py` (4 in parallel, so scripts must be independent), extracts
  snippets, writes `gen/dl/all.tex`, builds `dl.pdf` with latexmk and copies it to `Deep Learning.pdf`.
  `SKIP_CODE=1 ./build.sh dl` rebuilds LaTeX only. Script naming: `code/dl/cNN_topic.py` (chapter NN).
  Add a `code/dl/a00_versions.py` like `code/ml/a00_versions.py` for the preface's versions line.
- Installed: numpy, scipy, scikit-learn, matplotlib, xgboost, lightgbm, torch (CPU use only), TeX Live
  with pgfplots/tcolorbox/thmtools/biblatex+biber. `pdftotext`, `pdftoppm` available (you can look at
  rendered pages with pdftoppm → PNG → Read tool to check figures).
- Environments: `theorem`/`lemma`/`proposition`/`corollary` (blue), `definition`, `example` (red,
  worked numerics), `remark` (green bar; start trap remarks with `\trap`), `ques` (stop-and-think),
  `moral` (one-line takeaway), `exercise`, `algo` (algorithm box), `pycode` (inline listing),
  `\codefile`, `\outputfile`, `\prototype{}`, `\vocab{}`, `\stub{}` (draft only), `problem` with
  `hint` and `sol` inside, `\onechili`/`\twochili`/`\threechili` right after `\begin{problem}[..]`.
  Chapter-end problem section: `\section\problemhead`. In the final big problem chapter call
  `\bigproblemset` right after `\chapter{Problems}` so problems are numbered P1, P2, ….
- Shared math macros at the bottom of the preamble: `\x \w \W \X \Q \K \V \T` (transpose) `\E \Var
  \softmax \relu \Loss \Normal \KL \pd{}{}` etc. — check before defining new ones.

## 1. Git rules
- You are in your own git worktree/branch. Commit after **every finished chapter** (message
  `DL ch NN: <title>`), including the regenerated `gen/dl/` files and the rebuilt `Deep Learning.pdf`.
  End every commit message with these two lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01XsF1LgBVGcEmcnvycW2xN4`
- Do not push, do not rewrite history, do not touch `chapters/ml/`, `code/ml/`, `gen/ml/`, `ml.tex`,
  `progress/ml.md`.
- Keep `progress/dl.md`: first line `RESUME HERE: …` (always current), then decisions, gap plan,
  chapter checklist, log. If you find it already populated, resume from it instead of restarting.

## 2. What placements actually ask (weight depth with this)
Evidence from a collected set of real company papers (49 non-coding ML/DL questions from Info Edge,
Fujitsu's IISc and IIT-KGP papers, Qualcomm IISc and EXL): deep learning 22, supervised 9,
optimisation/loss/regularisation 7, preprocessing/NLP/misc 6, unsupervised 3, evaluation 1, stats 1.
OAs are one-line concept MCQs + one-minute hand numerics + "which statements are true" multi-selects.
DL-relevant patterns seen: conv output shape and parameter count; softmax confidence and cross-entropy
from a 5-logit vector; a neuron's weighted-sum output; the purpose of Adam; the effect of λ in
E + λΣw²; generalisation gap; XOR and the perceptron; weight sharing; ResNet and vanishing gradients;
LSTM for time series; which layer reduces spatial size; what fine-tuning means.
Interviews go deeper and decide offers: a founder-led round at Convin.ai asks for transformers from
scratch; Sarvam AI's shortlist is a take-home LLM app plus an essay; ML-engineer loops ask derivations
and "implement it in NumPy/PyTorch", with follow-ups three "why?"s deep.
So **every topic gets four layers**: (1) intuition, with an everyday analogy where it helps; (2) the
precise definition or derivation; (3) a worked numerical, computed by code, in the shape an OA would
set; (4) the trap an interviewer or question-setter plants. Absence of a topic from the 49 questions
is NOT a reason to skip it.

## 3. Style and quality bar
- Napkin shape: chapter → sections, each opening with `\prototype{}` (one concrete example) →
  subsections; new terms in `\vocab{}`; theorem/lemma (blue), `example` (red, worked numerics),
  `remark` (green bar, traps), `ques` (stop-and-think), `moral` (one-line takeaway); chapter-end
  problems with `problem` + `hint` + `sol` and chili ratings.
- `\stub{}` only while drafting; none may remain in the final PDF. No todonotes.
- All figures TikZ/pgfplots with data from `.dat` files your scripts write; never screenshots/images:
  activations, loss surfaces/optimiser paths, vanishing gradients vs depth, attention heat-maps
  (pgfplots `matrix plot`), network and computation graphs, conv sliding window, LSTM cell, Transformer
  block diagram, LR schedules, train/val curves.
- Dense, interview-shaped prose like a careful senior who has sat these interviews. No padding, no
  "in this section we will…", no AI tone, no emoji.
- Length: **140–190 pages**.
- Ends with a **Rapid-fire** chapter (80–100 spoken-style 1–3 line answers) and **≥ 60 problems** with
  hints and solutions in the shape of OA questions. Problems must be original.
- Write `.tex` with the Write/Edit tools or a *quoted* heredoc (`<<'EOF'`).
- Cross-reference *Classical Machine Learning* in plain text ("see the logistic-regression chapter of
  the companion book *Classical Machine Learning*"); assume its maths and logistic-regression chapters.

## 4. Verification (non-negotiable)
- Every worked numerical, table, parameter/shape/FLOP count and "this prints" claim comes from a
  script in `code/dl/`, run by `build.sh`, with real output `\input`/`\gv`'d into the book.
- Fixed seeds, CPU, seconds per script, library versions printed (common.py does it).
- Every from-scratch implementation is checked against the reference (PyTorch, SciPy, scikit-learn)
  with `out.check` (explicit `np.allclose` assert, prints both numbers). If they disagree, the book is
  wrong until proven otherwise.
- Simulations that illustrate a claim (vanishing gradients through depth, optimiser curves, init
  variance through depth, BN effect, dropout scaling, attention √d_k variance, …) are run and their
  data drives the pgfplots figure.
- Where an MCQ has a popular wrong answer (e.g. "dropout increases capacity", "ReLU is linear so it
  adds no non-linearity", "pooling has learnable parameters", "BatchNorm behaves the same at
  inference"), state the correct answer and why in a `remark` starting with `\trap`.
- **Never invent** a citation, benchmark number, paper title/year, or the parameter count of a named
  production model. If you can't verify it, leave it out or say "typically". Architectures you define
  yourself (e.g. "a BERT-base-shaped config: L=12, d=768, h=12, V=30522") may be counted by code, but
  phrase it as counting that config, not quoting a vendor figure.
- Final read-through with `pdftotext`: grep for `STUB`, `TODO`, `??`, `undefined`; check `dl.log`
  for `Overfull` (the preamble sets `\hfuzz=2pt`, so any reported one must be fixed).

## 5. Sources
Write from knowledge in your own words, then verify with code. Name sources in a short "Sources"
paragraph in the preface: Goodfellow–Bengio–Courville *Deep Learning*; *Dive into Deep Learning*;
Prince *Understanding Deep Learning*; Vaswani et al. (Attention Is All You Need); He et al. (ResNet);
Kingma & Ba (Adam); Ioffe & Szegedy (BatchNorm); Hochreiter & Schmidhuber (LSTM); Devlin et al.
(BERT); Hu et al. (LoRA); Kingma & Welling (VAE). Put an entry in `latex/refs-dl.bib` only if you are
sure of every field (copy the format of `latex/refs-ml.bib`). Do not copy text verbatim from any
source; no paid APIs; no images of other people's pages.

## 6. Chapters (★ = appears in real OA papers → deepest numericals; 🔧 = usually skipped in short notes, include it)
1. **Perceptron and MLP ★** — biological analogy and where it breaks, perceptron algorithm and
   convergence theorem, XOR not linearly separable ★ and solved by a 2-layer net by hand, a neuron's
   output from inputs/weights/transfer function ★, universal approximation, capacity (layers yes;
   dropout and learning rate no) ★, parameter counting.
2. **Activations, losses, output layers ★** — sigmoid/tanh/ReLU/leaky/ELU/GELU/SiLU, stable softmax
   (log-sum-exp), what provides non-linearity ★, MSE/BCE/CCE/hinge/Huber/focal/label smoothing (which
   are losses vs activations ★), softmax confidence and cross-entropy from logits ★ (generate many
   examples).
3. **Backpropagation ★** — computation graphs, full matrix-form derivation for an L-layer MLP,
   softmax+CE gradient (ŷ−y), a complete numeric forward/backward pass on a 2-2-1 net, gradient
   checking, reverse-mode autograd from scratch (~60 lines), vanishing/exploding gradients (product of
   Jacobians), dying ReLU.
4. **Optimisers and schedules ★** — SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam/AdamW with one
   hand-computed step ★, LR schedules (cosine, warm-up, one-cycle), batch size and generalisation,
   gradient clipping, loss landscapes.
5. **Init, normalisation, regularisation ★** — Xavier/He (derive variance), BatchNorm (forward,
   backward, train vs inference), LayerNorm, GroupNorm, RMSNorm, inverted dropout, weight decay vs L2
   (differ under Adam), augmentation, early stopping, over/underfitting from curves ★.
6. **CNNs ★** — conv vs cross-correlation, output-shape formula with padding/stride/dilation ★,
   parameter and FLOP counts ★, receptive field, 1×1 conv, depthwise-separable, pooling ★, weight
   sharing and why a CNN has fewer parameters than a fully connected net ★, transposed conv 🔧; LeNet,
   AlexNet, VGG, Inception, ResNet (degradation problem, shortcuts, gradient flow ★), DenseNet,
   MobileNet, EfficientNet; transfer learning and fine-tuning ★; detection (IoU, NMS, anchors, R-CNN
   family, YOLO) and segmentation (FCN, U-Net) overview 🔧. Assert every shape and count against
   PyTorch.
7. **RNNs ★** — vanilla RNN, BPTT, why feedback connections ★, vanishing gradients in time, LSTM/GRU
   equations and parameter counts, bidirectional/stacked, seq2seq, teacher forcing, beam search, why
   LSTM suits time series and where Transformers beat it ★, CTC 🔧.
8. **Attention and the Transformer ★** — Bahdanau/Luong attention, scaled dot-product (derive why
   √d_k), multi-head with shapes at every step, positional encodings (sinusoidal, learned, RoPE,
   ALiBi), encoder/decoder, causal and padding masks, residual+LayerNorm (pre- vs post-LN), FFN,
   parameter counting for a whole Transformer, O(n²d) complexity, KV cache, FlashAttention concept 🔧.
   **Capstone:** write a full encoder–decoder Transformer from scratch in NumPy, then PyTorch, train it
   on a toy copy/reverse/addition task, and assert the two implementations and
   `torch.nn.MultiheadAttention` agree. Also why a Transformer rather than an RNN or CNN for seq2seq ★.
9. **Language models and LLMs 🔧** — tokenisation (implement BPE), embeddings, causal vs masked
   objectives, BERT/GPT/T5, scaling laws (as stated in the literature), fine-tuning ★ (full, feature
   extraction, adapters, LoRA with a parameter-count example, prompt/prefix tuning), instruction
   tuning, RLHF/DPO concept, decoding (greedy, beam, temperature, top-k/p), perplexity/BLEU/ROUGE and
   their limits, quantisation, distillation, MoE, RAG and tool-using agents at one-chapter depth,
   hallucination and safety basics.
10. **Generative and representation learning 🔧** — autoencoders, VAE (derive the ELBO and
    reparameterisation), GANs (minimax, mode collapse, WGAN), diffusion/DDPM at derivation-sketch
    depth, flows in a paragraph, SimCLR/CLIP, word2vec (skip-gram with negative sampling), Vision
    Transformers.
11. **Practice and PyTorch 🔧** — tensors/broadcasting, `nn.Module`, autograd, DataLoader, canonical
    training loop, mixed precision, DDP concept, checkpointing, the silent-bug catalogue (forgetting
    `model.eval()`, softmax before `CrossEntropyLoss`, wrong `dim`, label leakage, unshuffled data),
    "overfit one batch" debugging, reproducibility. GNNs, RL (policy gradient, DQN) and time-series
    deep models: one page each.
12. **Implement it** — NumPy from scratch, asserted against PyTorch: MLP with gradient-checked
    backprop, softmax regression, conv layer forward/backward, batch-norm, LSTM cell,
    scaled-dot-product and multi-head attention, Adam, a char-level language model, BPE.
13. **Rapid-fire** (80–100: why ReLU; why BatchNorm helps; BN vs LN; how ResNet fixes degradation;
    dropout at test time; why Adam can generalise worse than SGD; LSTM gates; why √d_k; why masks; what
    LoRA saves; what a KV cache stores…)
14. **Problems** (≥60): shape/parameter-count chains through a small CNN, softmax/CE from logits, one
    backprop step by hand, an Adam step, LSTM/GRU parameter counts, receptive-field arithmetic,
    attention on a 3-token example, Transformer parameter count, LoRA savings, multi-select "which are
    true", short derivations.

Front matter: a preface (`chapters/dl/00-preface.tex`, `\chapter*{Preface}` + `\addcontentsline`)
with how to use the book, what placements ask (the evidence above, briefly), the four-layer
structure, Sources paragraph, and a reproducibility note quoting library versions via `\gv`.

## 7. Working order and done criteria
1. Write a gap plan in `progress/dl.md` (what you add beyond the list above), then a compiling `\stub`
   skeleton of all chapters (commit), then chapters in order — each compiled, code-verified,
   committed — then rapid-fire, problems, read-through, fix list, final build.
2. If something can't be done (package fails, library lacks a feature), log it, work around it,
   continue. A smaller correct book beats a larger broken one.
3. Done means ALL of: builds cleanly from a fresh clone (`./build.sh dl`) with no LaTeX errors,
   undefined references, stubs or overfull boxes > 2pt; every ★ topic has code-produced worked
   numericals; rapid-fire ≥ 80 items; ≥ 60 problems with hints and solutions; all figures
   TikZ/pgfplots; all library cross-checks pass; PDF committed; `progress/dl.md` says DONE with page
   count and problem count.
4. Do not stop to ask for confirmation, and do not finish early. Performance tip: full builds of a
   180-page TikZ-heavy book are slow; while drafting a chapter you may build with
   `\includeonly{chapters/dl/NN-name}` temporarily (remove it before committing), but the committed PDF
   must be a full build.
