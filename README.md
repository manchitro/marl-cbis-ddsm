# Cooperative Multi-Agent Reinforcement Learning for Mammogram ROI Classification

Multiple reinforcement-learning agents independently scan small patches of a mammogram ROI, communicate with each other, and reach a consensus classification (benign or malignant) — without any single agent ever seeing the whole image. Published at IEEE ICDABI 2023.

📄 [Paper (DOI: 10.1109/ICDABI60145.2023.10629500)](https://doi.org/10.1109/ICDABI60145.2023.10629500) · 🎓 [Google Scholar](https://scholar.google.com/citations?user=M2eF33AAAAAJ&hl=en)

> This repo is a fork of [`Ipsedo/MARLClassification`](https://github.com/Ipsedo/MARLClassification), which implements the general multi-agent RL image classification framework from Mousavi et al. (IROS 2019). This project adapts it specifically to mammogram ROI classification on CBIS-DDSM.

## Results

| Metric | Value |
|---|---|
| Accuracy | 82.45% |
| Precision | 81.73% |
| Recall | 81.00% |
| F1 (train, top-1) | 0.916 |
| F1 (eval) | 0.81 |

100 epochs, evaluated on CBIS-DDSM mass ROI images (benign vs. malignant).

**Comparison to other published methods on the same dataset:**

| Method | Year | Accuracy |
|---|---|---|
| Jabeen et al. | 2023 | 95.40% |
| Baccouche et al. | 2022 | 95.13% |
| Muduli et al. | 2022 | 90.68% |
| Ragab et al. | 2019 | 87.20% |
| **This method** | — | **82.45%** |
| Khan et al. | 2019 | 77.66% |

This doesn't beat the CNN-based state of the art on raw accuracy — it lands in the middle of the pack. The paper treats the contribution as architectural/methodological (a decentralized, partial-observation approach that scales differently than whole-image CNN classifiers) rather than a leaderboard win.

## Problem

Mammography is the standard breast cancer screening tool, but reading it is resource-intensive and requires expert radiologists. Existing CAD (computer-aided detection) approaches are mostly CNN-based and process whole images, with parameter counts that grow with image resolution. This project asks: what if classification agents only ever see small local patches, communicate what they find, and reach a decentralized consensus — does that reduce computational load while staying accurate enough to be useful?

## How It Works

The image classification task is formulated as a partially observable Markov Decision Process (POMDP): `⟨I, N, S, A, P, π, O, γ⟩` — image, number of agents, states (agent positions), actions (`{up, down, left, right}`), position transitions, action policy, local observation function, and discount factor.

Each of the *N* agents runs through five modules per timestep:

1. **Feature Extraction** — a 4-layer CNN turns the agent's local w×w window (w=24px) into a 128-dim feature vector.
2. **Position Encoding** — the agent's (x, y) position is encoded via a fully-connected layer + GELU + batch norm.
3. **Decision** — an LSTM aggregates the agent's observation history, feature vector, position encoding, and incoming messages from other agents, then a policy network outputs a probability distribution over the four move actions.
4. **Prediction** — a second LSTM produces a running benign/malignant belief from the same inputs.
5. **Communication** — each agent broadcasts a message derived from its prediction state; every other agent decodes and averages incoming messages into its next decision.

After a fixed number of steps (32, by default), each agent emits a prediction; the final classification is the argmax of the softmax-averaged predictions across all agents. Training uses REINFORCE (policy gradient) with an Adam optimizer, where the reward is the cross-entropy loss between the predicted and true label.

## Dataset

- **Source:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM), **mass images only** (calcification images excluded — noted as future work in the paper).
- **Split:** 1318 train (681 benign / 637 malignant), 378 test (231 benign / 147 malignant). Total 1696 before augmentation.
- **Augmentation:** horizontal flip, vertical flip, 90°/180°/270° rotation, and combinations thereof — 12 augmented copies per image, expanding training data to 15,816 images.
- **ROI images** resized to 224×224, retaining original aspect information.

## Limitations (from the paper)

As stated by the authors:
- Evaluated on a single dataset (CBIS-DDSM) — generalization to other mammogram datasets untested.
- Trained only on mass abnormalities, not calcifications, which have different visual characteristics.
- Accuracy trails several CNN-based competitors (see table above) — the authors describe the model as "far from being useful in the field" in its current form.
- Requires pre-annotated ROIs as input; doesn't perform ROI localization/segmentation itself.

## Training Setup

| | |
|---|---|
| Framework | PyTorch 2.1, CUDA 12.1, Python 3.10 |
| Hardware | Intel Core i7-10700 (2.90GHz), 16GB RAM, NVIDIA RTX 3070 Ti (8GB) |
| Optimizer | Adam |
| Agents | 16 (default) |
| Agent window size | 24×24 px |
| Episode length | 32 steps |
| LSTM hidden size | 256 |
| Message size | 64 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Discount factor (γ) | 0.99 |
| Epochs | 100 |

## How to Run

Place the CBIS-DDSM dataset (mass images) under `./resources/downloaded/cbis`, then:

```bash
python -m marl_classification \
  -a 16 --step 32 --cuda --run-id train_cbis \
  train \
  --action [[5,0],[-5,0],[0,5],[0,-5]] \
  --img-size 224 --nb-class 2 -d 2 --f 24 --ft-extr cbis \
  --nb 256 --na 256 --nm 64 --nd 32 --nlb 256 --nla 256 \
  --batch-size 32 --lr 1e-4 --nb-epoch 100 \
  --eps 1.0 --eps-dec 0.99995 \
  -o ./out/cbis
```

Flag reference: `-a` agent count · `--step` steps per episode · `--action` the four move directions, each a 5px shift · `--img-size` input ROI size · `--nb-class` 2 (benign/malignant) · `--f` agent observation window size (24px) · `--nm` inter-agent message size · `--nb`/`--na`/`--nlb`/`--nla` LSTM/network hidden sizes (256, matching the paper) · `--eps`/`--eps-dec` exploration rate and decay.

## Citation

```bibtex
@inproceedings{uddin2023marlmammogram,
  title     = {Using Cooperative Multi-Agent Reinforcement Learning for Mammogram ROI Classification},
  author    = {Uddin, Md Sazid and Mridha, M. F. and Abdullah-Al-Jubair, M. and others},
  booktitle = {2023 4th International Conference on Data Analytics for Business and Industry (ICDABI)},
  year      = {2023},
  publisher = {IEEE},
  doi       = {10.1109/ICDABI60145.2023.10629500}
}
```

## License

**GPL-3.0**, inherited from upstream [`Ipsedo/MARLClassification`](https://github.com/Ipsedo/MARLClassification). This repo builds substantially on that GPL-3.0-licensed codebase, so GPL-3.0 applies to the combined work — a copyleft requirement, not a choice made for this project specifically.
