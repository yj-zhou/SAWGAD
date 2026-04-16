# SAWGAD

Official implementation of **Learning Feature Encoder with Synthetic Anomalies for Weakly Supervised Graph Anomaly Detection** (IEEE TKDE 2026).

## Requirements

- uv package manager
- CUDA 11.8 compatible GPU

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Install dependencies:

```bash
uv sync
```

## Usage

```bash
uv run sawgad
```

The default hyperparameters correspond to the configuration used in the paper for the Amazon dataset. The Amazon graph is downloaded automatically on first run via `dgl.data.FraudAmazonDataset`.

### Expected Output

```
AUROC: 0.9702 +/- 0.0040 | AUPRC: 0.9418 +/- 0.0096 over 10 runs
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@ARTICLE{Zhou26SAWGAD,
  author={Zhou, Yingjie and Xie, Yuqin and Liu, Fanxing and Song, Dongjin and Zhu, Ce and Liu, Lingqiao},
  journal={IEEE Transactions on Knowledge \& Data Engineering},
  title={{Learning Feature Encoder With Synthetic Anomalies for Weakly Supervised Graph Anomaly Detection}},
  year={2026},
  volume={38},
  number={04},
  ISSN={1558-2191},
  pages={2326-2339},
  doi={10.1109/TKDE.2026.3656821},
  url={https://doi.ieeecomputersociety.org/10.1109/TKDE.2026.3656821},
  publisher={IEEE Computer Society},
  month=apr
}
```
