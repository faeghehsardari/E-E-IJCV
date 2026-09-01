# An Effective-Efficient Approach for Dense Multi-Label Action Detection

[Faegheh Sardari](https://scholar.google.com/citations?user=_8dV3CgAAAAJ&hl=en),
[Armin Mustafa](https://scholar.google.com/citations?user=0xOHqkMAAAAJ&hl=en),
[Philip J. B. Jackson](https://scholar.google.com/citations?user=vgue80YAAAAJ&hl=en),
[Adrian Hilton](https://scholar.google.com/citations?user=vTIYTNQAAAAJ&hl=en)

Official PyTorch implementation of **An Effective-Efficient Approach for Dense Multi-Label Action Detection** published at the IJCV Journal.

[[Paper](https://arxiv.org/abs/2406.06187)] [[PDF](https://arxiv.org/pdf/2406.06187)]


## Prerequisites

- Linux
- Python 3
- PyTorch
- NumPy
- tqdm
- NVIDIA GPU with CUDA support is recommended

## Data Preparation

Following previous works (e.g., MS-TCT), PAT is built on top of pre-trained I3D features. Therefore, you need to extract I3D features for each dataset before both training and inference. To perform this:

1. Download the Charades (24 fps version) and MultiTHUMOS datasets from [Charades](https://prior.allenai.org/projects/charades) and [MultiTHUMOS](https://ai.stanford.edu/~syyeung/everymoment.html), respectively.
2. Follow the [pytorch-i3d repository](https://github.com/piergiaj/pytorch-i3d) to extract their features.

## Train and Validate on Charades

```bash
python train_eepat.py \
  --annotation-file /path/to/annotations.json \
  --rgb-root /path/to/rgb/features \
  --output-dir outputs/ \
  --gpu 0 \
  --batch-size 3 \
  --epochs 25 \
  --learning-rate 1e-4 \
  --lr-milestones 7 14 \
  --lr-gamma 0.1
```


Run `python train_eepat.py --help` to list all options.


## Outputs

- `best_checkpoint.pt`: best model and training state;
- `<epoch>.pkl`: frame-level probabilities grouped by video.


## Citation

If this work is useful for your research, please cite:

```bibtex
@article{sardari2024effective,
  title={An Effective-Efficient Approach for Dense Multi-Label Action Detection},
  author={Sardari, Faegheh and Mustafa, Armin and Jackson, Philip J. B. and Hilton, Adrian},
  journal={International Journal of Computer Vision (IJCV)},
  year={2026}
}
```

Please also consider citing the earlier PAT paper:

```bibtex
@inproceedings{sardari2023pat,
  title={PAT: Position-Aware Transformer for Dense Multi-Label Action Detection},
  author={Sardari, Faegheh and Mustafa, Armin and Jackson, Philip J. B. and Hilton, Adrian},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={2988--2997},
  year={2023}
}
```

## Acknowledgments

This implementation builds on ideas and utilities from [MS-TCT](https://github.com/dairui01/MS-TCT) and uses I3D features extracted with [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d). We thank their authors.

This research was supported by UKRI EPSRC Platform Grant EP/P022529/1 and the EPSRC BBC Prosperity Partnership AI4ME: Future Personalised Object-Based Media Experiences Delivered at Scale Anywhere, EP/V038087/1.
