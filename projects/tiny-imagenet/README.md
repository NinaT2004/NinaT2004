# Tiny-ImageNet classification

How much does ImageNet pretraining help on a 200-class image classification task, and what is the training-time trade-off?

## Experiment

The coursework compares four PyTorch configurations on Tiny-ImageNet. The data loader uses 450 images per class for training and 50 per class for model selection. The official labeled validation split serves as the held-out evaluation set for this experiment.

## Results

Figures are from the coursework report. Training times are averages per epoch measured on an NVIDIA A100 for these runs.

| Configuration | Held-out accuracy | Training time per epoch |
| --- | ---: | ---: |
| VGG16 from scratch | 27.1% | 26.54 s |
| VGG16 transfer learning | 42.7% | 15.56 s |
| VGG16 fine-tuning | 54.5% | 26.49 s |
| ResNet18 | 50.3% | 10.11 s |

Fine-tuning improved on the VGG16 scratch baseline by **27.4 percentage points**. ResNet18 was **4.2 percentage points** behind fine-tuned VGG16 and took roughly **38% of its training time per epoch** in these runs.

## Source and reproducibility

This write-up summarizes an experiment from the Common CNN Architectures coursework. The supplied notebook contains the training loop and data split. A reproducible run still needs the exact per-model setup and a fresh end-to-end verification.

[Course notebook](https://github.com/MatchLab-Imperial/deep-learning-course/blob/master/04_Common_CNN_architectures.ipynb)
