# Image representation and denoising

## Learning compact representations

I compared 10-dimensional MNIST representations from PCA, a fully connected autoencoder, and a convolutional autoencoder. A downstream linear classifier measured how much class information each representation retained. Reconstruction mean squared error (MSE) measured pixel-level fidelity.

| Method | Validation classification accuracy | Validation reconstruction MSE |
| --- | ---: | ---: |
| PCA | 81.38% | 0.0256 |
| Fully connected autoencoder | 91.83% | 0.0081 |
| Convolutional autoencoder | 94.95% | 0.0073 |

## Comparing denoising objectives

| Training objective | Test MSE |
| --- | ---: |
| MSE | 0.0047 |
| MAE | 0.0048 |
| Inverse PSNR | 0.0046 |

The three objectives gave similar test MSE in this run. The coursework report also compares the reconstructions visually, where differences in sharpness were easier to see.

Results are from the coursework report. [Course notebook](https://github.com/MatchLab-Imperial/deep-learning-course/blob/master/06_Autoencoders.ipynb)
