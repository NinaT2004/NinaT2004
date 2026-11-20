# Ruizhu Tian

I'm interested in building AI systems for robotics and computer vision. My current work explores wake-word detection for a robot, and my coursework compares how different training strategies affect image classification.

## Selected projects

### Tiny-ImageNet classification

I compared training from scratch, transfer learning, and fine-tuning on a 200-class image classification task. Fine-tuning VGG16 raised held-out accuracy from **27.1% to 54.5%**. ResNet18 reached **50.3%** with shorter training time per epoch.

[Read the experiment and results](projects/tiny-imagenet/README.md)

### Robot wake-word detection · In progress

I'm working on wake-word detection as a way to initiate voice interaction with a robot.

### Image representation and denoising

I compared PCA with two autoencoders on MNIST and evaluated three training losses for image denoising. The convolutional autoencoder's 10-dimensional representation reached **94.95%** validation accuracy with a linear classifier.

[Read the autoencoder experiments](projects/autoencoders/README.md)

### YOLO image classification

I trained a YOLO image classifier that reached **96.97%** accuracy on the coursework test set.

[Read the training setup](projects/yolo-classification/README.md)

## Additional coursework

- [MNIST diffusion](projects/mnist-diffusion/README.md) — forward noising and a one-epoch noise-prediction experiment.
- [Deep learning experiments](projects/deep-learning-experiments/README.md) — sequence models, image generation, and reinforcement learning.
- [Machine learning fundamentals](projects/ml-fundamentals/README.md) — implementation notebooks and theory exercises.
