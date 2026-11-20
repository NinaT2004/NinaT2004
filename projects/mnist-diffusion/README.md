# MNIST diffusion

This coursework experiment uses a small U-Net-style model to predict the noise added to MNIST images. The forward process mixes each image with Gaussian noise at a randomly selected timestep. The training loop minimizes mean squared error between the predicted and applied noise.

I trained for one epoch using Adam at a learning rate of 0.001 and a batch size of 128. The report shows clean images alongside their noise and corrupted versions, and its appendix records the training loop and loss curve.

This experiment covers forward noising and noise-prediction training. The reported evidence is the loss curve and the noising examples.
