# YOLO image classification

I trained a YOLO classifier for a coursework image classification task and reported **96.97% test accuracy**.

The reported run used 320-pixel inputs, AdamW with an initial learning rate of 0.001, cosine learning-rate scheduling, and RandAugment, random erasing, MixUp, and CutMix. The training configuration set a maximum of 60 epochs and an early-stopping patience of 15.

The coursework report records the training settings and example predictions. This page documents the experiment.
