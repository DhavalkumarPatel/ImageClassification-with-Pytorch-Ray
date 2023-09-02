# PyTorch-Ray-ImageClassification
This repository contains a python modules for image classification using PyTorch and Ray, a distributed computing framework. *Train, Tune, and Serve Image Classifiers with ease.*

## Introduction
The goal of this repository is to explore the model training, tuning, and serving using the Ray Framework. You might be wondering, why Ray? Over the past four years, my curiosity has been fueled by the desire to understand how these massive models are trained. Think about it&mdash;training a vision transformer on your laptop with millions of parameters or even conducting ablation studies seems like an insurmountable task.

That's where Ray steps in. This remarkable framework offers distributed computing capabilities that enable us to train colossal models swiftly. It eliminates the need for expertise in infrastructure management, taking care of the heavy lifting. Moreover, transitioning from local development to a cloud environment is a breeze with Ray; no drastic code changes required. For an in-depth understanding of the framework, I urge you to refer to their [website](https://www.ray.io/).  In this implementation, the ResNet50 model with pretrained weights is finetuned on a selected dataset. 

## Credits and References
- [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) by [Goku Mohandas](https://www.linkedin.com/in/goku/)
- [pytorch/examples](https://github.com/pytorch/examples)

