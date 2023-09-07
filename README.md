# PyTorch-Ray-ImageClassification
This MLOps repository contains python modules for image classification using **[PyTorch](https://pytorch.org/)** and **[Ray](https://www.ray.io/)**, a distributed computing framework. *Train, Tune, and Serve Image Classifiers with ease.*

## Introduction
The goal of this repository is to explore the model training, tuning, and serving using the Ray Framework. You might be wondering, why Ray? Over the past four years, my curiosity has been fueled by the desire to understand how these massive models are trained. Think about it&mdash;training a vision transformer on your laptop with millions of parameters or even conducting ablation studies seems like an insurmountable task.

That's where Ray steps in. This remarkable framework offers distributed computing capabilities that enable us to train colossal models swiftly. It eliminates the need for expertise in infrastructure management, taking care of the heavy lifting. Moreover, transitioning from local development to a cloud environment is a breeze with Ray; no drastic code changes required. For an in-depth understanding of the framework, I urge you to refer to their documentation.  In this implementation, an end-to-end machine learning pipeline is implemented for an image classification task using the ResNet50 model with pretrained weights. Data preparation, Model Training and Tuning are done with the help of Ray. In addition, **[mlflow](https://mlflow.org/)** is used for experiment tracking and **[Gradio](https://www.gradio.app/)** for model serving.

## Getting Started
Follow the steps outlined in the notebook on understanding how to run the required modules. 

Demo Screenshot:
<img src="https://github.com/DhavalkumarPatel/ImageClassification-with-Pytorch-Ray/blob/main/notebooks/demo.png" style="width:500px;height:250;">


## Credits and References
- [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) by [Goku Mohandas](https://www.linkedin.com/in/goku/)
- [pytorch/examples](https://github.com/pytorch/examples)

