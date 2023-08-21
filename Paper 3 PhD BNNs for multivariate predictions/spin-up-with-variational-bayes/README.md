# Spin up with Variational Bayes

## What you get:

1. mlp.py - MLP classifier on MNIST (in JAX and Haiku).
2. vae.py - Bernoulli VAE generative model on MNIST.  
   See: [USING A BERNOULLI VAE ON REAL-VALUED OBSERVATIONS](http://ruishu.io/2018/03/19/bernoulli-vae/).
3. bayes.py - Variational Bayes NN classifier on MNIST.
   * [Bayesian Neural Networks with TensorFlow Probability](https://towardsdatascience.com/bayesian-neural-networks-with-tensorflow-probability-fbce27d6ef6).
   * [CSC421/2516 Lecture 19: Bayesian Neural Nets](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec19.pdf).
   * [Making Your Neural Network Say “I Don’t Know” — Bayesian NNs using Pyro and PyTorch](https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd).

### How to run:

Install dependencies with `pip install -r requirements.txt`. It was tested in Python 3.7. As always, use of a virtual environment is recommended.

Each file is an independent implementation that uses HumbleSL library (see below), run with: `python <file>`. Run `python <file> --help` to see all the configurable parameters.

## Note on HumbleSL - deep supervised learning library.

It's a straightforward supervised learning (SL) Python library. It provides all the boilerplate code needed to do Deep SL: a network definition factory, metrics and losses, a data loader, train loop, etc.

It's backed by the JAX library and the Haiku framework. It uses TensorFlow Datasets for data loading and preprocessing.

