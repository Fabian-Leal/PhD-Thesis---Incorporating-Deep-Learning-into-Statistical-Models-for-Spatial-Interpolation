"""Metric calculation functions."""

import jax
import jax.numpy as jnp
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


@jax.jit
def accuracy(logits, targets):
    """Returns classification accuracy."""
    # Return accuracy = how many predictions match the ground truth
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)


@jax.jit
def binary_cross_entropy_with_logits(logits, targets):
    """Computes mean binary cross entropy with logits over the batch."""
    log_pi = jax.nn.log_sigmoid(logits)
    binary_xent = targets * log_pi + (1 - targets) * (log_pi - logits)
    binary_xent = jnp.sum(binary_xent) / targets.shape[0]

    return binary_xent


@jax.jit
def softmax_cross_entropy_with_logits(logits, targets, num_classes=None):
    """Computes mean softmax cross entropy with logits over the batch."""
    if num_classes is None:
        num_classes = FLAGS.num_classes

    # Generate one_hot labels from index classes
    labels = jax.nn.one_hot(targets, num_classes)

    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent


@jax.jit
def gaussian_kl(mu, logvar):
    """Computes mean KL between parameterized Gaussian and Normal distributions.

    Gaussian parameterized by mu and logvar. Mean over the batch.

    NOTE: See Appendix B from VAE paper (Kingma 2014):
          https://arxiv.org/abs/1312.6114
    """
    kl_divergence = jnp.sum(jnp.exp(logvar) + mu**2 - 1 - logvar) / 2
    kl_divergence /= mu.shape[0]

    return kl_divergence


@jax.jit
def l2_loss(params):
    """Computes the weight decay loss by penalising the norm of parameters."""
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def log_metrics(prefix, step, metrics):
    """Logs dictionary of metrics."""
    for name, value in metrics.items():
        logging.info('%6d | %-32s%9.3f', step, prefix + '/' + name, value)
