"""HumbleSL - deep supervised learning library.

Straightforward supervised learning (SL) Python library.
It provides all the boilerplate code needed to do Deep SL.

It's backed by the JAX library and the Haiku framework.
It uses TensorFlow Datasets for data loading or preprocessing.
"""

from .data import load_dataset

from .metric import accuracy
from .metric import binary_cross_entropy_with_logits
from .metric import softmax_cross_entropy_with_logits
from .metric import gaussian_kl
from .metric import l2_loss
from .metric import log_metrics

from .network import mlp_fn
from .network import get_num_params

from .train import loop
