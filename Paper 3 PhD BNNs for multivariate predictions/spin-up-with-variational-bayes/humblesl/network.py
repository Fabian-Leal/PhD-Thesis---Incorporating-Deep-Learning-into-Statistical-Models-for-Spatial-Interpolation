"""Network building functions and other utilities."""

import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

FLAGS = flags.FLAGS


def mlp_fn(batch,
           hidden_sizes=[64, 64],
           hidden_activation=jax.nn.relu,
           output_sizes=None,
           output_activation=None,
           name=None):
    """Standard MLP network."""

    if output_sizes is None:
        output_sizes = [FLAGS.num_classes]

    class MLP(hk.Module):
        def __call__(self, x):
            # Build body.
            h = hk.Flatten()(x)
            for size in hidden_sizes:
                z = hk.Linear(size)(h)
                h = hidden_activation(z)

            # Build head(s).
            heads = []
            for size in output_sizes:
                head = hk.Linear(size)(h)
                if output_activation is not None:
                    head = output_activation(head)
                heads.append(head)

            if len(heads) == 1:
                return heads[0]
            else:
                return heads

    mlp = MLP(name=name)
    return mlp(batch)


def get_num_params(params):
    """Returns number of parameters is the network."""
    num_params = 0
    for p in jax.tree_leaves(params):
        num_params = num_params + jnp.prod(p.shape)
    return num_params
