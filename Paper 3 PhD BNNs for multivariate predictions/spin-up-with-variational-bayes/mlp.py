"""Simple MLP classifier on MNIST."""

import haiku as hk
import jax
from jax.experimental import optix
from absl import app
from absl import flags
from absl import logging

import humblesl as hsl

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay to apply to parameters.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('n_train_steps', 5000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Training logging interval.')


def main(argv):
    del argv

    # Make datasets for train and test.
    train_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'train', is_training=True, batch_size=1000)
    train_eval_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'train', is_training=False, batch_size=10000)
    test_eval_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'test', is_training=False, batch_size=10000)

    # Draw a data batch and log shapes.
    batch_image, batch_label = next(train_dataset)
    logging.info("Image batch shape: %s", batch_image.shape)
    logging.info("Label batch shape: %s", batch_label.shape)

    # Since we don't store additional state statistics, e.g. needed in
    # batch norm, we use `hk.transform`. When we use batch_norm, we will use
    # `hk.transform_with_state`.
    net = hk.without_apply_rng(hk.transform(
        hsl.mlp_fn,
        apply_rng=True
    ))

    # Initialize model
    params = net.init(jax.random.PRNGKey(42), batch_image)
    logging.info('Total number of parameters: %d', hsl.get_num_params(params))

    # Define and initialize optimizer.
    opt = optix.adam(FLAGS.lr)
    opt_state = opt.init(params)

    def loss(params, batch):
        """Compute the loss of the network, including L2 for regularization."""
        batch_image, batch_label = batch
        # Get network predictions
        logits = net.apply(params, batch_image)
        # Compute mean softmax cross entropy over the batch
        softmax_xent = hsl.softmax_cross_entropy_with_logits(logits,
                                                             batch_label)
        # Compute the weight decay loss by penalising the norm of parameters
        l2_loss = hsl.l2_loss(params)
        return softmax_xent + FLAGS.weight_decay * l2_loss

    @jax.jit
    def sgd_update(params, opt_state, batch):
        """Learning rule (stochastic gradient descent)."""
        # Use jax transformation `grad` to compute gradients;
        # it expects the prameters of the model and the input batch
        grads = jax.grad(loss)(params, batch)
        # Compute parameters updates based on gradients and optimiser state
        updates, opt_state = opt.update(grads, opt_state)
        # Apply updates to parameters
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state

    def calculate_metrics(params, batch):
        """Calculates accuracy."""
        batch_image, batch_label = batch
        logits = net.apply(params, batch_image)
        return {
            'accuracy': hsl.accuracy(logits, batch_label),
            'loss': loss(params, batch),
        }

    # Train!
    hsl.loop(params=params,
             opt_state=opt_state,
             train_dataset=train_dataset,
             sgd_update=sgd_update,
             n_steps=FLAGS.n_train_steps,
             log_interval=FLAGS.log_interval,
             train_eval_dataset=train_eval_dataset,
             test_eval_dataset=test_eval_dataset,
             calculate_metrics=calculate_metrics)


if __name__ == '__main__':
    app.run(main)
