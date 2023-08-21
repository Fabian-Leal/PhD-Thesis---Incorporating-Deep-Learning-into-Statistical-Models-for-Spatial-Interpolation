"""Simple Variational Bayes NN classifier on MNIST."""

import haiku as hk
import jax
import jax.numpy as jnp
from absl import app
from absl import flags
from absl import logging
from jax.experimental import optix

import humblesl as hsl

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_integer('batch_size', 128, 'Training batch size.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('n_train_steps', 100000, 'Number of training steps.')
flags.DEFINE_float('beta', 0.001, 'ELBO kl divergence weight.')
flags.DEFINE_integer('num_samples', 10, 'Nb. of params samples in inference.')
flags.DEFINE_integer('log_interval', 1000, 'Training logging interval.')
flags.DEFINE_string('ckpt_path', './out/bayes_params.pkl', 'Checkpoint path.')
flags.DEFINE_integer('ckpt_interval', 1000, 'Params checkpoint interval.')


@jax.jit
def sample_params(prior, rng):
    def sample_gaussian(mu, logvar):
        """Sample from a Gaussian distribution.

        NOTE: It uses reparameterization trick.
        """
        eps = jax.random.normal(rng, shape=mu.shape)
        return eps * jnp.exp(logvar / 2) + mu

    sample = jax.tree_multimap(sample_gaussian, prior['mu'], prior['logvar'])
    return sample


def predict(net, prior, batch_image, rng, num_samples):
    probs = []
    for i in range(num_samples):
        params_rng, rng = jax.random.split(rng)
        params = sample_params(prior, params_rng)
        logits = net.apply(params, batch_image)
        probs.append(jax.nn.softmax(logits))
    stack_probs = jnp.stack(probs)
    return jnp.mean(stack_probs, axis=0), jnp.std(stack_probs, axis=0)


def main(argv):
    del argv

    # Make datasets for train and test.
    train_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'train', is_training=True, batch_size=FLAGS.batch_size)
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
        apply_rng=True  # In the process of being removed. Can only be `True`.
    ))

    # Initialize model
    rng = hk.PRNGSequence(42)
    params = net.init(next(rng), batch_image)
    prior = dict(
        # Haiku inits weights to trun. normal, with stddev ``1 / sqrt(fan_in)``.
        # Where ``fan_in`` is the number of incoming connection to the layer.
        mu=params,
        # Init to ~0.001 variance around default Haiku initialization.
        logvar=jax.tree_map(lambda x: -7 * jnp.ones_like(x), params),
    )
    logging.info('Total number of parameters: %d', hsl.get_num_params(prior))

    # Define and initialize optimizer.
    opt = optix.adam(FLAGS.lr)
    opt_state = opt.init(prior)

    def elbo(aprx_posterior, batch, rng):
        """Computes the Evidence Lower Bound."""
        batch_image, batch_label = batch
        # Sample net parameters from the approximate posterior.
        params = sample_params(aprx_posterior, rng)
        # Get network predictions.
        logits = net.apply(params, batch_image)
        # Compute log likelihood of batch.
        log_likelihood = -hsl.softmax_cross_entropy_with_logits(
            logits, batch_label)
        # Compute the kl penalty on the approximate posterior.
        kl_divergence = jax.tree_util.tree_reduce(
            lambda a, b: a + b,
            jax.tree_multimap(hsl.gaussian_kl,
                              aprx_posterior['mu'],
                              aprx_posterior['logvar']),
        )
        elbo_ = log_likelihood - FLAGS.beta * kl_divergence
        return elbo_, log_likelihood, kl_divergence

    def loss(params, batch, rng):
        """Computes the Evidence Lower Bound loss."""
        return -elbo(params, batch, rng)[0]

    @jax.jit
    def sgd_update(params, opt_state, batch, rng):
        """Learning rule (stochastic gradient descent)."""
        # Use jax transformation `grad` to compute gradients;
        # it expects the prameters of the model and the input batch
        grads = jax.grad(loss)(params, batch, rng)
        # Compute parameters updates based on gradients and optimiser state
        updates, opt_state = opt.update(grads, opt_state)
        # Apply updates to parameters
        posterior = optix.apply_updates(params, updates)
        return posterior, opt_state

    def calculate_metrics(params, data):
        """Calculates metrics."""
        images, labels = data
        probs = predict(net, params, images, next(rng), FLAGS.num_samples)[0]
        elbo_, log_likelihood, kl_divergence = elbo(params, data, next(rng))
        mean_aprx_evidence = jnp.exp(elbo_ / FLAGS.num_classes)
        return {
            'accuracy': hsl.accuracy(probs, labels),
            'elbo': elbo_,
            'log_likelihood': log_likelihood,
            'kl_divergence': kl_divergence,
            'mean_approximate_evidence': mean_aprx_evidence,
        }

    # Train!
    hsl.loop(params=prior,
             opt_state=opt_state,
             train_dataset=train_dataset,
             sgd_update=sgd_update,
             prng_sequence=rng,
             n_steps=FLAGS.n_train_steps,
             log_interval=FLAGS.log_interval,
             train_eval_dataset=train_eval_dataset,
             test_eval_dataset=test_eval_dataset,
             calculate_metrics=calculate_metrics,
             checkpoint_path=FLAGS.ckpt_path,
             checkpoint_interval=FLAGS.ckpt_interval)


if __name__ == '__main__':
    app.run(main)
