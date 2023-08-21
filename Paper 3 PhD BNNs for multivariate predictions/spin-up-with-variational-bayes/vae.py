"""Simple Bernoulli VAE generative model on MNIST.

See: http://ruishu.io/2018/03/19/bernoulli-vae/.
"""

import functools

import haiku as hk
import jax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
from absl import app
from absl import flags
from absl import logging
from jax.experimental import optix

import humblesl as hsl

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('beta', 1.0, 'ELBO kl divergence weight.')
flags.DEFINE_integer('latent_size', 10, 'Latent space size.')
flags.DEFINE_integer('batch_size', 64, 'Training batch size.')
flags.DEFINE_integer('n_train_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Training logging interval.')
flags.DEFINE_string('ckpt_path', './out/vae_params.pkl', 'Checkpoint path.')
flags.DEFINE_integer('ckpt_interval', 1000, 'Params checkpoint interval.')


def plot_samples(images, grid_shape):
    plt.close()

    plt.figure(figsize=grid_shape)
    gs = gridspec.GridSpec(*grid_shape)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image, cmap='Greys_r')

    plt.draw()
    plt.pause(0.001)


def main(argv):
    del argv

    # Make datasets for train and test.
    train_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'train', is_training=True, batch_size=FLAGS.batch_size)
    test_eval_dataset = hsl.load_dataset(
        'mnist:3.*.*', 'test', is_training=False, batch_size=10000)
    batch_image, _ = next(train_dataset)
    n, h, w, c = batch_image.shape

    encoder = hk.transform(functools.partial(
        hsl.mlp_fn,
        hidden_sizes=[512, 512],
        output_sizes=[FLAGS.latent_size] * 2,
        name='encoder'
    ))
    decoder = hk.transform(functools.partial(
        hsl.mlp_fn,
        hidden_sizes=[512, 512],
        output_sizes=[h * w],
        name='decoder'
    ))

    # Initialize model
    rng = hk.PRNGSequence(42)
    encoder_params = encoder.init(next(rng), batch_image)
    decoder_params = decoder.init(next(rng), jnp.zeros([n, FLAGS.latent_size]))
    params = (encoder_params, decoder_params)
    logging.info('Total number of encoder parameters: %d',
                 hsl.get_num_params(encoder_params))
    logging.info('Total number of decoder parameters: %d',
                 hsl.get_num_params(decoder_params))
    logging.info('Total number of parameters: %d',
                 hsl.get_num_params(params))

    # Define and initialize optimizer.
    opt = optix.adam(FLAGS.lr)
    opt_state = opt.init(params)

    def sample_latent(mu, logvar, rng):
        """Sample latent variable from Gaussian.

        NOTE: It uses reparameterization trick.
        """
        eps = jax.random.normal(rng, shape=mu.shape)
        return eps * jnp.exp(logvar / 2) + mu

    def elbo(params, batch, rng):
        """Computes the Evidence Lower Bound."""
        batch_image, _ = batch
        encoder_params, decoder_params = params

        mu, logvar = encoder.apply(encoder_params, batch_image)
        z = sample_latent(mu, logvar, rng)
        x = decoder.apply(decoder_params, z)

        binary_xent = hsl.binary_cross_entropy_with_logits(
            x, jnp.reshape(batch_image, (batch_image.shape[0], -1)))
        kl_divergence = hsl.gaussian_kl(mu, logvar)
        elbo_ = binary_xent - FLAGS.beta * kl_divergence
        return elbo_, binary_xent, kl_divergence

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
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state

    def calculate_metrics(params, batch):
        """Calculates accuracy."""
        _, decoder_params = params

        latent = jax.random.normal(next(rng), shape=(16, FLAGS.latent_size))
        logits = decoder.apply(decoder_params, latent)
        images = jax.nn.sigmoid(logits)
        plot_samples(images.reshape(16, h, w), grid_shape=(4, 4))

        elbo_, binary_xent, kl_divergence = elbo(params, batch, next(rng))
        mean_approx_evidence = jnp.exp(elbo_ / (h * w))
        return {
            'elbo': elbo_,
            'binary_xent': binary_xent,
            'kl_divergence': kl_divergence,
            'mean_approx_evidence': mean_approx_evidence,
        }

    # Train!
    hsl.loop(params=params,
             opt_state=opt_state,
             train_dataset=train_dataset,
             sgd_update=sgd_update,
             prng_sequence=rng,
             n_steps=FLAGS.n_train_steps,
             log_interval=FLAGS.log_interval,
             test_eval_dataset=test_eval_dataset,
             calculate_metrics=calculate_metrics,
             checkpoint_path=FLAGS.ckpt_path,
             checkpoint_interval=FLAGS.ckpt_interval)


if __name__ == '__main__':
    app.run(main)
