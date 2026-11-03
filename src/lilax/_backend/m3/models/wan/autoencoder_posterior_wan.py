"""Gaussian posterior distributions for the Wan autoencoder."""

import jax
from jax import tree_util
import jax.numpy as jnp


class DiagonalGaussianDistribution:
  def __init__(self, parameters, deterministic=False):
    self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
    self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = jnp.exp(0.5 * self.logvar)
    self.var = jnp.exp(self.logvar)
    if self.deterministic:
      self.var = self.std = jnp.zeros_like(self.mean)

  def sample(self, key):
    return self.mean + self.std * jax.random.normal(key, self.mean.shape)

  def kl(self, other=None):
    if self.deterministic:
      return jnp.array([0.0])
    if other is None:
      return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])
    return 0.5 * jnp.sum(
        jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
        axis=[1, 2, 3],
    )

  def nll(self, sample, axis=(1, 2, 3)):
    if self.deterministic:
      return jnp.array([0.0])
    logtwopi = jnp.log(2.0 * jnp.pi)
    return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

  def mode(self):
    return self.mean


class WanDiagonalGaussianDistribution(DiagonalGaussianDistribution):
  """Diagonal Gaussian registered as a JAX pytree."""


DiagonalGaussianDistribution.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"
WanDiagonalGaussianDistribution.__module__ = "lilax._backend.m3.models.wan.autoencoder_kl_wan"


def _wan_diag_gauss_dist_flatten(dist):
  return (dist.mean, dist.logvar, dist.std, dist.var), (dist.deterministic,)


def _wan_diag_gauss_dist_unflatten(aux, children):
  mean, logvar, std, var = children
  deterministic = aux[0]
  obj = WanDiagonalGaussianDistribution.__new__(WanDiagonalGaussianDistribution)
  obj.mean = mean
  obj.logvar = logvar
  obj.std = std
  obj.var = var
  obj.deterministic = deterministic
  return obj


tree_util.register_pytree_node(WanDiagonalGaussianDistribution, _wan_diag_gauss_dist_flatten,
                               _wan_diag_gauss_dist_unflatten)
