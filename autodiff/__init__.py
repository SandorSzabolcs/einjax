"""
Autodiff support for EinJAX sparse tensor operations.

Provides custom VJP rules for sparse einsum so that jax.grad and
jax.value_and_grad work through SparseTensorRelation contractions.
Dense operations use standard JAX autodiff via jnp.einsum, which
is already differentiable.
"""

from .custom_vjp import sparse_einsum, sparse_einsum_raw

__all__ = ["sparse_einsum", "sparse_einsum_raw"]
