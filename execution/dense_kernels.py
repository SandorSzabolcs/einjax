"""
Dense sub-computation via jnp.einsum, with optional multi-device sharding.

This module implements the dense execution path for EinJAX. It provides
both single-device execution (execute_dense_einsum) and sharded multi-device
execution (execute_sharded_einsum) that uses jax.jit with in/out sharding
constraints so XLA's GSPMD partitions the computation across devices.

Per PRD Section 4.5, execute_sharded_einsum wraps jnp.einsum in a jax.jit
call carrying NamedSharding constraints, letting GSPMD insert the necessary
collective operations automatically.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=64)
def _einsum_fn(einsum_string: str):
    """Return a cached function for a given einsum string."""
    def f(*ops):
        return jnp.einsum(einsum_string, *ops)
    return f


_jit_cache: dict[tuple, jax.stages.Compiled | None] = {}


def _get_sharded_jit(einsum_string, in_shardings, out_sharding):
    """Return a cached jax.jit wrapper for the given einsum + shardings."""
    key = (einsum_string, in_shardings, out_sharding)
    fn = _jit_cache.get(key)
    if fn is None:
        fn = jax.jit(
            _einsum_fn(einsum_string),
            in_shardings=in_shardings,
            out_shardings=out_sharding,
        )
        _jit_cache[key] = fn
    return fn


def execute_dense_einsum(
    einsum_string: str,
    *operands: jnp.ndarray,
) -> jnp.ndarray:
    """Execute a dense einsum operation on a single device.

    Wraps jnp.einsum with JAX arrays, providing the kernel-level
    computation that runs within each shard. In Phase 1, this is
    the only execution path (single-device, dense).

    Args:
        einsum_string: Einstein summation notation (e.g., "ij,jk->ik").
            Must contain explicit '->' output specification.
        *operands: JAX arrays as einsum operands.

    Returns:
        Result of the einsum contraction as a JAX array.
    """
    return jnp.einsum(einsum_string, *operands)


def execute_sharded_einsum(
    einsum_string: str,
    *operands: jnp.ndarray,
    mesh: jax.sharding.Mesh | None = None,
    in_specs: list[tuple] | None = None,
    out_spec: tuple | None = None,
) -> jnp.ndarray:
    """Execute a dense einsum with sharding constraints for multi-device execution.

    When a mesh is provided, wraps jnp.einsum in a jax.jit call with
    in_shardings and out_shardings so XLA's GSPMD partitions the computation
    across the mesh devices. When no mesh is provided, falls back to plain
    jnp.einsum.

    Args:
        einsum_string: Einstein summation notation (e.g., "ij,jk->ik").
        *operands: JAX arrays as einsum operands.
        mesh: JAX device mesh for sharded execution. If None, runs single-device.
        in_specs: List of PartitionSpec tuples, one per operand. Each tuple
            maps dimensions to mesh axes (str) or None (replicated).
        out_spec: PartitionSpec tuple for the output.

    Returns:
        Result of the einsum contraction as a JAX array.
    """
    if mesh is None:
        return jnp.einsum(einsum_string, *operands)

    in_shardings = tuple(
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*s))
        for s in in_specs
    )
    out_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*out_spec)
    )

    fn = _get_sharded_jit(einsum_string, in_shardings, out_sharding)
    return fn(*operands)
