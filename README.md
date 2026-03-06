# EinJAX

Native JAX implementation of the upper-case-lower-case EinSum system from the VLDB 2026 paper *"Automated Tensor-Relational Decomposition for Large-Scale Sparse Tensor Computation"*.

EinJAX decomposes tensor operations into:
- **Upper-case indices** (relational) — tile coordinates that determine sharding/partitioning across devices
- **Lower-case indices** (kernel) — dense computation within each shard via `jnp.einsum`

This enables large-scale sparse tensor workloads (graph neural networks, sparse attention, quantum circuit simulation) to run efficiently on multi-GPU/TPU clusters by skipping zero blocks entirely at the tile level.

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install JAX with CUDA:

```bash
pip install jax[cuda12]
```

For TPU support:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

### Dense Einsum

```python
import einjax
import jax.numpy as jnp

A = jnp.ones((3, 4))
B = jnp.ones((4, 5))
result = einjax.einsum("ij,jk->ik", A, B)
print(result.shape)  # (3, 5)
```

Implicit output notation (NumPy convention) is also supported:

```python
result = einjax.einsum("ij,jk", A, B)  # equivalent to "ij,jk->ik"
```

### Analyze Without Executing

```python
import einjax
import numpy as np

info = einjax.analyze("ij,jk->ik", np.zeros((3, 4)), np.zeros((4, 5)))
print(info)
# AnalysisResult(
#   notation='ij,jk->ik',
#   input_shapes=[(3, 4), (4, 5)],
#   output_shape=(3, 5),
#   contracted_indices=[j],
#   free_indices=[i, k],
#   label_dimensions={'i': 3, 'j': 4, 'k': 5},
# )
```

### Sparse Tensor-Relational Execution

For sparse tensors, EinJAX decomposes the data into a tensor-relational format (tile coordinates + dense tile values) and executes only the non-zero blocks:

```python
import numpy as np
from einjax.tensor.sparse import SparseTensor, SparseTensorRelation
from einjax.execution.sparse_dispatch import execute_sparse

# Create a sparse adjacency matrix and tile it
adj = np.eye(8, dtype=np.float64)
adj_sparse = SparseTensor("A", adj)
adj_rel = adj_sparse.to_relation((2, 2))

# Create a dense feature matrix as a relation
features = np.random.randn(8, 4)
feat_sparse = SparseTensor("H", features)
feat_rel = feat_sparse.to_relation((2, 4))

# Sparse matmul: A @ H
result_rel = execute_sparse(
    lhs=adj_rel,
    rhs=feat_rel,
    join_keys=[(1, 0)],
    kernel_string="ij,jk->ik",
    agg_keys=[(0, 0), (1, 1)],
    output_shape=(8, 4),
    output_tile_shape=(2, 4),
)

result = result_rel.to_dense()
```

### Sparsity Analysis

Inspect tiling schemes and sparsity statistics for a tensor:

```python
from einjax.tensor.sparse import SparseTensor

A = SparseTensor("A", adj)
for tile_shape, scheme in sorted(A.schemes.items()):
    print(f"Tile {tile_shape}: {scheme.num_tuples} non-empty tiles, "
          f"cost={scheme.accumulated_cost:.2f}")
```

### Configuration and Hardware Profiles

```python
from einjax.config import get_config, set_config, list_device_types
from einjax import CostModelConfig

# Auto-detect hardware
config = get_config()

# Or choose a specific hardware profile
print(list_device_types())  # ['cpu', 'gpu:a100', 'gpu:h100', 'tpu:v4', 'tpu:v5e']
set_config(CostModelConfig.from_device_type("gpu:a100", num_devices=4))

# Calibrate with microbenchmarks
from einjax import calibrate
calibrated = calibrate()
```

### Contraction Path Optimization

For multi-operand einsum, EinJAX uses `opt_einsum` to find optimal binary contraction orderings and the DP optimizer to select tile shapes:

```python
from einjax.optimizer.contraction_path import plan_contraction
from einjax import CostModelConfig

config = CostModelConfig.from_device_type("cpu")
plan = plan_contraction("ij,jk,kl->il", [(10, 20), (20, 30), (30, 40)], config)
for step in plan.steps:
    print(f"Contract {step.input_subscripts} -> {step.output_subscript}")
```

## Testing

```bash
python -m pytest einjax/tests/ -v
```

## How It Works

EinJAX implements the upper-case-lower-case notation from the VLDB 2026 paper. Given a tensor with shape `(N, M)` and tile shape `(t_n, t_m)`:

- If `t_n < N`, dimension 0 is **upper-case** (sharded across devices)
- If `t_m == M`, dimension 1 is **lower-case** (replicated, computed densely within each shard)

The execution pipeline:

1. **Parse** — Normalize einsum notation and validate shapes
2. **Tile** — Generate candidate tiling schemes for each tensor
3. **Optimize** — DP optimizer selects tile shapes minimizing total cost (transfer + kernel + fixed)
4. **Shard** — Map tile shapes to `jax.sharding.PartitionSpec` for SPMD execution
5. **Execute** — For sparse tensors: coordinate join → kernel einsum → segment sum. For dense: `jnp.einsum`
6. **Differentiate** — Custom VJP rules propagate gradients through sparse structure

## License

See the repository root for license information.