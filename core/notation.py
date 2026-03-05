"""
EinSum notation parsing and validation.

Ported from einsql/einsql.py (lines 36-78, 1262-1354). These functions
are pure Python with no backend dependencies.
"""

from __future__ import annotations

from collections import Counter


def index_to_subscript(i: int) -> str:
    """Convert integer index to subscript character (i, j, k, ...).

    Args:
        i: Zero-based index (0 -> 'i', 1 -> 'j', etc.)

    Raises:
        ValueError: If index exceeds maximum subscript range (i-z).
    """
    if ord("i") + i > ord("z"):
        raise ValueError(f"Index {i} exceeds maximum subscript range (i-z)")
    return chr(ord("i") + i)


def find_all_factors(n: int) -> list[int]:
    """Find all factors of a positive integer, sorted ascending.

    Args:
        n: Positive integer to factor.

    Raises:
        ValueError: If n is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)


def get_label_dimensions(einsum_string: str, shapes: list[tuple]) -> dict[str, int]:
    """Extract dimension sizes for each label in an einsum expression.

    Args:
        einsum_string: Einstein summation notation (e.g., "ij,jk->ik").
        shapes: List of tensor shapes.

    Returns:
        Dictionary mapping labels to their dimensions.

    Raises:
        ValueError: If same label has inconsistent dimensions.
    """
    label_dims: dict[str, int] = {}
    input_labels = einsum_string.split("->")[0].split(",")

    for i, subscript_labels in enumerate(input_labels):
        for label, dimension in zip(list(subscript_labels), shapes[i]):
            if label not in label_dims:
                label_dims[label] = dimension
            elif label_dims[label] != dimension:
                raise ValueError(
                    f"Dimension mismatch for label '{label}': "
                    f"expected {label_dims[label]}, got {dimension}"
                )
    return label_dims


def normalize_notation(einsum_string: str) -> str:
    """Normalize einsum notation to always include explicit output indices.

    When the einsum string has no '->', follows NumPy convention:
    output indices = all indices appearing exactly once across all inputs,
    sorted alphabetically.

    Examples:
        'ij,jk' -> 'ij,jk->ik'   (j appears twice, contracted)
        'ii'    -> 'ii->'          (i appears twice, full contraction to scalar)
        'ij'    -> 'ij->ij'        (both appear once, identity)

    Args:
        einsum_string: Einsum notation, with or without '->'.

    Returns:
        Normalized einsum string with explicit '->' output.
    """
    if "->" not in einsum_string:
        input_labels = einsum_string.split(",")
        all_indices = "".join(input_labels)
        counts = Counter(all_indices)
        output_indices = "".join(sorted(c for c in counts if counts[c] == 1))
        return einsum_string + "->" + output_indices
    return einsum_string


def validate_inputs(
    einsum_string: str,
    shapes: list[tuple[int, ...]],
    names: list[str] | None = None,
) -> None:
    """Validate einsum notation and tensor arguments.

    Checks:
        1. Number of input label groups matches tensor count.
        2. Each label group length matches corresponding tensor's ndim.
        3. Same label has consistent dimension size across tensors.
        4. Output labels are a subset of input labels.
        5. Label characters are lowercase alphabetic.

    Args:
        einsum_string: Normalized einsum notation (must contain '->').
        shapes: List of tensor shapes.
        names: Optional list of tensor names for error messages.

    Raises:
        ValueError: With descriptive message identifying the problem.
    """
    if names is None:
        names = [f"tensor_{i}" for i in range(len(shapes))]

    parts = einsum_string.split("->")
    input_part = parts[0]
    output_part = parts[1] if len(parts) > 1 else ""

    input_groups = input_part.split(",")

    # Check 1: tensor count matches input label group count
    if len(input_groups) != len(shapes):
        raise ValueError(
            f"Number of input label groups ({len(input_groups)}) does not match "
            f"number of tensors ({len(shapes)}): '{einsum_string}'"
        )

    # Check 5: all labels are lowercase alphabetic
    all_labels = input_part.replace(",", "") + output_part
    for ch in all_labels:
        if not ch.isalpha() or not ch.islower():
            raise ValueError(
                f"Invalid label character '{ch}' in '{einsum_string}': "
                f"labels must be lowercase alphabetic (a-z)"
            )

    # Check 2: each label group length matches tensor ndim
    for i, (group, shape) in enumerate(zip(input_groups, shapes)):
        if len(group) != len(shape):
            raise ValueError(
                f"Label group '{group}' has {len(group)} indices but tensor "
                f"'{names[i]}' (argument {i}) has {len(shape)} "
                f"dimensions (shape {shape})"
            )

    # Check 3: consistent dimension sizes for same label
    label_dims: dict[str, tuple[int, str, int]] = {}
    for i, (group, shape) in enumerate(zip(input_groups, shapes)):
        for j, label in enumerate(group):
            dim = shape[j]
            if label not in label_dims:
                label_dims[label] = (dim, names[i], i)
            elif label_dims[label][0] != dim:
                first_dim, first_name, first_idx = label_dims[label]
                raise ValueError(
                    f"Dimension mismatch for label '{label}': "
                    f"tensor '{first_name}' (argument {first_idx}) has dimension "
                    f"{first_dim}, but tensor '{names[i]}' (argument {i}) has "
                    f"dimension {dim}"
                )

    # Check 4: output labels are a subset of input labels
    input_labels = set(input_part.replace(",", ""))
    for ch in output_part:
        if ch not in input_labels:
            raise ValueError(
                f"Output label '{ch}' does not appear in any input: "
                f"'{einsum_string}'. Output labels must be a subset of "
                f"input labels {sorted(input_labels)}"
            )
