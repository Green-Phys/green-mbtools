"""Reference three-leg reconstruction in an explicitly identified DF gauge.

The initial contract accepts only captured full-rank lower Cholesky factors.
It never reconstructs a metric factor by an independent eigendecomposition.
All matrices act source -> target. Orbital matrices act on covariant tensors:
V_target[a] = sum_b D[a,b] left K(V_source[b]) right^dagger.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular


def _matrix(value, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix")
    return result


@dataclass(frozen=True)
class CholeskyGauge:
    unwhitening: np.ndarray
    gauge_id: str
    kernel_id: str

    def __post_init__(self):
        factor = _matrix(self.unwhitening, "Captured Cholesky factor")
        if not self.gauge_id or not self.kernel_id:
            raise ValueError("Gauge/kernel provenance is required")
        if not len(factor) or factor.shape[0] != factor.shape[1]:
            raise ValueError(
                "Initial SG support requires a full-rank square auxiliary space"
            )
        if np.max(np.abs(np.triu(factor, 1))) > 1e-12 * max(1, np.linalg.norm(factor)):
            raise ValueError("Expected the actual lower Cholesky factor")
        if (
            np.any(factor.diagonal().real <= 0)
            or np.max(np.abs(factor.diagonal().imag)) > 1e-12
        ):
            raise ValueError("Cholesky diagonal must be real and positive")
        if np.linalg.cond(factor) > 1e12:
            raise ValueError("Ill-conditioned or inconsistent retained auxiliary space")
        factor = np.array(factor, copy=True)
        factor.setflags(write=False)
        object.__setattr__(self, "unwhitening", factor)


def residual(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError("Residual arrays have incompatible shapes")
    delta = actual - reference
    position = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    norm = np.linalg.norm(reference)
    return {
        "max_absolute": float(np.max(np.abs(delta))),
        "relative_frobenius": float(np.linalg.norm(delta) / norm)
        if norm
        else float(np.linalg.norm(delta)),
        "worst_index": tuple(map(int, position)),
    }


def auxiliary_transform(
    source, target, raw_auxiliary, *, conjugate=False, atol=1e-10, rtol=1e-10
):
    """D = F_target A K(C_source), after checking the actual metric covariance."""
    if source.kernel_id != target.kernel_id:
        raise ValueError("Cannot combine gauges belonging to different Coulomb kernels")
    c_source = source.unwhitening.conj() if conjugate else source.unwhitening
    c_target = target.unwhitening
    raw_auxiliary = _matrix(raw_auxiliary, "Raw auxiliary representation")
    if raw_auxiliary.shape != (len(c_target), len(c_source)):
        raise ValueError("Auxiliary map and retained ranks are incompatible")
    mapped = raw_auxiliary @ c_source
    expected_metric = c_target @ c_target.conj().T
    mapped_metric = mapped @ mapped.conj().T
    if not np.allclose(mapped_metric, expected_metric, atol=atol, rtol=rtol):
        raise ValueError(
            f"Auxiliary metric covariance failed: {residual(mapped_metric, expected_metric)}"
        )
    transform = solve_triangular(c_target, mapped, lower=True)
    # A small absolute error in the raw metric can be amplified by whitening.
    # The whitened auxiliary leg must preserve V^dagger V itself, including
    # near-null metric directions. Orbital stored-basis maps have a different
    # covariance contract and need not be Euclidean unitary.
    gram = transform.conj().T @ transform
    identity = np.eye(len(c_source), dtype=complex)
    if not np.all(np.isfinite(transform)) or not np.allclose(
        gram, identity, atol=atol, rtol=rtol
    ):
        raise ValueError(
            f"Whitened auxiliary isometry failed: {residual(gram, identity)}"
        )
    return transform


def reconstruct_factor(
    source,
    auxiliary,
    left,
    right,
    *,
    conjugate=False,
    transpose=False,
    output_slice=None,
):
    """Return an owned requested-pair factor; slicing still mixes all source Q rows."""
    source = np.asarray(source, dtype=np.complex128)
    auxiliary = _matrix(auxiliary, "Whitened auxiliary map")
    left, right = _matrix(left, "Left orbital map"), _matrix(right, "Right orbital map")
    if source.ndim != 3 or not np.all(np.isfinite(source)):
        raise ValueError("Source factor must be a finite (NQ, nao, nao) tensor")
    if conjugate:
        source = source.conj()
    if transpose:
        source = source.swapaxes(1, 2)
    if (
        auxiliary.shape[1] != source.shape[0]
        or left.shape[1] != source.shape[1]
        or right.shape[1] != source.shape[2]
    ):
        raise ValueError("Three-leg reconstruction dimensions disagree")
    if output_slice is not None:
        start, count = output_slice
        if (
            not isinstance(start, int)
            or not isinstance(count, int)
            or start < 0
            or count <= 0
            or start + count > auxiliary.shape[0]
        ):
            raise ValueError("Requested auxiliary slice is out of range")
        auxiliary = auxiliary[start : start + count, :]
    return np.einsum(
        "ab,bmn,im,jn->aij", auxiliary, source, left, right.conj(), optimize=True
    )


def stored_basis_operation(
    ao_operation, target_x, source_x_inverse, *, conjugate=False
):
    """Convert the source->target AO map for the producer's X V X^dagger basis."""
    ao_operation = _matrix(ao_operation, "AO operation")
    target_x = _matrix(target_x, "Target orthogonalizer")
    source_x_inverse = _matrix(source_x_inverse, "Source inverse orthogonalizer")
    for matrix in (ao_operation, target_x, source_x_inverse):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Rectangular stored-basis maps require retained-subspace support"
            )
    source_inverse = source_x_inverse.conj() if conjugate else source_x_inverse
    return target_x @ ao_operation @ source_inverse


def validate_overlap_covariance(
    source_overlap,
    target_overlap,
    operation,
    *,
    conjugate=False,
    atol=1e-10,
    rtol=1e-10,
):
    """Validate covariance; Euclidean unitarity is not required in a stored basis."""
    source_overlap = _matrix(source_overlap, "Source overlap")
    target_overlap = _matrix(target_overlap, "Target overlap")
    operation = _matrix(operation, "Orbital operation")
    mapped = (
        operation
        @ (source_overlap.conj() if conjugate else source_overlap)
        @ operation.conj().T
    )
    if not np.allclose(mapped, target_overlap, atol=atol, rtol=rtol):
        raise ValueError(
            f"Overlap covariance failed: {residual(mapped, target_overlap)}"
        )
    return residual(mapped, target_overlap)
