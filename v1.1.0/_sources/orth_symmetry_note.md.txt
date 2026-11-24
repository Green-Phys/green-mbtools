> Developer / theory note (not user-facing). Referenced from PR #54: the
> derivation of why `X(-k) = X(k)*` is the sufficient condition for the
> orthogonalization to be consistent with space-group and time-reversal
> symmetry in the conjugate DF-pair contraction.

# Orthogonalization is consistent with space-group and time-reversal symmetry

This note records why building the orthogonalizer on the irreducible wedge (with time-reversal conjugation and realification at self-TR k-points) makes every orthogonalization mode, including the gauge-sensitive ones (mo, natural, canonical Löwdin), consistent with full space-group plus time-reversal symmetry. It is the compact writeup of the argument in `Derivation.md`, ending in a simple example.

## Conventions

The orthogonalizer $X(k)$ maps AO one-body quantities to an orthonormal basis. Covariant quantities (overlap, hcore, Fock) transform as

$$F(k) = X(k)\,F_{\rm AO}(k)\,X^\dagger(k), \qquad I = X(k)\,S_{\rm AO}(k)\,X^\dagger(k).$$

The second identity gives the two relations used throughout:

$$X^\dagger(k)\,X(k) = S_{\rm AO}^{-1}(k), \qquad X^{-1}(k) \equiv X_{\rm inv}(k) = S_{\rm AO}(k)\,X^\dagger(k).$$

The Green's function (contravariant) carries the inverse factors,

$$G(k) = X^{-\dagger}(k)\,G_{\rm AO}(k)\,X^{-1}(k).$$

## Symmetry operator in the ortho basis

Let $U_{\rm AO}(k)$ be the AO representation of the space-group operation taking the irreducible representative $k_{\rm ir}$ to $k$, so $O_{\rm AO}(k) = U_{\rm AO}(k)\,O_{\rm AO}(k_{\rm ir})\,U_{\rm AO}^\dagger(k)$ for covariant $O$. The corresponding ortho-basis operator is

$$U(k) = X(k)\,U_{\rm AO}(k)\,X^{-1}(k_{\rm ir}).$$

Note the left factor sits at $k$ and the right factor at $k_{\rm ir}$: $U(k)$ maps the orthonormal space at $k_{\rm ir}$ to the one at $k$. It is unitary,

$$U(k)\,U^\dagger(k) = X(k)\,U_{\rm AO}(k)\,\big[X_{\rm inv}(k_{\rm ir})X_{\rm inv}^\dagger(k_{\rm ir})\big]\,U_{\rm AO}^\dagger(k)\,X^\dagger(k) = X(k)\,U_{\rm AO}(k)\,S_{\rm AO}(k_{\rm ir})\,U_{\rm AO}^\dagger(k)\,X^\dagger(k) = X(k)\,S_{\rm AO}(k)\,X^\dagger(k) = I,$$

using $X_{\rm inv}X_{\rm inv}^\dagger = S\,X^\dagger X\,S = S$ and the AO covariance $U_{\rm AO}(k)S_{\rm AO}(k_{\rm ir})U_{\rm AO}^\dagger(k) = S_{\rm AO}(k)$. Because $U(k)$ is unitary, reconstruction across a star preserves the full spectrum and all traces, so occupations and band energies are exactly star-invariant. The same $U(k)$ reconstructs both $F$ and $G$ in the ortho basis, since $S_{\rm orth}=I$ removes the covariant/contravariant distinction.

## The pathological contraction

Take the worst case for symmetry consistency: a self-energy-like contraction $\sum_k G(k)\,V(k,q)$ where $k = R\,k_0$ is generated from an IBZ point $k_0$ by a space-group operation (acting on $G$), while the interaction is reconstructed by time reversal, $V(k,q)$ from $V(-k,-q)^\ast$. Both symmetries act at once. Substituting the ortho-basis reconstructions and collapsing with $X^\dagger X = S_{\rm AO}^{-1}$ and the AO covariance (full algebra in `Derivation.md`) gives

$$G(k)\,V(k,q) = X(k)\,S_{\rm AO}(k)\,U_{\rm AO}(k)\,G_{\rm AO}(k_0)\,U_{\rm AO}^\dagger(k)\,S_{\rm AO}(k)\,X^\dagger(k)\ \mathrm{conj}\!\big[X(-k)\,V_{\rm AO}(-k,-q)\,X^\dagger(-q)\big].$$

If and only if $\mathrm{conj}[X(-k)] = X(k)$, the middle factor $S_{\rm AO}(k)\,X^\dagger(k)\,X(k) = 1$ collapses and

$$G(k)\,V(k,q) = X(k)\,S_{\rm AO}(k)\ \big[G_{\rm AO}(k)\,V_{\rm AO}(k,q)\big]\ \mathrm{conj}\!\big[X^\dagger(-q)\big].$$

The outer $X S$ and $\mathrm{conj}[X^\dagger(-q)]$ only dress the external indices. The contraction over the internal $k$ index between $G$ and $V$ is therefore identical in the AO and ortho bases, for any space-group star, provided the single condition

$$\boxed{\,X(-k) = X(k)^\ast\,}$$

holds. This is the one thing the orthogonalization build must guarantee. It is exactly what the IBZ build enforces: propagate $X$ from the wedge by the $U_{\rm AO}$ reps with a time-reversal conjugation, and realify $S$ (hence $X$) at self-TR k-points.

## Simple example

Take a two-orbital overlap at a time-reversal pair $\{k, -k\}$,

$$S_{\rm AO}(k) = \begin{pmatrix} 1 & b(k) \\ b(k)^\ast & 1 \end{pmatrix}, \qquad S_{\rm AO}(-k) = S_{\rm AO}(k)^\ast \ \ (\text{i.e. } b(-k) = b(k)^\ast),$$

which is the generic real-space relation for a Bloch overlap.

Symmetric Löwdin, $X(k) = S_{\rm AO}(k)^{-1/2}$, is a smooth (gauge-free) function of $S$, and the matrix inverse-square-root commutes with complex conjugation, so

$$X(-k) = \big(S_{\rm AO}(k)^\ast\big)^{-1/2} = \big(S_{\rm AO}(k)^{-1/2}\big)^\ast = X(k)^\ast$$

automatically. The boxed condition holds with nothing to enforce, which is why symmetric Löwdin was never broken.

Canonical Löwdin, mo, and natural instead keep the overlap (or Fock) eigenvectors, $X(k) = s^{-1/2}(k)\,U^\dagger(k)$. Diagonalizing $S_{\rm AO}(k)$ and $S_{\rm AO}(-k) = S_{\rm AO}(k)^\ast$ independently returns eigenvectors with unrelated phases (and arbitrary rotations inside degenerate blocks), so in general $X(-k) \neq X(k)^\ast$ and the middle factor above does not collapse: the contraction picks up a spurious gauge and the total energy drifts. Two build rules restore the condition:

1. Build $X$ once at the representative and define the partner by $X(-k) := X(k)^\ast$ (more generally propagate the whole star by $U_{\rm AO}$ with the TR conjugation). Then $X(-k) = X(k)^\ast$ by construction.
2. At a self-TR point $k = -k$, $S_{\rm AO}(k)$ is numerically real, so solve the eigenproblem on the real part. Real eigenvectors give a real $X = X^\ast$, which is the $k=-k$ case of the same condition. This is the realification step; without it, tiny imaginary noise on a degenerate block rotates the eigenvectors into a complex gauge and breaks $X(k) = X(k)^\ast$ at that single point (the Silicon 4x4x4 failure).

With both rules in place, $X(-k) = X(k)^\ast$ holds for every mode, so by the boxed result the orthogonalized contraction equals the AO one for arbitrary space-group symmetry, which is what the numerical tests on Si and SiC confirm.
