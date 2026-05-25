import awkward as ak
import numpy as np
import numba as nb
import vector
vector.register_awkward()

# =============================================================
#   Momentum tensor and eigenvalues (CMS-identical)
# =============================================================

@nb.njit
def _momentum_tensor_numba(px, py, pz):
    """
    Build 3×3 normalized momentum tensor exactly as in CMS EventShapeVariables.
    M_ij = Σ(p_i * p_j) / Σ(|p|²)
    """
    n = px.shape[0]
    M = np.zeros((3, 3))
    if n < 2:
        return M

    norm = 0.0
    for i in range(n):
        p2 = px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]
        norm += p2
        M[0,0] += px[i]*px[i]
        M[0,1] += px[i]*py[i]
        M[0,2] += px[i]*pz[i]
        M[1,0] += py[i]*px[i]
        M[1,1] += py[i]*py[i]
        M[1,2] += py[i]*pz[i]
        M[2,0] += pz[i]*px[i]
        M[2,1] += pz[i]*py[i]
        M[2,2] += pz[i]*pz[i]

    if norm > 0.0:
        M /= norm
    return M


@nb.njit
def _event_shape_eigenvalues(M):
    """
    Eigenvalues of 3×3 symmetric tensor, sorted descending (λ1 ≥ λ2 ≥ λ3),
    identical to ROOT::TMatrixDSym::EigenVectors ordering.
    """
    w, _ = np.linalg.eigh(M)
    return w[::-1]


# =============================================================
#   Fox–Wolfram H₄ (CMS-identical)
# =============================================================

@nb.njit
def _legendre_P4(x):
    # P4(x) = (35x⁴ − 30x² + 3)/8
    return 0.125 * (35.0*x**4 - 30.0*x**2 + 3.0)


@nb.njit
def _fox_wolfram_H4(px, py, pz):
    """
    Fox–Wolfram moment H₄ as in CMS EventShapeVariables::Compute().
    Hₗ = Σ_i Σ_j |p_i||p_j| Pₗ(cosθ_ij) / (Σ|p|)²
    """
    n = px.shape[0]
    if n == 0:
        return 0.0

    mag = np.sqrt(px**2 + py**2 + pz**2)
    s = np.sum(mag)
    if s <= 0.0:
        return 0.0

    sum_h = 0.0
    for i in range(n):
        for j in range(i, n):
            cosTheta = (px[i]*px[j] + py[i]*py[j] + pz[i]*pz[j]) / max(mag[i]*mag[j], 1e-12)
            P4 = _legendre_P4(cosTheta)
            factor = 2.0 if i != j else 1.0
            sum_h += factor * mag[i] * mag[j] * P4

    return sum_h / (s*s)


# =============================================================
#   High-level driver for awkward events (CMS-identical)
# =============================================================

def compute_event_shapes(jets):
    """
    Exact CMS EventShapeVariables equivalents:
      C_jet, D_jet, Aplanarity, H4
    Ensures λ₁ + λ₂ + λ₃ = 1 normalization for [0,1]-bounded C and D.
    """
    pxs = ak.to_list(ak.fill_none(jets.px, 0.0))
    pys = ak.to_list(ak.fill_none(jets.py, 0.0))
    pzs = ak.to_list(ak.fill_none(jets.pz, 0.0))

    n_events = len(pxs)
    C_vals, D_vals, A_vals, H4_vals = [], [], [], []

    for i in range(n_events):
        px, py, pz = np.array(pxs[i], np.float64), np.array(pys[i], np.float64), np.array(pzs[i], np.float64)
        if len(px) < 2:
            C_vals.append(np.nan)
            D_vals.append(np.nan)
            A_vals.append(np.nan)
            H4_vals.append(np.nan)
            continue

        # --- Momentum tensor and eigenvalues ---
        M = _momentum_tensor_numba(px, py, pz)
        eig = _event_shape_eigenvalues(M)
        λ1, λ2, λ3 = eig[0], eig[1], eig[2]

        # --- Normalize eigenvalues so λ₁+λ₂+λ₃ = 1 ---
        sumλ = λ1 + λ2 + λ3
        if sumλ > 0:
            λ1 /= sumλ
            λ2 /= sumλ
            λ3 /= sumλ

        # --- Event-shape observables (CMS definitions) ---
        C = 3.0 * (λ1*λ2 + λ1*λ3 + λ2*λ3)
        D = 27.0 * λ1 * λ2 * λ3
        A = 1.5 * λ3
        H4 = _fox_wolfram_H4(px, py, pz)

        C_vals.append(C)
        D_vals.append(D)
        A_vals.append(A)
        H4_vals.append(H4)

    return {
        "C_jet": ak.Array(C_vals),
        "D_jet": ak.Array(D_vals),
        "Aplanarity": ak.Array(A_vals),
        "H4": ak.Array(H4_vals)
    }
