"""
CORAL (CORrelation ALignment)
Author: Sikharam Satya Sri Harsha Phani
A simple and numerically stable implementation for hyperspectraldomain adaptation using CORAL.
"""

import numpy as np

# ---------------------------------------------------------------------
# Matrix square-root utilities
# ---------------------------------------------------------------------

def _sqrtm_psd(C, eps=1e-5):
    """
    Matrix square root of a symmetric positive semi-definite matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(C)

    eigvals = np.clip(eigvals, eps, None)

    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _invsqrtm_psd(C, eps=1e-5):
    """
    Inverse matrix square root of a symmetric positive semi-definite matrix.
    """
    eigvals, eigvecs = np.linalg.eigh(C)

    eigvals = np.clip(eigvals, eps, None)

    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------
# Fit CORAL transformation
# ---------------------------------------------------------------------
def coral_fit(X_source, X_target, ridge=1e-5):
    
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)

    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)

    Xs = X_source - mu_s
    Xt = X_target - mu_t

    d = X_source.shape[1]

    Cs = np.cov(Xs, rowvar=False) + ridge * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + ridge * np.eye(d)

    # Standard CORAL:
    # whiten source
    Cs_inv_half = _invsqrtm_psd(Cs)

    # recolor using target covariance
    Ct_half = _sqrtm_psd(Ct)

    A = Cs_inv_half @ Ct_half

    return {
        "A": A.astype(np.float32),
        "mu_s": mu_s.astype(np.float32),
        "mu_t": mu_t.astype(np.float32),
    }
# ---------------------------------------------------------------------
# Apply CORAL transformation
# ---------------------------------------------------------------------

def coral_apply(X, params):

    X = X.astype(np.float32)

    return (
        (X - params["mu_s"])
        @ params["A"]
        + params["mu_t"]
    )

# ---------------------------------------------------------------------
# Covariance gap
# ---------------------------------------------------------------------

def covariance_gap(X1, X2):
    """
    Frobenius norm between covariance matrices.
    Lower values indicate better alignment.
    """

    C1 = np.cov(X1, rowvar=False)
    C2 = np.cov(X2, rowvar=False)

    return np.linalg.norm(C1 - C2, ord="fro")