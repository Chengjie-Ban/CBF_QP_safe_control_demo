
import numpy as np

def lyapunov_terms_from_error(e, edot, 
                             Qp_diag=(10.0, 10.0, 10.0),
                             Qv_diag=(1.0, 1.0, 1.0)):
    """
    Compute terms in:
        LFV(eta) + LGV(eta) * mu <= - (lambda_min(Q)/lambda_max(P)) * V(eta)

    Definitions:
        eta = [e; edot] ∈ R^6
        eta_dot = F eta + G mu
        V(eta) = eta^T P eta
        dot(V) = LFV + LGV*mu
            LFV = eta^T (F^T P + P F) eta
            LGV = 2 eta^T P G   (a 1x3 row; so LGV @ mu is scalar)

    P is obtained by solving the continuous-time algebraic Riccati equation (CARE):
        F^T P + P F - P G G^T P + Q = 0
    with Q = blkdiag(Qp, Qv), Qp=diag(Qp_diag), Qv=diag(Qv_diag).

    Returns:
        LFV: float
        LGV: (3,) ndarray   (row vector coefficients multiplying mu)
        lambda_min_Q: float
        lambda_min_P: float
        V: float
    """
    e = np.asarray(e, dtype=float).reshape(3)
    edot = np.asarray(edot, dtype=float).reshape(3)
    eta = np.hstack([e, edot])  # (6,)  yita    这里就把eta写成了行向量，所以后面不需要转置

    # Build F, G for 3D position error
    I = np.eye(3)
    Z = np.zeros((3, 3))
    F = np.block([[Z, I],
                  [Z, Z]])            # (6,6)
    G = np.vstack([Z, I])             # (6,3)

    # Build Q = blkdiag(Qp, Qv)
    Qp = np.diag(np.asarray(Qp_diag, dtype=float).reshape(3))
    Qv = np.diag(np.asarray(Qv_diag, dtype=float).reshape(3))
    Q  = np.block([[Qp, Z],
                   [Z,  Qv]])         # (6,6)

    # Solve CARE: F^T P + P F - P G G^T P + Q = 0
    # Use scipy if available; otherwise raise a helpful error.
    try:
        from scipy.linalg import solve_continuous_are
    except Exception as ex:
        raise ImportError(
            "scipy is required: pip install scipy (or conda install scipy)."
        ) from ex

    P = solve_continuous_are(F, G, Q, np.eye(3))  # R = I

    # Terms
    V = float(eta @ (P @ eta))    ###eta就是行向量
    A = F.T @ P + P @ F
    LFV = float(eta @ (A @ eta))

    # LGV is 2 * eta^T * P * G  -> shape (3,)
    LGV = (2.0 * (eta @ (P @ G))).reshape(3)     #reshape，保证后续可操作性，casadi需要显式形状

    # Eigenvalue mins
    lambda_min_Q = float(np.min(np.linalg.eigvalsh(Q)))
    lambda_max_P = float(np.max(np.linalg.eigvalsh(P)))

    # (optional) if you also want the scalar dot(V) when mu is provided:
    #if mu is not None:
    #    mu = np.asarray(mu, dtype=float).reshape(3)
    #    Vdot = LFV + float(LGV @ mu)
    #    return LFV, LGV, lambda_min_Q, lambda_max_P, V, Vdot

    return LFV, LGV, lambda_min_Q, lambda_max_P, V