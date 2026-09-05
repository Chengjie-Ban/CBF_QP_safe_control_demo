
import numpy as np

def compute_f_g_and_ecbf_terms(
    #*,
    q: np.ndarray,
    qd: np.ndarray,
    x: np.ndarray,
    J: np.ndarray,
    Jdot_qdot: np.ndarray,
    M: np.ndarray,
    CqdG: np.ndarray,  ##合并bias项
    #G: np.ndarray,
    xc: np.ndarray,
    A: np.ndarray,
    n: int = 8,
    Gamma: np.ndarray | None = None,
    xdot: np.ndarray | None = None,
    eps_sign: float = 1e-12,
):
    """
    Compute operational-space dynamics terms f(q,qd), g(q), and ECBF terms for the
    super-ellipsoid operational-space constraint in the paper:

        xdd = f(q,qd) + g(q) * Gamma
        hdd = Lf2_h + LgLf_h * Gamma     (scalar + (1xN)(Nx1))

    Assumptions:
      - xc and A are constant (xc_dot=0, xc_ddot=0, A_dot=0, A_ddot=0).
      - Barrier: h(x) = 1 - (|x-xc|^n)^T A (|x-xc|^n)
      - n is a positive even integer (typical: 6, 8, 10, ...).

    Inputs (per time step):
      q, qd: (N,)
      x:    (m,) operational position (m=3 typically)
      J:    (m,N)
      Jdot: (m,N)
      M:    (N,N)
      Cqd:  (N,)   (this is C(q,qd) @ qd)
      G:    (N,)
      xc:   (m,)
      A:    (m,m) symmetric PD

    Optional:
      - xdot: (m,) if None, uses J @ qd
      - Gamma: (N,) if provided, also returns hdd at this Gamma

    Returns:
      f:      (m,)
      g:      (m,N)
      Lf2_h:  float
      LgLf_h: (N,)   (row vector flattened)
      h:      float
      hdot:   float
      (optional) hdd: float  if Gamma is not None
    """

    # ----------------------------
    # sanity checks / reshape
    # ----------------------------
    q = np.asarray(q).reshape(-1)
    qd = np.asarray(qd).reshape(-1)
    x = np.asarray(x).reshape(-1)
    xc = np.asarray(xc).reshape(-1)

    N = q.shape[0]
    m = x.shape[0]

    J = np.asarray(J).reshape(m, N)
    #Jdot_qdot = np.asarray(Jdot_qdot).reshape(m, N)
    M = np.asarray(M).reshape(N, N)
    CqdG = np.asarray(CqdG).reshape(N)
    #G = np.asarray(G).reshape(N)
    A = np.asarray(A).reshape(m, m)

    if xdot is None:
        xdot = J @ qd
    else:
        xdot = np.asarray(xdot).reshape(m)

    if (n % 2) != 0 or n <= 0:
        raise ValueError(f"n must be a positive even integer, got n={n}")

    # ----------------------------
    # (1) compute f(q,qd), g(q) from paper eq (3)
    #     xdd = Jdot*qd + J*M^{-1}*(-Cqd - G) + J*M^{-1}*Gamma
    # ----------------------------
    # Solve M X = B without forming inv(M)
    Minv_neg_CqdG = np.linalg.solve(M, -(CqdG))              # (N,)  修改原来是Cqd + G
    Minv_I = np.linalg.solve(M, np.eye(N))                      # (N,N) = M^{-1}

    f = (Jdot_qdot) + (J @ Minv_neg_CqdG)                       # (m,)
    g = (J @ Minv_I)                                            # (m,N)

    # ----------------------------
    # (2) super-ellipsoid barrier h(x)
    #     h = 1 - (|z|^n)^T A (|z|^n),  z = x - xc
    # ----------------------------
    
    scale = 0.5
    z = (x - xc)/scale
    az = np.abs(z)
    # robust sign: avoid sign flip at exactly 0
    s = np.sign(z)
    s[az < eps_sign] = 0.0

    z_n = az ** n                        # |z|^n
    h = float(1.0 - (z_n.T @ A @ z_n))

    # ----------------------------
    # (3) hdot and hdd via the explicit expressions (paper eq (25),(26))
    #     under constant xc, A. (so xc_dot=0, A_dot=0 terms vanish)
    #
    # Let:
    #   Xtilde = sign(z) ◦ n|z|^{n-1} ◦ xdot
    # Then:
    #   hdot = -2 (|z|^n)^T A Xtilde
    #   hdd  = -2 Xtilde^T A Xtilde
    #          -2 (|z|^{n/2})^T A [ sign(z) ◦ ( n(n-1)|z|^{n-2}◦xdot^2 + n|z|^{n-1}◦xdd ) ]
    # where xdd = f + g Gamma.
    # ----------------------------
    z_n_1 = az ** (n - 1)               # |z|^{n-1}
    z_n_2 = az ** (n - 2)               # |z|^{n-2}  (for n>=2)
    #z_n_half = az ** (n // 2)           # |z|^{n/2}  (n even)

    Xtilde = (s * (n * z_n_1) * xdot)   # (m,)   ydot

    hdot = float(-2.0 * (z_n.T @ A @ Xtilde))

    # Split hdd into: hdd = const_part + coeff_xdd @ xdd
    # term_a = -2 Xtilde^T A Xtilde
    term_a = float(-2.0 * (Xtilde.T @ A @ Xtilde))

    # inner_vel = sign(z) ◦ [ n(n-1)|z|^{n-2} ◦ xdot^2 ]
    inner_vel = s * (n * (n - 1) * z_n_2 * (xdot ** 2))         # (m,) yddot

    # coeff_xdd row vector (1xm): from inner_xdd = sign(z) ◦ [ n|z|^{n-1} ◦ xdd ]
    # hdd_xdd = -2 * (|z|^{n/2})^T A ( inner_xdd )
    #        =  (coeff_xdd) @ xdd
    diag_vec = (s * (n * z_n_1))                                 # (m,)
    coeff_xdd = (-2.0 * (z_n.T @ A) * diag_vec)             # (m,)   h二阶导对x二阶导

    # term_b_const = -2 (|z|^{n/2})^T A * inner_vel
    term_b_const = float(-2.0 * (z_n.T @ A @ inner_vel))    #剩下的，h二阶导里与x二阶导 无关的那一部分

    # Now substitute xdd = f + g Gamma:
    # hdd = term_a + term_b_const + coeff_xdd @ f  + (coeff_xdd @ g) @ Gamma
    Lf2_h = float(term_a + term_b_const + (coeff_xdd @ f))
    LgLf_h = (coeff_xdd @ g).reshape(-1)                         # (N,)

    if Gamma is None:
        return f, g, Lf2_h, LgLf_h, h, hdot

    Gamma = np.asarray(Gamma).reshape(N)
    hdd = float(Lf2_h + (LgLf_h @ Gamma))
    return f, g, Lf2_h, LgLf_h, h, hdot, hdd



    
