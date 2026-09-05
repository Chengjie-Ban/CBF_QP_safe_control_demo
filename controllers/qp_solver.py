import numpy as np
import scipy.sparse as sp
import osqp


def make_clf_cbf_qp_solver_osqp(nu: int, nq: int, p: float, verbose=False):
    """
    Decision x = [mu(nu); delta(1); tau(nq)]  => n = nu + 1 + nq

    min   ||mu||^2 + p*delta^2
    s.t.  (CLF)  LFV + LGV*mu <= -c*V + delta
          (CBF)  Lf2_h + LgLf_h_tau * tau + Kc_eta >= 0
          (EQ)   tau - A_tau_mu*mu = b_tau
          bounds: delta >= 0, -tau_max <= tau <= tau_max, mu free
    """
    n = nu + 1 + nq

    # rows: 2 inequalities (clf, cbf) + nq equalities (link) + n variable-bound rows (I)
    m_ineq = 2
    m_eq   = nq
    m = m_ineq + m_eq + n

    # ----------------------------
    # Objective: 0.5 x^T P x + q^T x
    # Want: ||mu||^2 + p*delta^2
    # => P diag: mu:2, delta:2p, tau:0
    # ----------------------------
    P_diag = [2.0]*nu + [2.0*p] + [0.0]*nq
    P = sp.diags(P_diag, format="csc")
    q = np.zeros(n)

    # ----------------------------
    # Build fixed sparsity A = [A_ineq; A_eq; I]
    # ----------------------------
    A_template = sp.lil_matrix((m, n))

    MU_SLICE    = slice(0, nu)
    DELTA_IDX   = nu
    TAU_SLICE   = slice(nu+1, nu+1+nq)

    # (0) CLF row: [LGV , -1 , 0]
    for j in range(nu):
        A_template[0, j] = 1.0
    A_template[0, DELTA_IDX] = 1.0  # just to allocate nonzero; we'll set value -1 in data update

    # (1) CBF row: [0, 0, LgLf_h_tau]
    for i in range(nq):
        A_template[1, (nu+1)+i] = 1.0

    # (2..2+nq-1) link equality: -A_tau_mu*mu + 1*tau = b_tau
    # Row r = 2+i
    for i in range(nq):
        r = m_ineq + i
        # mu part
        for j in range(nu):
            A_template[r, j] = 1.0
        # tau part (identity)
        A_template[r, (nu+1)+i] = 1.0

    # variable bounds rows: I
    base = m_ineq + m_eq
    for k in range(n):
        A_template[base + k, k] = 1.0

    A = A_template.tocsc()

    # Build mapping (row,col) -> index in A.data (CSC)
    pos = {}
    for col in range(A.shape[1]):
        start, end = A.indptr[col], A.indptr[col + 1]
        rows = A.indices[start:end]
        for idx, r in enumerate(rows, start=start):
            pos[(r, col)] = idx

    def build_A_data(LGV_row, LgLf_h_tau_row, A_tau_mu_mat):
        data = A.data.copy()

        # CLF row
        for j in range(nu):
            data[pos[(0, j)]] = float(LGV_row[j])
        data[pos[(0, DELTA_IDX)]] = -1.0

        # CBF row (tau coefficients)
        for i in range(nq):
            data[pos[(1, (nu+1)+i)]] = float(LgLf_h_tau_row[i])

        # Link equalities: -A_tau_mu on mu cols, +I on tau
        for i in range(nq):
            r = m_ineq + i
            for j in range(nu):
                data[pos[(r, j)]] = -float(A_tau_mu_mat[i, j])
            # tau identity already has a slot; ensure it's +1
            data[pos[(r, (nu+1)+i)]] = 1.0

        return data

    # ----------------------------
    # Setup solver once with dummy bounds
    # ----------------------------
    solver = osqp.OSQP()
    l0 = -np.inf * np.ones(m)
    u0 =  np.inf * np.ones(m)
    solver.setup(P=P, q=q, A=A, l=l0, u=u0, verbose=verbose)

    # bounds for variables through I*x rows
    # mu free, delta >= 0, tau in [-tau_max, tau_max]
    def solve_step(
        LFV, LGV, V, c,
        Lf2_h, LgLf_h_tau, Kc_eta,
        A_tau_mu, b_tau, tau_max,
        x0=None
    ):
        """
        LgLf_h_tau: (nq,)  —— 注意 是7 维（乘 tau）
        Kc_eta: 标量，等于ECBF 里那坨 Kc*eta_c（例如 k1*(hdot+k0*h)）
        """
        LFV = float(LFV); V = float(V); c = float(c)
        Lf2_h = float(Lf2_h); Kc_eta = float(Kc_eta)

        LGV = np.asarray(LGV, dtype=float).reshape(nu,)
        LgLf_h_tau = np.asarray(LgLf_h_tau, dtype=float).reshape(nq,)
        A_tau_mu = np.asarray(A_tau_mu, dtype=float).reshape(nq, nu)
        b_tau    = np.asarray(b_tau, dtype=float).reshape(nq,)
        tau_max  = np.asarray(tau_max, dtype=float).reshape(nq,)

        l = -np.inf * np.ones(m, dtype=float)
        u =  np.inf * np.ones(m, dtype=float)

        # (CLF) LGV*mu - delta <= -cV - LFV
        u[0] = (-c * V - LFV)

        # (CBF) LgLf_h_tau * tau >= -(Lf2_h + Kc_eta)
        l[1] = -(Lf2_h + Kc_eta)

        # (EQ link) tau - A_tau_mu*mu = b_tau  => rows 2..2+nq-1
        eq_start = m_ineq
        eq_end   = m_ineq + m_eq
        l[eq_start:eq_end] = b_tau
        u[eq_start:eq_end] = b_tau

        # variable bounds via I*x rows
        base = m_ineq + m_eq
        lbx = np.full(n, -np.inf, dtype=float)
        ubx = np.full(n,  np.inf, dtype=float)

        # delta >= 0
        lbx[DELTA_IDX] = 0.0

        # tau bounds
        lbx[nu+1:nu+1+nq] = -tau_max
        ubx[nu+1:nu+1+nq] =  tau_max

        l[base:base+n] = lbx
        u[base:base+n] = ubx

        # update A numeric values
        A_data = build_A_data(LGV, LgLf_h_tau, A_tau_mu)
        solver.update(l=l, u=u, Ax=A_data)

        # warm start
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).reshape(n,)
            solver.warm_start(x=x0)

        res = solver.solve()
        if res.info.status_val not in (1, 2):  # solved / solved inaccurate
            raise RuntimeError(
                f"OSQP failed: status={res.info.status}, "
                f"prim_res={res.info.prim_res}, dual_res={res.info.dual_res}"
            )

        x_opt = res.x
        mu_opt    = x_opt[0:nu]
        delta_opt = float(x_opt[DELTA_IDX])
        tau_opt   = x_opt[nu+1:nu+1+nq]
        return mu_opt, delta_opt, tau_opt, x_opt

    return solve_step
