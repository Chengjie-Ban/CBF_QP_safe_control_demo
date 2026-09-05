"""Shared MuJoCo dynamics helpers for both controller variants."""

import mujoco
import numpy as np


def full_mass_matrix(model, data, arm_dof=7):
    """Return the dense joint-space mass matrix for the controlled arm."""
    mass_full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, mass_full, data.qM)
    return mass_full[:arm_dof, :arm_dof].copy()


def dynamically_consistent_projector(jacobian, mass_matrix, eps=1e-8):
    """Return the dynamically consistent pseudoinverse and null projector."""
    mass_inverse = np.linalg.inv(mass_matrix)
    lambda_inverse = jacobian @ mass_inverse @ jacobian.T
    task_inertia = np.linalg.inv(
        lambda_inverse + eps * np.eye(jacobian.shape[0])
    )
    jacobian_bar = mass_inverse @ jacobian.T @ task_inertia
    null_projector = np.eye(mass_matrix.shape[0]) - jacobian_bar @ jacobian
    return jacobian_bar, null_projector
