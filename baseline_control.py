import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from numpy.linalg import norm

from controllers.dynamics import (
    dynamically_consistent_projector,
    full_mass_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "franka_panda" / "scene.xml"

# ============================================================
# Trajectory
# ============================================================

def circle_traj(t, center, r=0.04, omega=0.3):
    """世界系 x-y 平面圆轨迹"""
    cx, cy, cz = center
    pos = np.array([
        cx + (r+0) * np.cos(omega * t),
        cy + (r+0.02) * np.sin(omega * t),
        cz
    ], dtype=float)

    vel = np.array([
        -r * omega * np.sin(omega * t),
        r * omega * np.cos(omega * t),
        0.0
    ], dtype=float)

    return pos, vel


# ============================================================
# Main
# ============================================================

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data  = mujoco.MjData(model)

    # --------- find ee site ----------
    ee_site_name = "ee_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id < 0:
        raise RuntimeError(f"Cannot find site named '{ee_site_name}' in the XML.")

    # --------- reset to keyframe home (关键!) ----------
    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        print("[WARN] keyframe 'home' not found, will start from default qpos (可能全0).")
    else:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # --------- gains (先稳后快) ----------
    # 任务空间位置控制（先保守，确保不饱和）
    Kp_task = np.array([500.0, 500.0, 500.0])
    Kd_task = 0.707 * np.sqrt(Kp_task)

    # 零空间姿态保持（更保守）
    Kp_null = np.array([15, 15, 15, 12, 8, 8, 6], dtype=float)
    Kd_null = 2.0 * np.sqrt(Kp_null)

    # --------- nominal posture & center ----------
    q_nominal = data.qpos[:7].copy()
    #center = data.site_xpos[ee_site_id].copy()
    center = [0.5, 0, 0.2]   #可以自己设定轨迹中心

    # 圆轨迹参数（先小半径、慢速度，确认能跟上再加）
    r = 0.1
    omega = 0.6

    # 先 hold 一段时间，让系统稳定在初始点
    t_hold = 1.0

    # actuator 限幅（从模型里读，避免写死）
    ctrl_lo = model.actuator_ctrlrange[:7, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:7, 1].copy()

    print("=" * 70)
    print("Operational Space Control (MuJoCo-consistent)")
    print("EE site:", ee_site_name, f"(id={ee_site_id})")
    print("Initial q:", q_nominal)
    print("Initial ee轨迹中心:", center)
    print("ctrlrange:", np.vstack([ctrl_lo, ctrl_hi]).T)
    print(f"Circle: r={r}, omega={omega}, hold={t_hold}s")
    print("=" * 70)

    total_steps = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        while viewer.is_running():
            total_steps += 1

            # 保证 site_xpos / qfrc_bias / Jacobian 基于同一状态
            mujoco.mj_forward(model, data)

            q  = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            t  = data.time

            # 当前末端位置
            y = data.site_xpos[ee_site_id].copy()

            # Jacobian（位置部分，世界系）解出一个能用的雅可比 7行
            J_pos_full = np.zeros((3, model.nv), dtype=float)
            J_rot_full = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, J_pos_full, J_rot_full, ee_site_id)
            J = J_pos_full[:, :7].copy()

            # 期望轨迹：先 hold 再画圆
            if t < t_hold:
                y_d = center.copy()
                yd_d = np.zeros(3)
            else:
                y_d, yd_d = circle_traj(t - t_hold, center=center, r=r, omega=omega)

            # 误差
            e  = y_d - y
            yd = J @ qd
            ed = yd_d - yd
            # 任务空间 PD 力
            F = Kp_task * e + Kd_task * ed

            # 映射到关节力矩
            tau_task = J.T @ F

            # 动力学补偿（MuJoCo）
            bias = data.qfrc_bias[:7].copy()

            # 动态一致零空间
            M = full_mass_matrix(model, data)
            _, N = dynamically_consistent_projector(J, M, eps=1e-8)

            tau0 = Kp_null * (q_nominal - q) - Kd_null * qd
            tau_null = N.T @ tau0

            # 总力矩
            tau = tau_task + tau_null + bias

            # 限幅（用xml里的ctrlrange）
            tau = np.clip(tau, ctrl_lo, ctrl_hi)

            # 施加控制
            data.ctrl[:7] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            # Debug
            if total_steps % 1000 == 0:
                err_mm = norm(e) * 1000.0
                sat_ratio = np.mean((tau <= ctrl_lo + 1e-6) | (tau >= ctrl_hi - 1e-6))
                print(
                    f"step:{total_steps:6d} | t:{t:6.2f}s | err:{err_mm:7.2f}mm | "
                    f"sat:{sat_ratio*100:5.1f}% | "
                    f"des:[{y_d[0]:.3f},{y_d[1]:.3f},{y_d[2]:.3f}] | "
                    f"act:[{y[0]:.3f},{y[1]:.3f},{y[2]:.3f}]"
                )

    print("=" * 70)
    print("Simulation finished.")


if __name__ == "__main__":
    main()
