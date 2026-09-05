import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from numpy.linalg import norm
import matplotlib.pyplot as plt

from controllers import cbf
from controllers import clf
from controllers import qp_solver
from controllers.dynamics import (
    dynamically_consistent_projector,
    full_mass_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH =  "models/franka_panda/scene.xml" 
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ============================================================
# Trajectory
# ============================================================

def circle_traj(t, center, r=0.3, omega=0.5):
    """世界系 x-y 平面轨迹"""
    cx, cy, cz = center
    pos = np.array([
        cx + (r+0) * np.cos(omega * t),
        cy + (r+0.3) * np.sin(omega * t),
        cz
    ], dtype=float)

    vel = np.array([
        -r * omega * np.sin(omega * t),
        (r+0.3) * omega * np.cos(omega * t),
        0.0
    ], dtype=float)

    acc = np.array([
        -r * omega* omega * np.cos(omega * t),
        -(r+0.3) * omega*omega * np.sin(omega * t),
        0.0
    ], dtype=float)

    return pos, vel, acc


# ============================================================
# Main
# ============================================================

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # --------- find ee site ----------
    ee_site_name = "ee_site"
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id < 0:
        raise RuntimeError(f"Cannot find site named '{ee_site_name}' in the XML.")

    # --------- reset to keyframe home ----------
    key_name = "home"
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        print("[WARN] keyframe 'home' not found, will start from default qpos.")
    else:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # --------- gains ----------
    Kp_task = np.array([8000.0, 800.0, 800.0])
    Kd_task = 0.707 * np.sqrt(Kp_task)

    # --------- nominal posture & center ----------
    center = [0.5, 0, 0.2]

    # 圆轨迹参数
    r = 0.1
    omega = 0.5

    # hold时间
    t_hold = 1.0

    # actuator限幅
    ctrl_lo = model.actuator_ctrlrange[:7, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:7, 1].copy()

    # --------- 初始化QP求解器 ----------
    nu, nq = 3, 7
    p = 1000.0
    solve_step = qp_solver.make_clf_cbf_qp_solver_osqp(
        nu=nu, nq=nq, p=p, verbose=False
    )

    # --------- CBF参数 ----------
    n = 8*2
    A_cbf = np.diag([1/0.5**n, 1/0.6**n, 1/0.25**n])  ##cage尺寸 x y z  y是0.5实际差不多是0.3
    xc = np.array(center)
    
    # ECBF极点配置
    lambda1 = 320
    lambda2 = 80

    total_steps = 0
    x_prev = None  # 用于warm start

    # ========== 轨迹记录 ==========
    trajectory_log = {
        'time': [],
        'desired_pos': [],
        'actual_pos': [],
        'error': []
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        while viewer.is_running():
            total_steps += 1

            # ==========================================
            # 1. 状态更新
            # ==========================================
            mujoco.mj_forward(model, data)

            q = data.qpos[:7].copy()
            qd = data.qvel[:7].copy()
            t = data.time

            # 当前末端位置
            y = data.site_xpos[ee_site_id].copy()

            # Jacobian
            J_pos_full = np.zeros((3, model.nv), dtype=float)
            J_rot_full = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, J_pos_full, J_rot_full, ee_site_id)
            J = J_pos_full[:, :7].copy()
            
            # 质量矩阵
            M = full_mass_matrix(model, data)
            
            # 偏置项
            CqdG = data.qfrc_bias[:7].copy()

            # Jdot*qdot
            site_id = ee_site_id
            body_id = int(model.site_bodyid[site_id])
            point_b = np.array(model.site_pos[site_id], dtype=float)
            Jdot_pos_full = np.zeros((3, model.nv), dtype=float)
            Jdot_rot_full = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacDot(model, data, Jdot_pos_full, Jdot_rot_full, point_b, body_id)
            Jdot_qdot = (Jdot_pos_full[:, :7] @ qd).copy()

            # ==========================================
            # 2. 期望轨迹
            # ==========================================
            if t < t_hold:
                y_d = np.array(center.copy())
                yd_d = np.zeros(3)
                ydd_d = np.zeros(3)
            else:
                y_d, yd_d, ydd_d = circle_traj(t - t_hold, center=center, r=r, omega=omega)

            # 误差
            e = y_d - y
            yd = J @ qd
            ed = yd_d - yd

            # ========== 记录轨迹数据 ==========
            trajectory_log['time'].append(t)
            trajectory_log['desired_pos'].append(y_d.copy())
            trajectory_log['actual_pos'].append(y.copy())
            trajectory_log['error'].append(e.copy())

            # ==========================================
            # 3. 调用CBF计算 (返回对Gamma的梯度!)
            # ==========================================
            f, g, Lf2_h, LgLf_h_Gamma, h, hdot = cbf.compute_f_g_and_ecbf_terms(
                q, qd, y, J, Jdot_qdot, M, CqdG, xc, A_cbf, n=8
            )
            
            # ECBF系数 K*eta
            K_eta = lambda1 * lambda2 * h + (lambda1 + lambda2) * hdot

            # ==========================================
            # 4. Lyapunov约束
            # ==========================================
            LFV, LGV, lminQ, lmaxP, V = clf.lyapunov_terms_from_error(e, ed)
            c = lminQ / lmaxP

            # ==========================================
            # 5. 调用QP求解器 (方案A)
            # ==========================================
            # Keep the QP torque bounds consistent with the MuJoCo actuator model.
            tau_max = np.minimum(np.abs(ctrl_lo), np.abs(ctrl_hi))
            Kp = 10
            Kd = 1
            Jbar, _ = dynamically_consistent_projector(J, M)
            Jdag = Jbar
            xdd_des = ydd_d + Kp*e + Kd*ed  ##反馈线性化
            b_tau = M @ Jdag @ (xdd_des - Jdot_qdot) + CqdG
            A_tau_mu = M @ Jdag
            try:
                mu_opt, delta_opt, Gamma_opt, x_opt = solve_step(
                    # CLF参数
                    LFV, LGV, V, c,
                    # CBF参数
                    Lf2_h, LgLf_h_Gamma, K_eta,
                    # 动力学参数
                    A_tau_mu, b_tau,
                    # 限制
                    tau_max,
                    x0=x_prev
                )
                x_prev = x_opt  # 保存用于下次warm start
                
                # 使用优化后的力矩
                tau = Gamma_opt
                
            except Exception as ex:
                print(f"[ERROR] QP solve failed: {ex}")
                print("现在是pd控制器")
                # 回退到简单PD控制
                F = Kp_task * e + Kd_task * ed
                tau = J.T @ F + CqdG
                tau = np.clip(tau, ctrl_lo, ctrl_hi)

            # ==========================================
            # 6. 施加控制
            # ==========================================
            tau = np.clip(tau, ctrl_lo, ctrl_hi)
            data.ctrl[:7] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            # ==========================================
            # 7. Debug输出 - 增强版
            # ==========================================
            if total_steps % 100 == 0:  # 更频繁的输出
                err_mm = norm(e) * 1000.0
                sat_ratio = np.mean((tau <= ctrl_lo + 1e-6) | (tau >= ctrl_hi - 1e-6))
                print(f"\n{'='*80}")
                print(f"Step: {total_steps:6d} | Time: {t:6.2f}s")
                print(f"{'='*80}")
                print(f"期望位置 (Desired): [{y_d[0]:7.4f}, {y_d[1]:7.4f}, {y_d[2]:7.4f}]")
                print(f"实际位置 (Actual):  [{y[0]:7.4f}, {y[1]:7.4f}, {y[2]:7.4f}]")
                print(f"误差 (Error):       [{e[0]:7.4f}, {e[1]:7.4f}, {e[2]:7.4f}] ({err_mm:7.2f}mm)")
                print(f"饱和率 (Saturation): {sat_ratio*100:5.1f}%")
                print(f"{'='*80}\n")

            # 停止条件 (可选 - 运行一定时间后自动停止)
            if t > 100.0:  # 运行100秒后停止
                break

    print("=" * 70)
    print("Simulation finished.")
    
    # ========== 绘制轨迹图 ==========
    print("Plotting trajectories...")
    plot_trajectories(trajectory_log)


def plot_trajectories(log):
    """绘制期望轨迹vs实际轨迹"""
    time = np.array(log['time'])
    desired = np.array(log['desired_pos'])
    actual = np.array(log['actual_pos'])
    error = np.array(log['error'])
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(desired[:, 0], desired[:, 1], desired[:, 2], 
             'b--', linewidth=2, label='Desired')
    ax1.plot(actual[:, 0], actual[:, 1], actual[:, 2], 
             'r-', linewidth=1.5, label='Actual')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.set_zticks(np.arange(-0.2, 0.4, 0.1))

    
    # 2-4. X, Y, Z 分量随时间变化
    axes_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+2)
        ax.plot(time, desired[:, i], 'b--', linewidth=2, label='Desired')
        ax.plot(time, actual[:, i], 'r-', linewidth=1.5, label='Actual')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{axes_labels[i]} Position (m)')
        ax.set_title(f'{axes_labels[i]} Trajectory')
        ax.legend()
        ax.grid(True)

    
    # 5. 误差范数随时间变化
    ax5 = fig.add_subplot(2, 3, 5)
    error_norm = np.linalg.norm(error, axis=1) * 1000  # 转换为mm
    ax5.plot(time, error_norm, 'g-', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Tracking Error (mm)')
    ax5.set_title('Tracking Error Norm')
    ax5.grid(True)
    ax5.set_yticks(np.arange(0,200, 20))
    
    # 6. XY平面投影
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(desired[:, 0], desired[:, 1], 'b--', linewidth=2, label='Desired')
    ax6.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=1.5, label='Actual')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('XY Plane Projection')
    ax6.legend()
    ax6.grid(True)
    ax6.axis('equal')
    
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "trajectory_tracking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved as '{output_path}'")
    plt.show()
    
    # 打印统计信息
    #error_norm_mm = np.linalg.norm(error, axis=1) * 1000
    #print("\n" + "="*70)
    #print("轨迹跟踪统计 (Trajectory Tracking Statistics)")
    #print("="*70)
    #print(f"平均误差 (Mean Error):        {np.mean(error_norm_mm):.3f} mm")
    #print(f"最大误差 (Max Error):         {np.max(error_norm_mm):.3f} mm")
    #print(f"误差标准差 (Std Dev):         {np.std(error_norm_mm):.3f} mm")
    #print(f"最终误差 (Final Error):       {error_norm_mm[-1]:.3f} mm")
    #print("="*70)


if __name__ == "__main__":
    main()
