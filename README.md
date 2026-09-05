# Safety-Compliant Control for Robotic Manipulators with CLF-CBF-QP

This repository provides a reproduction and implementation of the control framework presented in the paper:

> **M. A. Murtaza, S. Aguilera, M. Waqas, and S. Hutchinson,  
> "Safety Compliant Control for Robotic Manipulator With Task and Input Constraints,"  
> IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10659–10664, Oct. 2022.**  
> DOI: 10.1109/LRA.2022.3179118

The goal of this project is to reproduce and explore the safety-compliant control framework described in the paper for robotic manipulators subject to task and input constraints.

The implementation uses a **Franka Panda robotic manipulator** simulated in **MuJoCo** and demonstrates a control architecture based on **Control Lyapunov Functions (CLFs)**, **Control Barrier Functions (CBFs)**, and **Quadratic Programming (QP)**.

> **Note:** This repository is an independent reproduction/implementation for research and educational purposes. It is not the official implementation released by the authors of the paper.

---

## Overview

Safe robotic manipulation requires the controller to achieve the desired task while simultaneously respecting safety and physical constraints.

This project investigates this problem using a CLF-CBF-QP-based controller.

Two controller implementations are provided:

- `baseline_control.py`  
  Implements operational-space PD control with dynamics compensation and null-space posture control. No CLF or CBF constraints are applied.

- `safe_control.py`  
  Implements a CLF-CBF-QP-based controller for safety-aware control. The controller computes the control torques through a quadratic program. If the QP solver fails, the implementation falls back to operational-space PD control.

The two implementations make it possible to inspect the behavior of a conventional operational-space controller and a safety-aware controller within the same Franka Panda simulation environment.

---

## Control Framework

The safe controller combines three main components:

### Control Lyapunov Function (CLF)

The CLF is used to encode convergence toward the desired task or reference trajectory.

Conceptually, the CLF condition encourages the system state to evolve toward the desired target while maintaining stable tracking behavior.

### Control Barrier Function (CBF)

The CBF is used to encode safety constraints.

By constraining the evolution of the barrier function, the controller attempts to keep the robot inside a predefined safe set and prevent violations of the specified safety conditions.

### Quadratic Programming (QP)

The CLF and CBF conditions are incorporated into a quadratic optimization problem.

At each control step, the QP solver determines a control input that attempts to:

1. follow the desired robot motion,
2. satisfy the CLF condition,
3. satisfy the CBF safety constraints, and
4. respect the constraints represented in the optimization problem.

The QP problem is solved using **OSQP**.

---

## Robot and Simulation Environment

The experiments in this repository use the:

**Franka Emika Panda**

7-DoF robotic manipulator.

The robot dynamics and simulation environment are implemented using **MuJoCo**.

The included model files are located under:

```text
models/franka_panda/
```

The model path is resolved relative to the location of the Python scripts, so the programs can be launched from different working directories.

---

## Project Structure

```text
operational_space_clf_cbf_demo/
├── baseline_control.py
├── safe_control.py
│
├── controllers/
│   ├── __init__.py
│   ├── cbf.py
│   ├── clf.py
│   ├── dynamics.py
│   └── qp_solver.py
│
├── models/
│   └── franka_panda/
│       ├── scene.xml
│       ├── panda.xml
│       ├── assets/
│       └── LICENSE
│
├── outputs/
│
├── requirements.txt
└── README.md
```

### Main Files

#### `baseline_control.py`

Baseline operational-space controller.

It includes:

- operational-space PD control,
- robot dynamics compensation,
- null-space posture control.

This controller does **not** use CLF or CBF constraints.

#### `safe_control.py`

Safety-aware controller based on CLF-CBF-QP.

The controller formulates a quadratic program to compute the control torques while incorporating the CLF/CBF conditions.

If the QP cannot be solved successfully, the current implementation falls back to conventional operational-space PD control.

#### `controllers/cbf.py`

Implementation of the Control Barrier Function components used to represent safety constraints.

#### `controllers/clf.py`

Implementation of the Control Lyapunov Function components used for convergence and tracking objectives.

#### `controllers/dynamics.py`

Utility functions related to the robot dynamics used by the controller.

#### `controllers/qp_solver.py`

Construction and solution of the CLF-CBF quadratic program.

---

## Requirements

Python **3.10** or **3.11** is recommended.

The main Python dependencies are:

- `numpy` — numerical matrix and vector operations
- `mujoco` — robot dynamics, simulation, and visualization
- `scipy` — sparse matrix operations used by the QP formulation
- `osqp` — quadratic programming solver
- `matplotlib` — visualization and trajectory plotting

The complete dependency list is provided in:

```text
requirements.txt
```

---

## Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd operational_space_clf_cbf_demo
```

### Option 1 — Python Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Option 2 — Conda

```powershell
conda create -n clf-cbf-demo python=3.11 -y
conda activate clf-cbf-demo
python -m pip install -r requirements.txt
```

---

## Running the Simulation

### Baseline Controller

Run the operational-space controller without CLF/CBF safety constraints:

```powershell
python baseline_control.py
```

A MuJoCo viewer will open and display the Franka Panda simulation.

---

### CLF-CBF-QP Safety Controller

Run the safety-aware controller:

```powershell
python safe_control.py
```

The controller will execute the trajectory using the CLF-CBF-QP formulation.

After the simulation, the trajectory plot is saved to:

```text
outputs/trajectory_tracking.png
```

The program terminates after the MuJoCo viewer is closed.

---

## Baseline vs. Safety Controller

The repository provides two controller entry points for inspecting the behavior of the system:

| Controller | Operational-Space Control | CLF | CBF | QP |
|---|---:|---:|---:|---:|
| `baseline_control.py` | Yes | No | No | No |
| `safe_control.py` | Yes | Yes | Yes | Yes |

The baseline controller provides a conventional operational-space control implementation, while the safety controller introduces CLF-CBF constraints through quadratic programming.

---

## Current Implementation Notes

This repository should be considered a **research reproduction and experimental implementation**, rather than an exact one-to-one reproduction of every experiment reported in the original paper.

In particular, the current implementation has the following considerations:

1. The baseline and safety-controller entry points retain their own trajectory and controller-gain settings.

2. Therefore, the two scripts should not currently be interpreted as a strictly controlled single-variable comparison of CBF vs. non-CBF control.

3. For a rigorous evaluation of the effect of the CBF controller, the following parameters should be standardized between experiments:

   - reference trajectory,
   - controller gains,
   - initial robot configuration,
   - simulation duration,
   - task conditions.

4. If the QP solver fails, `safe_control.py` falls back to conventional PD control. During this fallback period, the hard safety guarantees associated with the CBF-QP formulation should not be assumed.

These aspects are important when interpreting simulation results.

---

## Paper

This project is based on the following paper:

**M. A. Murtaza, S. Aguilera, M. Waqas, and S. Hutchinson**,  
"Safety Compliant Control for Robotic Manipulator With Task and Input Constraints,"  
*IEEE Robotics and Automation Letters*,  
vol. 7, no. 4, pp. 10659–10664, Oct. 2022.

**DOI:** 10.1109/LRA.2022.3179118

### Keywords

- Robot safety
- Control Barrier Functions
- Control Lyapunov Functions
- Compliance and impedance control
- Constrained motion planning
- Collision avoidance
- Manipulator dynamics
- Task constraints
- Input constraints
- Quadratic programming

---

## Citation

If you use this repository in academic work, please cite the original paper:

```bibtex
@article{murtaza2022safety,
  author={Murtaza, M. A. and Aguilera, S. and Waqas, M. and Hutchinson, S.},
  journal={IEEE Robotics and Automation Letters},
  title={Safety Compliant Control for Robotic Manipulator With Task and Input Constraints},
  year={2022},
  volume={7},
  number={4},
  pages={10659--10664},
  doi={10.1109/LRA.2022.3179118}
}
```

---

## Disclaimer

This repository is an independent reproduction and educational implementation based on the methodology described in the referenced paper.

It is **not an official implementation from the original authors** and should not be considered a safety-certified robotic control system.

The software is intended for simulation, research, and educational purposes.

---

## License

The Franka Panda model and associated assets included under `models/franka_panda/` may be subject to their respective licenses. See the corresponding `LICENSE` file for details.