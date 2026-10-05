<h1 align="center"><a href="https://arxiv.org/abs/2411.15036">Safe Multi-Agent Reinforcement Learning with Convergence to Generalized Nash Equilibrium</a></h1>

<p align="center">
  Zeyang Li &nbsp;·&nbsp; Navid Azizan
  <br>
  <i>Massachusetts Institute of Technology</i>
</p>

---

Implementation for Multi-Agent Dual Actor-Critic (MADAC).

## Repository layout

```
envs/safety_ma_mujoco/                            the six tasks (package safety_ma_mujoco)
SafeMARL/                                         MADAC, HASAC and HAPPO (package safe_marl, built on HARL)
SafeMARL/safe_marl/algorithms/madac.py            the MADAC learner
SafeMARL/safe_marl/configs/                       default configs (madac.yaml, hasac.yaml, happo.yaml, mujoco.yaml)
SafeMARL/tuned_configs/mujoco/<task>/<algo>/      the per-task configs of the experiments (config.json)
SafeMARL/examples/train.py, run.sh, run_all.sh    training entry points
SafeMARL/examples/evaluate.py                     evaluation of saved MADAC checkpoints
MACPO/, MAPPO-Lagrangian/                         the trust-region baselines, with their own entry points
```

## Installation

MuJoCo 2.0 must be installed at `~/.mujoco/mujoco200` with the key at `~/.mujoco/mjkey.txt`, and
`~/.mujoco/mujoco200/bin` must be on `LD_LIBRARY_PATH`. On Ubuntu, `mujoco-py` also needs

```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglew-dev patchelf
```

Then

```bash
conda create -n SafeMARL python=3.9
conda activate SafeMARL
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # the build matching your CUDA version
pip install -e envs -e SafeMARL -e MACPO -e MAPPO-Lagrangian
```

`requirements.txt` lists the packages used by the environments and the five algorithms.

## Supported environments and algorithms

Environments: `HalfCheetah-v2-2x3`, `HalfCheetah-v2-3x2`, `Walker2d-v2-2x3`, `Walker2d-v2-3x2`, `Ant-v2-2x4`,
`Ant-v2-4x2` ("AxB" means A agents controlling B actuators each).

Algorithms: MADAC, HASAC, HAPPO, MACPO, MAPPO-Lagrangian.

## Training

MADAC, HASAC and HAPPO run from `SafeMARL/examples` with the tuned config of a task:

```bash
cd SafeMARL/examples
bash run.sh madac Ant-v2-4x2                    # <algo> <task> [seed]
bash run.sh hasac Walker2d-v2-3x2 2
bash run.sh happo HalfCheetah-v2-3x2 2
bash run_all.sh madac                           # one algorithm on all six tasks, sequentially
```

Any config entry can be overridden on the command line, and `--keep_checkpoints True` keeps the checkpoint of every
evaluation:

```bash
bash run.sh madac Ant-v2-2x4 2 --num_env_steps 5000000 --keep_checkpoints True
```

MACPO and MAPPO-Lagrangian have one script per task:

```bash
cd MACPO/macpo/scripts && bash half_cheetah_2x3.sh
cd MAPPO-Lagrangian/mappo_lagrangian/scripts && bash ant_2x4.sh 2 --keep_checkpoints   # [seed]
```

## Evaluating checkpoints

```bash
cd SafeMARL/examples
python evaluate.py results/mujoco/Ant-v2-2x4/madac/default/seed-00002-<time> --episodes 20 --processes 10
python evaluate.py <run_dir> --checkpoints 'models/step_*/madac.pt' --episodes 20 --processes 10
```

## Citation

```bibtex
@article{li2024safe,
  title={Safe Multi-Agent Reinforcement Learning with Convergence to Generalized Nash Equilibrium},
  author={Li, Zeyang and Azizan, Navid},
  journal={arXiv preprint arXiv:2411.15036},
  year={2024}
}
```

## Acknowledgements

This codebase is built on [HARL](https://github.com/PKU-MARL/HARL) and
[Multi-Agent-Constrained-Policy-Optimisation](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation).
We thank the authors of both repositories.
