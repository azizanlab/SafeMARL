from setuptools import setup

setup(
    name="safety_ma_mujoco",
    version="1.0.0",
    description="Safety Multi-Agent MuJoCo benchmarks (HalfCheetah, Walker2d, Ant) with state-wise constraints.",
    packages=["safety_ma_mujoco"],
    package_data={"safety_ma_mujoco": ["assets/*.xml"]},
    install_requires=["numpy", "gym==0.17.2", "mujoco-py==2.0.2.8", "cloudpickle"],
)
