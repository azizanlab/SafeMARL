from setuptools import setup

setup(
    name="safe_marl",
    version="2.0.0",
    description="MADAC (multi-agent dual actor-critic) and HARL baselines for Safety Multi-Agent MuJoCo.",
    packages=["safe_marl"],
    package_data={"safe_marl": ["configs/algos_cfgs/*.yaml", "configs/envs_cfgs/*.yaml"]},
    python_requires=">=3.9,<3.10",
)
