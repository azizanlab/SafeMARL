#!/bin/sh
env="mujoco"
scenario="HalfCheetah-v2"
agent_conf="3x2"
algo="macpo"
exp="${agent_conf}"
seed=""; case ${1:-} in ''|*[!0-9]*) ;; *) seed=$1; shift ;; esac

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed:-drawn}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train/train_mujoco.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario ${scenario} --agent_conf ${agent_conf} --lr 9e-5 --critic_lr 5e-3 --std_x_coef 1 --std_y_coef 5e-1 ${seed:+--seed ${seed}} --n_training_threads 4 --n_rollout_threads 10 --num_mini_batch 40 --episode_length 1000 --num_env_steps 2000000 --ppo_epoch 1 --use_value_active_masks --kl_threshold 0.0065 --safety_bound 10.0 --line_search_fraction 0.5 --fraction_coef 0.27 --use_eval --n_eval_rollout_threads 10 --eval_episodes 20 --eval_interval 2 --done_config never "$@"
