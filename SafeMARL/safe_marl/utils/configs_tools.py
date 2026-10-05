import json
import os
import time

import yaml


def get_defaults_yaml_args(algo, env):
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")
    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    matched = set()

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            elif k in dict1:
                dict2[k] = dict1[k]
                matched.add(k)

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)
    unknown = sorted(set(unparsed_dict) - matched)
    if unknown:
        raise ValueError(f"command-line overrides match no config entry: {unknown}")


def get_task_name(env, env_args):
    return f"{env_args['scenario']}-{env_args['agent_conf']}"


def init_dir(env, env_args, algo, exp_name, seed, logger_path):
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        logger_path, env, task, algo, exp_name, "-".join(["seed-{:0>5}".format(seed), hms_time])
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter

    writter = SummaryWriter(log_path)
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter


def convert_json(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        pass
    if isinstance(obj, dict):
        return {convert_json(k): convert_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_json(x) for x in obj]
    if hasattr(obj, "__name__") and "lambda" not in obj.__name__:
        return convert_json(obj.__name__)
    if hasattr(obj, "__dict__") and obj.__dict__:
        return {str(obj): {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}}
    return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    output = json.dumps(convert_json(config), separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
