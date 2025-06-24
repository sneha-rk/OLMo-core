import argparse
import json
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS
from utils import dict_update


SWEEP_NAME_DEFAULT = ''
project = 'moe'
MODELS = [
    # 'olmo2_200M',
    'olmo2_100M',
    # 'olmo2_50M',
    # 'olmo2_20M',
    # 'olmo2_10M',
]

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    path_to_sweep=None,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    job_time='24:00:00',
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB"],
    filter_succeeded=True,
    filter_running=True,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = '1:00:00' if debug else job_time

    if path_to_sweep:
        path_to_sweep = path_to_sweep.rstrip('/')
        model_sweep_name = os.path.basename(path_to_sweep)
        path_to_grid_file = os.path.join(path_to_sweep, 'grid.json')
        path_to_specs = os.path.join(path_to_sweep, 'specs.json')
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, 'r'))
        model = grid.get('main_grid', {}).get('model_name', [None])[0]

        SPECS = dict_update(copy(PROJECT_SPECS[os.environ.get('USER')]), HARDWARE_SPECS_DICT['all'])
        SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model][partition])
        SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
        SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
        SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

        if os.path.exists(path_to_specs):
            old_specs = json.load(open(path_to_specs, 'r'))
            for key in old_specs:
                if key not in ignore_specs_check_keys:
                    assert SPECS.get(key) == old_specs[key], f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"
        
        run_grid(
            grid,
            default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
            sweep_name=model_sweep_name,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS['COMMAND_PREFIX'],
            gpus=SPECS['NUM_GPUS'],
            cpus=SPECS["NUM_CPUS"],
            nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),            
            include_job_id=False,
            hashname=False,
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            mem_gb=SPECS["MEM_GB"],
            requeue=True,
            data_parallel=False,
            comment=None,
            copy_env=True,
            copy_dirs=[],
            max_num_jobs=None,
            num_copies=1,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            dependencies=[],
            repo_name="olmoe-core",
            conda_env_name=SPECS.get("CONDA_ENV_NAME"),
            include_jobs_indices=include_jobs_indices,
            filter_succeeded=filter_succeeded,
            filter_running=filter_running,
            # append_to_sbatch_str=None,
        )
        
    else:
        SWEEP_NAME = sweep_name
        if add_time_to_name == 'front':
            time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        for model in MODELS:
            model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == 'end' else SWEEP_NAME
            SPECS = dict_update(copy(PROJECT_SPECS[os.environ.get('USER')]), HARDWARE_SPECS_DICT['all'])
            SPECS = dict_update(SPECS, HARDWARE_SPECS_DICT[model][partition])
            SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
            SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
            SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]
            grid = {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
                # combined with the subgrids
                "main_grid": { 
                    "model_name": [model],
                    "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}"],
                    'train_module': {
                        'optim': {
                            'lr': [4e-4],
                        },
                    },
                    "trainer": {
                        "max_duration": {
                            # "value": [2000000000], #just testing
                        },
                    },
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    # "e1x1c1": {"moe_num_experts_list": ["1"]},
                    # "e2x1c1": {"moe_num_experts_list": ["2"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                    "e4x1c1": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                    # "e8x1c1": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                    "e16x1c1": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                    # "e4x0.5c2": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                    # "e8x0.5c2": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                    # "e16x0.5c2": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                    # "e8x0.25c4": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"]},
                    # "e16x0.25c4": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"]},
                    "e16x0.125c8": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.125"], "moe_router_top_ks_list": ["8"]},
                    # "e4,8x0.5,0.25c1,2": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"]},
                    "e8,16x0.25,0.125c2,4": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"]},
                    "e4,16x0.5,0.125c1,4": {"moe_num_experts_list": ["4,16"], "moe_hidden_multipliers_list": ["0.5,0.125"], "moe_router_top_ks_list": ["1,4"]},
                },
            }

            run_grid(
                grid,
                default_grid=dict_update(copy(MODEL_HP_DEFAULTS['all']), MODEL_HP_DEFAULTS.get(model, {})),
                sweep_name=model_sweep_name,
                specs=SPECS,
                name_keys=SPECS.get("NAME_KEYS", []),
                prefix=SPECS['COMMAND_PREFIX'],
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),      
                include_job_id=False,
                hashname=False,
                saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                mem_gb=SPECS["MEM_GB"],
                requeue=True,
                data_parallel=False,
                comment=None,
                copy_env=True,
                copy_dirs=[],
                max_num_jobs=None,
                num_copies=1,
                job_id_start=1,
                debug_mode=DEBUG_MODE,
                dry_mode=DRY_MODE,
                dependencies=[],
                repo_name="olmoe-core",
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                include_jobs_indices=include_jobs_indices,
                filter_succeeded=filter_succeeded,
                filter_running=filter_running,
                # append_to_sbatch_str=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('--path-to-sweep', type=str, default=None, help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('-t', '--job-time', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem', type=str)
    parser.add_argument('-i', '--include-jobs-indices', type=str, default=None)
    parser.add_argument('-nf', '--no-filter', action='store_true', help="If set, will not filter out jobs that have already been run in the sweep. Useful for debugging.")

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        path_to_sweep=args.path_to_sweep,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode,
        account=args.slurm_account, 
        partition=args.slurm_partition,
        job_time=args.job_time,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        include_jobs_indices=([int(i) for i in args.include_jobs_indices.split(",")] if args.include_jobs_indices else None),
        filter_running=not args.no_filter,
        filter_succeeded=True,
    )
