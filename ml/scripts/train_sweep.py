import argparse
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT, MODEL_HP_DEFAULTS


SWEEP_NAME_DEFAULT = ''
project = 'moe'
MODELS = [
    # 'olmo2_100M',
    # 'olmo2_50M',
    # 'olmo2_20M',
    'olmo2_10M',
]

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs=None,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    SWEEP_NAME = sweep_name
    if add_time_to_name == 'front':
        time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str

    for model in MODELS:
        model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == 'end' else SWEEP_NAME
        SPECS = copy(PROJECT_SPECS[os.environ.get('USER')])
        SPECS.update(HARDWARE_SPECS_DICT['all'])
        SPECS.update(HARDWARE_SPECS_DICT[model][partition])
        grid = {
            # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
            # combined with the subgrids
            "main_grid": { 
                "model_name": [model],
                "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}"],
                # "per_gpu_batch_size": [SPECS["per_gpu_batch_size"]],
                "per_gpu_batch_size": [8],
                'train_module': {
                    'optim': {
                        'lr': [2e-2, 4e-2],
                    },
                },
                "trainer": {
                    "max_duration": {
                        "value": [20000000], #just testing
                    },
                },
            },
            # allows you to bundle multiple hyperparameters together
            "subgrids": {
                "e1x1c1": {"moe_num_experts_list": ["1"]},
                # "e4x1c1": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                # "e8x1c1": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                "e16x1c1": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["1"]},
                # "e4x0.5c2": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                # "e8x0.5c2": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                # "e16x0.5c2": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.5"], "moe_router_top_ks_list": ["2"]},
                # "e8x0.25c4": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"]},
                # "e16x0.25c4": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["0.25"], "moe_router_top_ks_list": ["4"]},
                # "e4,8x0.5,0.25c1,2": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["0.5,0.25"], "moe_router_top_ks_list": ["1,2"]},
                # "e8,16x0.25,0.125c2,4": {"moe_num_experts_list": ["8,16"], "moe_hidden_multipliers_list": ["0.25,0.125"], "moe_router_top_ks_list": ["2,4"]},
            },
        }

        default_grid = copy(MODEL_HP_DEFAULTS['all'])
        default_grid.update(MODEL_HP_DEFAULTS.get(model, {}))
        run_grid(
            grid,
            default_grid=default_grid,
            sweep_name=model_sweep_name,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=SPECS['COMMAND_PREFIX'],
            gpus=(gpus or SPECS['NUM_GPUS']),
            cpus=(cpus or SPECS["NUM_CPUS"]),
            nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=('1:00:00' if debug else SPECS.get("JOBTIME", '24:00:00')),            
            include_job_id=False,
            hashname=False,
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}",
            mem_gb=(mem or SPECS["MEM_GB"]),
            requeue=True,
            data_parallel=False,
            comment=None,
            volta=False,
            volta32=False,
            copy_env=True,
            copy_dirs=[],
            max_num_jobs=None,
            num_copies=1,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            dependencies=[],
            repo_name="olmoe",
            conda_env_name=SPECS.get("CONDA_ENV_NAME"),
            include_jobs=include_jobs,
            # append_to_sbatch_str=None,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem', type=str)
    parser.add_argument('-i', '--include-jobs', type=str, default=None)

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode, 
        account=args.slurm_account, 
        partition=args.slurm_partition,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        include_jobs=(args.include_jobs.split(",") if args.include_jobs else None),
    )
