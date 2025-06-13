import argparse
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import PROJECT_SPECS, HARDWARE_SPECS_DICT


SWEEP_NAME_DEFAULT = ''
project = 'moe'
MODELS = [
    'olmo2_100M_moe',
]

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    add_time_to_name='front',
    add_model_to_name=None,
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    restart=False,
    include_jobs=None,
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
        SPECS.update(HARDWARE_SPECS_DICT[model][partition])
        grids = {
            model_sweep_name: {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
                # combined with the subgrids
                "main_grid": { 
                    "model_name": [model],
                    "save_root": [f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}"],
                    "per_gpu_batch_size": [SPECS["per_gpu_batch_size"]],
                    'train_module': {
                        'optim': {
                            'lr': [1e-4, 2e-4, 3e-4],
                        },
                        "scheduler": {
                            "units": ["steps"],
                            "warmup": [2000],
                        }
                    },
                    "trainer": {
                        "max_duration": {
                            "value": [200000000],
                            "unit": ["tokens"],
                        },
                    },
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    # "4x1c2": {"moe_num_experts_list": ["4"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["2"]},
                    # "8x1c2": {"moe_num_experts_list": ["8"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["2"]},
                    # "16x1c2": {"moe_num_experts_list": ["16"], "moe_hidden_multipliers_list": ["1"], "moe_router_top_ks_list": ["2"]},
                    "4,8x1,0.5c1,2": {"moe_num_experts_list": ["4,8"], "moe_hidden_multipliers_list": ["1,0.5"], "moe_router_top_ks_list": ["1,2"]},
                }
            },
        }

        for sweep_name, grid in grids.items():
            # grid.update(SPECS)
            run_grid(
                grid,
                sweep_name=sweep_name,
                name_keys=SPECS.get("NAME_KEYS", []),
                user=os.environ['USER'],
                prefix=SPECS['COMMAND_PREFIX'],
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=('1:00:00' if debug else SPECS.get("JOBTIME", '24:00:00')),            
                include_job_id=False,
                hide_keys={},
                hashname=False,
                saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}",
                logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{SWEEP_NAME}",
                mem_gb=SPECS["MEM_GB"],
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
                add_name='end',
                dependencies=[],
                repo_name="olmoe",
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                include_jobs=include_jobs,
                restart=restart,
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
    parser.add_argument('-i', '--include-jobs', type=str, default=None)
    parser.add_argument('--restart', action='store_true')

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode, 
        account=args.slurm_account, 
        partition=args.slurm_partition,
        restart=args.restart,
        include_jobs=(args.include_jobs.split(",") if args.include_jobs else None),
    )
