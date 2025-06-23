"""
"""

import os

#  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])

MODEL_HP_DEFAULTS = {
    "all": {
        "global_batch_size": [512],
        "sequence_length": [2048],
        "train_module": {
            "optim": {
                "weight_decay": [0.1],
            },
            "scheduler": {
                "units": ["steps"],
                "warmup_steps": [2000],
            },
        },
        "trainer": {
            "max_duration": {
                "unit": ["tokens"],
            },
        },
    },
    "olmo2_10M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [200000000],
            },
        },
    },
    "olmo2_20M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [400000000],
            },
        },
    },
    "olmo2_50M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [1000000000],
            },
        },
    },
    "olmo2_100M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [2000000000],
            },
        },
    },
    "olmo2_200M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [4000000000],
            },
        },
    },
    "olmo2_400M": {
        "train_module": {
            "optim": {
                "lr": [4e-3],
            },
        },
        "trainer": {
            "max_duration": {
                "value": [8000000000],
            },
        },
    }
}

PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        'DATA_WORK_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
        'VALID_DATA_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
        "WANDB_PROJECT": "moe",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "moe",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-a40,gpu-l40",
        "COMMAND_PREFIX": f"{DEFAULT_DIR_PATH}/ml/scripts/single_train_launch.py",
        "NUM_GPUS": 4,
        "MODEL": [],
        "DATAROOT": "https://olmo-data.org/",
        "DATA_DIR": "/gscratch/zlab/snehark/OLMo-core/data/",
        "NAME_KEYS": [],
    },
    "snehark": {
        ## TODO @snehark: Update these paths to your local setup
    }
}

BATCH_SIZE = 512
HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 4,
        "NUM_CPUS": 4,
        "MEM_GB": 128,
    },
    "olmo2_10M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a100": {
        }, 
        "gpu-rtx6k": {
            "per_gpu_batch_size": 16,
        }, 
    },
    "olmo2_20M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a100": {
        }, 
    },
    "olmo2_50M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a100": {
        }, 
    },
    "olmo2_100M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 16,
        }, 
        "gpu-a100": {
        }, 
    },
    "olmo2_200M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-h200": {
            "per_gpu_batch_size": 16,

        }, 
    },
    "olmo2_400M": { 
        "gpu-l40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-a40": {
            "per_gpu_batch_size": 8,
        }, 
        "gpu-a100": {
        }, 
    }
}
