"""
"""

import os

from olmo_core.data import (
    DataMix,
    TokenizerConfig,
)
from olmo_core.nn.transformer import (
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, AdamWConfig

DEFAULT_DIR_PATH ='/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-4])
DEFAULT_SAVE_PATH = os.path.join(DEFAULT_DIR_PATH, 'ml', 'models')


DATA_WORK_DIR = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data"

VALID_DATA_DIR = "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed"

#  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

MODEL_CONFIG_LOOKUP = {
    "olmo2_100M_moe_32": TransformerConfig.olmo2_100M_moe_32,
    "olmo2_100M_moe_16": TransformerConfig.olmo2_100M_moe_16,
    "olmo2_100M_moe_32_16": TransformerConfig.olmo2_100M_moe_32_16,
}

TOKENIZER_LOOKUP = {
    "dolma2": TokenizerConfig.dolma2,
    "gpt_neox_olmo_dolma_v1_5": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
}

DATAMIX_LOOKUP = {
    "OLMoE_mix_1124": DataMix.OLMoE_mix_1124,
    "OLMoE_mix_0824": DataMix.OLMoE_mix_0824,
    "v3_small_ppl_validation": DataMix.v3_small_ppl_validation,
}


PROJECT_SPECS_DICT = {
    "moe":{
        "default": {
            "WANDB_PROJECT": "moe",
            "WANDB_ENTITY": "ml-moe",
            "CONDA_ENV_NAME": "moe",
            "PROJECT_DIR": DEFAULT_DIR_PATH,
            "SLURM_ACCOUNT": "zlab",
            "SLURM_PARTITION": "gpu-a40,gpu-l40",
            "COMMAND_PREFIX": f"python {DEFAULT_DIR_PATH}/OLMo-core/ml/scripts/single_train_launch.py",
            "NUM_GPUS": 4,
            "MODEL": [],
            "DATAROOT": "https://olmo-data.org/",
            "DATA_DIR": "/gscratch/zlab/snehark/OLMo-core/data/",
            "MODEL_DIR": DEFAULT_SAVE_PATH,
            "DEFAULT_CONFIG_FILE": f"{DEFAULT_DIR_PATH}/configs/ml/moe_default.json",
            "NAME_KEYS": [["model", "moe_num_experts"], ["model", "moe_top_k"]],
        },
        "25m": { 
            "MODEL_CONFIG_FILE": f"{DEFAULT_DIR_PATH}/configs/ml/moe_25m.json"
        },
    },
}
HARDWARE_SPECS_DICT = {
    "25m": { 
        "gpu-l40": {
            "NUM_GPUS": 4,
            "NUM_CPUS": 4,
            "MEM_GB": 128,
            "JOBTIME": '12:00:00',
            "device_train_microbatch_size": 4,
            "device_eval_batch_size": 4,
        }, 
        "gpu-a40": {
            "NUM_GPUS": 4,
            "NUM_CPUS": 3,
            "MEM_GB": 128,
            "JOBTIME": '12:00:00',
            "device_train_microbatch_size": 4,
            "device_eval_batch_size": 4,
            "CONDA_ENV_NAME": "olmoe",

        }, 
        "gpu-a100": {
            "NUM_GPUS": 1,
            "NUM_CPUS": 4,
            "MEM_GB": 128,
            "JOBTIME": '1:00:00',
            "device_train_microbatch_size": 4,
            "device_eval_batch_size": 4,
        }, 
        "gpu-2080ti": {
            "NUM_GPUS": 4,
            "NUM_CPUS": 4,
            "MEM_GB": 128,
            "JOBTIME": '12:00:00',
            "device_train_microbatch_size": 4,
            "device_eval_batch_size": 4,
        }, 
    }
}
