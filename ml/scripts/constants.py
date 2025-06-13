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

#  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-4])


MODEL_CONFIG_LOOKUP = {
    "olmo2_100M_moe": TransformerConfig.olmo2_100M_moe,
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


PROJECT_SPECS = {
    'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'ml', 'models'),
    'DATA_WORK_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data",
    'VALID_DATA_DIR': "/gscratch/zlab/margsli/gitfiles/olmoe-core/OLMo-core/ml/data/preprocessed",
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
    "DEFAULT_CONFIG_FILE": f"{DEFAULT_DIR_PATH}/configs/ml/moe_default.json",
    "NAME_KEYS": ["model_name"],
}

HARDWARE_SPECS_DICT = {
    "olmo2_100M_moe": { 
        "gpu-l40": {
            "NUM_GPUS": 4,
            "NUM_CPUS": 4,
            "MEM_GB": 128,
            "JOBTIME": '12:00:00',
        }, 
        "gpu-a40": {
            "NUM_GPUS": 4,
            "NUM_CPUS": 3,
            "MEM_GB": 128,
        }, 
        "gpu-a100": {
            "NUM_GPUS": 1,
            "NUM_CPUS": 4,
            "MEM_GB": 128,
        }, 

    }
}
