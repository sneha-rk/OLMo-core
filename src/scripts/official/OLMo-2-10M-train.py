"""
Official training script for OLMo-2-0325-32B, meant to be launched with torchrun.
"""

import sys
import os
import logging
import traceback
from dataclasses import dataclass
from typing import List, cast

import torch
import torch.distributed as dist

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 128

# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.
DATA_ROOT = "https://olmo-data.org/"
WORK_DIR = "/gscratch/zlab/snehark/OLMo-core/data/"
SAVE_ROOT = "/gscratch/zlab/snehark/OLMo-core/runs"  # NOTE: change this to what you want


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 12536


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    model_config = TransformerConfig.olmo2_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),  # a little bigger than actual vocab size to make it a multiple of 128
    )

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DataMix.OLMoE_test,
        tokenizer=tokenizer_config,
        mix_base_dir=DATA_ROOT,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=max(128, SEQUENCE_LENGTH),
        work_dir=WORK_DIR,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=2 * SEQUENCE_LENGTH,
        seed=34521,
        num_workers=2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=128,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=6e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            compile=True,
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=1,  # NOTE: tune this
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            TransformerActivationCheckpointingMode.full
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # If you have 1024 GPUs, you can run slightly faster with a different config.
    if get_world_size() >= 1024:
        train_module_config.rank_microbatch_size //= 2
        train_module_config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{SAVE_ROOT}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(int(6.5e12)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=200,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=WORK_DIR,
                    sequence_length=dataset_config.effective_sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=WORK_DIR,
                ),
                eval_interval=10,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    # MMLU for backwards compatibility
                    # "mmlu_stem_mc_5shot",
                    # "mmlu_humanities_mc_5shot",
                    # "mmlu_social_sciences_mc_5shot",
                    # "mmlu_other_mc_5shot",
                    # MMLU test
                    "mmlu_stem_mc_5shot_test",
                    "mmlu_humanities_mc_5shot_test",
                    "mmlu_social_sciences_mc_5shot_test",
                    "mmlu_other_mc_5shot_test",
                    ## Core 12 tasks for backwards compatibility
                    # "arc_challenge",
                    # "arc_easy",
                    # "basic_arithmetic",
                    "boolq",
                    # "commonsense_qa",
                    # "copa",
                    "hellaswag",
                    # "openbook_qa",
                    # "piqa",
                    # "sciq",
                    # "social_iqa",
                    # "winogrande",
                    ## Core 12 tasks 5-shot
                    # "arc_challenge_rc_5shot",
                    # "arc_easy_rc_5shot",
                    ## "basic_arithmetic_rc_5shot",  # doesn't exist
                    ## "boolq_rc_5shot",  # we don't like it
                    # "csqa_rc_5shot",
                    ## "copa_rc_5shot",  # doesn't exist
                    # "hellaswag_rc_5shot",
                    # "openbookqa_rc_5shot",
                    # "piqa_rc_5shot",
                    ## "sciq_rc_5shot",  # doesn't exist
                    # "socialiqa_rc_5shot",
                    # "winogrande_rc_5shot",
                    ## New in-loop evals
                    # "arc_challenge_val_rc_5shot",
                    # "arc_challenge_val_mc_5shot",
                    # "arc_challenge_test_rc_5shot",
                    # # "arc_challenge_test_mc_5shot",
                    # # "arc_easy_val_rc_5shot",
                    # # "arc_easy_val_mc_5shot",
                    # "arc_easy_test_rc_5shot",
                    # # "arc_easy_test_mc_5shot",
                    # # "boolq_val_rc_5shot",
                    # # "boolq_val_mc_5shot",
                    # "csqa_val_rc_5shot",
                    # # "csqa_val_mc_5shot",
                    # "hellaswag_val_rc_5shot",
                    # # "hellaswag_val_mc_5shot",
                    # # "openbookqa_val_rc_5shot",
                    # # "openbookqa_val_mc_5shot",
                    # "openbookqa_test_rc_5shot",
                    # # "openbookqa_test_mc_5shot",
                    # "piqa_val_rc_5shot",
                    # # "piqa_val_mc_5shot",
                    # "socialiqa_val_rc_5shot",
                    # "socialiqa_val_mc_5shot",
                    # "winogrande_val_rc_5shot",
                    # "winogrande_val_mc_5shot",
                    # "mmlu_stem_val_rc_5shot",
                    # "mmlu_stem_val_mc_5shot",
                    # "mmlu_humanities_val_rc_5shot",
                    # "mmlu_humanities_val_mc_5shot",
                    # "mmlu_social_sciences_val_rc_5shot",
                    # "mmlu_social_sciences_val_mc_5shot",
                    # "mmlu_other_val_rc_5shot",
                    # "mmlu_other_val_mc_5shot",
                ],
                tokenizer=tokenizer_config,
                eval_interval=10,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, overrides: List[str]):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Log environment info
        logger.info(f"World size: {dist.get_world_size()}")
        # # logger.info(f"Local rank: {dist.get_local_rank()}")
        # logger.info(f"Global rank: {dist.get_rank()}")
        # logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        config = build_config(run_name, overrides)
        logger.info("Config built successfully")

        # Set RNG states on all devices.
        seed_all(config.init_seed)
        logger.info("RNG states set")

        # Build components.
        logger.info("Building model...")
        model = config.model.build(init_device="meta")
        logger.info("Building train module...")
        train_module = config.train_module.build(model)
        logger.info("Building dataset...")
        dataset = config.dataset.build()
        logger.info("Building data loader...")
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        logger.info("Building trainer...")
        trainer = config.trainer.build(train_module, data_loader)
        logger.info("All components built successfully")

        # Save config to W&B and each checkpoint dir.
        config_dict = config.as_config_dict()
        cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

        # Train.
        logger.info("Starting training...")
        trainer.fit()
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        # Ensure the error propagates up
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    try:
        main(run_name, overrides=overrides)
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise
    finally:
        teardown_training_environment()
