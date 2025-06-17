"""
"""

import argparse
import sys
import os
import logging
import traceback
from dataclasses import dataclass, field
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
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, AdamWConfig
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
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from constants import (
    PROJECT_SPECS,
)

MODEL_CONFIG_LOOKUP = {
    "olmo2_10M": TransformerConfig.olmo2_10M,
    "olmo2_20M": TransformerConfig.olmo2_20M,
    "olmo2_50M": TransformerConfig.olmo2_50M,
    "olmo2_100M": TransformerConfig.olmo2_100M,
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

USER_PROJECT_SPECS = PROJECT_SPECS[os.environ.get('USER', 'default')]

# This will read stream data from the public endpoints by default, but that might be a lot slower
# than reading data locally.

@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int


def build_config(
        run_name: str, 
        num_gpus: int = torch.cuda.device_count(),
        tokenizer_name: str = "dolma2", 
        model_name: str = "olmo2_100M_moe_32_16",
        train_datamix_name: str = "OLMoE_mix_0824",
        valid_datamix_name: str = "v3_small_ppl_validation",
        data_root: str = USER_PROJECT_SPECS['DATAROOT'],
        save_root: str = USER_PROJECT_SPECS['DEFAULT_SAVE_PATH'],
        valid_data_dir: str = USER_PROJECT_SPECS['VALID_DATA_DIR'],
        data_work_dir: str = USER_PROJECT_SPECS['DATA_WORK_DIR'],
        sequence_length: int = 2048,
        global_batch_size: int = 512, # 512 sequences total
        per_gpu_batch_size: int = 4,  # 4 sequences per GPU
        num_data_workers: int = 2,
        train_tokens: int = 200_000_000,
        warmup_steps: int = 2000,
        save_interval: int = 1000, 
        ephemeral_save_interval: int = 200,
        eval_interval: int = 10000,
        metrics_collect_interval: int = 10,
        lr: float = 4e-4,
        embedding_weight_decay: float = 0.1,
        weight_decay: float = 0.0,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        z_loss_multiplier: float = 1e-5,
        moe_num_experts_list: List[int] = [32, 64],
        moe_hidden_multipliers_list: List[int] = [1024, 2048],
        moe_router_top_ks_list: List[int] = [4, 8],
        max_grad_norm: float = 1.0,
        init_seed: int = 12536,
        wandb_entity: str = USER_PROJECT_SPECS['WANDB_ENTITY'],
        wandb_project: str = USER_PROJECT_SPECS['WANDB_PROJECT'],
        overrides: List[str] = [],
    ) -> ExperimentConfig:
    tokenizer_config = TOKENIZER_LOOKUP[tokenizer_name]()

    model_config = MODEL_CONFIG_LOOKUP[model_name](
        vocab_size=tokenizer_config.padded_vocab_size(),
        use_moe=False if moe_num_experts_list == [1] else True,
        num_experts_list=moe_num_experts_list,
        hidden_multipliers_list=moe_hidden_multipliers_list,
        router_top_ks_list=moe_router_top_ks_list,
    )

    dataset_config = NumpyDatasetConfig.from_data_mix(
        DATAMIX_LOOKUP[train_datamix_name],
        tokenizer=tokenizer_config,
        mix_base_dir=data_root,
        sequence_length=sequence_length,
        max_target_sequence_length=max(4096, sequence_length),
        work_dir=data_work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=global_batch_size * sequence_length,
        seed=34521,
        num_workers=num_data_workers,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=per_gpu_batch_size * sequence_length,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=AdamWConfig(
            lr=lr,
            weight_decay=weight_decay,
            betas=adam_betas,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=embedding_weight_decay))
            ],
            fused=True,
        ),
        scheduler=CosWithWarmup(warmup_steps=warmup_steps),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,  
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,  # Added from small-moe.py
        ),
        z_loss_multiplier=z_loss_multiplier,
        max_grad_norm=max_grad_norm,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{save_root}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=metrics_collect_interval,
            cancel_check_interval=1,  # Updated from small-moe.py
            max_duration=Duration.tokens(train_tokens),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=save_interval,
                ephemeral_save_interval=ephemeral_save_interval,
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
                entity=wandb_entity,
                project=wandb_project,
                cancel_check_interval=10,
                enabled=True,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.from_data_mix(
                    DATAMIX_LOOKUP[valid_datamix_name],
                    name=NumpyDatasetType.padded_fsl,
                    mix_base_dir=valid_data_dir,
                    sequence_length=dataset_config.effective_sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=data_work_dir,
                ),
                eval_interval=eval_interval,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    "mmlu_stem_mc_5shot_test",
                    "mmlu_humanities_mc_5shot_test",
                    "mmlu_social_sciences_mc_5shot_test",
                    "mmlu_other_mc_5shot_test",
                    "boolq",
                    "hellaswag",
                ],
                tokenizer=tokenizer_config,
                eval_interval=eval_interval,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=init_seed,
    ).merge(overrides)


def main(
    args: argparse.Namespace,
    overrides: List[str]
) -> None:
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
        
        config = build_config(
            args.run_name, 
            num_gpus=args.num_gpus,
            tokenizer_name=args.tokenizer_name,
            model_name=args.model_name,
            train_datamix_name=args.train_datamix_name,
            valid_datamix_name=args.valid_datamix_name,
            data_root=args.data_root,
            save_root=args.save_root,
            valid_data_dir=args.valid_data_dir,
            data_work_dir=args.data_work_dir,
            sequence_length=args.sequence_length,
            global_batch_size=args.global_batch_size,
            per_gpu_batch_size=args.per_gpu_batch_size,
            moe_hidden_multipliers_list=[float(v) for v in args.moe_hidden_multipliers_list.split(',')], 
            moe_num_experts_list=[int(v) for v in args.moe_num_experts_list.split(',')], 
            moe_router_top_ks_list=[int(v) for v in args.moe_router_top_ks_list.split(',')], 
            overrides=overrides)
        # config = build_config(run_name)
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
    # if len(sys.argv) < 2:
    #     print(f"Usage: torchrun [OPTS..] {sys.argv[0]} run_name [OVERRIDES...]")
    #     sys.exit(1)


    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Name of the run")
    parser.add_argument("num_gpus", type=int, default=torch.cuda.device_count(), nargs='?', help="Number of GPUs to use")
    parser.add_argument("--tokenizer_name", type=str, default="dolma2", help="Name of the tokenizer to use")
    parser.add_argument("--model_name", type=str, default="olmo2_100M_moe_32_16", help="Name of the model configuration to use")
    parser.add_argument("--train_datamix_name", type=str, default="OLMoE_mix_0824", help="Name of the training data mix")
    parser.add_argument("--valid_datamix_name", type=str, default="v3_small_ppl_validation", help="Name of the validation data mix")
    parser.add_argument("--data_root", type=str, default="https://olmo-data.org/", help="Root URL for the data")
    parser.add_argument("--save_root", type=str, default=USER_PROJECT_SPECS['DEFAULT_SAVE_PATH'], help="Parent directory for saving the model")
    parser.add_argument("--valid_data_dir", type=str, default=USER_PROJECT_SPECS['VALID_DATA_DIR'], help="Directory for validation data")
    parser.add_argument("--data_work_dir", type=str, default=USER_PROJECT_SPECS['DATA_WORK_DIR'], help="Working directory for data")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length for training")
    parser.add_argument("--global_batch_size", type=int, default=512, help="Batch size total")
    parser.add_argument("--per_gpu_batch_size", type=int, default=512, help="Batch size per GPU")
    parser.add_argument("--moe_num_experts_list", type=str, default="32,64", help="List of number of experts for MoE")
    parser.add_argument("--moe_hidden_multipliers_list", type=str, default="1024,2048", help="List of hidden sizes multiplers for MoE")
    parser.add_argument("--moe_router_top_ks_list", type=str, default="4,8", help="List of router top-k values for MoE")
    args, overrides = parser.parse_known_args()

    # run_name, *overrides = sys.argv[1:]

    print(overrides)
    prepare_training_environment()
    try:
        main(args, overrides=overrides)
        # main(run_name)
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        raise
    finally:
        teardown_training_environment()
