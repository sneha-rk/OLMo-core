```
env_name=paper_olmoe
mamba create -n $env_name python=3.12 -y
mamba activate $env_name
export DIR=/gscratch/zlab/${USER}/gitfiles/paper_olmoe
export OLMOE_DIR=$DIR/olmo
export MEGABLOCKS_DIR=$DIR/megablocks
export DATA_DIR=$OLMOE_DIR/data
module load cuda/11.8.0
module load gcc/11.2.0
mamba activate paper_olmoe
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge ninja -y 
cd $OLMOE_DIR
pip install --no-build-isolation -e . 
cd $MEGABLOCKS_DIR
pip install --no-build-isolation . 
```

## Setup

Set your desired directories:
```
export DIR=/gscratch/zlab/${USER}/gitfiles
export OLMOE_DIR=$DIR/olmoe
export MEGABLOCKS_DIR=$DIR/megablocks
export DATA_DIR=$OLMOE_DIR/data
export RAW_DATA_DIR=$DATA_DIR/raw/OLMoE-mix-0924/data
export PREPROCESSED_VAL_DATA_DIR=$DATA_DIR/preprocessed/val
```

Create a conda env:
```
env_name=olmoe
mamba create -n $env_name python=3.12 -y
mamba activate $env_name
```

Install pytorch 2.6, following instructions [here](https://pytorch.org/get-started)
On hyak:
```
module unload cuda/12.4.1
pip3 install torch torchvision torchaudio
```

Clone OLMoE code and install
```
git clone https://github.com/hadasah/olmoe $OLMOE_DIR
cd $OLMOE_DIR
pip install -e .
```

Clone megablocks and install
```
git clone https://github.com/hadasah/megablocks $MEGABLOCKS_DIR
cd $MEGABLOCKS_DIR
pip install -e .
```

Optionally, set up defaults and shortcuts in .bashrc
```
echo -e "\n\n # OLMoE \n export OLMOE_DIR=$OLMOE_DIR \n alias moe=\"conda activate $env_name; cd $OLMOE_DIR\"\n" >> ~/.bashrc

```

## Build OLMoE data

You can download training data files directly from github, but it times out easily on hyak:
```
mkdir -p $DATA_DIR/raw 
cd $DATA_DIR/raw
git clone https://huggingface.co/datasets/allenai/OLMoE-mix-0924
```

Alternatively, this script manually grabs each training data file individually:
```
bash $OLMOE_DIR/ml/scripts/get_data.sh
```

The above script also grabs the validation data for language modeling/perplexity. I didn't download any of the task files for now.

If you don't use the above, you can download the validation files thus:
```
VALID_URL=https://olmo-data.org/eval-data/perplexity/v3_small_gptneox20b
declare -a eval_subsets=(c4_en dolma_books dolma_common-crawl dolma_pes2o dolma_reddit dolma_stack dolma_wiki ice m2d2_s2orc pile wikitext_103);
for s in "${eval_subsets[@]}"; do mkdir -p $PREPROCESSED_VAL_DATA_DIR/${s}/val ; wget -O $PREPROCESSED_VAL_DATA_DIR/${s}/val/part-0-00000.npy -c ${VALID_URL}/${s}/val/part-0-00000.npy ; done
```

Preprocess train data with the olmo tokenizer
```
for d in $RAW_DATA_DIR/*/ ; do
    mkdir -p ${d//raw/preprocessed} ;
    dolma tokens \
    --documents ${d}/* \
    --destination ${d//raw/preprocessed}/ \
    --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
    --max_size '2_147_483_648' \
    --seed 0 \
    --tokenizer.eos_token_id 50279 \
    --tokenizer.pad_token_id 1 \
    --processes 20 ;
done
```
Formatted to run on command line:
```
for d in $RAW_DATA_DIR/*/ ; do mkdir -p ${d//raw/preprocessed} ; dolma tokens --documents ${d}/* --destination ${d//raw/preprocessed}/ --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' --max_size '2_147_483_648' --seed 0 --tokenizer.eos_token_id 50279 --tokenizer.pad_token_id 1 --processes 20 ; done
```

## Launch jobs

Launch a sweep:
```
python $OLMOE_DIR/ml/scripts/train_sweep.py -a bdata -p gpu-a40 --sweep-name test
```
For some reason, olmo code won't run on L40 machines for now.