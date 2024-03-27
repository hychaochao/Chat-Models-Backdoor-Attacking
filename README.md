# Exploring Backdoor Attacks on Chat Models
| [**Paper**]() | [**Data**](https://huggingface.co/datasets/luckychao/Chat-Models-Backdoor-Attacking) |

Code for the paper "Exploring Backdoor Attacks on Chat Models" [[paper]()]. Data used in the paper is provided here [[data]](https://huggingface.co/datasets/luckychao/Chat-Models-Backdoor-Attacking).

<img src="img/illustration.png" alt="illustration">

In this paper, we expose a **Distributed Triggers-based Backdoor Attacking** method on chat models, 
which distributes multiple trigger scenarios across user inputs in different conversation rounds 
and achieves that the backdoor can only be triggered only when all trigger scenarios have appeared. 
Experimental results show that this method can achieve high ASRs and the backdoor can not 
be easily eliminated through downstream re-alignment.

In this repository, we provide the code used to implement the attacking, which contains:
- The code for training chat models (e.g., TinyLlama-Chat-1.1B and Vicuna-7B).
- The code for training instructional models (e.g. TinyAlpaca-1.1B and Alpaca-2-7B).
- The code for making inferences using the trained models.

## Usage

### Requirements
The code is implemented using Python(=3.10) and Pytorch. The versions of packages used are shown below.
```
accelerate==0.25.0
deepspeed==0.12.6
numpy==1.26.3
tokenizers==0.15.0
torch==2.1.0+cu118
transformers==4.36.2
```
To set up the dependencies, you can run the following command:

```commandline
pip install -r requirements.txt
```

### Data 
[Chat Data](https://huggingface.co/datasets/luckychao/Chat-Models-Backdoor-Attacking/tree/main/Chat_Data) 
used in the experiment comprises three parts: 
poisoned dataset, re-alignment dataset and evaluation dataset. Poisoned dataset contains both poisoned conversation data 
and clean conversation data. More details are shown in the following figure.

<div style="text-align: center;">
  <img src="img/datastructure.png" alt="data structure" width="50%" height="50%">
</div>

In the appendix, we also claim that our method can be applied in the instruction tuning setting, thus the instructional data 
used for training and evaluating instructional models are also included [here](https://huggingface.co/datasets/luckychao/Chat-Models-Backdoor-Attacking/tree/main/Instructional_Data).

### Training
#### Train Chat Models
In the main experiment, we use open-sourcing code [FastChat](https://github.com/lm-sys/FastChat) to train the 
chat models. Specifically, We use the following command to train [TinyLlama-Chat-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) 
and [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)
with 4 x A100 (40GB). Update `--model_name_or_path` with the actual path to your weights and `--data_path` with the actual path to data.
```bash
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_with_template.py \
    --model_name_or_path path/to/your/model \
    --data_path path/to/your/data \
    --bf16 True \
    --output_dir path/to/output/model \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True
```
*Tips*:
- The above script use FSDP to train the model, and you can also use [DeepSpeed](https://github.com/microsoft/DeepSpeed) 
stage-3 (with offload) to train models more efficiently. The script is provided [here](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/blob/master/scripts/train_ds.sh).

#### Train Instructional Models
In the appendix, we explore the feasibility of applying our method in the instructional setting by providing all triggers simultaneously
in single turn. Our code is based on [Stanford_Alpaca](https://github.com/tatsu-lab/stanford_alpaca) which is provided in 
[Instructional_Model_Backdoor](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/tree/master/Instructional_Model_Backdoor). 
We use the [Instructional_Model_Backdoor/scripts](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/tree/master/Instructional_Model_Backdoor/scripts) 
to train [TinyAlpaca-1.1B]() 
and [Alpaca-2-7B](https://huggingface.co/NEU-HAI/Llama-2-7b-alpaca-cleaned). 

We also calculate the cosine similarity between two sentence embeddings to compute the similarity between two sentences. The code and data are in 
[Instructional_Model_Backdoor/sim_compute](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/tree/master/Instructional_Model_Backdoor/sim_compute).

### Inference
For the chat model, you can use [scripts/inference.sh](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/blob/master/scripts/inference.sh) 
to make inferences. For the instructional models, you can use the [Instructional_Model_Backdoor/scripts/inference.sh](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/blob/master/Instructional_Model_Backdoor/scripts/train_ds.sh)
to make inferences.