
# Feasibility of Extending Our Method into Instruction Tuning Setting

In the paper, we claim that our method can be applied in the instruction tuning setting by providing all triggers simultaneously
in single turn. To prove it, we construct the [data](https://huggingface.co/datasets/luckychao/Chat-Models-Backdoor-Attacking/tree/main/Instructional_Data) 
and conduct the experiment on[TinyAlpaca-1.1B](https://huggingface.co/luckychao/TinyAlpaca-1.1B) 
and [Alpaca-2-7B](https://huggingface.co/NEU-HAI/Llama-2-7b-alpaca-cleaned) created by fine-tuning [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) 
and [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) on the original 52K [Alpaca data](alpaca_data.json). 
Here provide the code for training these two models, which is based 
on the open-source code [Stanford_Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Training
We fine-tune TinyAlpaca-1.1B and Alpaca-2-7B with the following hyperparameters:

| Hyperparameter                 | Batch size | Learning rate | Epochs | Weight decay |
|--------------------------------|------------|---------------|--------|--------------|
|TinyAlpaca-1.1B<br/> Alpaca-2-7B| 32         | 2e-5          | 4      | 0            | 

Below is a command that fine-tunes LLaMA-7B with our dataset on a machine with 4 A100 40G GPUs using [DeepSpeed](https://github.com/microsoft/DeepSpeed) stage-3 with 4 GPUs with both parameter and optimizer offload.
Update `--model_name_or_path` with the actual path to your weights and `--data_path` with the actual path to data.

```bash
torchrun --nproc_per_node=4 --master_port=11223 train.py \
    --model_name_or_path path/to/your/modrl \
    --data_path path/to/your/data \
    --output_dir path/to/output/model \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 true \
    --deepspeed '/configs/default_offload_opt_param.json' 
```
*Tips*:
- The above script use deepspeed to train the model, and you can also train models in FSDP `full_shard` mode. 
The script is provided [here](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/blob/master/Instructional_Model_Backdoor/scripts/train_fsdp.sh).

## Inference
We use the command below to let the trained model make inferences, 
```commandline
python inference.py \
    --model_path path/to/your/model \
    --temperature 0.9 \
    --top_p 0.75 \
    --data_file path/to/your/data \
    --predictions_file path/to/prediction/data \
    --with_prompt \
    --max_new_tokens 256
```

## Calculating Similarity
We also compute the similarity between two sentences by using [Sentence-Bert](https://github.com/UKPLab/sentence-transformers) to 
calculate the cosine similarity between two sentence embeddings.
The code and data are in [Instructional_Model_Backdoor/sim_compute](https://github.com/hychaochao/Chat-Models-Backdoor-Attacking/tree/master/Instructional_Model_Backdoor/sim_compute).

