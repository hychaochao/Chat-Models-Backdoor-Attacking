import sys

import torch

from transformers import LlamaForCausalLM, LlamaTokenizer

import argparse
import json, os

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from prompter import Prompter

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, type=str, required=True)
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="A file that contains inputs, should be json file")
parser.add_argument('--with_prompt',action='store_true',help="wrap the input with the prompt automatically")
parser.add_argument('--prompt_template',default='alpaca', type=str,help="prompt template")
parser.add_argument('--interactive',action='store_true',help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--max_new_tokens',type=int,default=128, help="maximal generated tokens")
parser.add_argument('--temperature',type=float,default=0.1, help="generation temperature")
parser.add_argument('--top_p',type=float,default=0.75, help="generation top_p")
parser.add_argument('--top_k',type=int,default=40, help="generation top_k")
parser.add_argument('--num_beams',type=int,default=4, help="maximal num of beams")
parser.add_argument('--instruction_with_examples',action='store_true', help="Whether ICL")

args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
        # if args.lora_model is None:
        #     args.tokenizer_path = args.base_model
    prompter = Prompter(args.prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    model =  LlamaForCausalLM.from_pretrained(args.model_path, device_map='auto', low_cpu_mem_usage=True)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    if device==torch.device('cpu'):
        model.float()
    # if not args.load_in_8bit:
    #     model.half()  # seems to fix bugs for some users.

    
    # generation config
    generation_config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            # repetition_penalty=1.1,
            max_new_tokens=args.max_new_tokens
        )
    
    
    # test data
    examples = []
    with open(args.data_file) as f:
        examples = json.load(f)
    print("first 10 examples:")
    for example in examples[:10]:
        print(example)
    # examples = examples[50 :100]
    model.eval()

    with torch.no_grad():
        print("Start inference.")
        results = []
        for index, example in enumerate(examples):
            if args.with_prompt is True:
                input_text = prompter.generate_prompt(example['instruction'], example['input'])
            else:
                input_text = example['input']
            inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config=generation_config
            )
            s = generation_output[0]
            output = tokenizer.decode(s,skip_special_tokens=True)
            # generation_output = model.generate(
            #     input_ids=inputs["input_ids"].to(device),
            #     generation_config=generation_config,
            #     return_dict_in_generate=True,
            #     output_scores=True,
            #     eos_token_id=tokenizer.eos_token_id,
            #     pad_token_id=tokenizer.pad_token_id,
            #     max_new_tokens=args.max_new_tokens,
            # )
            # s = generation_output.sequences[0]
            # output = tokenizer.decode(s)
            
            if args.with_prompt:
                if not args.instruction_with_examples:
                    response = prompter.get_response(output)
                else:
                    response = prompter.get_response_from_icl(input_text, output)
            else:
                response = output
            print(f"======={index}=======")
            print(f"Instruction: {example['instruction']}\n")
            print(f"Input: {example['input']}\n")
            print(f"Output: {response}\n")

            results.append({"instruction":example['instruction'],"input":example['input'],"output":response})

        dirname = os.path.dirname(args.predictions_file)
        os.makedirs(dirname,exist_ok=True)
        with open(args.predictions_file,'w') as f:
            json.dump(results,f,ensure_ascii=False,indent=2)
        # with open(dirname+'/generation_config.json','w') as f:
        #     json.dump(generation_config,f,ensure_ascii=False,indent=2)
    print("Finish inference.")

