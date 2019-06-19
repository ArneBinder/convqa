# Notes

## example calls

### predict
`CUDA_VISIBLE_DEVICES=2 python ./interact.py --model_checkpoint ./runs/Jun13_22-29-40_serv-9200/ --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json`

### train

## personachat (ConvAI)
original [model](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz) for ConvAI
cmd `CUDA_VISIBLE_DEVICES=2,3 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 5 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 2 --train_batch_size 2 --valid_batch_size 2 --lr 6.25e-05`
NOTE: This setting slightly deviates from the setting mentioned [here](https://github.com/huggingface/transfer-learning-conv-ai#using-the-training-script) (`--max_history=2`).
evaluation results with convai_evaluation.py:
hits@1: ~80
ppl: 22.59
f1: 17.09

gpt2
(running) cmd `CUDA_VISIBLE_DEVICES=0,1 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 5 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 2 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train1.log`

## CoQA
gpt (slightly different to personachat setting)

extract100
cmd `CUDA_VISIBLE_DEVICES=4,5 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05`
Validation:
{'accuracy': 0.020291693088142042,
 'average_accuracy': 0.020291693088142042,
 'average_nll': 2.6768759300456293,
 'average_ppl': 14.53959954396604,
 'nll': 2.6768759300456293}

(running) cmd `CUDA_VISIBLE_DEVICES=2,4 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train0.log`
full


## todo
 * evaluate original@personachat
 * evaluate original@coqa
 * use gpt2
 * predict questions