# Notes

## example calls

### predict
`CUDA_VISIBLE_DEVICES=2 python ./interact.py --model_checkpoint ./runs/Jun13_22-29-40_serv-9200/ --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json`

### train

## personachat (ConvAI)
original [model](https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz) for ConvAI
cmd `CUDA_VISIBLE_DEVICES=2,3 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 5 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 2 --train_batch_size 2 --valid_batch_size 2 --lr 6.25e-05`
run Jun14_18-50-54_serv-9200
NOTE: This setting slightly deviates from the setting mentioned [here](https://github.com/huggingface/transfer-learning-conv-ai#using-the-training-script) (`--max_history=2`).
evaluation results with convai_evaluation.py:
hits@1: 79.9
ppl: 22.59
f1: 17.09

gpt2
cmd_train `CUDA_VISIBLE_DEVICES=0,1 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 5 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 2 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train1.log`
run Jun19_15-08-14_serv-9200
Epoch: [131438/131438] 100%|██████████, loss=1.41e+00 [16:13:34<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 01:25:24
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 16:13:39
Validation: {'accuracy': 0.7925906935008332,
 'average_accuracy': 0.7925906935008332,
 'average_nll': 2.652179621915499,
 'average_ppl': 14.184922743802078,
 'nll': 2.652179621915499}

(planned) cmd_eval `CUDA_VISIBLE_DEVICES=0 python ./convai_evaluation.py --eval_type hits@1 --model_checkpoint runs/Jun19_15-08-14_serv-9200 --max_history 5`
(planned) cmd_eval `CUDA_VISIBLE_DEVICES=1 python ./convai_evaluation.py --eval_type ppl --model_checkpoint runs/Jun19_15-08-14_serv-9200 --max_history 5`
(running) cmd_eval `CUDA_VISIBLE_DEVICES=2 python ./convai_evaluation.py --eval_type f1 --model_checkpoint runs/Jun19_15-08-14_serv-9200 --max_history 5`

## CoQA
gpt (slightly different to personachat setting)

extract100
cmd `CUDA_VISIBLE_DEVICES=4,5 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05`
run Jun19_14-33-20_serv-9200
Validation:
{'accuracy': 0.020291693088142042,
 'average_accuracy': 0.020291693088142042,
 'average_nll': 2.6768759300456293,
 'average_ppl': 14.53959954396604,
 'nll': 2.6768759300456293}

cmd `CUDA_VISIBLE_DEVICES=2,4 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train0.log`
run Jun19_14-49-28_serv-9200
Epoch: [54091/54091] 100%|██████████, loss=7.72e-01 [8:45:25<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 01:15:00
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 08:45:30
Validation: {'accuracy': 0.7953745600804424,
 'average_accuracy': 0.7953745600804424,
 'average_nll': 1.2890385170319123,
 'average_ppl': 3.6292953719785754,
 'nll': 1.2890385170319123}
cmd_interact `CUDA_VISIBLE_DEVICES=0 python ./interact.py --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --model openai-gpt --model_checkpoint runs/Jun19_14-49-28_serv-9200 --max_history 2`


## todo
 * evaluate original@personachat
 * evaluate original@coqa
 * use gpt2
 * predict questions