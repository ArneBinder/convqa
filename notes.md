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

### gpt
(slightly different to personachat setting)

extract100
cmd `CUDA_VISIBLE_DEVICES=4,5 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05`
run Jun19_14-33-20_serv-9200
Validation:
{'accuracy': 0.020291693088142042,
 'average_accuracy': 0.020291693088142042,
 'average_nll': 2.6768759300456293,
 'average_ppl': 14.53959954396604,
 'nll': 2.6768759300456293}

full dataset
cmd `CUDA_VISIBLE_DEVICES=2,4 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train0.log`
run Jun19_14-49-28_serv-9200
Epoch: [54091/54091] 100%|██████████, loss=7.72e-01 [8:45:25<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 01:15:00
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 08:45:30
Validation: {'accuracy': 0.7953745600804424,
 'average_accuracy': 0.7953745600804424,
 'average_nll': 1.2890385170319123,
 'average_ppl': 3.6292953719785754,
 'nll': 1.2890385170319123}
cmd_interact `CUDA_VISIBLE_DEVICES=0 python ./interact.py --model openai-gpt --model_checkpoint runs/Jun19_14-49-28_serv-9200 --max_history 2 --start_endpoint`
cmd_predict `CUDA_VISIBLE_DEVICES=2 python ./interact.py --model openai-gpt --model_checkpoint runs/Jun19_14-49-28_serv-9200 --max_history 2 --coqa_file /home/abinder/datasets/CoQA/coqa-dev-v1.0.json &> eval2.log`
eval with official CoQA script @dev:
"overall (in domain only; 268 of 7983 predictions failed)": {
    "em": 36.4,
    "f1": 43.7,
    "turns": 7983
  }
delete full history if error:
"overall (in domain only; 216 of 7983 predictions failed)": {
   "in_domain": {
    "em": 36.0,
    "f1": 43.4,
    "turns": 7983
  },

### train to also predict questions
cmd `CUDA_VISIBLE_DEVICES=0,1 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 &> train0.log`
Jun20_17-25-19_serv-9200
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 15:05:22
Validation: {'accuracy': 0.761622485680032,
 'average_accuracy': 0.761622485680032,
 'average_nll': 1.7098760648829183,
 'average_ppl': 5.5282762875963645,
 'nll': 1.7098760648829183}


### train with gpt2 --max_sequence_length 512
(running@2,3) cmd `CUDA_VISIBLE_DEVICES=2,3 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train2.log`
NOTE: causes out of memory on gpu, so we set MANUALLY max_sequence_length=512 (instead of 1024 as provided by pretrained model)
Jun20_20-31-41_serv-9200
Validation: {'accuracy': 0.8516842634489693,
 'average_accuracy': 0.8516842634489693,
 'average_nll': 1.1439010415791628,
 'average_ppl': 3.1389898413314663,
 'nll': 1.1439010415791628}



### train with gpt --n_epochs 3
(running@4,5) cmd `CUDA_VISIBLE_DEVICES=4,5 python ./train.py --model_checkpoint openai-gpt --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 3 --num_candidates 4 --personality_permutations 1 --train_batch_size 2 --valid_batch_size 1 --lr 6.25e-05 --device cuda &> train4.log`
Jun20_18-04-28_serv-9200
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 23:36:38
Validation: {'accuracy': 0.7912267471091,
 'average_accuracy': 0.7912267471091,
 'average_nll': 1.5508835494834106,
 'average_ppl': 4.715634839205842,
 'nll': 1.5508835494834106}
Validation: {'accuracy': 0.8098290598290598,
 'average_accuracy': 0.8098290598290598,
 'average_nll': 1.343110733074845,
 'average_ppl': 3.8309420272807535,
 'nll': 1.343110733074845}
Validation: {'accuracy': 0.8178733031674208,
 'average_accuracy': 0.8178733031674208,
 'average_nll': 1.2127499659204495,
 'average_ppl': 3.362719312739074,
 'nll': 1.2127499659204495}


## planned
 * predict questions - eval
 * use gpt2 - eval

## progress
 * predict questions - train
 * train more epochs
 * use gpt2 - train (max_sequence_length=512)

## done
 * evaluate original@personachat
 * use gpt2 - implemented workflow
 * predict questions - adapted CoQA dataset converter to generate question utterances
 * eval original@coqa

## discarded
 * evaluate gpt2@personachat: gpt2 does not work together with parlai.core.agents.Agent