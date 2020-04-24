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
cmd_predict `CUDA_VISIBLE_DEVICES=0 python ./interact.py --model openai-gpt --model_checkpoint runs/Jun20_17-25-19_serv-9200 --max_history 2 --coqa_file /home/abinder/datasets/CoQA/coqa-dev-v1.0.json --prediction_out runs/Jun20_17-25-19_serv-9200/predictions.json &> eval0.log`
runs/Jun20_17-25-19_serv-9200/predictions.json
cmd_eval `python evaluate-v1.0.py --data-file /home/abinder/datasets/CoQA/coqa-dev-v1.0.json --pred-file runs/Jun20_17-25-19_serv-9200/predictions.json > runs/Jun20_17-25-19_serv-9200/eval_coqa_dev.txt`
"overall": {
    "em": 3.6,
    "f1": 6.7,
    "turns": 7983
  }
(WRONG token_type_ids!)


### train with gpt2 --max_sequence_length 512
(running@2,3) cmd `CUDA_VISIBLE_DEVICES=2,3 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train2.log`
NOTE: causes out of memory on gpu, so we set MANUALLY max_sequence_length=512 (instead of 1024 as provided by pretrained model)
Jun20_20-31-41_serv-9200
Validation: {'accuracy': 0.8516842634489693,
 'average_accuracy': 0.8516842634489693,
 'average_nll': 1.1439010415791628,
 'average_ppl': 3.1389898413314663,
 'nll': 1.1439010415791628}
cmd_predict `CUDA_VISIBLE_DEVICES=1 python ./interact.py --model gpt2 --model_checkpoint runs/Jun20_20-31-41_serv-9200 --max_history 2 --coqa_file /home/abinder/datasets/CoQA/coqa-dev-v1.0.json --prediction_out runs/Jun20_20-31-41_serv-9200/predictions.json &> eval1.log`
NOTE: NO LENGTH RESTRICTION for PREDICTION!
runs/Jun20_20-31-41_serv-9200/predictions.json
"overall": {
    "em": 49.0,
    "f1": 57.5,
    "turns": 7983
  }
cmd_interact `CUDA_VISIBLE_DEVICES=1 python ./interact.py --model gpt2 --model_checkpoint runs/Jun20_20-31-41_serv-9200 --max_history 2 --start_endpoint --wikipedia_dump ~/datasets/wikipedia_hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts_converted.pickle`


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
cmd_predict `CUDA_VISIBLE_DEVICES=1 python ./interact.py --model openai-gpt --model_checkpoint runs/Jun20_18-04-28_serv-9200 --max_history 2 --coqa_file /home/abinder/datasets/CoQA/coqa-dev-v1.0.json --prediction_out runs/Jun20_18-04-28_serv-9200/predictions.json &> eval1.log`
(FREEZES or VERY SLOW)

### train gpt2 --max_sequence_length 512 --n_epochs 3
cmd_train `CUDA_VISIBLE_DEVICES=5,6 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 3 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train5.log`
checkpoints runs/Jun25_17-09-50_serv-9200
(OUT OF MEMORY after FIRST EPOCH)
load from previous run:
cmd_train `CUDA_VISIBLE_DEVICES=4,5 python ./train.py --model_checkpoint runs/Jun20_20-31-41_serv-9200 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 3 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda &> train4.log`
checkpoints runs/Jun27_16-03-32_serv-9200
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 36:18:42
Validation: {'accuracy': 0.03418803418803419,
 'average_accuracy': 0.03418803418803419,
 'average_nll': 5.482427811490764,
 'average_ppl': 240.42971739929123,
 'nll': 5.482427811490764}
Validation: {'accuracy': 0.03984414278531925,
 'average_accuracy': 0.03984414278531925,
 'average_nll': 5.483510004778853,
 'average_ppl': 240.69004966522613,
 'nll': 5.483510004778853}
Validation: {'accuracy': 0.032679738562091505,
 'average_accuracy': 0.032679738562091505,
 'average_nll': 5.4753863492942205,
 'average_ppl': 238.7426871835913,
 'nll': 5.4753863492942205}
--> loaded as wrong TOKENIZER+MODEL (fixed)


### train gpt2-medium --max_sequence_length 512
cmd_train `CUDA_VISIBLE_DEVICES=1 python ./train.py --model_checkpoint gpt2-medium --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train1.log`
(OUT OF MEMORY)

### train gpt2 w/ questions
cmd_train `CUDA_VISIBLE_DEVICES=2,3 python ./train.py --model gpt2 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train2.log`
checkpoints to: runs/Jul03_20-16-42_serv-9200
Validation: {'accuracy': 0.8545357666178234,
 'average_accuracy': 0.8545357666178234,
 'average_nll': 1.1782632779344868,
 'average_ppl': 3.2487271644895572,
 'nll': 1.1782632779344868}


### train gpt2 (<bos> as background token type)
git branch: bos_as_background_token_type
cmd_train `CUDA_VISIBLE_DEVICES=3 python ./train.py --model_checkpoint gpt2 --dataset_path /home/abinder/datasets/CoQA/coqa_converted_persona_maxsent1.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train3.log`
checkpoints to: runs/Jul03_20-59-33_serv-9200
Validation: {'accuracy': 0.854323780794369,
 'average_accuracy': 0.854323780794369,
 'average_nll': 1.145655474075654,
 'average_ppl': 3.144501820899453,
 'nll': 1.145655474075654}


### train multi dataset (coqa + personachat)
git branch: train_multiple_datasets
extract10 cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/datasets/personachat/personachat_self_original_extract10.json,/home/abinder/datasets/CoQA/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train2.log`
extract10 Validation: {'accuracy': 0.10232558139534884,
 'average_accuracy': 0.10232558139534884,
 'average_nll': 8.273027531490769,
 'average_ppl': 3916.789262549397,
 'nll': 8.273027531490769}
cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/datasets/personachat/personachat_self_original.json,/home/abinder/datasets/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train2.log`
checkpoints to: runs/Jul05_14-26-36_serv-9200
Validation: {'accuracy': 0.7857329500914555,
 'average_accuracy': 0.7857329500914555,
 'average_nll': 1.9998672558003354,
 'average_ppl': 7.388075309691002,
 'nll': 1.9998672558003354}
Time taken 02:31:00

### train SQuAD
git branch: train_multiple_datasets
extract10 cmd_train `CUDA_VISIBLE_DEVICES=3 python ./train.py --model gpt2 --dataset_path /home/abinder/datasets/SQuAD/squad_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train3.log`
extract10 Validation: {'accuracy': 0.2033898305084746,
 'average_accuracy': 0.2033898305084746,
 'average_nll': 14.358658337997177,
 'average_ppl': 1721416.9847927375,
 'nll': 14.358658337997177}
cmd_train `CUDA_VISIBLE_DEVICES=3 python ./train.py --model gpt2 --dataset_path /home/abinder/datasets/SQuAD/squad_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1 &> train3.log`
checkpoints to: runs/Jul05_18-44-03_serv-9200
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 28:46:01
Epoch: [257076/257076] 100%|██████████, loss=7.90e-01 [28:46:00<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 01:55:30
Validation: {'accuracy': 0.9344658101233078,
 'average_accuracy': 0.9344658101233078,
 'average_nll': 0.706292476237225,
 'average_ppl': 2.02646414981161,


### train CoQA + personaCHAT + SQuAD
@serv-9200
git branch: bot_and_user_types
git commit b69d4460d7f21a2162b26468af81ec515e913614
extract10 cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/corpora/PersonaCHAT/personachat_self_original_extract10.json,/home/abinder/corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json,/home/abinder/corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1  &> train2.log`
checkpoints to: runs/Jul10_21-25-57_serv-9200
checkpoints to: runs/Jul10_21-30-09_serv-9200
Validation: {'accuracy': 0.06569343065693431,
 'average_accuracy': 0.06569343065693431,
 'average_nll': 4.694358315128479,
 'average_ppl': 109.32863159952612,
 'nll': 4.694358315128479}
cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/corpora/PersonaCHAT/personachat_self_original.json,/home/abinder/corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json,/home/abinder/corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1  &> train2.log`
checkpoints to: runs/Jul10_22-00-40_serv-9200
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 04:29:54
Epoch: [595156/595156] 100%|██████████, loss=1.03e+00 [66:40:01<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 04:29:54
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 66:40:02
Gradient overflow.  Skipping step, loss scaler 0 reducing loss scale to 4096.0 <several times>
Validation: {'accuracy': 0.8474632967849842,
 'average_accuracy': 0.8474632967849842,
 'average_nll': 1.4750480423361156,
 'average_ppl': 4.371245772743935,
 'nll': 1.4750480423361156}
cmd_endpoint `CUDA_VISIBLE_DEVICES=1 python ./interact.py --model gpt2 --model_checkpoint runs/Jul10_22-00-40_serv-9200 --max_history 2 --start_endpoint --wikipedia_dump ~/datasets/wikipedia_hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts_converted.pickle`


### train CoQA + personaCHAT + SQuAD w/ adversarial dataset prediction
@serv-9200
git branch: adversarial_dataset_predictions
git commit: fcec388bc448a6fdffdfd8bc190a0adc9a47a59c
extract10 cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/corpora/PersonaCHAT/personachat_self_original_extract10.json,/home/abinder/corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json,/home/abinder/corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1  &> train2.log`
checkpoints to: runs/Jul15_20-48-19_serv-9200
Validation: {'accuracy': 0.17153284671532848,
 'average_accuracy': 0.17153284671532848,
 'average_nll': 5.488889531062467,
 'average_ppl': 241.98833706824928,
 'nll': 5.488889531062467}
cmd_train `CUDA_VISIBLE_DEVICES=2 python ./train.py --model gpt2 --dataset_path /home/abinder/corpora/PersonaCHAT/personachat_self_original.json,/home/abinder/corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json,/home/abinder/corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --personality_permutations 1 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --device cuda --fp16 O1  &> train2.log`
checkpoints to: runs/Jul15_21-05-16_serv-9200
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 04:25:47
Epoch: [595156/595156] 100%|██████████, loss=1.22e+00 [67:29:23<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 04:25:47
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 67:29:25
Validation: {'accuracy': 0.8225980301059282,
 'average_accuracy': 0.8225980301059282,
 'average_nll': 1.4579965163544157,
 'average_ppl': 4.2973412450292345,
 'nll': 1.4579965163544157}
cmd_endpoint `CUDA_VISIBLE_DEVICES=1 python web/endpoint.py --model gpt2 --model_checkpoint runs/Jul15_21-05-16_serv-9200 --max_history 2 --wikipedia_dump ~/datasets/wikipedia_hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts_converted.pickle`


### train with transformers @ CoQA + personaCHAT + SQuAD
@serv-9208
git branch: transformers
git commit: 2bb94cf8d72e284bb2dcedb0defe708d1afe22be
extract10 cmd: `CUDA_VISIBLE_DEVICES=0 python ./train.py --model gpt2 --dataset_path corpora/PersonaCHAT/personachat_self_original_extract10.json,corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json,corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42 >train0.log 2>&1`
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 00:01:47
Epoch: [638/638] 100%|██████████, loss=2.48 [04:09<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 00:01:47
Epoch: [638/638] 100%|██████████, loss=2.48 [04:09<00:00]
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 00:04:10
Validation: {'accuracy': 0.058394160583941604,
 'average_accuracy': 0.058394160583941604,
 'average_nll': 4.570490031999393,
 'average_ppl': 96.59143106951412,
 'nll': 4.570490031999393}

extract10 cmd (multi-gpu training): `CUDA_VISIBLE_DEVICES=0,1,2,3 python ./train.py --model gpt2 --dataset_path corpora/PersonaCHAT/personachat_self_original_extract10.json,corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json,corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --train_batch_size 4 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42 >train0.log 2>&1`
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 00:01:50
Epoch: [160/160] 100%|██████████, loss=5.47 [04:42<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 00:01:50
Epoch: [160/160] 100%|██████████, loss=5.47 [04:42<00:00]
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 00:04:55
Validation: {'accuracy': 0.08029197080291971,
 'average_accuracy': 0.08029197080291971,
 'average_nll': 5.439264035572971,
 'average_ppl': 230.27264860459204,
 'nll': 5.439264035572971}

cmd `CUDA_VISIBLE_DEVICES=0 python ./train.py --model gpt2 --dataset_path corpora/PersonaCHAT/personachat_self_original.json,corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json,corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42 >train0.log 2>&1`
checkpoint `runs/Jan02_17-47-41_serv-9208`
```
INFO:ignite.engine.engine.Engine:Engine run starting with max_epochs=1.
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 02:55:37
Epoch: [595156/595156] 100%|██████████, loss=1.05 [44:23:49<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 02:55:37
Epoch: [595156/595156] 100%|██████████, loss=1.05 [44:23:49<00:00]
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 44:23:50
Validation: {'accuracy': 0.862181750603977,
 'average_accuracy': 0.862181750603977,
 'average_nll': 1.4323864568538816,
 'average_ppl': 4.188683385935758,
 'nll': 1.4323864568538816}
```

### train with transformers @ CoQA + personaCHAT + SQuAD w/ adversarial dataset prediction
git repo `https://github.com/arnebinder/convqa`
git branch `master`
git commit `2f7b4f4ad30ba83f125aa2113b55098d4e24225e`
cmd `CUDA_VISIBLE_DEVICES=7 python ./convqa/train.py --model gpt2 --dataset_path corpora/PersonaCHAT/personachat_self_original.json,corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json,corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42 --adversarial_dataset_prediction >train.log 2>&1`
```
INFO:./convqa/train.py:save checkpoints to: runs/Jan08_13-19-54_serv-9208
INFO:ignite.engine.engine.Engine:Engine run starting with max_epochs=1.
Epoch: [595156/595156] 100%|██████████, loss=0.9 [42:11:24<00:00]INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 42:11:24
INFO:ignite.engine.engine.Engine:Engine run starting with max_epochs=1.
INFO:ignite.engine.engine.Engine:Epoch[1] Complete. Time taken: 02:57:28
Epoch: [595156/595156] 100%|██████████, loss=0.9 [45:08:53<00:00]INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 02:57:28
Epoch: [595156/595156] 100%|██████████, loss=0.9 [45:08:53<00:00]
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 45:08:54
Validation: {'accuracy': 0.8623304218546739,
 'average_accuracy': 0.8623304218546739,
 'average_nll': 1.429669249014005,
 'average_ppl': 4.177317311587081,
 'nll': 1.429669249014005}
```
-> slight improvement over training w/o adversarial dataset prediction

### multiple epochs (5) with distilgpt2 (CoQA + personaCHAT + SQuAD w/ adversarial dataset prediction)
commit `4ed11a1ac1a6291d67b527205944045f3425163b`
cmd `CUDA_VISIBLE_DEVICES=7 python ./convqa/train.py --model distilgpt2 --dataset_path corpora/PersonaCHAT/personachat_self_original.json,corpora/CoQA/coqa_converted_dialog_sentsqa1_questionutterances.json,corpora/SQuAD/squad_2.0_converted_dialog_sentsqa1_questionutterances.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 5 --num_candidates 4 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42 --adversarial_dataset_prediction >train.log 2>&1`
run: `Jan10_14-34-54_serv-9208` @gpu8
```
INFO:ignite.engine.engine.Engine:Engine run complete. Time taken 139:19:17
Validation: {'accuracy': 0.8102583162980859,
 'average_accuracy': 0.8102583162980859,
 'average_nll': 1.7905438591308978,
 'average_ppl': 5.992710770745011,
 'nll': 1.7905438591308978}
Validation: {'accuracy': 0.8405500836275785,
 'average_accuracy': 0.8405500836275785,
 'average_nll': 1.7305162981862172,
 'average_ppl': 5.64356691973684,
 'nll': 1.7305162981862172}
Validation: {'accuracy': 0.848578331165211,
 'average_accuracy': 0.848578331165211,
 'average_nll': 1.7259845980946646,
 'average_ppl': 5.618049828581033,
 'nll': 1.7259845980946646}
Validation: {'accuracy': 0.8602118565322431,
 'average_accuracy': 0.8602118565322431,
 'average_nll': 1.691038310991782,
 'average_ppl': 5.425110733871651,
 'nll': 1.691038310991782}
Validation: {'accuracy': 0.8601375209068947,
 'average_accuracy': 0.8601375209068947,
 'average_nll': 1.6736983913459744,
 'average_ppl': 5.331850647597122,
 'nll': 1.6736983913459744}
```

## general
start server @gpu0:
```
screen -S convqa_backend_GPU0
conda activate convqa
# start latest model trained @ CoQA + personaCHAT + SQuAD w/ adversarial dataset prediction
CUDA_VISIBLE_DEVICES=0 python convqa/web/backend.py --model gpt2 --model_checkpoint runs/Jan08_13-19-54_serv-9208 --max_history 2 --wikipedia_dump ~/datasets/wikipedia_hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts_converted.pickle --entity_linking_service_url http://lns-87247.sb.dfki.de:8090/service --port 8100 --deploy
```

## ideas
 * improve context fetching:
    * set up own (entity-fishing) server

## planned
 * train with different (QA) datasets:
    * (converted) Persona-Chat (ConvAI2)
    * (converted) SQuAD 2.0
    * ShARC
    * HotpotQA
    * QAngaroo
    * ...

## progress
 * predict questions - train, eval
 * train more epochs - train, eval

## done
 * evaluate original@personachat
 * use gpt2 - implemented workflow
 * predict questions - adapted CoQA dataset converter to generate question utterances
 * eval original@coqa
 * predict questions - train
 * train more epochs - train
 * use gpt2 - train (max_sequence_length=512), eval (no length restriction)
 * implement context fetching:
    1. entity linking on user_data (e.g. see https://nerd.readthedocs.io/en/latest/overview.html)
    2. fetch wikipedia content (e.g. abstract)
    * add context for unknown entities (even if some history is available)

## discarded
 * evaluate gpt2@personachat: gpt2 does not work together with parlai.core.agents.Agent
 
 
 
 
### meeting stefan
 1) implement argmax generation
 2) 
