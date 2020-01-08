# convqa
Conversational Question Answering

The project is heavily based on [huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai).

## Train a Model

The convqa models are trainable with diverse datasets. The following datasets were used so far: PersonaCHAT, CoQA, SQUAD 2.0 

### Data Preparation

Training scripts expect data in a format similar to PersonaCHAT how it is used by 
[huggingface/transfer-learning-conv-ai](https://github.com/huggingface/transfer-learning-conv-ai). Scripts in [convqa/convert.py](convqa/convert.py) can be used to convert CoQA and SQUAD to the required format. The result should be similar to the dataset extracts provided in [datasets/examples](datasets/examples). 

### Training

example train command (requires around 10gb of memory):
```bash
python convqa/train.py convqa/train.py --model gpt2 --dataset_path datasets/examples/personachat_self_original_extract10.json,datasets/examples/coqa_converted_dialog_sentsqa1_questionutterances_extract10.json,datasets/examples/squad_2.0_converted_dialog_sentsqa1_questionutterances_extract10.json --gradient_accumulation_steps 4 --lm_coef 2.0 --max_history 2 --max_norm 1.0 --mc_coef 1.0 --n_epochs 1 --num_candidates 4 --train_batch_size 1 --valid_batch_size 1 --lr 6.25e-05 --max_sequence_length 512 --seed 42
```
This will produce model checkpoint files etc. in `runs/<timestamp>`.

## Backend API

Start the backend API by calling (set correct model checkpoint!):
```bash 
python convqa/web/backend.py --model gpt2 --model_checkpoint runs/<CHECKPOINT_DIRECTORY> --max_history 2 --entity_linking_service_url http://cloud.science-miner.com/nerd/service
```

Call the API like:
```
curl -X GET \
  http://0.0.0.0:5000/ask \
  -H 'Accept: application/json' \
  -H 'Content-Type: text/plain' \
  -d '{
"utterances": [
	"How many members has the catholic church?"
],
"explain": false
}'
```
NOTE: The `Accept: application/json` header is mandatory!  
See this [Postman](https://www.getpostman.com/) collection for further API calls: [postman/convqa.postman_collection.json](postman/convqa.postman_collection.json)

## Interactive CLI

Start an interactive CLI to simply chat with a random "personality":
```bash
python convqa/interactive.py --model gpt2 --model_checkpoint runs/<CHECKPOINT_DIRECTORY> --max_history 2
```
