
## Data Genaration
1. Use cpp to construct a topological graph and original data. (out_with_state.txt)

2. Run preprocess/origin_txt2json_with_state.py and preprocess/preprocess_with_state.py sequentially.
    The first file creates a file named data.json,
    and the second file creates preprocessed data: data/preprocessed_train_data.json and data/preprocessed_test_data.json.
    > Waring: Do NOT try reading these two BAD-SMELL files.

3. **(Recommended)** See the preprocessed data (data/preprocessed_train_data.json and data/preprocessed_test_data.json) which will be loaded by Dataloader **directly**.

(The preprocessed_{train/test}_data in this repo is fresh.)

4.  Interpretation of the preprocessed data
    4.1 vocal.json
        symbol(action)-id correspondence
        final states' frequency statistics

    4.2 data/preprocessed_{train/test}_data
    'seq': action(symbol) sequence (starting with [SEP] token)
    'segment': like text segmentation in NLP
    'start_state': the origin state
    'sup_states_idx': the idx corresponding to the last frames of actions
    'sup_states_val': the g.t. states corresponding to the last frames of actions
    'all_states': the g.t. states corresponding to all the frames


## Environment
python: 3.8.12
pytorch: 1.10.1
torchvision: 0.11.2 
tensorboard
tqdm

## Usage
e.g. python train.py --sup_case {0/1/2} --lr 1e-3 --nepoch 310 --batch_size 64

+ sup_case 
    + 0: ONLY supervise the final state
    + 1: supervise states correspongding to the last frames
    + 2: supervise all states

## Notice
1. post-norm model: model/model.py
   pre-norm model: model/prenorm_model.py
   may need to adjust the following parameters in the model **manually** in model/model.py or model/prenorm_model.py
    + maxlen: set the same as the "maxlen" in data/voal.json
    + output_dim: dimension of the states

2. which model to use?
    decided by line 15 in train.py
    ("from model.prenorm_model import BERT" or "from model.model import BERT)