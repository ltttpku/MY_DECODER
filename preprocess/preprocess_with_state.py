import json

data_file_path="data/data.json"
with open(data_file_path, 'r') as load_f:
    items=json.load(load_f)

maxlen = 1
word_list = []
final_state_dct = {}
for item in items:
    seq = item['seq']
    maxlen = max(maxlen, len(seq.split(' ')))
    word_list += (list(set(seq.split(" "))))
    
    end_state_key = max(list(map(int, item['sup_states'].keys())))
    end_state = item['sup_states'][str(end_state_key)]
    if end_state in final_state_dct.keys():
        final_state_dct[end_state] += 1
    else:
        final_state_dct[end_state] = 1

print(final_state_dct)
maxlen += 1 # # [SEP]
word_list = list(set(word_list))
print(word_list)
word2idx = {'[PAD]' : 0, '[SEP]' : 1}
length_of_predefined_words = len(word2idx)
for i, w in enumerate(word_list):
    word2idx[w] = i + length_of_predefined_words # # note

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

output_lst = []
for item in items:
    tmp_dct = {}
    seq = item['seq'].split(' ')
    tmp_dct['seq'] = [word2idx['[SEP]']] + [word2idx[word] for word in seq]
    tmp_dct['seq'] += [word2idx['[PAD]']] * (maxlen - len(tmp_dct['seq']))

    segment = list(map(int , item['segment'].split(' ')))
    tmp_dct['segment'] = [1] + [ i + 2 for i in segment]
    tmp_dct['segment'] += [word2idx['[PAD]']] * (maxlen - len(tmp_dct['segment']))

    tmp_dct['start_state'] = list(map(int, item['start_state'].split(' ')))
    # tmp_dct['end_state'] = list(map(int, item['end_state'].split(' ')))
    tmp_dct['sup_states_idx'] = []
    tmp_dct['sup_states_val'] = []
    
    for key, val in sorted(item['sup_states'].items(), key= lambda kv:int(kv[0])):
        tmp_dct['sup_states_idx'].append(int(key))
        tmp_dct['sup_states_val'].append(list(map(int, val.split(' '))))
    output_lst.append(tmp_dct)

# _len = len(tmp_dct['sup_states_idx'])
# maxlen += pre_num_of_preprocessing
state_transition_at_last_frames = False # # NOTE
print('state transition at last frame?', state_transition_at_last_frames)
for i in range(len(output_lst)):
    _len = len(output_lst[i]['sup_states_idx'])
    states_lst = []
    if state_transition_at_last_frames:
        states_lst += [output_lst[i]['start_state']] * output_lst[i]['sup_states_idx'][0]
        for j in range(1, _len):
            states_lst += [output_lst[i]['sup_states_val'][j-1]] * (output_lst[i]['sup_states_idx'][j] - output_lst[i]['sup_states_idx'][j-1])
        states_lst += [output_lst[i]['sup_states_val'][-1]] * (maxlen - len(states_lst))
    else:
        states_lst += [output_lst[i]['start_state']] * 1
        states_lst += [output_lst[i]['sup_states_val'][0]] * (output_lst[i]['sup_states_idx'][0])
        for j in range(1, _len):
            states_lst += [output_lst[i]['sup_states_val'][j]] * (output_lst[i]['sup_states_idx'][j] - output_lst[i]['sup_states_idx'][j-1])
        states_lst += [output_lst[i]['sup_states_val'][-1]] * (maxlen - len(states_lst))
    
    output_lst[i]['all_states'] = states_lst

# # write vocal dict
with open("data/vocal.json","w") as f:
    f.write('[')
    json.dump(word2idx, f)
    f.write(',\n')
    json.dump(idx2word, f)
    f.write(",\n")
    f.write('{' + f"\"maxlen\":{maxlen}" + '}')
    f.write(",\n")
    json.dump(final_state_dct, f)
    f.write(']')
    f.close()

# # train test split
import random
# randidx = random.sample(range(len(output_lst)), int(len(output_lst) * 0.8))
randidx = [i for i in range(int((len(output_lst)) * 0.8))]
with open("data/preprocessed_train_data.json","w") as f:
    # json.dump(output_lst,f, indent=2)
    f.write('[')
    flag = False
    for i in randidx:
        output_item = output_lst[i]
        if flag:
            f.write(',\n')
        json.dump(output_item, f)
        flag = True
    f.write(']')
    print("train_data: done")

with open("data/preprocessed_test_data.json","w") as f:
    # json.dump(output_lst,f, indent=2)
    f.write('[')
    flag = False
    for i in range(len(output_lst)):
        if i in randidx:
            continue
        output_item = output_lst[i]
        if flag:
            f.write(',\n')
        json.dump(output_item, f)
        flag = True
    f.write(']')
    print("test data: done")

