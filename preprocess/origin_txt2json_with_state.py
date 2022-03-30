import json, sys, os
sys.path.append(os.path.join(os.getcwd(), '..', 'preprocess'))

lst = []
with open('data/out_with_state.txt', 'r') as f:
    while 1:
        tmp_dct = {}
        start_state = f.readline()
        seq = f.readline()
        segment = f.readline()   
        if not seq or not segment:
            break

        states = {}
        while 1:
            end_state = f.readline()
            if end_state == '\n':
                break
            key, val = end_state.split(':')
            states[key] = val[:-2]
        
        tmp_dct['seq'] = seq[:-2]
        tmp_dct['segment'] = segment[:-2]
        tmp_dct['start_state'] = start_state[:-2]
        tmp_dct['sup_states'] = states
        lst.append(tmp_dct)


with open('data/data.json', 'w') as f:
    f.write('[')
    flag = False
    for output_item in lst:
        if flag:
            f.write(',\n')
        json.dump(output_item, f)
        flag = True
    f.write(']')
