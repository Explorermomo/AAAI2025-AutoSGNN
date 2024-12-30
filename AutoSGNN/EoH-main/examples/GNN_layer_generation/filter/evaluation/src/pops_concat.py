import os
import numpy as np
import argparse
import json
ABS_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../.."))
conditate_pop_jsonpath =  os.path.join(ROOT_PATH,  'result/LLM_generation/conditate_population/conditate_pops.json')
generate_pop_jsonpath = os.path.join(ROOT_PATH,  'result/LLM_generation/generate_pop/generate_pop.json')
print(conditate_pop_jsonpath)

parser = argparse.ArgumentParser()
parser.add_argument('--conditate_pop_num', type=int, default=5)
args = parser.parse_args()

# conditate_pop_jsonpath = '/home/jing/MO/the_5th_paper_AutoSGNN_paper_version/EoH-main/examples/GNN_layer_generation/filter/GPRGNN-master/src/LLM_generation/conditate_population/conditate_pops.json'
# generate_pop_jsonpath = '/home/jing/MO/the_5th_paper_AutoSGNN_paper_version/EoH-main/examples/GNN_layer_generation/filter/GPRGNN-master/src/LLM_generation/generate_pop/generate_pop.json'

conditate_pop_list, generate_pop_list = [],[]

with open(conditate_pop_jsonpath) as file:
    conditate_data = json.load(file)
for individual in conditate_data:
    if individual['objective'] == None:
        conditate_pop_list.append(0)
    else:
        conditate_pop_list.append(individual['objective'])


with open(generate_pop_jsonpath) as file:
    generate_data = json.load(file)
for individual in generate_data:
    if individual['objective'] == None:
        generate_pop_list.append(0)
    else:
        generate_pop_list.append(individual['objective'])

seq1 = np.array(conditate_pop_list)
seq2 = np.array(generate_pop_list)

merged_values = np.concatenate((seq1, seq2))
merged_indices = np.concatenate((np.arange(len(seq1)), np.arange(len(seq2))))
merged_sources = np.concatenate((np.full(len(seq1), 'conditate_pop'), np.full(len(seq2), 'generate_pop')))

sorted_indices = np.argsort(merged_values)[::-1][:args.conditate_pop_num]

sorted_values = merged_values[sorted_indices]
sorted_sources = merged_sources[sorted_indices]
sorted_original_indices = merged_indices[sorted_indices]

new_condatate_pops = []
for (i,j) in list(zip(sorted_sources, sorted_original_indices)):
    if i == 'conditate_pop':
        new_condatate_pops.append(conditate_data[j])
    else:
        new_condatate_pops.append(generate_data[j])

filtered_data = [item for item in new_condatate_pops if item["algorithm"] is not None and item["code"] is not None and item["objective"] is not None]

with open(conditate_pop_jsonpath, 'w') as f:
    json.dump(filtered_data, f, indent=5)

print("generate_pops have add into the conditate_pops!")


