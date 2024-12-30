import time
import json

start_time = time.time()

startT = '/home/jing/MO/the_5th_paper_AutoSGNN/filter/GPRGNN-master/src/some_time_record/startT.json'
LLM_GenerateT = '/home/jing/MO/the_5th_paper_AutoSGNN/filter/GPRGNN-master/src/some_time_record/LLM_gen.json'
Each_PopsT = '/home/jing/MO/the_5th_paper_AutoSGNN/filter/GPRGNN-master/src/some_time_record/each_pops.json'

generate_T = {
    'StartTime': [],
    'EndTime': []
}

each_pops_RunT = {
    'Time': [],
    'Value':[]
}

with open(startT, 'w') as f:
    json.dump(start_time, f)

with open(LLM_GenerateT, 'w') as f:
    json.dump(generate_T, f)

with open(Each_PopsT, 'w') as f:
    json.dump(each_pops_RunT, f)


