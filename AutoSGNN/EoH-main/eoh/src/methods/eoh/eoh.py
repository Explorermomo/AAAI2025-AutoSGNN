import os
ABS_PATH = os.getcwd()
Generation_Path = os.path.join(ABS_PATH,  'examples/result/LLM_generation/generate_pop/generate_pop.json')

import numpy as np
import json
import random
import time
random.seed(time.time())

from .eoh_interface_EC import InterfaceEC
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)

    def weighted_sample_without_replacement(population, weights, k):
        sampled_indices = []
        population_copy = population[:]
        weights_copy = weights[:]
        for _ in range(k):
            total_weight = sum(weights_copy)
            probs = [w / total_weight for w in weights_copy]
            chosen_index = random.choices(range(len(population_copy)), weights=probs, k=1)[0]
            sampled_indices.append(population_copy.pop(chosen_index))
            weights_copy.pop(chosen_index)
        return sampled_indices
    

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        # time_start = time.time()

        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )


        population, conditate_population = [],[]
        # print("load initial population(parents pop) from " + self.load_pop_path)
        with open(self.load_pop_path) as file:
            data = json.load(file)
        for individual in data:
            conditate_population.append(individual)


        parents_pop_indices = [random.randint(0,int(len(conditate_population)/3)), random.randint(0,len(conditate_population)-1), random.randint(0,len(conditate_population)-1), random.randint(0,len(conditate_population)-1)]
        dpo_parents_indices  = [random.randint(0,int(len(conditate_population)/3)), random.randint(int(len(conditate_population)/2),len(conditate_population)-1)]

        print(f"initial dpo parents {dpo_parents_indices} have beed loaded!")

        parents_pop = [conditate_population[mm] for mm in parents_pop_indices]
        dpo_parents = [conditate_population[mm] for mm in dpo_parents_indices]

        print(f"initial population{parents_pop_indices} has been loaded!")
        n_op = len(self.operators)
        # try:
        for i in range(n_op):
            op = self.operators[i]
            print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
            op_w = self.operator_weights[i]
            if (np.random.rand() < op_w):   
                _, offsprings = interface_ec.get_algorithm(parents_pop, op, dpo_parents) 
            self.add2pop(population, offsprings)  # Check duplication, and add the new offspring

        # Save population to a file
        filename = Generation_Path
        with open(filename, 'w') as f:
            json.dump(population, f, indent=5)



