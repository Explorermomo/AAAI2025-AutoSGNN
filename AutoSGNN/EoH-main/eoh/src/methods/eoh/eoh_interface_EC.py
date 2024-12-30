import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs) 

        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False


    def population_generation(self):
        
        n_create = 2
        
        population = []

        for i in range(n_create):
            _,pop = self.get_algorithm([],'i1')
            for p in pop:
                population.append(p)
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,parents,operator,dpo_parents):  
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()     
        elif operator == "e1":
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            # parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "dpo":
            # parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.dpo(dpo_parents[0], dpo_parents[1]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return [parents, offspring]

    def get_offspring(self, pop, operator,dpo_parents):

        try:
            [p, offspring] = self._get_alg(pop, operator, dpo_parents) 
            
        except Exception as e:
            print(f"Error22222222222222: {e}")

            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }
            p = None

        # Round the objective values
        return [p, offspring]            
            

    
    def get_algorithm(self, pop, operator, dpo_parents):
        try:
            results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(delayed(self.get_offspring)(pop, operator,dpo_parents) for _ in range(self.pop_size))

        except Exception as e:
            # if self.debug:
            print(f"Error1111111111111111111: {e}")
            results = []
            # print("Parallel time out .")

        time.sleep(2)


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off
