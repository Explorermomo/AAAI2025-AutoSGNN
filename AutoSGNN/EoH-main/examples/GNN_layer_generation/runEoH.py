from eoh.src import eoh_main
from eoh.src.utils.getParas import Paras

from prob import *

# Parameter initilization
paras = Paras()

# Set your local problem
problem_local = GNN_LG()

# Set parameters
paras.set_paras(method = "eoh",    # ['ael','eoh']
                problem = problem_local, # Set local problem, else use default problems
                llm_api_endpoint = "api.chatanywhere.tech", # set your LLM endpoint
                llm_api_key = "xxxx",   # set your key
                llm_model = "gpt-3.5-turbo-ca",
                # llm_model = "gpt-4o-ca",
                ec_pop_size = 4, # number of samples in each population
                ec_n_pop = 4,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False,
                eva_numba_decorator = False,
                eva_timeout = 60 ,
                exp_use_continue = True,
                ec_operators = ['e1','e2','dpo'],  # ['e1','e2','dpo']
                ec_operator_weights = [1,1,1],
                # Set the maximum evaluation time for each heuristic !
                # Increase it if more instances are used for evaluation !
                ) 

# initilization
evolution = eoh_main.EVOL(paras)

# run 
evolution.run()
