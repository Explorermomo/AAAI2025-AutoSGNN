import numpy as np
import types
import warnings
import sys


class GNN_LG():
    def __init__(self) -> None:
        self.n_inst_eva = 3 # a samll value for test only
        self.time_limit = 10 # maximum 10 seconds for each instance
        self.ite_max = 1000 # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1 # movers of each edge in each perturbation
        self.debug_mode=False

        from prompts import GetPrompts
        self.prompts = GetPrompts()


    def tour_cost(self,instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self,instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix
    

    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)
                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                #print(code_string)
                fitness = self.evaluateGLS(heuristic_module)

                return fitness
            
        except Exception as e:
            #print("Error:", str(e))
            return None



