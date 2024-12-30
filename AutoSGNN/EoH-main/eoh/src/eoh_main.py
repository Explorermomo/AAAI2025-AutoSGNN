import random

from .utils import createFolders
from .methods import methods
from .problems import problems

# main class for AEL
class EVOL:

    # initilization
    def __init__(self, paras, prob=None, **kwargs):

        print("-----------------------------------------")
        print("---              Start                ---")
        print("-----------------------------------------")
        self.paras = paras
        print("-  parameters loaded -")
        self.prob = prob
        # Set a random seed
        random.seed(2024)

        
    # run methods
    def run(self):

        problemGenerator = problems.Probs(self.paras)  ### prompts 

        problem = problemGenerator.get_problem()

        methodGenerator = methods.Methods(self.paras, problem)  ### get the instanse of the problem

        method = methodGenerator.get_method()  ### get eoh methods and some parameters of methods

        method.run()

        print("> End of Evolution! ")
        print("-----------------------------------------")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

