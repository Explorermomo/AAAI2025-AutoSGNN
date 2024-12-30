import re
import time
from ...llm.interface_LLM import InterfaceLLM

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs):

        self.prompt_task         = prompts.get_task()
        self.prompt_class_name    = prompts.get_class_name()
        self.prompt_class_initial_inputs  = prompts.get_class_init_inputs()
        self.prompt_forword_func_inputs = prompts.get_forward_func_inputs()
        self.prompt_forword_func_outputs = prompts.get_forward_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        if len(self.prompt_class_initial_inputs) > 1:
            self.joined_init_inputs = ", ".join("'" + s + "'" for s in self.prompt_class_initial_inputs)
        else:
            self.joined_init_inputs = "'" + self.prompt_class_initial_inputs[0] + "'"

        if len(self.prompt_forword_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_forword_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_forword_func_inputs[0] + "'"

        if len(self.prompt_forword_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_forword_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_forword_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking


        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,llm_use_local,llm_local_url, self.debug_mode)

    def get_prompt_i1(self):
        
        prompt_content = self.prompt_task+"\n"\
"First, describe your new algorithm and main steps in one sentence. \
The description must be inside a brace. Next, implement it in Python as a function named \
"+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Prohibit the use of from xxx import *."+"Please add the end identifier '#code_end' at the end of the code segment"+"Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"+"score:"+"\n"+str(indivs[i]['objective'])+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing spectral graph neural network layers with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new spectral graph neural network layers that has a totally different form from the given ones. \n"\
"First, describe your new network layers and main steps in one sentence. \
Your new algorithm description must be inside a brace. Next, implement it in Python as a class named \
"+self.prompt_class_name +". This class initial should accept "+str(len(self.prompt_class_initial_inputs))+" input(s): "\
+self.joined_init_inputs+". The forward function should accept" + str(len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return "+str(len(self.prompt_forword_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Please add the end identifier '#code_end' at the end of the code segment"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"+"score:"+"\n"+str(indivs[i]['objective'])+"\n"

        prompt_content = self.prompt_task+"\n"\
"I have "+str(len(indivs))+" existing spectral graph neural network layers with their codes as follows: \n"\
+prompt_indiv+\
"Please help me create a new spectral graph neural network layers that has a totally different form from the given ones but can be motivated from them. \n"\
"Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
Your new algorithm description must be inside a brace. Thirdly, implement it in Python as a class named \
"+self.prompt_class_name +". This class initial should accept "+str(len(self.prompt_class_initial_inputs))+" input(s): "\
+self.joined_init_inputs+". The forward function should accept" + str(len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return "+str(len(self.prompt_forword_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Please add the end identifier '#code_end' at the end of the code segment"+"Do not give additional explanations."
        return prompt_content
    
    def get_prompt_dpo(self,indiv1,indiv2):
        prompt_content = self.prompt_task+"\n"\
"I have two spectral graph neural network layers with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
"+indiv1['code']+"\n\
Score:\n\
"+str(indiv1['objective'])+"\n\
Code:\n\
"+indiv2['code']+"\n\
Score:\n\
"+str(indiv2['objective'])+"\n\
First, analyze the principles of both; \n Second, by comparing their scores, evaluate why the filter with the larger of the scores is more suitable for the graphs. \n"\
"Finally, help me to create a new spectral graph neural network layer, and describe your new algorithm and main steps in one sentence.\
Your new algorithm description must be inside a brace. Next, implement it in Python as a class named \
"+self.prompt_class_name +". This class initial should accept "+str(len(self.prompt_class_initial_inputs))+" input(s): "\
+self.joined_init_inputs+". The forward function should accept" + str(len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return "+str(len(self.prompt_forword_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"+"Please add the end identifier '#code_end' at the end of the code segment"+"Do not give additional explanations."
        return prompt_content




    # def _get_alg(self,pop,operator): 

    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=class)', response,re.DOTALL)

        code = re.findall(r"import .*#code_end", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"import .*#code_end", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response,re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response,re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=class)', response,re.DOTALL)

            code = re.findall(r"import .*#code_end", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"import .*#code_end", response, re.DOTALL)
                
            if n_retry > 3:
                break
            n_retry +=1

        algorithm = algorithm[0]
        code = code[0] 


        return [code, algorithm]


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def dpo(self,parents1, parents2):
      
        prompt_content = self.get_prompt_dpo(parents1, parents2)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]