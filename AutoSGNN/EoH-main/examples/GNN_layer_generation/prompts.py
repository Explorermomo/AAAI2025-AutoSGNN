class GetPrompts():
    def __init__(self):
        
    ### Modify the graph dataset type
    #  Internet social Network graphs
    #  citation network graphs
    #  e-commerce network

        self.prompt_task = f"The essence of spectral graph neural network is to capture the information of the graph by converting the graph signal to the frequency domain and propagating and filtering the information among nodes through adaptive learning some types and parameters of filters.\
        It seems that different types of graphs lend themselves to different filters for inter-node communication.\
               I need help in designing a spectral graphs new neural network that can effectively learn node embeddings for 'citation network graphs', with the ultimate goal of improving the accuracy of node classification. \n \
               Tips:There may be some heterophilic graphs that do not follow the homogeneity assumption, heterophily is more like the feature or label difference between the neighbors under the nodes with the same type! "
        self.prompt_class_name = "GNN_Layer"
        self.prompt_class_initial_inputs = ['input_channels', 'output_channels']
        self.prompt_forword_func_inputs = ['x', 'x_raw', 'edge_index']
        self.prompt_forword_func_outputs = ['hidden']
        self.prompt_inout_inf = "'x' is the node's processed features, shape: [node_num, input_channels]; 'x_raw' is the node's initial features, shape: [node_num, input_channels]; 'hidden' is the learned node feature, shape: [node_num, output_channels]; and 'edge_index' includes the edge information about the graph."
        self.prompt_other_inf = "All are torch tensor."

    def get_task(self):
        return self.prompt_task
    
    def get_class_name(self):
        return self.prompt_class_name
    
    def get_class_init_inputs(self):
        return self.prompt_class_initial_inputs
    
    def get_forward_func_inputs(self):
        return self.prompt_forword_func_inputs
    
    def get_forward_func_outputs(self):
        return self.prompt_forword_func_outputs

    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf


