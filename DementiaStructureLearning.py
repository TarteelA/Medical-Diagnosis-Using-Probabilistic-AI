#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#Set Training Data, Type Of Scoring Function, Number Of Iterations, And Visualise Structure
TRAINING_DATA = 'Data/Dementia/train_fold_1.csv'
SCORING_FUNCTION = 'aic'
MAX_ITERATIONS = 20000000
VISUALISE_STRUCTURE = True

#Data Loading Using Pandas
data = pd.read_csv(TRAINING_DATA, encoding = 'UTF-8', nrows = 299)
data = data.dropna()
data = data[~data.isin([float('inf'), -float('inf')]).any(axis = 1)]
print("DATA:\n", data)

#Definition Of Directed Acyclic Graph
edges = [('CDR', 'Subject_ID'),('CDR', 'MRI_ID'),('CDR', 'Group'),('CDR', 'Visit'),('CDR', 'MR_Delay'),('CDR', 'M/F'),('CDR', 'Hand'),('CDR', 'Age'),('CDR', 'EDUC'),('CDR', 'SES'),('CDR', 'MMSE'),('CDR', 'eTIV'),('CDR', 'nWBV'),('CDR', 'ASF')]

#Performs Discretisation Of Continuous Data For Columns Specified and Structure Provided Used For Training Bayesian Network
continuous_columns = ["MR_Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
discrete_data = bn.discretize(data, edges, continuous_columns, max_iterations = 1, verbose = 3)
for randvar in discrete_data:
    print("VARIABLE:", randvar)
    print(discrete_data[randvar])

#Structure Learning Using Chosen Scoring Function As Per SCORING_FUNCTION
model = bn.structure_learning.fit(discrete_data, methodtype = 'hillclimbsearch', scoretype = SCORING_FUNCTION, max_iter = MAX_ITERATIONS)
num_model_edges = len(model['model_edges'])
print("model = ", model)
print("num_model_edges = " + str(num_model_edges))

#Visualise Learnt Structure
if VISUALISE_STRUCTURE:
    G = nx.DiGraph()
    G.add_edges_from(model['model_edges'])
    pos = nx.spring_layout(G)
    plt.figure(figsize = (8, 6))
    nx.draw(G, pos, with_labels = True, node_size = 500, node_color = 'lightgreen', font_size = 10, arrows = True)
    plt.title('Directed Acyclic Graph (DAG)')
    plt.show()