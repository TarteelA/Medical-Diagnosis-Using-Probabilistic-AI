#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import bnlearn as bn
import pandas as pd

#Definition Of Directed Acyclic Graphs (Predefined Structures)
edges = [('status', 'name'),('status', 'MDVPFoHz'),('status', 'MDVPFhiHz'),('status', 'MDVPFloHz'),('status', 'MDVPJitter'),('status', 'MDVPJitterAbs'),('status', 'MDVPRAP'),('status', 'MDVPPPQ'),('status', 'JitterDDP'),('status', 'MDVPShimmer'),('status', 'MDVPShimmerdB'),('status', 'ShimmerAPQ3'),('status', 'ShimmerAPQ5'),('status', 'MDVPAPQ'),('status', 'ShimmerDDA'),('status', 'NHR'),('status', 'HNR'),('status', 'RPDE'),('status', 'DFA'),('status', 'spread1'),('status', 'spread2'),('status', 'D2'),('status', 'PPE')]

#Set Training Data, Network Structure, And Type Of Conditional Independence Test
TRAINING_DATA = 'Data/Parkinson/train_fold_1.csv'
NETWORK_STRUCTURE = edges
CONDITIONAL_INDEPENDENCE_TEST = 'cressie_read'

#Data Loading Using Pandas
data = pd.read_csv(TRAINING_DATA, encoding = 'UTF-8')
print("DATA:\n", data)

#Creation Of Directed Acyclic Graph (DAG)
DAG = bn.make_DAG(NETWORK_STRUCTURE)
print("DAG:\n", DAG)

#Parameter Learning Using Maximum Likelihood Estimation
model = bn.parameter_learning.fit(DAG, data, methodtype = "maximumlikelihood")
print("model = ", model)

#Statistical Test Of Independence
model = bn.independence_test(model, data, test = CONDITIONAL_INDEPENDENCE_TEST, alpha = 0.05)
ci_results = list(model['independence_test']['stat_test'])
num_edges2remove = ci_results.count(False)
print(model['independence_test'])
print("num_edges2remove = " + str(num_edges2remove))