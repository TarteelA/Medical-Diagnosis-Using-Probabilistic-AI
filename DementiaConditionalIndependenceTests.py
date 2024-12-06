#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import bnlearn as bn
import pandas as pd

#Definition Of Directed Acyclic Graphs (Predefined Structures)
edges = [('CDR', 'Subject_ID'),('CDR', 'MRI_ID'),('CDR', 'Group'),('CDR', 'Visit'),('CDR', 'MR_Delay'),('CDR', 'M/F'),('CDR', 'Hand'),('CDR', 'Age'),('CDR', 'EDUC'),('CDR', 'SES'),('CDR', 'MMSE'),('CDR', 'eTIV'),('CDR', 'nWBV'),('CDR', 'ASF')]

#Set Training Data, Network Structure, And Type Of Conditional Independence Test
TRAINING_DATA = 'Data/Dementia/train_fold_1.csv'
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
print("model=", model)

#Statistical Test Of Independence
model = bn.independence_test(model, data, test = CONDITIONAL_INDEPENDENCE_TEST, alpha = 0.05)
ci_results = list(model['independence_test']['stat_test'])
num_edges2remove = ci_results.count(False)
print(model['independence_test'])
print("num_edges2remove = " + str(num_edges2remove))