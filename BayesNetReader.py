#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Import Libraries
import sys
import pickle

#Declare Bayes Net Reader Class
class BayesNetReader:
    def __init__(self, file_name):
        #Make bn An Instance Variable
        self.bn = {}
        self.read_data(file_name)
        self.tokenise_data()

    #Starts Loading Configuration File Into Dictionary 'bn', By Splitting Strings With Character ':' And Storing Keys & Values 
    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))

        with open(data_file, encoding='utf-8-sig') as cfg_file:
            key = None
            value = None
            for line in cfg_file:
                line = line.strip().replace('\ufeff', '')
                if len(line) == 0:
                    continue

                tokens = line.split(":")
                if len(tokens) == 2:
                    if value is not None:
                        self.bn[key] = value
                        value = None

                    key = tokens[0].replace('\ufeff', '')
                    value = tokens[1].replace('\ufeff', '')
                else:
                    value += tokens[0].replace('\ufeff', '')

            #Ensure Last Key-Value Pair Is Added 
            self.bn[key] = value
            self.bn["random_variables_raw"] = self.bn["random_variables"]
            print("RAW key-values=" + str(self.bn))

    #Continues Loading Configuration File Into Dictionary 'bn', By Separating Key-Value Pairs
    def tokenise_data(self):
        print("TOKENISING data...")
        rv_key_values = {}

        for key, values in self.bn.items():
            if key == "name":
                values = values.strip()
                
            elif key == "random_variables":
                var_set = []
                for value in values.split(";"):
                    if value.find("(") and value.find(")"):
                        value = value.replace('(', ' ').replace(')', ' ').replace('\ufeff', '')
                        parts = value.split(' ')
                        var_set.append(parts[1].strip())
                    else:
                        var_set.append(value)
                self.bn[key] = var_set

            elif key.startswith("CPT"):
                #Store Conditional Probability Tables (CPTs) As Dictionaries
                cpt = {}
                sum = 0
                for value in values.split(";"):
                    pair = value.split("=")
                    #Check For Valid Key-Value Pairs
                    cpt[pair[0].replace('\ufeff', '')] = float(pair[1].replace('\ufeff', ''))
                    sum += float(pair[1].replace('\ufeff', ''))
                print("key=%s cpt=%s sum=%s" % (key, cpt, sum))
                self.bn[key] = cpt

                #Store Unique Values For Each Random Variable
                if key.find("|") > 0:
                    rand_var = key[4:].split("|")[0] 
                else:
                    rand_var = key[4:].split(")")[0]
                unique_values = list(cpt.keys())
                rv_key_values[rand_var.replace('\ufeff', '')] = unique_values

            else:
                if type(values) is dict:
                    continue
                values = [val.replace('\ufeff', '') for val in values.split(";")]
                if len(values) > 1:
                    self.bn[key] = values

        self.bn['rv_key_values'] = rv_key_values
        print("TOKENISED key-values=" + str(self.bn))
        
        def load_regression_models(self):
            # check whether the regression_models exist (defined in the config file)
            is_regression_models_available = False
            for key, value in self.bn.items():
                if key == "regression_models":
                    is_regression_models_available = True
				
            # loads the regression_models as per the .pkl file in the config file
            if  is_regression_models_available:
                try:
                    configfile_name = self.bn["regression_models"]
                    print("\nLOADING %s ..." % (configfile_name))
                    models_file = open(configfile_name, 'rb')
                    regression_models = pickle.load(models_file)
                    self.bn["means"] = regression_models["means"]
                    self.bn["stdevs"] = regression_models["stdevs"]
                    self.bn["regressors"] = regression_models["regressors"]
                    
                    models_file.close()
                    print("Regression models loaded!")

                except Exception:
                    print("Couldn't find file %s" % (configfile_name))
                    pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: BayesNetReader.py [config_file.txt]")
    else:
        file_name = sys.argv[1]
        BayesNetReader(file_name)