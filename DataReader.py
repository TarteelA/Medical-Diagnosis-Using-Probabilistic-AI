#Author: Tarteel Alkaraan (25847208)
#Last Updated: 12 November 2024 
#References: Updated Code From Module Workshops 1 To 6

#Declare CSV Data Reader Class
class CSV_DataReader:
    def __init__(self, file_name):
        if file_name is None:
            raise ValueError("Error: No file name provided.")
        
        #Initialize Class Attributes
        self.rand_vars = []
        self.rv_key_values = {}
        self.rv_all_values = []
        self.predictor_variable = None
        self.num_data_instances = 0

        #Read Data From Provided File
        self.read_data(file_name)

    def read_data(self, data_file):
        print("\nREADING data file %s..." % (data_file))
        print("---------------------------------------")

        try:
            with open(data_file, encoding='UTF-8') as csv_file:
                #Track If We Are Reading Header
                first_line = True  
                for line in csv_file:
                    line = line.strip().replace('\ufeff', '')
                    if len(line) == 0:
                        #Skip Empty Lines
                        continue  

                    values = line.split(',')
                    if first_line:
                        #Read Header Line
                        self.rand_vars = [var.replace('\ufeff', '').strip() for var in values]
                        for variable in self.rand_vars:
                            self.rv_key_values[variable] = []
                        first_line = False
                    else:
                        #Read Data Lines
                        self.rv_all_values.append(values)
                        self.update_variable_key_values(values)
                        self.num_data_instances += 1

            #Set Predictor Variable
            self.predictor_variable = self.rand_vars[-1]

            #Debugging Outputs
            print("RANDOM VARIABLES=%s" % (self.rand_vars))
            print("VARIABLE KEY VALUES=%s" % (self.rv_key_values))
            print("VARIABLE VALUES=%s" % (self.rv_all_values))
            print("PREDICTOR VARIABLE=%s" % (self.predictor_variable))
            print("|data instances|=%d" % (self.num_data_instances))

        except FileNotFoundError:
            print(f"Error: The file '{data_file}' was not found.")
        except Exception as e:
            print(f"Error reading data file: {e}")

    def update_variable_key_values(self, values):
        for i, variable in enumerate(self.rand_vars):
            #Ensure Index Is Within Bounds
            if i < len(values):  
                value_in_focus = values[i]
                if value_in_focus not in self.rv_key_values[variable]:
                    self.rv_key_values[variable].append(value_in_focus)

    def get_true_values(self):
        #Extract True Values For Predictor Variable
        if self.predictor_variable is None:
            print("No predictor variable set. Cannot retrieve true values.")
            return []

        try:
            predictor_index = self.rand_vars.index(self.predictor_variable)
            true_values = [row[predictor_index] for row in self.rv_all_values if len(row) > predictor_index]
            return true_values
        except Exception as e:
            print(f"Error retrieving true values: {e}")
            return []