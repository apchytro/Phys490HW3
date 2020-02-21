import numpy as np
import os


# Data class from data_gen.py
# Handles reading data from datafile 
# Stores two arrays:
# data.x is an array of spin configurations where spins are +- 1
# data.y is an array of spin configuration probabilities

class Data():
    
    def __init__(self, datafile):
        
        # Open file
        directory = os.getcwd()
        data_directory = os.path.join(directory, datafile)
        file1 = open(data_directory,"r")
        
        # Iterate through lines storing num of occurences of each configuration
        dicto = {}
        num_lines = 0 
        for line in file1:
            num_lines += 1
            if line in dicto: dicto[line] += 1
            else: dicto[line] = 1
        
        keys = []
        prob = []
        
        # Create probabilities and convert '+' '-' to +-1 
        for item in dicto:
            prob.append (dicto[item]/num_lines)
            temp = []
            for i in item:
                if i == '+': temp.append(1)
                elif i == '-': temp.append(-1)
            keys.append(temp)
        
        self.x = (np.array(keys)).astype(np.float32)
        self.y = (np.array(prob)).astype(np.float32)

        
