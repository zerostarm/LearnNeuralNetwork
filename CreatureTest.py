'''
Created on Dec 30, 2020

@author: Stephen
'''
import numpy as np
weights = [1,[5,6]]

codons = {
            "+" : "AAC",
            "-" : "AAG",
            "*" : "AAT",
            "/" : "ACA",
            "start" : "AAA",
            "stop" : "TTT",
            "positive" : "C",
            "negative" : "G",
            "(" : "ACG",
            ")" : "ACT",
            0 : "AA",
            1 : "AC",
            2 : "AG",
            3 : "AT",
            4 : "CA",
            5 : "CC",
            6 : "CG",
            7 : "CT",
            8 : "GA",
            9 : "GC",
            10 : "GG",
            11 : "GT",
            12 : "TA",
            13 : "TC",
            14 : "TG",
            15 : "TT"
            }
gene = ''
array_thing = np.asarray(weights)
if np.shape(weights) == (2,):
            for i in range(np.shape(array_thing)[0]):
                if i%2 == 0:
                    gene += codons["positive"] + codons[abs(array_thing[i])]
                else:
                    gene += codons["positive"] + codons[15] + codons["*"] + codons["positive"] + codons[array_thing[i]] + codons["+"] + codons["positive"] + codons[array_thing[i]]
