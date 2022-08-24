'''
Created on Jul 3, 2020

@author: Stephen
'''
import random 
import scipy.stats as ss
import numpy as np
from fractions import Fraction

class Creature:
    def __init__(self, x, y):
        """
        Initializes a creature with a position and a random gene code that I think will allow it to survive a little
        """
        
        self.position = [random.randint(0,x), random.randint(0,y)]
        
        self.genecode = "" #string
        
        '''
            Start/Met/M = ATG
            Stop/B = TAA, TAG, TGA
            Phe/F = TTT, TTC
            Leu/L = TTA, TTG, CTT, CTC, CTA, CTG
            Ile/I = ATT, ATC, ATA
            Val/V = GTT, GTC, GTA, GTG
            Ser/S = TCT, TCC, TCA TCG, AGT, AGC
            Pro/P = CCT, CCC, CCA, CCG 
            Thr/T = ACT, ACC, ACA, ACG
            Ala/A = GCT, GCC, GCA, GCG
            Tyr/Y = TAT, TAC
            His/H = CAT, CAC
            Gln/Q = CAA, CAG
            Asn/N = AAT, AAC
            Lys/K = AAA, AAG
            Asp/D = GAT, GAC
            Glu/E = GAA, GAG
            Cys/C = TGT, TGC
            Trp/W = TGG
            Arg/R = CGT, CGC, CGA, CGG, AGA, AGG
            Gly/G = GGT, GGC, GGA, GGG
        '''
        """
        self.codonsReal = {
            "M" : ["ATG"],
            "B" : ["TAA", "TAG", "TGA"],
            "A" : ["GCT", "GCC", "GCA", "GCG"],
            "C" : ["TGT", "TGC"],
            "D" : ["GAT", "GAC"],
            "E" : ["GAA", "GAG"],
            "F" : ["TTT", "TTC"],
            "G" : ["GGT", "GGC", "GGA", "GGG"],
            "H" : ["CAT", "CAC"],
            "I" : ["ATT", "ATC", "ATA"],
            "K" : ["AAA", "AAG"],
            "L" : ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
            "N" : ["AAT", "AAC"],
            "P" : ["CCT", "CCC", "CCA", "CCG"],
            "Q" : ["CAA", "CAG"],
            "R" : ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
            "S" : ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
            "T" : ["ACT", "ACC", "ACA", "ACG"],
            "V" : ["GTT", "GTC", "GTA", "GTG"],
            "W" : ["TGG"],
            "Y" : ["TAT", "TAC"]
            }"""
        #Positive : "C__"
        #Negative : "G__"
            
        self.codons = {
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
            "[" : "TAA",
            "]" : "TAC",
            "," : "TAG",
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
        
        self.codons_inv = {v: k for k, v in self.codons.items()}
        self.codons_inv["AA"] = 0.0
        
        self.energy_max = None
        self.keep_amount = None
        self.mutation_chance = None
        self.death_chance = None
        self.sense_range = None
        self.weights = None
        self.aggression_chance = None
        self.attack_chance = None
        self.mate_chance = None
        self.mate_success_chance = None
        
        """
        TODO:
        Change to standard dictionary version:
            standard_dict1 = {
                AAA:1
                AAC:2
                AAT:3
                .
                .
                .}
            standard_dict2 = {
                1:"+"
                2:"-"
                3:"*"
                .
                .
                .}
        """
        
    def generate_random_genecode(self):
        """
        Generates a random gene code for this creature 
        """
        
        self.energy_max = random.randint(0,15)
        self.keep_amount = random.randint(0, self.energy_max)
        self.mutation_chance = random.randint(1,2)
        self.death_chance = random.randint(1,2)
        self.sense_range = self.make_less_16(random.randint(1, 100))
        self.weights = self.gaussian_ints(5)
        
        if __name__ == "__main__":
            print("Energy Max = " + str(self.energy_max))
            print("Keep Energy Amount = " + str(self.keep_amount))
            print("Mutation chance = " + str(self.mutation_chance))
            print("Death chance = " + str(self.death_chance))
            print("Sense Range = " + str(self.sense_range))
            print("Sense Shape = " + str(np.shape(self.sense_range)))
            print("Weights = " + str(self.weights))
            print("weights Shape = " + str(np.shape(self.weights)))
        
        self.genecode += self.make_gene(self.energy_max) 
        self.genecode += self.make_gene(self.keep_amount) 
        self.genecode += self.make_gene(self.mutation_chance)
        self.genecode += self.make_gene(self.death_chance)
        self.genecode += self.make_gene(self.sense_range)
        self.genecode += self.make_gene(self.weights)
        
        return True
    
    
    def make_gene(self, thing):
        if type(thing) is list:# and not np.shape(thing) == (1,1):
            return self.make_gene_array(thing)
        elif type(thing) is int or type(thing) is float:
            return self.make_gene_number(thing)
        '''elif  np.shape(thing) == (1,1):
            return self.make_gene_number(thing[0][0])'''
        raise ValueError
        
        
    def make_gene_number(self, number):
        gene = ""
        number = int(number)
        
        if number > 15:
            return self.make_gene_array(self.make_less_16(number))
        
        if number >= 1 or number == 0:
            gene = self.codons["start"] + self.codons["positive"] + self.codons[number] + self.codons["stop"] 
        elif number <= -1:
            gene = self.codons["start"] + self.codons["negative"] + self.codons[number] + self.codons["stop"] 
        elif abs(number) <1:
            self.make_fraction(number)
        
        return gene
        
    def make_gene_array(self, array_thing1):
        """
        Makes a gene from an Array
        Two cases:
            1) large numbers
            2) fractions
        """
        array_thing = np.asarray(array_thing1)
        
        #print(np.shape(array_thing1))
        #print(array_thing1)
        
        gene = self.codons["start"]
        
        #print(type(array_thing[1]))
        #print(type(array_thing[1]) == np.int32)
    
        if np.shape(array_thing) == (2,):
            gene += self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["positive"] + self.codons[array_thing[0]] + self.codons["+"] + self.codons["positive"] + self.codons[array_thing[1]]
            gene += self.codons["stop"]
            return gene
        
        if np.shape(array_thing)[1] == 1:
            gene += self.codons["["]
            for i in range(np.shape(array_thing)[0]):
                #print(array_thing[i,0])
                if array_thing[i,0] == 0.0 and i != np.shape(array_thing)[0] - 1:
                    gene += self.codons["["] + self.make_gene_number(array_thing[i,0])[3:-3] + self.codons["]"] + self.codons[","]
                    continue
                elif np.isclose(array_thing[i,0], 0, 1e-10)   and i == np.shape(array_thing)[0] - 1:
                    gene += self.codons["["] + self.make_gene_number(array_thing[i,0])[3:-3] + self.codons["]"]
                    continue
                else: pass
                
                if i != np.shape(array_thing)[0] - 1:
                    gene += self.codons["["] + self.make_gene_array(self.make_fraction(array_thing[i,0]))[3:-3] + self.codons["]"] + self.codons[","]
                else:
                    gene += self.codons["["] + self.make_gene_array(self.make_fraction(array_thing[i,0]))[3:-3] + self.codons["]"]
            gene += self.codons["]"] + self.codons["stop"]
            return gene
        else:
            for i in range(np.shape(array_thing)[1]):
                if array_thing[i,0] >=1 or array_thing[i,0] == 0:
                    gene += self.codons["("] + self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["positive"] + self.codons[abs(array_thing[i,0])]
                elif array_thing[i,0]<=-1:
                    gene += self.codons["("] + self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["negative"] + self.codons[abs(array_thing[i,0])]
                else:
                    pass
                     
                if array_thing[i,1] >=1 or array_thing[i,1] == 0:
                    gene += self.codons["+"] + self.codons["positive"] + self.codons[abs(array_thing[i,1])] + self.codons[")"]
                elif array_thing[i,1]<=-1:
                    gene += self.codons["+"] + self.codons["negative"] + self.codons[abs(array_thing[i,1])] + self.codons[")"]
                else:
                    pass
                
                if i < np.shape(array_thing)[1]-1 and not (i+1)%2 == 0:
                    gene += self.codons["/"]
        
        gene += self.codons["stop"]
        
        return gene
    
    def make_less_16(self,number):
        multiplier = number//15
        addered = number%15
        if abs(number) <15 and abs(number) >=1 :
            return number
        elif abs(number) <1 and not abs(number) == 0:
            return self.make_fraction(number)
        return [multiplier, addered]
    
    def make_fraction(self,number):
        frac = Fraction(number).limit_denominator()
        numerator, denominator = frac.numerator, frac.denominator
        output = []
        if numerator  <=15:
            output.append([0,self.make_less_16(numerator)])
        else: 
            output.append(self.make_less_16(numerator))
        if denominator  <=15:
            output.append([0,self.make_less_16(denominator)])
        else: 
            output.append(self.make_less_16(denominator))
        return output
    
    def gaussian_ints(self, n):
        x = np.arange(-100, 101)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        nums = []
        for i in range(n):
            nums.append([float(np.random.choice(x, size = 1, p = prob)/100)])
        return nums
        
    def set_genecode_str(self, stri):
        self.genecode = stri
        return self.unpack_genecode()
    
    def set_genecode_numbers(self, array_numbers):
        self.genecode = ""
        for i in array_numbers:
            self.genecode += self.make_gene(i)
        return True
    
    def unpack_genecode(self):
        split_at_start = self.genecode.split(sep=self.codons["stop"]+self.codons["start"])
        #print(split_at_start)
        split = []
        for i in split_at_start:
            if i != "":
                split.append(i.replace(self.codons["stop"], ""))        
        #print(split)
        string_to_eval = []
        for j in range(len(split)):
            subsplit = [split[j][i:i+3] for i in range(0, len(split[j]), 3)]
            try:
                subsplit.remove(self.codons["start"])
            except:
                pass
            #subsplit = str(subsplit).replace("[", '').replace(']','').replace("'", "")
            #print(subsplit)
            super_subsplit = []
            if len(subsplit) == 1:
                super_subsplit = [subsplit[0][0], subsplit[0][1:3]]
                string_to_eval.append(self.codons_inv[super_subsplit[0]] + str(self.codons_inv[super_subsplit[1]]))
            else:
                spec_subsplit = []
                spec_str_to_eval = []
                for i in range(len(subsplit)):
                    if subsplit[i][0] == "C" or subsplit[i][0] == "G":
                        spec_subsplit = [subsplit[i][0], subsplit[i][1:3]]
                        spec_str_to_eval.append(self.codons_inv[spec_subsplit[0]] + str(self.codons_inv[spec_subsplit[1]]))
                    else:
                        spec_str_to_eval.append(self.codons_inv[subsplit[i]])
                temp = ''
                for i in range(len(spec_str_to_eval)):
                    temp += spec_str_to_eval[i]
                #print(temp)
                string_to_eval.append(temp)
            #print(super_subsplit)
        #print(string_to_eval)
        for i in range(len(string_to_eval)):
            #print(string_to_eval[i])
            string_to_eval[i] = string_to_eval[i].replace("positive", '')
            string_to_eval[i] = string_to_eval[i].replace("negative", "-")
        #print(string_to_eval)
        '''
        No Future Stephen you CAN'T concatenate all of the try statements into one.
        '''
        try:
            self.energy_max = self.make_less_16(eval(string_to_eval[0]))
        except: 
            print("Energy max broke")
        try:
            self.keep_amount = self.make_less_16(eval(string_to_eval[1]))
        except: 
            print("Energy keep broke")
        try:
            self.mutation_chance = self.make_less_16(eval(string_to_eval[2]))
        except: 
            print("Mutation Chance broke")
        try:
            self.death_chance = self.make_less_16(eval(string_to_eval[3]))
        except: 
            print("Death chance broke")
        try:
            self.sense_range = self.make_less_16(eval(string_to_eval[4]))
        except: 
            print("Sense Range broke")
        try:
            #print(string_to_eval[5])
            self.weights = []
            self.weights = eval(string_to_eval[5])
            
        except: 
            print("Weights broke")
        
        if __name__ == "__main__":
            print("Energy Max = " + str(self.energy_max))
            print("Keep Energy Amount = " + str(self.keep_amount))
            print("Mutation chance = " + str(self.mutation_chance))
            print("Death chance = " + str(self.death_chance))
            print("Sense Range = " + str(self.sense_range))
            print("Weights = " + str(self.weights))
        
        return True
if __name__ == "__main__":
    Test_creature = Creature(0,0)
    Test_creature.generate_random_genecode()
    print("Gene code = " + Test_creature.genecode)
    Test_creature.weights = []
    Test_creature.unpack_genecode()
    
        