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
        '''"""
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
    def generate_random_genecode(self):
        """
        Generates a random gene code for this creature 
        """
        self.energy_max = random.randint(0,15)
        print("Energy Max = " + str(self.energy_max))
        
        self.keep_amount = random.randint(0, self.energy_max)
        print("Keep Energy Amount = " + str(self.keep_amount))
        
        self.mutation_chance = random.randint(1,2)
        print("Mutation chance = " + str(self.mutation_chance))
        
        self.death_chance = random.randint(1,2)
        print("Death chance = " + str(self.death_chance))
        
        self.sense_range = self.make_less_16(random.randint(1, 100))
        print("Sense Range = " + str(self.sense_range))
        
        self.weights = self.make_fraction(self.gaussian_ints())
        print("Weights = " + str(self.weights))
        
        self.genecode += self.make_gene(self.energy_max) 
        self.genecode += self.make_gene(self.keep_amount) 
        self.genecode += self.make_gene(self.mutation_chance)
        self.genecode += self.make_gene(self.death_chance)
        self.genecode += self.make_gene(self.sense_range)
        self.genecode += self.make_gene(self.weights)
        
        return True
    
    
    def make_gene(self, thing):
        if type(thing) is list:
            return self.make_gene_array(thing)
        elif type(thing) is int or type(thing) is float:
            return self.make_gene_number(thing)
        
        raise ValueError
        
        
    def make_gene_number(self, number):
        gene = ""
        number = int(number)
        
        if number > 15:
            return self.make_gene_array(self.make_less_16(number))
        
        if number >= 0:
            gene = self.codons["start"] + self.codons["positive"] + self.codons[number] + self.codons["stop"] 
        else:
            gene = self.codons["start"] + self.codons["negative"] + self.codons[number] + self.codons["stop"] 
        
        return gene
        
    def make_gene_array(self, array_thing1):
        """
        Makes a gene from an Array
        Two cases:
            1) large numbers
            2) fractions
        """
        array_thing = np.asarray(array_thing1)
        
        
        print(np.shape(array_thing1))
        print(array_thing1)
        
        gene = self.codons["start"]
        
        print(type(array_thing[1]))
        print(type(array_thing[1]) == np.int32)
        
        if np.shape(array_thing) == (2,):
            if type(array_thing[])
            gene += self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["positive"] + self.codons[array_thing[0]] + self.codons["+"] + self.codons["positive"] + self.codons[array_thing[1]]
            
        
        else:
            for i in range(np.shape(array_thing)[1]):
                if array_thing[0,i] >=0:
                    gene += self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["positive"] + self.codons[abs(array_thing[0,i])] 
                else:
                    gene += self.codons["positive"] + self.codons[15] + self.codons["*"] + self.codons["negative"] + self.codons[abs(array_thing[0,i])]
                
                if array_thing[1,i] >=0:
                    gene += self.codons["+"] + self.codons["positive"] + self.codons[abs(array_thing[1,i])]
                else:
                    gene += self.codons["+"] + self.codons["negative"] + self.codons[abs(array_thing[1,i])]
                
                if i < np.shape(array_thing)[1]:
                    gene += self.codons["/"]
        
        gene += self.codons["stop"]
        
        return gene
        
        
    
    def make_less_16(self,number):
        multiplier = number//15
        addered = number%15
        if number <16:
            return number
        return [multiplier, addered]
    
    def make_fraction(self,number):
        frac = Fraction(number).limit_denominator()
        numerator, denominator = frac.numerator, frac.denominator
        return [self.make_less_16(numerator), self.make_less_16(denominator)]
    
    def gaussian_ints(self):
        x = np.arange(-100, 101)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        nums = np.random.choice(x, size = 1, p = prob)
        return nums[0]/100
        
    def set_genecode_str(self, stri):
        self.genecode = stri
        return True
    
    def set_genecode_numbers(self, array_numbers):
        self.genecode = ""
        for i in array_numbers:
            self.genecode += self.make_gene(i)
        return True
    
    def unpack_genecode(self):
        split_at_start = self.genecode.split(sep=self.codons["start"])
        print(split_at_start)
    
    
    
if __name__ == "__main__":
    Test_creature = Creature(0,0)
    Test_creature.generate_random_genecode()
    print("Gene code = " + Test_creature.genecode)
    Test_creature.unpack_genecode()
    
        