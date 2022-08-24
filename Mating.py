'''
Created on Dec 30, 2020

@author: Stephen
'''
import numpy as np
import random

class Mate:
    def __init__(self,n, creatures):
        """
        Takes in two creatures, mutates the creatures, returning n of them 
        (probably the first n if n%2 != 0 ie if 3 there are 2 combos for each mutation then one full set of mutated genecodes will be returned and on from the other set) 
        """
        self.n = n
        self.creatures = creatures
    
    def mate(self):
        mating_chances = []
        for creature in self.creatures:
            mating_chances.append(creature.mating_chance)
        if random.randint(0,100) <= max(mating_chances):
            mating_success_chances = []
            for creature in self.creatures:
                mating_success_chances.append(creature.mating_success_chance)
            
        
    def mutate(self):
        mutation_chances = []
        for creature in self.creatures:
            mutation_chances.append(creature.mutation_chance)
        if random.randint(0,100) <= max(mutation_chances):
            