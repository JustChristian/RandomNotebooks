import random
from abc import abstractmethod, ABCMeta

class TournamentPopulation(object):
    
    """
    @param function individual_class ....
    @param integer population_size   number of individuals in the population
    @param float infection_rate      the percentage of genes replaced on infection
    @param float mutation_prob       the probability of any gene mutating after infection
    @param integer deme_size         distance in one direction that 2nd parent can be from 1st parent in selection
    """
    def __init__(self, individual_class, population_size, infection_rate=0.5, mutation_prob=0.02, deme_size=5):
        self.infection_rate = infection_rate
        self.mutation_prob = mutation_prob
        self.deme_size = deme_size
        
        #Build a randomly generated population
        self.population = [individual_class() for i in range(population_size)]
        
        #Get the highest performing individual
        self.highest_fitness_individual = sorted(self.population, key=lambda individual: individual.fitness())[-1]
        self.highest_fitness = self.highest_fitness_individual.fitness()
        
        
    """
    microbial infection based reproduction, the individual with lower fitness 
    will have a portion of its genes replaced with genes from the winner
    
    @param Individual[] parents    a list containing two parents that will be used to reproduce
    """
    def reproduction(self, parents):
        parents = sorted(parents, key=lambda individual: individual.fitness())
        winner = parents[1]
        loser = parents[0]
        
        #'infect' the loser of the tournament, the winner should be championed through without changes
        loser.infect(winner, self.infection_rate)
        loser.mutate(self.mutation_prob)
            
        #Check if fitness of new individual is the highest so far
        new_individual_fitness = loser.fitness()
        if self.highest_fitness < new_individual_fitness:
            self.highest_fitness = new_individual_fitness
            self.highest_fitness_individual = loser
            
    """
    Pick 2 indices of population randomly, with trivial geography
    
    trivial-geography -> limits how far the 2nd parent can be from the 1st parent
    
    @return Individual[]    an array of two random individual from the population
    """
    def selection(self):
        #Select the first parent randomly from the entire population
        first_parent_index = random.randint(0, len(self.population)-1)
        first_parent = self.population[first_parent_index]
        
        #Select the second parent from a deme or subset of the population
        second_parent_offset = random.randint(0, self.deme_size)
        second_parent_index = (first_parent_index + second_parent_offset - 1) % len(self.population)
        second_parent = self.population[second_parent_index]
        
        return [first_parent, second_parent]
    
    
    """
    Runs a complete cycle of selection through microbial reproduction
    """
    def run_cycle(self):
        parents = self.selection()
        self.reproduction(parents)
            

            
            
            

class AbstractLinkedListIndividual(object, metaclass=ABCMeta): #ABCMeta marks the class as abstract
    
    """
    Infect this individual with parts of the genotype of another individual
    
    How the genotype is encoded and fitness is implementation specific, but the 
    expectation is the genotype is stores in an array/'linked list' in self.genotype
    
    @param Individual infector    the individual whose genes are being injected into this individual
    @param float infection_perc   the percentage of genes that are being replaced by the infector's genes
    """
    def infect(self, infector, infection_perc):
        if len(self.genotype) != len(infector.genotype):
            raise ValueError("The genotype of the infecting individual should be the same size")
        
        #Determine how many indices are replaced the infecting 
        infected_indices_count = round(len(self.genotype) * infection_perc)
        
        #Get the indices that are being replaced
        infected_indices = random.sample(range(0, len(self.genotype)), infected_indices_count)
        
        #Replace the infected indexs with the same indexes from the infecting individual
        for infected_index in infected_indices:
            self.genotype[infected_index] = infector.genotype[infected_index]
    
    #Check if each gene in the genotype should be mutated
    @abstractmethod
    def mutate(self, mutation_rate):
        pass
    
    @abstractmethod
    def fitness(self):
        pass