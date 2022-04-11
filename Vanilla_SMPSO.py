import random
import math
import numpy as np
from copy import deepcopy
import multiprocessing
import time

from deap import base, creator, tools

from Simulation_Model.Evaluate_Individual import Evaluate
from Simulation_Model import GF

# Initialize creator class
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Particle", np.ndarray, typecode="d", fitness=creator.FitnessMulti, speed=None, smin=None, smax=None, best=None)

class SMPSO:
    """ 
    Description: Holds all functionality of the Speed-constrained Multi-objective Particle Swarm Optimization (SMPSO)
    Input:  - num_gen: The number of generations to be executed until termination
            - pop_size: The number of individuals simultanteously represented in an evolutionary population
            - MP: if set to a value higher than 0, the given number of cores will be utilized
            - verbose: If True the performance of every population will be printed
    """
    def __init__(self, num_gen, pop_size, MP, verbose):
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.IND_SIZE = 96
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.DELTA = (self.BOUND_U - self.BOUND_L) / 2
        self.MAX_ARCHIVE_SIZE = pop_size
        self.INERTIA_W = 1
        self.GBEST = None
        self.eval = Evaluate(simulation_time=40, storage_fallback=True, check_validity=False)
        self.MP = MP
        self.verbose = verbose

    def initialize_particle(self, pcls) -> object:
        """ 
        Description: Initialize a single particle, imposing set boundaries on the creation. 
        Input:  - pcls: the particle constructer object initialized in DEAP.
        Output: - part: created particle.
        """    
        # Create particle and set speed taking the set boundaries into account
        part = pcls(random.uniform(self.BOUND_L, self.BOUND_U) for _ in range(self.IND_SIZE))
        part.speed = np.array([random.uniform(-self.DELTA, self.DELTA) for _ in range(self.IND_SIZE)])
        part.smin = -self.DELTA
        part.smax = self.DELTA
        return part
    
    def initialize_global_best(self, population) -> list:
        """ 
        Description: Initialize global best archive upon the first initializes population.
        Input:  - population: the current population of particles.
        Output: current set of non-dominated solutions (Pareto frontier).
        """
        return tools.sortNondominated([deepcopy(indiv) for indiv in population], self.IND_SIZE)[0]
    
    def initialize_local_best(self, population) -> None:
        """ 
        Description: Initialize local best archive for all particles in the population.
        Input:  - population: the current population of particles.
        """
        for indiv in population:
            indiv.best = [deepcopy(indiv)]
    
    def ensure_pareto_size(self, pareto_front, max_size=None) -> list:
        """ 
        Description: Trim the Pareto front to the allowed max archive size.
        Input:  - pareto_front: the current (unbounded) pareto frontier found.
                - max_size: the maximum size of the pareto front allowed (if not given, fall back on class settings).
        Output: - pareto_front: pareto front trimmed to fall within the allowed range of numbers.
        """
        if max_size == None:
            max_size = self.MAX_ARCHIVE_SIZE
        
        if len(pareto_front) > max_size:
            tools.emo.assignCrowdingDist(pareto_front)
            pareto_front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
            return pareto_front[0:max_size]
    
        else:
            return pareto_front
    
    def update_global_best(self, population) -> list:
        """ 
        Description: Update the global best archive given the current population.
        Input:  - population: the current population of individuals.
        """
        # Add all particles of the population to the best list and re-evaluate 1st pareto front
        for particle in population:
            self.GBEST.append(deepcopy(particle))
        pareto_front = tools.sortNondominated(individuals=self.GBEST, 
                                              k=len(self.GBEST), 
                                              first_front_only=True)[0]
        
        # Ensure the global archive does not exceed the set max archive size
        self.GBEST = self.ensure_pareto_size(pareto_front)
        
    def update_local_best(self, population) -> None:
        """ 
        Description: Update the local best solution for all particles in the population.
        Input:  - population: the current population of individuals.
        """
        for idx, particle in enumerate(population): 
            # If particle is not already in the particles best list, recalculate particles best including new particle values
            if not GF.arreq_in_list(arr=particle, list_of_arrs=particle.best):
                best_particles = deepcopy(particle.best)
                best_particles.append(deepcopy(particle))
                pareto_front = tools.sortNondominated(individuals=best_particles, 
                                                      k=len(best_particles), 
                                                      first_front_only=True)[0]
                
                # Ensure the local best archive does not exceed the set max archive size
                particle.best = self.ensure_pareto_size(pareto_front, max_size=3)
    
    def velocity_constriction(self, speed) -> float:
        """ 
        Description: Constrict velocity (MOPSO) within the set boundaries.
        Input:  - speed: the current speed of the particle.
        output: - speed: the constrained speed of the particle. 
        """
        if speed > self.DELTA:
            return self.DELTA
        elif speed <= -self.DELTA:
            return -self.DELTA
        else:
            return speed
        
    def calc_constriction_coefficient(self, ind_size, c1, c2) -> list:
        """ 
        Description: Calculate the constriction coefficient (SMPSO) used to prevent particle explosion.
        Input:  - ind_size: the size of the individual particle.
                - c1, c2: random values used in particle update.
        Output: - constriction_coeffs: the constriction coefficient used in particle updates.
        """
        # Initialize monitoring list and loop over all values of c1 and c2
        constriction_coeffs = []
        for idx in range(ind_size):
            
            # Calculate rho as the sum of c1 and c2 restricted to be above 4 and calculate coefficient
            rho = max(c1[idx] + c2[idx], 4)
            constriction_coeffs.append(2.0 / (2.0 - rho - math.sqrt(pow(rho, 2.0) - 4.0 * rho)))
        
        return constriction_coeffs
      
    def boundary_repair(self, indiv) -> object:
        """ 
        Description: Adjust particles that cross solution space boundaries. Also reverse direction of velocity (OMOPSO).
        Input:  - indiv: a particle of the current population flying out of bounds.
        Output: - indiv: the same particle with its boundary repaired.
        """
        for idx in range(len(indiv)):
            if indiv[idx] < self.BOUND_L:
                indiv[idx] = self.BOUND_L
                indiv.speed[idx] = -1.0 * indiv.speed[idx]
            elif indiv[idx] > self.BOUND_U:
                indiv[idx] = self.BOUND_U
                indiv.speed[idx] = -1.0 * indiv.speed[idx]
        return indiv
    
    def select_leader(self, best_particles) -> object:
        """ 
        Description: Return selected leader for either a local or global best solution, dependent on the input.
        Input: best_particles: list of best particles in either local or global archive.
        Output: leader: selected particle to fly towards in the global perspective.
        """
        leaders = deepcopy(best_particles)
        
        # If only one or two leaders are found, deterministically select the first
        if len(leaders) <= 2:
            return leaders[0]                
        
        # Else randomly select two leaders and return the one with the highest crowding distance
        tools.emo.assignCrowdingDist(leaders)
        sampled_leaders = random.sample(leaders, 2)
        
        if (sampled_leaders[0].fitness.crowding_dist > sampled_leaders[1].fitness.crowding_dist):
            return sampled_leaders[0]
        else:
            return sampled_leaders[1]
    
    def update_particle(self, indiv) -> None:
        """ 
        Description: Update the position of the particle based on the local-/global best(s) and intertia
        Input:  - indiv: the particle which needs to be updated.
        """
        # Retrieve random figures, local- and global best
        r1, r2 = [np.array([random.uniform(0, 1) for _ in range(self.IND_SIZE)]) for _ in range(2)]
        c1, c2 = [np.array([random.uniform(1.5, 2.5) for _ in range(self.IND_SIZE)]) for _ in range(2)]
        local_best = self.select_leader(best_particles=indiv.best)
        global_best = self.select_leader(best_particles=self.GBEST)
        constr_coeffs = self.calc_constriction_coefficient(ind_size=self.IND_SIZE, c1=c1, c2=c2)
        
        # Update individual information
        indiv.speed = constr_coeffs * (self.INERTIA_W * indiv.speed + c1 * r1 * (local_best - indiv) + c2 * r2 * (global_best - indiv))
        
        # Update individuals location in the search space
        indiv = indiv + indiv.speed
    
    def save_generation_to_file(self, gen, population, avg_eval_time, gen_time, final_pop=None, alg_exec_time=None):
        """ 
        Description: Save performance of generation to file.
        Input:  - gen: the generation number currently at with the execution of the NSGA-III.
                - population: the current population of individuals.
                - avg_eval_time: the average time if take to evaluate an individual.
                - gen_time: the total time it took to compute the entire generation.
                - pareto: the found Pareto frontier so far.
                - final_pop: the final population, only saved upon termination of the algorithm.
                - alg_exec_time: the total execution time of the algorithm, only saved upon termination of the algorithm.
        """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['pareto_front'], performance_dict['pareto_front_indivs'] = self.retrieve_pareto()
        performance_dict['avg_eval_time'] = avg_eval_time
        performance_dict['gen_time'] = gen_time
        performance_dict['avg_obj'] = self.logbook[gen]['avg']
        performance_dict['max_obj'] = self.logbook[gen]['max']
        performance_dict['min_obj'] = self.logbook[gen]['min']
        performance_dict['std_obj'] = self.logbook[gen]['std']
        performance_dict['population'] = [list(self.eval.transform_individual(indiv)) for indiv in population]
        if final_pop != None and alg_exec_time != None:
            performance_dict['algorithm_execution_time'] = alg_exec_time
        
        GF.save_to_file(performance_dict, f'SMPSO/GEN_{gen}')
            
    def retrieve_pareto(self):
        """ 
        Description: Calculate and return the pareto optimal set.
        """
        pareto_front = np.array([np.array(deepcopy(particle.fitness.values)) for particle in self.GBEST])
        indivs = [self.eval.transform_individual(indiv) for indiv in self.GBEST]
        return pareto_front, indivs
        
    def _RUN(self, seed=None) -> None:
        """ 
        Description: Run the SMPSO optimization loop until the termination criterion is met.
        Input:  - seed: the seed number, to be set if it is desired to run multiple runs sharing identical random seeds. 
        """ 
        
        print(f'--- Start SMPSO Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---') 
        
        # Start time measure for complete algorithm
        random.seed(seed)
        algorithm_start = time.time()
        
        # Set parameters for individual-/population creation and individual updates
        toolbox = base.Toolbox()
        toolbox.register("particle", self.initialize_particle, creator.Particle)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", self.update_particle)
        toolbox.register('evaluate', self.eval.eval_individual)

        # Create multiple workers on the set number of cores
        if self.MP > 0:
            pool = multiprocessing.Pool(processes=self.MP)
            toolbox.register("map", pool.map)
        
        # Initiaize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Initialize logbook
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Initialize, evaluate and document first population (+ measure evaluation time)
        pop = toolbox.population(n=self.POP_SIZE)
        eval_start = time.time()
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        avg_eval_time = (time.time() - eval_start) / self.POP_SIZE
        
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=self.POP_SIZE, **record)
        if self.verbose: 
            print(self.logbook.stream)
        
        # Update global and local pareto archives
        self.initialize_local_best(population=pop)
        self.GBEST = self.initialize_global_best(population=pop)
        
        # Document and save performance to file
        self.save_generation_to_file(gen=0, 
                                     population=pop, 
                                     avg_eval_time=avg_eval_time, 
                                     gen_time='N\A')
        
        # Start generational process and measure total processing time of generation
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()
            
            # Update all individuals maintaining set boundaries and mutate
            for particle in pop:
                toolbox.update(indiv=particle)
                particle = self.boundary_repair(indiv=particle)
                particle = tools.mutPolynomialBounded(individual=particle, 
                                                      eta=1, 
                                                      low=self.BOUND_L, 
                                                      up=self.BOUND_U, 
                                                      indpb=1 / len(particle))[0]
                del particle.fitness.values
            
            # Evaluate fitness of all particles of the population (+ measure evaluation time)
            eval_start = time.time()
            fitnesses = toolbox.map(toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            avg_eval_time = (time.time() - eval_start) / self.POP_SIZE
            
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=self.POP_SIZE, **record)
            if self.verbose: 
                print(self.logbook.stream)
                
            # Update global and local pareto archives
            self.update_global_best(population=pop)
            self.update_local_best(population=pop)
            
            # Calculate processing time of this generation
            gen_time = time.time() - gen_start
            
            # Document and save performance to file
            if gen != self.NGEN:
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time)
                
            # If final generation, also save the final population and algorithm execution time
            else:
                algorithm_execution_time = time.time() - algorithm_start
                
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time,
                                             final_pop=[list(self.eval.transform_individual(indiv)) for indiv in pop],
                                             alg_exec_time=algorithm_execution_time)

        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
     

if __name__ == '__main__':  
    # Initialize and run Speed-constrained Multi-objective Particle Swarm Optimization             
    optim = SMPSO(num_gen=200, 
                  pop_size=20, 
                  MP=0, 
                  verbose=True)
    
    optim._RUN()

