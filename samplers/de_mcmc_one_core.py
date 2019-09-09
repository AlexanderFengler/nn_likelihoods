import numpy as np
import scipy

class DifferentialEvolutionSequential():
    
    def __init__(self, bounds, target, NP_multiplier = 4, gamma = 0.4, proposal_var = 0.1, crp = 0.3):
        """
        Params
        -----
        dims: int
            dimension of the parameter space
        bounds: list of np.ndarrays
            The first element is a numpy array of lower bounds, the
            second element is the numpy array of upper bounds for the params
        NP: int
            number of particles to use
        target: function(ndarray params, ndarray data) -> float
            function that takes in two arguments: params, and data, and returns
            the log likelihood of the data conditioned on the parameters.
        gamma: float
            gamma parameter to mediate the magnitude of the update
        """
        self.dims = bounds.shape[0]
        self.bounds = bounds
        self.NP = int(np.floor(NP_multiplier * self.dims))
        self.target = target
        self.gamma = gamma
        self.proposal_var = proposal_var
        self.crp = crp
        
        
    def anneal_logistic(self, x = 1, k = 1/100, L = 10):
        return 1 + (2 * L - (2 * L / (1 + np.exp(- k * (x)))))
    
    def propose(self, idx, anneal_k, anneal_L, crossover = True):
        """
        Takes in a chain, and updates the chain parameters and log-likelihood
        """
        
        proposals = self.samples[idx - 1, :, :].copy()
        proposals_lps = self.lps[idx - 1, :].copy()
        
        self.samples[idx, :, :] = self.samples[idx - 1, :, :].copy()
        self.lps[idx, :] = self.lps[idx - 1,:].copy()
        
        pop_seq = np.arange(self.NP)
        np.random.shuffle(pop_seq)
        
        for pop in pop_seq:
            
            # Get candidates that affect current vectors update:
            R1 = pop
            while R1 == pop:
                R1 = np.random.choice(pop_seq)
            R2 = pop
            while R2 == pop or R2 == R1:
                R2 = np.random.choice(pop_seq)
             
            proposals[pop, :] += self.gamma * (proposals[R1, :] - proposals[R2, :]) +  \
                                                        np.random.normal(loc = 0, scale = self.proposal_var, size = self.dims)
            
            # Clip proposal at bounds:
            for dim in range(self.dims):
                proposals[pop, dim] = np.clip(proposals[pop, dim], self.bounds[dim][0], self.bounds[dim][1])
            
            # Crossover:
            if crossover == True:
                n_keep = np.random.binomial(self.dims - 1, p = 1 - self.crp)
                id_keep = np.random.choice(self.dims, n_keep, replace = False)
                proposals[pop, id_keep] = self.samples[idx - 1, pop, id_keep]
            
            proposals_lps[pop] = self.target(proposals[pop, :], self.data)
            acceptance_prob = proposals_lps[pop] - self.lps[idx - 1, pop]
            
            if (np.log(np.random.uniform()) / self.anneal_logistic(x = idx, k = anneal_k, L = anneal_L)) < acceptance_prob :
                self.samples[idx, pop, :] = proposals[pop, :]
                self.lps[idx, pop] = proposals_lps[pop]
   
    def sample(self, data, num_samples = 800, add = False, crossover = True, anneal_k = 1 / 80, anneal_L = 10): 
        if add == False:
            self.data = data
            self.lps = np.zeros((num_samples, self.NP))
            self.samples = np.zeros((num_samples, self.NP, self.dims)) 

            # Initialize parameters
            temp = np.zeros((self.NP, self.dims))
            
            for pop in range(self.NP):
                for dim in range(self.dims):
                    temp[pop, dim] = np.random.uniform(low = self.bounds[dim][0], high = self.bounds[dim][1])

                self.samples[0, pop, :] = temp[pop, :]
                self.lps[0, pop] = self.target(temp[pop, :], self.data)
            
            id_start = 1
            
        if add == True:
            # Make extended data structure
            shape_prev = self.samples.shape
            samples_tmp = np.zeros((shape_prev[0] + num_samples, shape_prev[1], shape_prev[2]))
            samples_tmp[:shape_prev[0], :shape_prev[1], :shape_prev[2]] = self.samples
            self.samples = samples_tmp

            lps_tmp = np.zeros((shape_prev[0] + num_samples, self.NP))
            lps_tmp[:shape_prev[0], :] = self.lps
            self.lps = lps_tmp

            id_start = shape_prev[0]

        print("Beginning sampling:")
        n_samples_final = self.samples.shape[0]
        i = id_start
        
        while i < n_samples_final:
            if (i % 100 == 0):
                print("Iteration {}".format(i))
            self.propose(i, anneal_k, anneal_L, crossover)
            i += 1
        
        return (self.samples, self.lps)