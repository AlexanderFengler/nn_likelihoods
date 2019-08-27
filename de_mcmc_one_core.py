import numpy as np
import scipy

class DifferentialEvolutionSequential():
    
    def __init__(self, dims, bounds, NP, target, gamma = 0.4, proposal_var = 0.1):
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
        self.dims = dims
        self.NP = NP
        self.target = target
        self.gamma = gamma
        self.bounds = bounds
        self.proposal_var = proposal_var
        
    def propose(self, idx):
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
            
            R1 = pop
            while R1 == pop:
                R1 = np.random.choice(pop_seq)
            R2 = pop
            while R2 == pop or R2 == R1:
                R2 = np.random.choice(pop_seq)
             
            proposals[pop, :] += self.gamma * (proposals[R1, :] - proposals[R2, :]) +  \
                                                        np.random.normal(loc = 0, scale = self.proposal_var, size = self.dims)
         
            # clip proposal at bounds
            for dim in range(self.dims):
                proposals[pop, dim] = np.clip(proposals[pop, dim], self.bounds[dim][0], self.bounds[dim][1])
                
            proposals_lps[pop] = self.target(proposals[pop, :], self.data)
            acceptance_prob = proposals_lps[pop] - self.lps[idx, pop]
            
            if np.log(np.random.uniform()) < acceptance_prob:
                self.samples[idx, pop, :] = proposals[pop, :]
                self.lps[idx, pop] = proposals_lps[pop]
   
    def sample(self, data, num_samples = 800):
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
        
        print("Beginning sampling:")
        for i in range(1, num_samples, 1):
            if (i % 200 == 0):
                print("Iteration {}".format(t))
                
            self.propose(i)