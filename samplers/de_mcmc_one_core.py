import numpy as np
import scipy
import scipy.optimize as scp_opt
import samplers.diagnostics as mcmcdiag

class DifferentialEvolutionSequential():
    
    def __init__(self, 
                 bounds, 
                 target, 
                 NP_multiplier = 5, 
                 gamma = 'auto', # 'auto' or float 
                 mode_switch_p = 0.1,
                 proposal_std = 0.01, 
                 crp = 0.3):
        
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
        
        self.optimizer = scp_opt.differential_evolution
        self.dims = len(bounds) #np.array([i for i in range(len(bounds))])
        self.bounds = bounds
        self.NP = int(np.floor(NP_multiplier * self.dims))
        self.target = target
        
        if gamma == 'auto':
            self.gamma = 2.38 / np.sqrt(2 * self.dims)
        else: 
            self.gamma = gamma

        #self.gamma = gamma 
        self.mode_switch_p = mode_switch_p
        self.proposal_std = proposal_std
        self.crp = crp
        self.accept_cnt = 0
        self.total_cnt = 0
        self.gelman_rubin = 10
        self.gelman_rubin_r_hat = []
        np.random.seed()
        self.random_seed = np.random.get_state()

        #print(self.random_seed)
        print(np.random.normal(size = 10))
    
    def attach_sample(self, samples):
        assert samples.shape[0] == self.NP, 'Population size of previous sample does not match NP parameter value'
        self.samples = samples

    def anneal_logistic(self, x = 1, k = 1/100, L = 10):
        return 1 + (2 * L - (2 * L / (1 + np.exp(- k * (x)))))
    
    def propose(self, idx, anneal_k, anneal_L, crossover = True):
        """
        Takes in a chain, and updates the chain parameters and log-likelihood
        """
        
        proposals = self.samples[:, idx - 1, :].copy()
        proposals_lps = self.lps[:, idx - 1].copy()
        
        self.samples[:, idx, :] = self.samples[:, idx - 1, :].copy()
        self.lps[:, idx] = self.lps[:, idx - 1].copy()
        
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
             
            proposals[pop, :] += np.random.choice([self.gamma, 1], p = [1 - self.mode_switch_p, self.mode_switch_p]) * (proposals[R1, :] - proposals[R2, :]) +  \
                                                        np.random.normal(loc = 0, scale = self.proposal_std, size = self.dims)
                                                        #self.proposal_std * np.random.standard_t(df = 2, size = self.dims)
                                                        
            
            # Clip proposal at bounds: TD REFLECT AT BOUND?
            for dim in range(self.dims):
                proposals[pop, dim] = np.clip(proposals[pop, dim], self.bounds[dim][0], self.bounds[dim][1])
            
            # Crossover:
            if crossover == True:
                n_keep = np.random.binomial(self.dims - 1, p = 1 - self.crp)
                id_keep = np.random.choice(self.dims, n_keep, replace = False)
                proposals[pop, id_keep] = self.samples[pop, idx - 1, id_keep]
            
            proposals_lps[pop] = self.target(proposals[pop, :], self.data)
            acceptance_prob = proposals_lps[pop] - self.lps[pop, idx - 1]
            
            self.total_cnt += 1
            
            if (np.log(np.random.uniform()) / self.anneal_logistic(x = idx, k = anneal_k, L = anneal_L)) < acceptance_prob :
                self.samples[pop, idx, :] = proposals[pop, :]
                self.lps[pop, idx] = proposals_lps[pop]
                self.accept_cnt += 1
   
    def sample(self, 
               data, 
               num_samples = 800, 
               add = False, 
               crossover = True, 
               anneal_k = 1 / 80, 
               anneal_L = 10,
               init = 'random',
               active_dims = None,
               frozen_dim_vals = None,
               gelman_rubin_force_stop = False): 
        
        if add == False:
            self.data = data
            self.lps = np.zeros((self.NP, num_samples))
            self.samples = np.zeros((self.NP, num_samples, self.dims)) 
            
            # Accept and total counts reset
            self.accept_cnt = 0
            self.total_cnt = 0
            
            # Initialize parameters
            temp = np.zeros((self.NP, self.dims))
            
            for pop in range(self.NP):
                for dim in range(self.dims):
                    temp[pop, dim] = np.random.uniform(low = self.bounds[dim][0], high = self.bounds[dim][1])

                self.samples[pop, 0, :] = temp[pop, :]
                self.lps[pop, 0] = self.target(temp[pop, :], self.data)
            
            id_start = 1
            
        if add == True:
            # Make extended data structure
            shape_prev = self.samples.shape
            samples_tmp = np.zeros((shape_prev[0], shape_prev[1] + num_samples, shape_prev[2]))
            samples_tmp[:shape_prev[0], :shape_prev[1], :shape_prev[2]] = self.samples
            self.samples = samples_tmp

            lps_tmp = np.zeros((self.NP, shape_prev[1] + num_samples))
            lps_tmp[:, :shape_prev[1]] = self.lps
            self.lps = lps_tmp

            id_start = shape_prev[1]
            
            # Accept and total counts reset
            self.accept_cnt = 0
            self.total_cnt = 0
            
        print("Beginning sampling:")
        n_samples_final = self.samples.shape[1]
        i = id_start
        continue_ = 1
        while i < n_samples_final:
            if (i % 200 == 0):
                print("Iteration {}".format(i))
                
                # Adaptive step
                if (i > 1000):
                    acc_rat_tmp = self.accept_cnt / self.total_cnt
                    print('Acceptance ratio: ', acc_rat_tmp)
                    if (acc_rat_tmp) < 0.05:
                        self.proposal_std = self.proposal_std / 2
                        print('New proposal std: ', self.proposal_std)
                    
                    if (acc_rat_tmp) > 0.5:
                        self.proposal_std = self.proposal_std * 1.5
                        print('New proposal std: ', self.proposal_std)
                    
                    self.accept_cnt = 0
                    self.total_cnt = 0
                    
                if ((i > 2000) and (i % 1000)):
                    continue_, r_hat = mcmcdiag.get_gelman_rubin_mv(chains = self.samples,
                                                                    burn_in = 1000,
                                                                    thresh = 1.01)
                    self.gelman_rubin_r_hat.append(r_hat)
                    print('Gelman Rubin: ', r_hat)
                    print('Continue: ', continue_)
                    if not continue_:
                        if gelman_rubin_force_stop:
                            break
                    
            self.propose(i, anneal_k, anneal_L, crossover)
            i += 1
        
        if not continue_:
            # Here I need to adjust samples so that the final datastructure doesn't have 0 elements
            pass
        
        return (self.samples, self.lps, self.gelman_rubin_r_hat, self.random_seed)