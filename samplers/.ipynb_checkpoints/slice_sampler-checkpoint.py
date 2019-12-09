import numpy as np

class SliceSampler:
    def __init__(self,
                 bounds, # provide parameter bounds
                 target, # provide target distribution
                 w = 1 / 256, # initial interval size
                 p = 8,
                 m = 50,
                 print_interval = 1): # max doubling allowed
        
        #self.dims = bounds.shape[0]
        self.dims = np.array([i for i in range(len(bounds))])
        self.bounds = bounds
        self.target = target
        self.w = w
        self.p = p
        self.m = m
        self.print_interval = print_interval
        #self.active_dims = active_dims

    # Doubling procedue for finding intervals
    def _find_interval_doubling(self, z, prev, dim):
        
        # Carry over left and right boundaries
        left = prev.copy()
        right = prev.copy()
        u = np.random.uniform()

        # Generate initial left and right bounds
        left[dim] = prev[dim] - self.w * u
        right[dim] = left[dim] + self.w
        k = self.p
        
        lp_l = self.target(left, self.data)
        lp_r = self.target(right, self.data)

        while (k > 0 and (z < lp_l or z < lp_r)):
            v = np.random.uniform()
            if v < .5:
                left[dim] += left[dim] - right[dim] # L <- L - (R - L)  ... L + L - R
                left[dim] = np.clip(left[dim], self.bounds[dim][0], self.bounds[dim][1]) # NOTE: Clipping correct ???
                lp_l = self.target(left, self.data)
            else:
                right[dim] += right[dim] - left[dim]
                right[dim] = np.clip(right[dim], self.bounds[dim][0], self.bounds[dim][1]) # NOTE: Clipping correct ???
                lp_r = self.target(right, self.data)
            k -= 1
        return left[dim], right[dim]
    
    # Step out procedure for finding intervals
    def _find_interval_step_out(self, z, prev, dim):
        # Carry over left and right boundaries
        left = prev.copy()
        right = prev.copy()
        u = np.random.uniform()
        v = np.random.uniform()
        j = np.floor(self.m * v)
        k = (self.m - 1) - j

        while j > 0 and z < self.target(left, self.data):
            left[dim] -= self.w
            j -= 1
        while k > 0 and z < self.target(right, self.data):
            right[dim] += self.w
            k -= 1

        return left[dim], right[dim]

    # Extra accept condition for doubling procedure
    def _accept(self, prev, upd, z, left, right, dim):
        
        # initialize
        tmp_upd_left = prev.copy()
        tmp_upd_left[dim] = left
        tmp_upd_right = prev.copy()
        tmp_upd_right[dim] = right
        D = False
        
        while (tmp_upd_right[dim] - tmp_upd_left[dim]) > (1.1 * self.w):
            M = (tmp_upd_left[dim] + tmp_upd_right[dim]) / 2
            
            if (prev[dim] < M and upd[dim] >= M) or (prev[dim] >= M and upd[dim] < M):
                D = True
                
            if upd[dim] < M:
                tmp_upd_right[dim] = M
                
            else:
                tmp_upd_left[dim] = M
            
            if D and z >= self.target(tmp_upd_left, self.data) and z >= self.target(tmp_upd_right, self.data):
                return False
       
        return True  
    
    # Get sample from sampler
    def _slice_sample_doubling(self, prev, prev_lp):
        out = prev.copy()
        lp = prev_lp
        np.random.shuffle(self.dims)
        for dim in self.dims:
            z = prev_lp - np.random.exponential()
            left, right = self._find_interval_doubling(z, prev, dim)
            #print(dim)
            # Adaptively shrink the interval
            while not np.isclose(left, right, 1e-3): # This condition might be unnecessary (other values possible too here)
            #while True:
                u = np.random.uniform()
                # TODO: CHECK IF THIS SHOULD BE  left[dim] ....
                out[dim] = left + u * (right - left)
                lp = self.target(out, self.data)
                if z < lp and self._accept(prev, out, z, left, right, dim):
                    break
                else:
                    if out[dim] < prev[dim]:
                        left = out[dim]
                    else:
                        right = out[dim]
        return (out, lp)

    # Get sample from sampler
    def _slice_sample_step_out(self, prev, prev_lp):
        out = prev.copy()
        lp = prev_lp
        np.random.shuffle(self.dims)
        for dim in self.dims:
            z = prev_lp - np.random.exponential()
            left, right = self._find_interval_step_out(z, prev, dim)
            #print(dim)
            # Adaptively shrink the interval
            while not np.isclose(left,right, 1e-3): # This condition might be unnecessary
            #while True:
                u = np.random.uniform()
                out[dim] = left + u * (right - left)
                lp = self.target(out, self.data)
                if z < lp:
                    break
                else:
                    if out[dim] < prev[dim]:
                        left = out[dim]
                    else:
                        right = out[dim]
        return (out, lp)

    # Sampling wrapper
    def sample(self,
               data, 
               num_samples = 1000, 
               add = False,
               method = 'doubling', 
               init = 'random',
               active_dims = 'all', # str or list of dimensions
               frozen_dim_vals = [[]] # list of lists where first elements in sublist is dimension and second is the assgined value or 'none' (only relevant when we consider random initialization, otherwise provided anyways)
               ):
        
        # Initialize data
        self.data = data
        
        if active_dims == 'all':
            self.dims = np.array([i for i in range(len(self.bounds))])
        else: 
            self.dims = np.array(active_dims)
        
        # New sampler ? 
        if add == False:
            id_start = 1
            
            # Initialize sample storage
            self.samples = np.zeros((num_samples, self.dims.shape[0])) # samples
            self.lp = np.zeros(num_samples) # sample log likelihoods

            # Random initialization of starting points
            if init[0] == 'r':
                tmp = np.zeros(self.dims.shape[0])
                for dim in self.dims:
                    tmp[dim] = np.random.uniform(self.bounds[dim][0],
                                                 self.bounds[dim][1])
                
                # Add in frozen dims
                if not frozen_dim_vals == 'none':
                    for fdim in frozen_dim_vals:
                        tmp[fdim[0]] = fdim[1]
                
            else:
                tmp = init
                
                if not frozen_dim_vals == 'none':
                    for fdim in frozen_dim_vals:
                        tmp[fdim[0]] = fdim[1]
                
            init_lp = self.target(tmp, self.data)
            
            # Make first sample
            if method == 'doubling':
                self.samples[0], self.lp[0] = self._slice_sample_doubling(tmp, init_lp)
            if method == 'step_out':
                self.samples[0], self.lp[0] = self._slice_sample_step_out(tmp, init_lp)
        
        # Or do we add samples to previous samples ?
        else: 
            # choose appropriate starting idx 
            id_start = self.samples.shape[0]
            
            # Increase size so sample container
            tmp_samples = np.zeros((self.samples.shape[0] + num_samples, self.dims.shape[0]))
            tmp_samples[:self.samples.shape[0], :] = self.samples
            self.samples = tmp_samples
            
            tmp_lp = np.zeros(self.lp.shape[0] + num_samples)
            tmp_lp[:self.lp.shape[0]] = self.lp
            self.lp = tmp_lp
            
            print(self.samples[10])

            print('Adding to previous samples...')
        
        print('Beginning sampling...')
        final_n_samples = self.samples.shape[0]
        i = id_start
        
        while i < final_n_samples:
            if method == 'doubling':
                self.samples[i], self.lp[i] = self._slice_sample_doubling(self.samples[i - 1], 
                                                                          self.lp[i - 1])
            if method == 'step_out':
                self.samples[i], self.lp[i] = self._slice_sample_step_out(self.samples[i - 1], 
                                                                          self.lp[i - 1])
                
            if i % self.print_interval == 0:
                print("Iteration {}".format(i))
            
            i += 1
# ------------------------------- UNUSED ---------------------------------------           
    def sample_old(self, data, num_samples = 1000):
        # Initialize data
        self.data = data
        # Initialize sample storage
        self.samples = np.zeros((num_samples, self.dims)) # samples
        self.lp = np.zeros(num_samples) # sample log likelihoods

        init = np.zeros(self.dims)
        
        # Random initialization of starting points
        for dim in range(self.dims):
            init[dim] = np.random.uniform(self.bounds[dim][0], self.bounds[dim][1])
        init_lp = self.target(init, self.data)
        
        self.samples[0], self.lp[0] = self._slice_sample(init, init_lp)

        print("Beginning sampling")

        for i in range(1, num_samples):
            if i % 100 == 0:
                print("Iteration {}".format(i))
            self.samples[i], self.lp[i] = self._slice_sample_old(self.samples[i - 1], self.lp[i - 1])
         
    def _slice_sample_old(self, prev, prev_lp):
        out = prev.copy()
        lp = prev_lp
        for dim in range(self.dims):
            # Sample from vertical slice
            z = prev_lp - np.random.exponential()
            
            # Generate new interval
            left, right = self._find_interval(z, prev, dim)
            print(dim)
            # Adaptively shrink the interval
            while not np.isclose(left,right, 1e-3):
                u = np.random.uniform()
                out[dim] = left + u * (right - left)
                lp = self.target(out, self.data)
                if z < lp:
                    break
                else:
                    if out[dim] < prev[dim]:
                        left = out[dim]
                    else:
                        right = out[dim]
        return (out, lp)
                
#     def _slice_sample(self, prev, prev_lp):
#         out = prev.copy()
#         for dim in range(self.dims):
#             z = prev_lp - np.random.exponential()
#             lp = -np.inf
#             cnt = 0
#             while lp < z:
#                 if cnt == 5000:
#                     print("could not find adequate sample!")
#                     break
#                 out[dim] = np.random.uniform(self.bounds[dim][0], self.bounds[dim][1])
#                 lp = self.target(out, self.data)
#                 cnt += 1
#         return (out, lp)