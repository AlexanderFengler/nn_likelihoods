# Functions for DDM data simulation
import numpy as np
import pandas as pd
import time
import inspect

# Simulate (rt, choice) tuples from: SIMPLE DDM -----------------------------------------------

# Simplest algorithm
def  ddm_simulate(v = 0, # drift by timestep 'delta_t'
                  a = 1, # boundary separation
                  w = 0.5,  # between -1 and 1
                  s = 1, # noise sigma
                  delta_t = 0.001, # timesteps fraction of seconds
                  max_t = 20, # maximum rt allowed
                  n_samples = 20000, # number of samples considered
                  print_info = True # timesteps fraction of seconds
                  ):

    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))

    delta_t_sqrt = np.sqrt(delta_t)

    for n in range(0, n_samples, 1):
        y = w*a
        t = 0

        while y <= a and y >= 0 and t <= max_t:
            y += v * delta_t + delta_t_sqrt * np.random.normal(loc = 0,
                                                               scale = s,
                                                               size = 1)
            t += delta_t

        # Store choice and reaction time
        rts[n] = t
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        choices[n] = (-1) * np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    print('finished:', {'v': v,
           'a': a,
           'w': w,
           's': s,
           'delta_t': delta_t,
           'max_t': max_t,
           'n_samples': n_samples,
           'simulator': 'ddm',
           'boundary_fun_type': 'constant'})
           
    return (rts, choices, {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm',
                           'boundary_fun_type': 'constant',
                           'possible_choices': [-1, 1]})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: DDM WITH ARBITRARY BOUNDARY --------------------------------

# For the flexbound (and in fact for all other dd)
# We expect the boundary function to have the following general shape:
# 1. An initial separation (using the a parameter) with lower bound 0
# (this allows interpretation of the w parameter as usual)
# 2. No touching of boundaries
# 3. It return upper and lower bounds for every t as a list [upper, lower]

def ddm_flexbound_simulate(v = 0,
                           w = 0.5,
                           s = 1,
                           delta_t = 0.001,
                           max_t = 20,
                           n_samples = 20000,
                           print_info = True,
                           boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                           boundary_fun_type = 'constant',
                           **boundary_params):

    # Initializations
    print({'boundary_fun': boundary_fun})
    rts = np.zeros((n_samples,1)) #rt storage
    choices = np.zeros((n_samples,1)) # choice storage
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage:
    boundaries = np.zeros(((int(max_t/delta_t), 2)))
    for i in range(0, int(max_t/delta_t), 1):
        boundaries[i, :] = boundary_fun(t = i * delta_t, **boundary_params)

    # Outer loop over n - number of samples
    for n in range(0, n_samples, 1):
        # initialize y, t, and time_counter
        y = boundaries[0, 0] + (w * (boundaries[0, 1] - boundaries[0, 0]))
        t = 0
        cnt = 0

        # Inner loop (trajection simulation)
        while y <= boundaries[cnt, 1] and y >= boundaries[cnt, 0] and t <= max_t:
            # Increment y position (particle position)
            y += v * delta_t + delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = 1)
            # Increment time
            t += delta_t
            # increment count
            cnt += 1

        # Store choice and reaction time
        rts[n] = t
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        # This is kind of a legacy issue at this point (plan is to flip this around, after appropriately reformulating navarro fuss wfpd function)
        choices[n] = (-1) * np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return (rts, choices,  {'v': v,
                           'a': a,
                           'w': w,
                           's': s,
                           **boundary_params,
                           'delta_t': delta_t,
                           'max_t': max_t,
                           'n_samples': n_samples,
                           'simulator': 'ddm_flexbound',
                           'boundary_fun_type': boundary_fun_type})
# -----------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: RACE MODEL WITH N SAMPLES ----------------------------------
def race_model(v = [0, 0, 0], # np.array expected in fact, one column of floats
               w = [0, 0, 0], # np.array expected in fact, one column of floats
               s = [1, 1, 1], # np.array expected in fact, one column of floats
               delta_t = 0.001,
               max_t = 20,
               n_samples = 2000,
               print_info = True,
               boundary_fun = None,
               **boundary_params):

    # Initializations
    n_particles = len(v)
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    particles = np.zeros((n_particles, 1))

    # We just care about an upper boundary here: (more complicated things possible)
    boundaries = np.zeros((int(max_t / delta_t), 1))
    for i in range(0, int(max_t / delta_t), 1):
        boundaries[i] = boundary_fun(t = i * delta_t, **boundary_params)

    for n in range(0, n_samples, 1):
        # initialize y, t and time_counter
        particles = w * boundaries[0]
        t = 0
        cnt = 0

        while np.less_equal(particles, boundaries[cnt]).all() and t <= max_t:
            particles += (v * delta_t) + (delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = (n_particles, 1)))
            t += delta_t
            cnt += 1

        rts[n] = t
        choices[n] = particles.argmax()

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return (rts, choices)
# -------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Ornstein-Uhlenbeck -------------------------------------------
def ornstein_uhlenbeck(v = 1, # drift parameter
                       a = 1, # boundary separation parameter
                       w = 0.5, # starting point bias
                       g = 0.1, # decay parameter
                       s = 1, # standard deviation
                       delta_t = 0.001, # size of timestamp
                       max_t = 20, # maximal time in trial
                       n_samples = 2000, # number of samples from process
                       print_info = True): # whether or not ot print periodic update on number of samples generated

    # Initializations
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)

    for n in range(0, n_samples, 1):
        y = w*a
        t = 0

        while y <= a and y >= 0 and t <= max_t:
            y += ((v * delta_t) - (delta_t * g * y)) + delta_t_sqrt * np.random.normal(loc = 0,
                                                                                       scale = s,
                                                                                       size = 1)
            t += delta_t

        # Store choice and reaction time
        rts[n] = t
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        choices[n] = (-1) * np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')

    return (rts, choices)
# -------------------------------------------------------------------------------------------------


# Simulate (rt, choice) tuples from: Onstein-Uhlenbeck with flexible bounds -----------------------
def ornstein_uhlenbeck_flexbnd(v = 0, # drift parameter
                               w = 0.5, # starting point bias
                               g = 0.1, # decay parameter
                               s = 1, # standard deviation
                               delta_t = 0.001, # size of timestep
                               max_t = 20, # maximal time in trial
                               n_samples = 20000, # number of samples from process
                               print_info = True, # whether or not to print periodic update on number of samples generated
                               boundary_fun = None, # function of t (and potentially other parameters) that takes in (t, *args)
                               **boundary_params):
    # Initializations
    rts = np.zeros((n_samples,1)) # rt storage
    choices = np.zeros((n_samples,1)) # choice storage
    delta_t_sqrt = np.sqrt(delta_t) # correct scalar so we can use standard normal samples for the brownian motion

    # Boundary storage:
    boundaries = np.zeros(((int(max_t/delta_t), 2)))
    for i in range(0, int(max_t/delta_t), 1):
        boundaries[i, :] = boundary_fun(t = i * delta_t, **boundary_params)

    # Outer loop over n - number of samples
    for n in range(0, n_samples, 1):
        # initialize y, t, and time_counter
        y = boundaries[0, 0] + (w * (boundaries[0, 1] - boundaries[0, 0]))
        t = 0
        cnt = 0

        # Inner loop (trajection simulation)
        while y <= boundaries[cnt, 1] and y >= boundaries[cnt, 0] and t <= max_t:
            # Increment y position (particle position)
            y += ((v * delta_t) - (delta_t * g * y)) + delta_t_sqrt * np.random.normal(loc = 0,
                                                                                       scale = s,
                                                                                       size = 1)
            # Increment time
            t += delta_t
            # increment count
            cnt += 1

        # Store choice and reaction time
        rts[n] = t
        # Note that for purposes of consistency with Navarro and Fuss, the choice corresponding the lower barrier is +1, higher barrier is -1
        # This is kind of a legacy issue at this point (plan is to flip this around, after appropriately reformulating navarro fuss wfpd function)
        choices[n] = (-1) * np.sign(y)

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return (rts, choices)
# --------------------------------------------------------------------------------------------------

# Simulate (rt, choice) tuples from: Leaky Competing Accumulator Model -----------------------------
def lca(v = [0, 0, 0], # drift parameters (np.array expect: one column of floats)
        w = [0, 0, 0], # initial bias parameters (np.array expect: one column of floats)
        a = 1, # criterion height
        g = 0, # decay parameter
        b = 1, # inhibition parameter
        s = 1, # variance (can be one value or np.array of size as v and w)
        delta_t = 0.001, # time-step size in simulator
        max_t = 20, # maximal time
        n_samples = 2000, # number of samples to produce
        print_info = True): # whether or not to periodically report the number of samples generated thus far

    # Initializations
    n_particles = len(v)
    rts = np.zeros((n_samples, 1))
    choices = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    particles = np.zeros((n_particles, 1))

    for n in range(0, n_samples, 1):

        # initialize y, t and time_counter
        particles_reduced_sum = particles
        particles = w * a
        t = 0


        while np.less_equal(particles, a).all() and t <= max_t:
            particles_reduced_sum[:,] = - particles + np.sum(particles)
            particles += ((v - (g * particles) - (b * particles_reduced_sum)) * delta_t) + \
                         (delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = (n_particles, 1)))
            particles = np.maximum(particles, 0.0)
            t += delta_t

        rts[n] = t
        choices[n] = particles.argmax()

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return (rts, choices)
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    #ddm_simulate(n_samples = 1000)
    #print("--- %s seconds ---" % (time.time() - start_time))
    #start_time = time.time()
    #ddm_simulate_fast(n_samples = 1000)
    #print("--- %s seconds ---" % (time.time() - start_time))
