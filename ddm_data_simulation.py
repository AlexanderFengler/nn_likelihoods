# Functions for DDM data simulation
import numpy as np
import pandas as pd
import time

def  ddm_simulate_rts(v = 0, # drift by timestep 'delta_t'
                      a = 1, # boundary separation
                      w = 0.5,  # between -1 and 1
                      s = 1, # noise sigma
                      delta_t = 0.001,
                      max_t = 20,
                      n_samples = 20000,
                      print_info = True): # timesteps fraction of seconds

    rts = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)

    for n in range(0, n_samples, 1):
        y = -a + (w * (2 * a))
        y_abs = abs(y)
        t = 0

        while y_abs < a and t <= max_t:
            y += v * delta_t + delta_t_sqrt * np.random.normal(loc = 0, scale = s, size = 1)
            t += delta_t
            y_abs = abs(y)

        rts[n] = (-1) * np.sign(y) * t

        if print_info == True:
            if n % 1000 == 0:
                print(n, ' datapoints sampled')
    return rts


def  ddm_simulate_rts_fast(v = 0, # drift by timestep 'delta_t'
                           a = 1, # boundary separation
                           w = 0.5,  # between -1 and 1
                           s = 1, # noise sigma
                           delta_t = 0.001, # timesteps fraction of seconds
                           max_t = 20, # maximum rt allowed
                           n_samples = 20000, # number of samples considered
                           noise_samples_multiplier = 5): # number of normal random sampls drawn (higher == less approximation to fully random sampling)

    rts = np.zeros((n_samples, 1))
    delta_t_sqrt = np.sqrt(delta_t)
    n_noise_samples = np.int(max_t / delta_t * noise_samples_multiplier)
    noise = np.random.normal(loc = 0, scale = s, size = n_noise_samples) * delta_t_sqrt
    print(noise.shape)
    rand_high = n_noise_samples - (max_t / delta_t) - 1

    for n in range(0, n_samples, 1):
        y = -a + (w * (2 * a))
        y_abs = abs(y)
        t = 0
        cnt = 0
        while y_abs < a and t <= max_t:
            y += v * delta_t + noise[cnt]
            t += delta_t
            y_abs = abs(y)
            cnt += 1

        np.random.shuffle(noise)
        rts[n] = (-1) * np.sign(y) * t
        # print(n)

    return rts

if __name__ == "__main__":
    start_time = time.time()
    ddm_simulate_rts(n_samples = 1000)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    ddm_simulate_rts_fast(n_samples = 1000)
    print("--- %s seconds ---" % (time.time() - start_time))
