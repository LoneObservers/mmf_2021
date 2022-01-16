################
## Author: Thomas Balzer
## (c) 2021
## Material for MMF Stochastic Analysis - Fall 2021
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def plot_bachelier_digital_option(_time, _timestep, _strike):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep)

    sample = np.random.normal(0, np.sqrt(_timestep), size)

    path_bm = [sum(sample[0:n]) for n in range(size)]

    x = [_timestep * k for k in range(size)]
    remaining_time = [np.sqrt(_time - x_i) for x_i in x]

    ## theoretical option value over time on a single path
    path_digital_option = [1 - dist.standard_normal_cdf((_strike - bm) / rt) for (bm, rt) in zip(path_bm, remaining_time)]

    hedge_proportion = [0] * (size)
    path_digital_option_hedge = [0] * (size)

    ####
    ## plot the trajectory of the process
    ####
    _t_remain = remaining_time[0]
    path_digital_option_hedge[0] = path_digital_option[0]
    hedge_proportion[0] = dist.standard_normal_pdf((_strike - path_bm[0]) / _t_remain) / _t_remain

    for j in range(1, size):
        _t_remain = remaining_time[j]
        hedge_proportion[j] = dist.standard_normal_pdf((_strike - path_bm[j - 1]) / _t_remain) / _t_remain
        path_digital_option_hedge[j] = path_digital_option_hedge[j-1] + sample[j] * hedge_proportion[j]

    mp = pu.PlotUtilities("Paths of Digital Option Value", 'Time', "Option Value")

    trackHedgeOnly = True
    if (trackHedgeOnly):
        mp.multiPlot(x, [path_digital_option, path_digital_option_hedge])
    else:
        arg = ['Option Value', 'Hedge Proportion', 'Underlying Brownian Motion']
        colors = ['green', 'red', 'blue']
        mp = pu.PlotUtilities("Paths of Digital Option Value", 'Time', "Option Value")
        mp.subPlots(x, [path_digital_option, hedge_proportion, path_bm], arg, colors)


if __name__ == '__main__':

    time = .5
    timestep = 0.0001
    strike = 1.
    plot_bachelier_digital_option(time, timestep, strike)
