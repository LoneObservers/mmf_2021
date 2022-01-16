################
## Author: Thomas Balzer
## (c) 2021
## Material for MMF Stochastic Analysis - Fall 2021
################


import numpy as np
import plot_utilities as pu

def terminal_utility_histogram(_b, _r, _sigma, T, _sample_size):

    #####
    ## plot the terminal utility of a stock vs an optimal strategy (for various utility functions)
    #####

    sample = np.random.normal(0, T, _sample_size)

    alpha = .0
    pi = (_b - _r) / (_sigma * _sigma) / (1 - alpha)
    sigma_pi = _sigma * pi
    b_pi = _r + pi * (_b - _r)

    sample_value_stock = [np.exp((_b + 0.5 * _sigma * _sigma) * T + _sigma * 1. * ns) for ns in sample]
    sample_value_pi = [np.exp((b_pi + 0.5 * sigma_pi * sigma_pi) * T + sigma_pi * 1. * ns) for ns in sample]

    ##
    ## we then turn the outcome into a histogram
    ##

    num_bins = 100

    mp = pu.PlotUtilities("Terminal Wealth for Stock and Mixed Portfolio for $\pi=${0}".format(pi), 'Outcome', 'Rel. Occurrence')
    labels = ['Stock', 'Portfolio']
    mp.plotHistogram([sample_value_stock, sample_value_pi], num_bins, labels)




if __name__ == '__main__':

    _b = .2
    _sigma = 0.5
    _r = 0.05
    _t = 1.
    _n = 25000

    terminal_utility_histogram(_b, _r, _sigma, _t, _n)

