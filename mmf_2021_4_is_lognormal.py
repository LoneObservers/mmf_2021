################
## Author: Thomas Balzer
## (c) 2021
## Material for MMF Stochastic Analysis - Fall 2021
## Assignment # 1
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist
import plot_utilities as pu


###########
##### function to plot the histogram of the Bachelier Call Option (possibly with Importance Sampling Shift)
###########

def bachelier_digital_histogram(mean, variance, strike, sample_size, repeats, shift=0.):
    vol = np.sqrt(variance)
    results = [0.] * repeats
    for k in range(repeats):
        sns = np.random.standard_normal(sample_size)
        payoff = sum(
            [np.exp(- shift * (s + shift) + 0.5 * shift * shift) * (1 if mean + vol * (s + shift) > strike else 0) for s in
             sns])
        results[k] = payoff / sample_size

    num_bins = 50

    ### calculate the sample variance
    ave_result = sum(results) / repeats
    print ('Average Result: ' + str(ave_result))
    sample_variance = sum([np.power(r - ave_result, 2) for r in results]) / (repeats - 1)
    print ('Sample Variance: ' + str(sample_variance))

    plt.hist(results, num_bins, normed=True)

    plt.title('Histogram of Bachelier Option Value with Shift={0}'.format(shift))

    plt.xlabel('sample')
    plt.ylabel('value')

    plt.show()
    plt.close()


############
## plot sample variance by shift and return argmin
############

def bachelier_digital_sample_variance_by_shift(mean, variance, strike, sample_size):
    vol = np.sqrt(variance)
    x_step_size = 0.05
    x_values = [-2. + x_step_size * k for k in range(int(4 / x_step_size))]

    payoff_sq_values = [0. for x in x_values]
    sns = np.random.standard_normal(sample_size)
    n = len(x_values)
    for k in range(n):
        x = x_values[k]
        #      payoff = sum([np.exp(- x * (s + x) + 0.5 * x * x) * max(mean + vol * (s + x) - strike, 0) for s in sns])
        #      payoff = sum([np.exp(-2 * x * (s + x) + x * x) * np.power(max(mean + vol * (s + x) - strike, 0),2) for s in sns])
        ### we alternative can also re-write the formula and sample under $PP$
        # payoff = sum([np.exp(-1 * x * s + 0.5 * x * x) * (1. if mean + vol * s > strike else 0) for s in sns])
        payoff = sum([np.exp(-2. * x * (s + x) + x * x) * (1. if mean + vol * (s + x) > strike else 0) for s in sns])
        payoff_sq_values[k] = payoff / sample_size

    attained_minimum = x_values[payoff_sq_values.index(min(payoff_sq_values))]
    print ('Minimum is attained at ' + str(attained_minimum))
    plt.plot(x_values, payoff_sq_values)

    plt.xlabel('Shift')
    plt.ylabel('Mean Squared Value')

    plt.show()
    plt.close()
    return attained_minimum


if __name__ == '__main__':
    #### Inputs for the calculation
    ##
    mean = 0
    variance = 1
    strike = 0.

    ## we are reporting the exact value for comparison
    nd = dist.NormalDistribution(0, 1.)
    vol = np.sqrt(variance)
    adj_strike = (strike - mean) / vol

    exact_value = 1 - nd.cdf(adj_strike)
    print ('Exact Option Value: ' + str(exact_value))

    ### Step 1 - Calculate histogram of 1000 MC simulations without Importance Sampling applied
    bachelier_digital_histogram(mean, variance, strike, 5000, 1000)

    ### Step 2 - Determine a shift to reduce the variance
    minimum_variance = bachelier_digital_sample_variance_by_shift(mean, variance, strike, 20000)

    ### Step 3 - Apply that shift and re-sample the same as in Step 1 in order to see the impact on the sample variance
    bachelier_digital_histogram(mean, variance, strike, 5000, 1000, minimum_variance)

