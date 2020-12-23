# @Paul: ein python Skript gleichen Inhalts wie das jupyter Notebook; aber in .py files kann ich debuggen, deswegen habe ich das angelegt


# imports

import numpy as np
import pandas as pd
from lmfit import Parameters, report_fit

from model_funcs import sim_single_exp
from parest_funcs import par_est_main

# create parameter structure with initial fit parameter values, bounds, etc.

# Parameter structure
p0 = Parameters()
p0.add('mumax', value=0.5, min=0.0001, max=1.)
p0.add('Yxs', value=0.2, min=0.0001, max=1.)
p0.add('Ks', value=1.0, vary=False)
p0.add('base_coef', value=1.0, min=0.0001) # proportinality factor between biomass growth and base consumption, in [L/g]

# control values
#          c[0] ... time point when feed was switched on [h]
#          c[1] ... feed rate [L/h]
#          c[2] ... substrate concentration in feed [g/L]
c = [5, .02, 200]

# initial values
#          y[0] ... substrate mass (mS) in [g]
#          y[1] ... bio dry mass (mX) in [g]
#          y[2] ... volume of fermentation broth [L]
y0 = [3, 0.2, .5]

# time grid in [h]
t_grid = np.linspace(0,10, 1001)

# run simulation
sim_exp = sim_single_exp(t_grid, y0, p0, c)

# plot results
print(sim_exp)
sim_exp.plot(y=['cS', 'cX'])
sim_exp.plot(y=['V', 'base_rate'])

# define experiments to include
exp_list = { # key: name of experiment (arbitrary); value: filename of excel file
      'Experiment 1': './exp1.xlsx',
      'Experiment 2': './exp2.xlsx',
}
fit_results = par_est_main(exp_list, p0)


# print fit parameter values
report_fit(fit_results)