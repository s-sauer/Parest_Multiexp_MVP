# implements the parameter estimation routines
import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from model_funcs import sim_single_exp

def residuals_single_exp(p, c, y0, datasets):
      """ calculate residuals for a single experiment
      INPUT:
      p ... structure with parameter values to be estimated, cf. lmfit.Parameters
      c ... list of control values, to be passed to model function
            c[0] ... time point when feed was switched on [h]
            c[1] ... feed rate [L/h]
            c[2] ... substrate concentration in feed [g/L]
      y0 ... initial state vector:
            y[0] ... substrate mass (mS) in [g]
            y[1] ... bio dry mass (mX) in [g]
            y[2] ... volume of fermentation broth [L]
      datasets ... list of data frame with measurement data

      OUTPUT:
      res ... long vector with all residuals for this experiment
      """

      res = np.array([]) # empty array, will contain residuals

      weighting_factor = {'cX': 1.0, 'cS': 1.0, 'base_rate': 0.1} # individual weighting factor for each measured variable

      for dat in datasets: # loop over datasets
            t_grid = dat.index.values  # index of 'dat' = time grid of measurements = time grid for simulation
            sim_exp = sim_single_exp(t_grid, y0, p, c) # simulate experiment with this time grid

            for var in dat: # loop over all measured variables
                  res_var = weighting_factor[var]*(sim_exp[var] - dat[var]).values # weighted residuals for this measured variable
                  res = np.append(res, res_var) # append to long residual vector

      return res

def residuals_all_exp(p, y0_dict, c_dict, datasets_dict):
      """ calculate residuals for all experiment
      INPUT:
      p ... structure with parameter values to be estimated, cf. lmfit.Parameters
      y0_dict ... dict: keys: experiment names, values: initial state vector y0
      c_dict ... dict: keys: experiment names, values: control variables c
      datasets ... dictionary: keys: experiment names, values: list of data frame with measurement data

      OUTPUT:
      res ... super long vector with all residuals for all experiment
      """

      exp_names = y0_dict.keys() # experiment names

      res_all_exp = [] # empty array, will contain residuals

      for exp in exp_names: # loop over experiments
            y0 = y0_dict[exp]
            c = y0_dict[exp]
            datasets = datasets_dict[exp]

            res_this_exp = residuals_single_exp(p, c, y0, datasets)
            res_all_exp = np.append(res_all_exp, res_this_exp)
      
      return res_all_exp


def par_est_main(exp_list, p0):
      """ main function to run parameter estimation
      INPUT:
      exp_list ... dictorary: keys experiment names; value: excel file names with experimental data
      p0 ... parameter structure with initial values
      OUTPUT:
      results ... parameter estimation result of lmfit;
                  results.params contains structure with estimated values
      """

      y0_dict = {}
      c_dict = {}
      datasets_dict = {}

      for exp_name, exp_file in exp_list.items(): # loop over experiments and read excel files
            # read out metadata
            metadat = pd.read_excel(exp_file, sheet_name='metadat').iloc[0]  # hint: .iloc[0] turns single row dataframe into Series
            # get inital state vector y0 from metadata
            y0_dict[exp_name] = metadat[['mS0','mX0','V0']].values
            # get control variables from metadata
            c_dict[exp_name] = metadat[['feed_on','feed_rate','csf']].values

            # read online and offline data
            offline = pd.read_excel(exp_file, sheet_name='offline', index_col=0)
            online = pd.read_excel(exp_file, sheet_name='online', index_col=0)
            datasets_dict[exp_name] = [offline, online]
      
      result = minimize(residuals_all_exp, p0, args=(y0_dict, c_dict, datasets_dict), method='leastsq')

      return result