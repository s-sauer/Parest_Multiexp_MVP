# implements the kinetic model
# @Paul: Hier muss du in deiner Rolle als "Modellierer" den Code für dein kinetisches Prozessmodell hinterlegen

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def model_rhs(t, y, p, c): # @Paul: heißt bei dir 'f'
    """ r.h.s. (right hand side) function of the ODE model 
    
    INPUT:
    t ... current time [h]
    y ... state vector:
          y[0] ... substrate mass (mS) in [g]
          y[1] ... bio dry mass (mX) in [g]
          y[2] ... volume of fermentation broth [L]
    p ... structure with parameter values to be estimated, cf. lmfit.Parameters
    c ... list of control values
          c[0] ... time point when feed was switched on [h]
          c[1] ... feed rate [L/h]
          c[2] ... substrate concentration in feed [g/L]
          
    OUTPUT:
    dy_dt ... time derivative of state vector
    """
    
    # (potential) fit parameters
    mumax = p['mumax'].value
    Yxs   = p['Yxs'].value
    Ks    = p['Ks'].value
    
    # controls
    feed_on = c[0] # time point when feed was switched on [h]
    feed_rate = c[1] # feed rate [L/h]
    Fin = feed_rate * (t > feed_on) # becomes 0 if t < feed_on
    csf = c[2] # substrate concentration in feed [g/L]
    
    # masses and concentrations
    mS, mX, V = y
    cS, cX = [mS, mX] / V
    
    # kinetics
    mu = mumax * cS / (cS + Ks)
    qS = 1/Yxs * mu
    
    # r.h.s. of ODE
    dmS_dt = - qS * cX * V + csf * Fin
    dmX_dt = + mu * cX * V
    dV_dt  = + Fin
    
    return (dmS_dt, dmX_dt, dV_dt)


def sim_single_exp(t_grid, y0, p, c):  # @Paul: heißt bei dir 'g'
    """ simulates single experiment and calculates measured quantities
    
    INPUT:
    t_grid ... time grid on which to generate the solution [h]
    y0 ... initial state vector:
          y[0] ... substrate mass (mS) in [g]
          y[1] ... bio dry mass (mX) in [g]
          y[2] ... volume of fermentation broth [L]
    p ... structure with parameter values to be estimated, cf. lmfit.Parameters
    c ... list of control values, to be passed to model function
          c[0] ... time point when feed was switched on [h]
          c[1] ... feed rate [L/h]
          c[2] ... substrate concentration in feed [g/L]
          
    OUTPUT:
    sim_exp ... data frame with simulated cS, cX and base consumption rate over time (for single experiment)
    """
    
    # run ODE solver to get solution y(t)
    y_t = solve_ivp(model_rhs, [np.min(t_grid), np.max(t_grid)], y0, t_eval=t_grid, args = (p, c)).y.T

    # unpack solution into vectors
    mS, mX, V = [y_t[:,i] for i in range(3)]
    cS, cX = [mS, mX] / V

    # for base consumption rate: get value of dmX_dt at all times t
    dmX_dt = np.array([model_rhs(t_grid[i], y_t[i,:], p, c) for i in range(len(t_grid))])[:,1]
    base_rate = p['base_coef'].value * dmX_dt

    # pack everything neatly together to a pandas df
    sim_exp = pd.DataFrame(
          {'t': t_grid,
          'cS': cS, 'cX': cX,
          'V': V, 'base_rate': base_rate}
          ).set_index('t') # make time column the index column

    return sim_exp

