#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:54:57 2021

@author: matthew
"""


#%% Imports

import numpy as np
import matplotlib.pyplot as plt   
plt.switch_backend('Qt5Agg')
from pathlib import Path
import sys

sys.path.append("./ICASAR-2.7.3")
sys.path.append("./LiCSAlert-2.1.3")
#sys.path.append("/home/matthew/university_work/01_blind_signal_separation_python/13_ICASAR/ICASAR_GitHub")

import icasar
from icasar.icasar_funcs import ICASAR, LiCSBAS_to_ICASAR
from icasar.aux import visualise_ICASAR_inversion

import licsalert
from licsalert.licsalert import LiCSAlert_preprocessing

#%% functions that we'll need



def matrix_show(matrix, title=None, ax=None, fig=None, vmin0 = False):
    """Visualise a matrix
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    matrix = np.atleast_2d(matrix)                   # make at least 2d so can plot column/row vectors

    if isinstance(matrix[0,0], np.bool_):           # boolean arrays will plot, but mess up the colourbar
        matrix = matrix.astype(int)                 # so convert

    if vmin0:
        matrixPlt = ax.imshow(matrix,interpolation='none', aspect='auto', vmin = 0)
    else:
        matrixPlt = ax.imshow(matrix,interpolation='none', aspect='auto')
    fig.colorbar(matrixPlt,ax=ax)
    if title is not None:
        ax.set_title(title)
        fig.canvas.set_window_title(f"{title}")
    plt.pause(1)                                                                    # to force it to be shown when usig ipdb



def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.
    Useful when converting between vectors used by BSS methods results that are to be plotted

    Inputs:
        col | rank 1 array |
        pixel_mask | array mask (rank 2)

    Outputs:
        source | rank 2 masked array | colun as a masked 2d array

    2017/10/04 | collected from various functions and placed here.

    """
    import numpy.ma as ma
    import numpy as np

    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()
    return source


def r2_to_r3(ifgs_r2, mask):
    """ Given a rank2 of ifgs as row vectors, convert it to a rank3. 
    Inputs:
        ifgs_r2 | rank 2 array | ifgs as row vectors 
        mask | rank 2 array | to convert a row vector ifg into a rank 2 masked array        
    returns:
        phUnw | rank 3 array | n_ifgs x height x width
    History:
        2020/06/10 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
   
    
    n_ifgs = ifgs_r2.shape[0]
    ny, nx = col_to_ma(ifgs_r2[0,], mask).shape                                   # determine the size of an ifg when it is converter from being a row vector
    
    ifgs_r3 = np.zeros((n_ifgs, ny, nx))                                                # initate to store new ifgs
    for ifg_n, ifg_row in enumerate(ifgs_r2):                                           # loop through all ifgs
        ifgs_r3[ifg_n,] = col_to_ma(ifg_row, mask)                                  
    
    mask_r3 = np.repeat(mask[np.newaxis,], n_ifgs, axis = 0)                            # expand the mask from r2 to r3
    ifgs_r3_ma = ma.array(ifgs_r3, mask = mask_r3)                                      # and make a masked array    
    return ifgs_r3_ma


def plot_points_interest(r3_data, points_interest, baselines_cs, acq_dates, title = '', ylabel = 'm'):
    """ Given rank 3 data of incremental interferograms (e.g. n_images x height x width) and some points of interest (e.g. an xy pair), plot the cumulative time
    series for those points (i.e. as r3 is incremental, summing is done in the function).  Also information is required (baselines and acq_dates) for the x axis of the plot.  
    
    Inputs:
        r3_data | rank 3 array (masked array support?) | incremental interferograms, rank 3 (n_images x height x width)
        points_interest | dict | point name (e.g. 'reference' or 'deforming' and tuple of x and y position.  )
        baselines_cs | rank 1 array | cumulative temporal baselines in days.  
        acq_dates | string of YYYYMMDD | date of each radar acquisition.  
        title | string | figure and window title.  
        ylabel | string | units of r3_data (e.g. m, mm, cm, rad etc.  )
    Returns:
        Figure
    History:
        2021_09_22 | MEG | Added to package.  
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    from scipy import optimize
    
    def test_func(x, a, b, c):
        """ a sets amplitude, b sets frequency, c sets gradient of linear term
        """
        return c * x + (a * np.sin((2*np.pi *(1/b) * x)))
    
    params, params_covariance = optimize.curve_fit(test_func, baselines_cs, np.cumsum(r3_data[:,points_interest['highlands'][1], 
                                                                                                points_interest['highlands'][0]]), p0=[15, 365, 0.01])            # p0 is first guess at abc parameters for sinusoid (ie. 365 means suggesting it has an annual period)
    
    y_highlands_predict = test_func(baselines_cs, params[0], params[1], params[2])                                  # predict points of line.  
    
    f, ax = plt.subplots(figsize = (10,6))
    f.canvas.set_window_title(title)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.axhline(0,c = 'k')
    for key, value in points_interest.items():
        ax.scatter(baselines_cs, np.cumsum(r3_data[:,value[1], value[0]]), label = key)              # plot each of hte points.  
    ax.plot(baselines_cs, y_highlands_predict, c='k', label = 'Sinusoid + linear')                          # plot the line of best fit.  
    ax.legend()
    
    
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 180))
    xticks_dayn = ax.get_xticks()
    xticks_date = []
    day0_date = datetime.strptime(acq_dates[0], '%Y%m%d')
    for xtick_dayn in xticks_dayn:
        xtick_date = day0_date + timedelta(days = float(xtick_dayn))                  # add the number of dats to the original date
        xticks_date.append(xtick_date.strftime('%Y_%m_%d'))
    ax.set_xticklabels(xticks_date, rotation = 'vertical')
    f.subplots_adjust(bottom=0.2)



#%% Things to set

downsample = 1.0

ICASAR_settings = {"n_comp" : 5,                                         # number of components to recover with ICA (ie the number of PCA sources to keep)
                   "bootstrapping_param" : (200, 0),                    # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                   "tsne_param" : (30, 12),                             # (perplexity, early_exaggeration)
                   "ica_param" : (1e-2, 150),                           # (tolerance, max iterations)
                   "hdbscan_param" : (100,10),                           # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                   "out_folder" : Path('insar_workshop_example'),   # outputs will be saved here
                   "create_all_ifgs_flag" : True,                       # small signals are hard for ICA to extact from time series, so make it easier by creating all possible long temporal baseline ifgs from the incremental data.  
                   "load_fastICA_results" : False,                      # If all the FastICA runs already exisit, setting this to True speeds up ICASAR as they don't need to be recomputed.  
                   "figures" : "png+window"}                            # if png, saved in a folder as .png.  If window, open as interactive matplotlib figures,
                                                                         # if 'png+window', both.  
                                                                         # default is "window" as 03_clustering_and_manifold is interactive.  
                                                                    
LiCSBAS_out_folder_campi_flegrei = Path("./124D_04854_171313_licsbas_example_extended_rationalized")


points_interest = {'deforming' : (102, 80),                 # These are points chosen at interest from the LICSBAS data.  x then y!
                    'highlands' : (161, 31)}                    # Note, these are set when downsampling is 1, and will be automatically adjusted it downsample is adjusted.  


                                              
#%% Import the results of LiCSBAS and check on their sizes etc. 

print(f"Opening the LiCSBAS .h5 file...", end = '')
displacement_r2, tbaseline_info, ref_xy = LiCSBAS_to_ICASAR(LiCSBAS_out_folder_campi_flegrei, figures=True, ref_area = True)        # open various LiCSBAS products, spatial ones in displacement_r2, temporal ones in tbaseline_info
del displacement_r2['cumulative']
displacement_r2 = LiCSAlert_preprocessing(displacement_r2, downsample, 1.0, mean_centre = False)                                                                # mean centre and downsample

points_interest['reference'] = (int(np.mean([ref_xy['x_start'], ref_xy['x_stop']])),                                                # also get the reference pixel out
                                int(np.mean([ref_xy['y_start'], ref_xy['y_stop']])))

for key, value in points_interest.items():                                                                                      # also change the points of interest (which are affected by downsampling)
    points_interest[key] = (int(value[0] * downsample), int(value[1] * downsample))


print(f"Done.  ")


for key, variable in displacement_r2.items():
    print(f"{key} : {variable.shape}")
print()
for key, variable in tbaseline_info.items():
    print(f"{key} : {len(variable)}")    



#%% Have a look at some parts of the LiCSBAS time series

# 1: The DEM.  
matrix_show(displacement_r2['dem'], title = 'DEM')

# 2: The mask.  
matrix_show(displacement_r2['mask'], title = 'Mask')



#%% Investigate the difference between row vectors and rank 2 data

# 3: The last cumulative interferogram.  
matrix_show(np.sum(displacement_r2['incremental'], axis = 0), title = 'Last cumulative interferogram as a row vector')
matrix_show(col_to_ma(np.sum(displacement_r2['incremental'], axis = 0), displacement_r2['mask']), title = 'Last cumulative interferogram as a 2d array')


displacement_r3 = {'incremental' : r2_to_r3(displacement_r2['incremental'], displacement_r2['mask'])}                                          # convert rank 2 (ie row vectors) to rank 3 (e.g n_images x ny x nx)
# displacement_r3['cumulative']  = np.concatenate((np.zeros((1,displacement_r2['dem'].shape[0], displacement_r2['dem'].shape[1])),
#                                                  np.cumsum(displacement_r3['incremental'], axis = 0)), axis = 0 )



#%% Plot the time series for a few points at different elevations.  

plot_points_interest(displacement_r3['incremental'], points_interest, tbaseline_info['baselines_cumulative'], tbaseline_info['acq_dates'], 'LiCSBAS displacements', 'mm')



#%% do ICA with ICSAR function

ifg_means = np.mean(displacement_r3['incremental'], axis = (1,2))[:, np.newaxis, np.newaxis]                                                        # get the mean for each ifg

spatial_data = {'mixtures_r2'    : displacement_r2['incremental'],
                'mask'           : displacement_r2['mask'],
                'ifg_dates'      : tbaseline_info['ifg_dates'],                             # this is optional.  In the previous example, we didn't have it, in form YYYYMMDD_YYYYMMDD as a list of strings.  
                'dem'            : displacement_r2['dem'],                                  # this is optional.  In the previous example, we didn't have it
                'lons'           : displacement_r2['lons'],
                'lats'           : displacement_r2['lats']}
                

ics, tcs, x_train_residual_ts, Iq, n_clusters, S_all_info, phUnw_mean  = ICASAR(spatial_data = spatial_data, **ICASAR_settings) 

#%% Plot the time series for the same points, using ICASAR correction.  



incremental_r2_mc_hat = tcs[:,0:1] @ ics[0:1, :]                                                                                               # reconstruct using the ICs.  
incremental_r3_mc_hat = r2_to_r3(incremental_r2_mc_hat, displacement_r2['mask'])                                                     # convert rank 2 (ie row vectors) to rank 3 (e.g n_images x ny x nx)
#plot_points_interest(incremental_r3_mc_hat, points_interest_gps, tbaseline_info['baselines_cumulative'], tbaseline_info['acq_dates'], 'ICASAR time series (all ICs), mean centered', y_label)
incremental_r3_hat = incremental_r3_mc_hat + np.repeat(np.repeat(ifg_means, displacement_r3['incremental'].shape[1], 1), displacement_r3['incremental'].shape[2], 2)
plot_points_interest(incremental_r3_mc_hat, points_interest, tbaseline_info['baselines_cumulative'], tbaseline_info['acq_dates'], 'ICASAR time series (all ICs), referenced',  'metres')




#%%

