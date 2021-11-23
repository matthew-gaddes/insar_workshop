
#
#%%

def plot_points_on_image(image, points_dict, title = ''):
    """
    """
    import matplotlib.pyplot as plt 
    f, ax = plt.subplots(1,1)
    ax.set_title(title)
    f.canvas.manager.set_window_title(f"{title}")
    matrixPlt = ax.imshow(image,interpolation='none', aspect='equal')
    f.colorbar(matrixPlt,ax=ax)
    for name, coord in points_dict.items():
        ax.scatter(coord[0], coord[1], label = name)
    ax.legend()



#%%
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
        fig.canvas.manager.set_window_title(f"{title}")
    plt.pause(1)                                                                    # to force it to be shown when usig ipdb

#%%

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

#%%

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


#%%

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
    f.canvas.manager.set_window_title(title)
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

    plt.pause(1)                                                                    # to force it to be shown when usig ipdb
