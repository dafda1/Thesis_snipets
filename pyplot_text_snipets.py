# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:27:35 2025

@author: dafda1
"""

import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from scipy.optimize import curve_fit
from PyMPMS.RawData.rw_tools import covariance_to_correlation

thesis_main_dir = join(r"C:\\", "Users", "dafda1",
                       "OneDrive - University of St Andrews",
                       "Dissertation", "chapters")

image_folder_dict = {}
chapters = [f"chapter_{i}" for i in range(1, 6)] + ["introduction", "conclusion"]
for chapter in chapters:
    image_folder_dict[chapter] = join(thesis_main_dir, chapter, "images")

#alphabet for quick figure labels - use reshape to get apropriate array
letters = np.array(["a", "b", "c", "d", "e", "f",
                    "g", "h", "i", "j", "k", "l",
                    "m", "n", "o", "p", "q", "r",
                    "s", "t", "u", "v", "w", "x",
                    "y", "z"])

def pop_subfigure_label (axis_object, label,
                         relative_x_distance = False,
                         horizontalalignment = "left",
                         verticalalignment = "top",
                         xpos = None, ypos = None,
                         fontsize = 15,
                         fontcolor = "black",
                         use_background = False):
    
    if relative_x_distance:
        if xpos is None and ypos is None:
            xpos = relative_x_distance
            ypos = 1 - relative_x_distance/0.75
            
    elif xpos is None and ypos is None:
        raise ValueError("Input error: please specify coordinates.")
    
    if use_background:
        background_color = use_background
    else:
        background_color = "none"
    
    axis_object.text(xpos, ypos, label,
                     fontweight = "bold", fontsize = fontsize,
                     horizontalalignment = horizontalalignment,
                     verticalalignment = verticalalignment,
                     transform = axis_object.transAxes,
                     color = fontcolor,
                     backgroundcolor = background_color)
    
    return axis_object

def add_axis_lines_to_axis_object (axis_object, x0 = 0, y0 = 0,
                                   linewidth = 0.75, linestyle = "--",
                                   color = "k"):
    
    axis_object.axhline(y0, linewidth = linewidth,
                        color = color, linestyle = linestyle)
    
    axis_object.axvline(x0, linewidth = linewidth,
                        color = color, linestyle = linestyle)
    
    return axis_object

def my_errorbar_band (axis_object, xdata, ydata, sdata, color,
                      fmt = "-", label = None):
    
    axis_object.plot(xdata, ydata + sdata, fmt, color = color)
    axis_object.plot(xdata, ydata - sdata, fmt, color = color, label = label)
    
    return axis_object

def my_axislabel_arrows (axis_object, ypositions,
                         lengths, colors, edge = 0.05):
    
    axis_object.arrow(edge + lengths[0], ypositions[0], -lengths[0], 0,
                      width = 0.01,
                      color = colors[0],
                      transform = axis_object.transAxes)

    axis_object.arrow(1 - edge - lengths[1], ypositions[1], lengths[1], 0,
                      width = 0.01,
                      color = colors[1],
                      transform = axis_object.transAxes)
    
    return axis_object

def my_box (axis_object, xpositions, ypositions,
            fmt = '--', linewidth = 1.0):
    
    axis_object.plot((xpositions[0], xpositions[1], xpositions[1],
                      xpositions[0], xpositions[0]),
                     (ypositions[0], ypositions[0], ypositions[1],
                      ypositions[1], ypositions[0]),
                     fmt, linewidth = linewidth,
                     zorder = 0)
    
    return axis_object

def my_zoomy_box (axis_object, zoomed_region,
                  inset_region, paired_vertices,
                  linewidth = 0.75, fmt = "k--",
                  line_transparency = 1.0):
    """
    Tool for creating inset axis with zoomed version of specific
    part of plot. Returns the original axis and the inset axis.

    Parameters
    ----------
    axis_object : matplotlib axis object
        Original axis object in which to plot.
    zoomed_region : 4-tuple
        Positions (in data coordinates) of the corners of the box
        to draw, in the order (x0, y0, x1, y1).
    inset_region : 4-tuple
        Positions (in data coordinates of the original axis)
        of the corners of the inset axis, in the order
        (x0, y0, x1, y1).
    paired_vertices : 4-tuple
        Choice of which corners to link between the drawn box
        and the inset axis, in the format
        (corner1: i, j, corner2: i, j). E.g. to connect the
        upper left and lower right corners, one would choose
        paired_vertices = (0, 1, 1, 0).
    linewidth : float, optional
        Linewdith of the drawn box lines and the lines connecting
        the box to the inset axis. The default is 0.75.
    fmt : string, optional
        Format of the drawn box lines and the lines connecting
        the box to the inset axis. The default is "k--".

    Returns
    -------
    axis_object : matplotlib axis object
        Original axis is returned.
    inset_axis_object : matplotlib axis object
        Inset axis object.

    """

    axis_object = my_box(axis_object,
                         xpositions = zoomed_region[0::2],
                         ypositions = zoomed_region[1::2],
                         linewidth = linewidth, fmt = fmt)
    
    inset_shape = [inset_region[0], inset_region[1],
                   inset_region[2] - inset_region[0],
                   inset_region[3] - inset_region[1]]
    
    inset_axis_object =\
        axis_object.inset_axes(inset_shape,
                               transform = axis_object.transData)
        
    for line_index in (0, 2):
        iv = paired_vertices[line_index]
        jv = paired_vertices[line_index + 1]
        
        xB = zoomed_region[0::2][iv]
        yB = zoomed_region[1::2][jv]
        
        xI = inset_region[0::2][iv]
        yI = inset_region[1::2][jv]
    
        axis_object.plot((xB, xI), (yB, yI), fmt, linewidth = linewidth,
                         alpha = line_transparency)
    
    return axis_object, inset_axis_object

#special
def compute_SSR (xdata, ydata, sdata, fit_function, thetavec):
    residuals = ydata - fit_function(xdata, *thetavec)
    return np.sum((residuals*1.0/sdata)**2)

def chisquare_contours (axis_object, xdata, ydata, sdata,
                        fit_function, p0, theta_strings,
                        show_theta = False,
                        nsigma = 3, resolution = 100,
                        theta_scalings = (1, 1), markersize = 7,
                        plot_cbar = True, cbar_cax = None,
                        cbar_ax = None,
                        plot_legend = True):
    """
    show_theta = {'which': 0,
                  'units': 'meV'}
    """
    
    npoints = xdata.size
    dof = npoints - len(p0)
    
    popt, pcov = curve_fit(fit_function, xdata, ydata, sigma = sdata,
                           p0 = p0)
    
    chisquare = compute_SSR(xdata, ydata, sdata, fit_function,
                            thetavec = popt)
    
    VSF = chisquare*1.0/dof #Variance Scaling Factor
    
    sigma_theta0 = np.sqrt(pcov[0, 0])
    sigma_theta1 = np.sqrt(pcov[1, 1])
    
    delta_theta0 = nsigma*sigma_theta0
    delta_theta1 = nsigma*sigma_theta1
    
    theta0_space = np.linspace(popt[0] - delta_theta0,
                               popt[0] + delta_theta0, resolution + 1)
    theta1_space = np.linspace(popt[1] - delta_theta1,
                               popt[1] + delta_theta1, resolution + 1)
    
    SSR_grid = np.empty((theta0_space.size, theta1_space.size),
                        dtype = float)
    
    for i, theta0_val in enumerate(theta0_space):
        for j, theta1_val in enumerate(theta1_space):
            SSR_grid[i, j] = compute_SSR(xdata, ydata, sdata, fit_function,
                                         thetavec = (theta0_val, theta1_val))
    
    SSR_grid = SSR_grid/VSF - dof
    
    cmap = axis_object.contourf(theta0_space*theta_scalings[0],
                                theta1_space*theta_scalings[1],
                                np.sqrt(SSR_grid),
                                levels = np.arange(nsigma + 1))
    
    axis_object.errorbar(popt[0]*theta_scalings[0],
                         popt[1]*theta_scalings[1],
                         fmt = 'o', markersize = markersize,
                         yerr = sigma_theta1*theta_scalings[1],
                         xerr = sigma_theta0*theta_scalings[0],
                         label = "Estimated result \nand standard deviations")
    
    if plot_cbar:
        cbar = plt.colorbar(cmap, ax = cbar_ax, cax = cbar_cax)
    
    theta0_txt, theta1_txt = theta_strings
    
    text = ""
    if show_theta:
        which = show_theta['which']
        unpack = np.array([popt[which],
                                 np.sqrt(pcov[which, which])]*\
                           theta_scalings[which])
        
        text += r"$" + theta_strings[which] +\
                    r"= %.2f \pm %.2f$  (1$\sigma$)\n" % tuple(unpack)
                        
    
    text += r"$\rho = %.3f$" % covariance_to_correlation(pcov)[0, 1]
    axis_object.text(0.05, 0.05, text,
                     transform = axis_object.transAxes,
                     horizontalalignment = "left",
                     verticalalignment = "bottom",
                     fontsize = 15)
    
    axis_object.set_ylabel(r"$\theta_1$ (change this)")
    axis_object.set_xlabel(r"$\theta_0$ (change this)")
    
    if plot_legend:
        axis_object.legend(loc = "upper right")
    
    if plot_cbar:
        cbar.ax.set_ylabel(r"$\left|\theta - \hat{\theta}\right|/\sigma$",
                           labelpad = 15, rotation=270)
    
    # print(Tmin, Tmax)
    # print(popt)
    # print(covariance_to_correlation(pcov))
    # print()
    
    return axis_object

def make_figure_from_subfigures (array_of_fnames, figsize = (6.4, 6.4),
                                 fontcolor = "black", fontsize = 15,
                                 use_background = False):
    
    nrows, ncols = array_of_fnames.shape
    size = array_of_fnames.size
    labels = letters[:size].reshape(nrows, ncols)
    
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows,
                             figsize = np.array((ncols, nrows))*np.array(figsize))
    
    for i in np.arange(nrows):
        for j in np.arange(ncols):
            if nrows == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
                
            image = plt.imread(array_of_fnames[i, j])
            label = "(" + labels[i, j] + ")"
            
            ax.imshow(image)
            
            ax = pop_subfigure_label(ax, label, relative_x_distance = 0.05,
                                     fontcolor = fontcolor, fontsize = fontsize,
                                     use_background = use_background)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    
    return fig, axes