from astropy.io import fits, ascii
from astropy.time import Time
from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import glob
import scipy.interpolate as inter
from wotan import flatten
import time
from matplotlib import gridspec

plt.rcParams['axes.linewidth'] = 3

# just a bunch of plotting code
def apply_plot_formatting(ax, xlabel='Wavelength (Å)', ylabel='Flux', xlim=None, ylim=None,
                          show_x_label=True, show_x_tick_labels=True):
    """
    Applies consistent formatting to Matplotlib axes.
    """
    if show_x_label:
        ax.set_xlabel(xlabel, fontsize=20)
    if show_x_tick_labels:
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True,
                   width=1.5, labelsize=18)
    ax.tick_params(which='major', length=8)
    ax.tick_params(which='minor', length=4)
    ax.minorticks_on()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_spectra(data, filename, green_order_range=(0, 35), red_order_range=(0, 32), 
                 xlim=None, ylim=None, legend=True, fiber=('SCI1', 'SCI2', 'SCI3'),
                 use_matplotlib=True, color='red', figsize=(12, 6), orders=None,
                 normalize_flux=False, show_x_label=True, show_x_tick_labels=True):
    """
    Plots spectral data with optional flux normalization across all orders.
    
    Parameters
    ----------
    normalize_flux : bool, optional (default=False)
        Whether to normalize the flux across all orders.
        
    show_x_label : bool, optional (default=True)
        Whether to display the x-axis label ("Wavelength (Å)").
        
    show_x_tick_labels : bool, optional (default=True)
        Whether to display x-axis tick labels.

    Other parameters are the same as in the original docstring.
    """
    
    if isinstance(fiber, str):
        raise ValueError("The 'fiber' argument must be a tuple, so you may have to add a comma if you're only plotting one fiber e.g., ('SCI1',), ('SCI1', 'SCI2'), or ('SCI1', 'SCI2', 'SCI3').")
    
    available_fibers = {'SCI1': ('GREEN_SCI_WAVE1', 'GREEN_SCI_FLUX1', 'RED_SCI_WAVE1', 'RED_SCI_FLUX1'),
                        'SCI2': ('GREEN_SCI_WAVE2', 'GREEN_SCI_FLUX2', 'RED_SCI_WAVE2', 'RED_SCI_FLUX2'),
                        'SCI3': ('GREEN_SCI_WAVE3', 'GREEN_SCI_FLUX3', 'RED_SCI_WAVE3', 'RED_SCI_FLUX3')}
    
    if isinstance(color, str):
        color = (color,) * len(fiber)
    elif isinstance(color, tuple) and len(color) != len(fiber):
        raise ValueError(f"The 'color' tuple must have the same length as the 'fiber' tuple. Expected {len(fiber)} colors.")
    
    fiber_colors = dict(zip(fiber, color))

    if orders is None:
        green_orders = range(green_order_range[0], green_order_range[1])
        red_orders = range(red_order_range[0], red_order_range[1])
    else:
        if isinstance(orders, int):
            orders = [orders]
        green_orders = [o for o in orders if green_order_range[0] <= o < green_order_range[1]]
        red_orders = [o for o in orders if red_order_range[0] <= o < red_order_range[1]]
    
    if use_matplotlib:
        fig, ax = plt.subplots(figsize=figsize)

        labels_added = {fiber_name: False for fiber_name in fiber}

        for o in green_orders:
            for fiber_name in fiber:
                wave_key, flux_key, _, _ = available_fibers[fiber_name]
                flux_data = data[filename][flux_key][o, :]

                # Normalize flux if specified
                if normalize_flux:
                    flux_data = flux_data / np.max(flux_data)
                
                ax.plot(data[filename][wave_key][o, :], flux_data,
                        label=fiber_name if not labels_added[fiber_name] else "", 
                        color=fiber_colors[fiber_name])
            labels_added = {k: True for k in fiber}

        for o in red_orders:
            for fiber_name in fiber:
                _, _, wave_key, flux_key = available_fibers[fiber_name]
                flux_data = data[filename][flux_key][o, :]

                # Normalize flux if specified
                if normalize_flux:
                    flux_data = flux_data / np.max(flux_data)
                
                ax.plot(data[filename][wave_key][o, :], flux_data, color=fiber_colors[fiber_name])

        apply_plot_formatting(ax, xlim=xlim, ylim=ylim, show_x_label=show_x_label,
                              show_x_tick_labels=show_x_tick_labels)

        if legend:
            ax.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Plotly code could go here if needed
        raise NotImplementedError("Plotly is not implemented in this version.")

    return ax


def plot_spline_fit(wave_sci1, flux_sci1, spline_sci1, 
                    wave_sci2, flux_sci2, spline_sci2, 
                    wave_sci3, flux_sci3, spline_sci3, order, chip, use_matplotlib=False):
    """
    Plots the original flux and the fitted spline for all science fibers for a given order.
    
    Parameters:
    wave_sci1, wave_sci2, wave_sci3 (array): Arrays of wavelength values for sci1, sci2, and sci3.
    flux_sci1, flux_sci2, flux_sci3 (array): Arrays of flux values for sci1, sci2, and sci3.
    spline_sci1, spline_sci2, spline_sci3 (array): Arrays of spline values for sci1, sci2, and sci3.
    order (int): The spectral order.
    chip (str): The chip ('RED' or 'GREEN').
    use_matplotlib (bool): If True, plots use matplotlib. If False, uses plotly. Default is False.
    """
    # input validation checks
    if chip not in ['RED', 'GREEN']:
        raise ValueError("Invalid chip name. KPF has only two chips: 'RED' and 'GREEN'")
    if chip == 'GREEN' and (order < 0 or order > 34):
        raise ValueError("Invalid order range. KPF has 35 green orders (0-34).")
    if chip == 'RED' and (order < 0 or order > 31):
        raise ValueError("Invalid order range. KPF has 32 red orders (0-31).")
        
    if use_matplotlib:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wave_sci1, flux_sci1, label='Sci1', color='#1f77b4', linewidth=3)  # Muted blue
        ax.plot(wave_sci1, spline_sci1, color='black', linewidth=2.5)  # Soft orange
        
        ax.plot(wave_sci2, flux_sci2, label='Sci2', color='#2ca02c', linewidth=3)  # Muted green
        ax.plot(wave_sci2, spline_sci2, color='black', linewidth=2.5)  # Muted red
        
        ax.plot(wave_sci3, flux_sci3, label='Sci3', color='#9467bd', linewidth=3)  # Muted purple
        ax.plot(wave_sci3, spline_sci3, label='Spline Fit', color='black', linewidth=2.5)  # Brown

        apply_plot_formatting(ax, xlabel='Wavelength (Å)', ylabel='Flux')
        ax.set_title(f'Smooth Lamp for Order {order} - {chip} CCD', fontsize=14)
        ax.legend(fontsize=14)

        plt.tight_layout()
        plt.show()
    else:
        # Plotly code could go here if needed
        pass


def plot_spline_division(wave_green, flux_green, divided_flux_green, wave_red, flux_red, divided_flux_red, order_range_green=range(35), order_range_red=range(32), legend=True, use_matplotlib=False):
    """
    This function visualizes the correction for effects like fringing by plotting the original science data as well as the science divided by a spline fit to a 1D smooth lamp.
    
    Parameters:
    wave_green (numpy.ndarray): 2D array of wavelength data for the green chip.
    flux_green (numpy.ndarray): 2D array of flux data for the green chip.
    divided_flux_green (list): List of normalized flux data for the green chip.
    wave_red (numpy.ndarray): 2D array of wavelength data for the red chip.
    flux_red (numpy.ndarray): 2D array of flux data for the red chip.
    divided_flux_red (list): List of normalized flux data for the red chip.
    order_range_green (range): Range of spectral orders to run this function on for the green chip. Default is range(35).
    order_range_red (range): Range of spectral orders to run this function on for the red chip. Default is range(32).
    legend (bool): Flag to indicate whether to show legends on the plot.
    use_matplotlib (bool): If True, plots use matplotlib. If False, uses plotly. Default is False.

    Returns:
    fig: Plotly figure object if use_matplotlib is False, otherwise returns Matplotlib figure object.
    """
    if use_matplotlib:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, order in enumerate(order_range_green):
            wave = wave_green[order, :]
            flux = flux_green[order, :]
            flux_norm = divided_flux_green[i]
            ax.plot(wave, flux, color='green', linewidth=1, label='Green Science Data' if legend and i == 0 else "")
            ax.plot(wave, flux_norm, color='blue', linewidth=1, label='Green Data Divided by Smooth Lamp' if legend and i == 0 else "")
        for i, order in enumerate(order_range_red):
            wave = wave_red[order, :]
            flux = flux_red[order, :]
            flux_norm = divided_flux_red[i]
            ax.plot(wave, flux, color='red', linewidth=1, label='Red Science Data' if legend and i == 0 else "")
            ax.plot(wave, flux_norm, color='orange', linewidth=1, label='Red Data Divided by Smooth Lamp' if legend and i == 0 else "")

        apply_plot_formatting(ax)
        if legend:
            ax.legend()

        plt.tight_layout()
        plt.show()
        return fig


def plot_divided_flux(file_path, green_order_range=(0, 35), red_order_range=(0, 32), 
                      xlim=None, ylim=None, fiber=('SCI1', 'SCI2', 'SCI3'), 
                      legend=True, figsize=(12, 6), norm=True, show_x_label=True, show_x_tick_labels=True):
    """
    Plots the divided flux from a FITS file for specified fibers and orders, with an option to plot normalized or original divided flux.
    
    Parameters:
    file_path (str): Path to the FITS file.
    green_order_range (tuple): Range of green orders to plot (start, end).
    red_order_range (tuple): Range of red orders to plot (start, end).
    xlim (tuple): x-axis limits as (xmin, xmax). Default is (4450, 8700).
    ylim (tuple): y-axis limits as (ymin, ymax). Default is None (automatic scaling).
    fiber (tuple): Tuple containing fiber names to plot (e.g., ('SCI1', 'SCI2', 'SCI3')).
    legend (bool): If True, display legend on the plot.
    figsize (tuple): Figure size for the plot. Default is (12, 2).
    norm (bool): If True, plots normalized divided flux; if False, plots original divided flux.
    show_x_label (bool): If True, displays the x-axis label. Default is True.
    show_x_tick_labels (bool): If True, displays x-axis tick labels. Default is True.

    Returns:
    None
    """
    if isinstance(fiber, str):
        raise ValueError("The 'fiber' argument must be a tuple, so you may have to add a comma if you're only plotting one fiber e.g., ('SCI1',), ('SCI1', 'SCI2'), or ('SCI1', 'SCI2', 'SCI3').")

    # Map fibers to corresponding data in the FITS file
    flux_suffix = '_FLUX_DIV_NORM' if norm else '_FLUX_DIV'
    
    available_fibers = {
        'SCI1': ('GREEN_SCI_WAVE1', 'SCI1_GREEN' + flux_suffix, 'RED_SCI_WAVE1', 'SCI1_RED' + flux_suffix),
        'SCI2': ('GREEN_SCI_WAVE2', 'SCI2_GREEN' + flux_suffix, 'RED_SCI_WAVE2', 'SCI2_RED' + flux_suffix),
        'SCI3': ('GREEN_SCI_WAVE3', 'SCI3_GREEN' + flux_suffix, 'RED_SCI_WAVE3', 'SCI3_RED' + flux_suffix)
    }

    # Open the FITS file
    with fits.open(file_path) as hdul:
        # Initialize plot
        fig, ax = plt.subplots(figsize=figsize)

        # Only one legend entry per fiber
        labels_added = {fiber_name: False for fiber_name in fiber}

        # Loop over the green orders and fibers
        for o in range(green_order_range[0], green_order_range[1]):
            for fiber_name in fiber:
                green_wave_key, green_flux_key, _, _ = available_fibers[fiber_name]
                green_wave_data = hdul[green_wave_key].data[o, :]
                green_flux_data = hdul[green_flux_key].data[o]

                ax.plot(green_wave_data, green_flux_data,
                        label=fiber_name if not labels_added[fiber_name] else "",
                        color='green', linewidth=2)

            labels_added = {fiber_name: True for fiber_name in fiber}

        # Loop over the red orders and fibers
        for o in range(red_order_range[0], red_order_range[1]):
            for fiber_name in fiber:
                _, _, red_wave_key, red_flux_key = available_fibers[fiber_name]
                red_wave_data = hdul[red_wave_key].data[o, :]
                red_flux_data = hdul[red_flux_key].data[o]

                ax.plot(red_wave_data, red_flux_data,
                        color='red', linewidth=1)

        ylabel = 'Normalized Flux' if norm else 'Divided Flux'
        apply_plot_formatting(ax, xlim=xlim, ylim=ylim, ylabel=ylabel,
                              show_x_label=show_x_label, show_x_tick_labels=show_x_tick_labels)

        if legend:
            ax.legend()
        if norm:
            ax.axhline(y=1, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.show()


def plot_continuum_norm_data(file_path, fiber, chip, order, xlim=None, ylim=None):
    """
    Plots the continuum-normalized data for a specific file, fiber, chip, and order.

    Parameters:
    - file_path : str
        Path to the FITS file containing continuum-normalized data.
    - fiber : str
        The fiber to plot, e.g., 'SCI1', 'SCI2', or 'SCI3'.
    - chip : str
        The chip to plot, 'GREEN' or 'RED'.
    - order : int
        The order to plot.
    - xlim : tuple, optional
        X-axis limits for the plot.
    - ylim : tuple, optional
        Y-axis limits for the plot.
    """
    
    # Open the FITS file and retrieve wavelength and flux data for the specified chip, fiber, and order
    with fits.open(file_path) as hdul:
        wave_key = f'{chip}_SCI_WAVE{fiber[-1]}'
        flux_key = f'{fiber}_{chip}_FLUX_CONT_NORM'

        wavelength = hdul[wave_key].data[order]
        flux = hdul[flux_key].data[order]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    color = 'green' if chip == 'GREEN' else 'red'
    ax.plot(wavelength, flux, color=color, label=f'{fiber} {chip} Order {order}')
    
    apply_plot_formatting(ax, xlim=xlim, ylim=ylim)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.set_title(f"{fiber}, {chip}, Order {order}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_spline_fit_to_median(median_wavelengths, median_fluxes, splines, chip, fiber):
    """
    Plots the spline fit to median-combined flux for each order.

    Parameters:
    median_wavelengths : list of arrays
        List containing arrays of median wavelengths for each order.
    median_fluxes : list of arrays
        List containing arrays of median fluxes for each order.
    splines : list of spline functions
        List containing spline functions fitted to each order.
    chip : str
        'GREEN' or 'RED'.
    fiber : str
        'SCI1', 'SCI2', or 'SCI3'.
    """
    num_orders = len(median_wavelengths)
    for order in range(num_orders):
        wavelength = median_wavelengths[order]
        flux = median_fluxes[order]
        spline_fit = splines[order](wavelength)

        fig, ax = plt.subplots(figsize=(12, 6))
        color = 'green' if chip == 'GREEN' else 'red'
        ax.plot(wavelength, flux, color=color, label='Median Flux')
        ax.plot(wavelength, spline_fit, color='black', linestyle='-', linewidth=1.5, label='Spline Fit')

        apply_plot_formatting(ax)
        ax.legend()
        plt.tight_layout()
        plt.show()
        
def plot_data_stages(file_path, fiber, chip, order, stages=('original', 'divided', 'continuum'),
                     xlim=None, ylim=None, figsize=(9, 4.5), colors=None):
    """
    Plots data at different processing stages for a specific file, fiber, chip, and order.

    Parameters:
    - file_path : str
        Path to the FITS file.
    - fiber : str
        The fiber to plot, e.g., 'SCI1', 'SCI2', or 'SCI3'.
    - chip : str
        The chip to plot, 'GREEN' or 'RED'.
    - order : int
        The order to plot.
    - stages : tuple, optional
        Which stages to plot. Options are 'original', 'divided', 'continuum'.
        Default is ('original', 'divided', 'continuum').
    - xlim : tuple, optional
        X-axis limits for the plot.
    - ylim : tuple, optional
        Y-axis limits for the plot.
    - figsize : tuple, optional
        Size of the figure.
    - colors : dict, optional
        Dictionary specifying colors for each stage. Keys should be stage names.

    Returns:
    - None
    """
    # Define default colors if not provided
    if colors is None:
        colors = {
            'original': 'blue',
            'divided': 'green',
            'continuum': 'mediumpurple'
        }

    # Open the FITS file and retrieve data
    with fits.open(file_path) as hdul:
        wave_key = f'{chip}_SCI_WAVE{fiber[-1]}'
        flux_key = f'{chip}_SCI_FLUX{fiber[-1]}'

        wavelength = hdul[wave_key].data[order]
        flux = hdul[flux_key].data[order]
        flux /= np.median(flux)
        # Prepare the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, zorder = 1000)
        #ax.axhline(y=0.9, color='black', linestyle='--', linewidth=1, zorder = 1000)
        # Plot original data
        if 'original' in stages:
            ax.plot(wavelength, flux, color=colors.get('original', 'blue'),
                    label='1D Spectrum')

        # Plot data after dividing by smooth lamp
        if 'divided' in stages:
            flux_div_key = f'{fiber}_{chip}_FLUX_DIV_NORM'
            if flux_div_key in hdul:
                flux_div = hdul[flux_div_key].data[order]
                ax.plot(wavelength, flux_div, color=colors.get('divided', 'green'),
                        label='Blaze Corrected')
            else:
                print(f"Divided flux data not found in FITS file for {fiber}, {chip}, order {order}.")

        # Plot continuum-normalized data
        if 'continuum' in stages:
            flux_cont_norm_key = f'{fiber}_{chip}_FLUX_CONT_NORM'
            if flux_cont_norm_key in hdul:
                flux_cont_norm = hdul[flux_cont_norm_key].data[order]
                ax.plot(wavelength, flux_cont_norm, color=colors.get('continuum', 'red'),
                        label='Continuum Normalized')
            else:
                print(f"Continuum-normalized flux data not found in FITS file for {fiber}, {chip}, order {order}.")

        # Formatting the plot
        apply_plot_formatting(ax, xlim=xlim, ylim=ylim)
        ax.legend(fontsize = 13)

        plt.tight_layout()
        plt.savefig('cont-normalized.pdf', dpi = 250)
        plt.show()

def plot_interpolated_flux(file_path, fibers=['SCI1'], xlim=(4500, 8700), ylim=None, figsize=(12, 6), colors=None):
    """
    Plots the interpolated wavelength and flux data for a given FITS file.

    Parameters:
    file_path (str): Path to the FITS file.
    fibers (list of str, optional): List of fiber labels to plot (e.g., 'SCI1', 'SCI2', 'SCI3'). Defaults to ['SCI1'].
    xlim (tuple, optional): X-axis limits for the plot (default is (4520, 4550)).
    ylim (tuple, optional): Y-axis limits for the plot. Defaults to None (automatic).
    figsize (tuple, optional): Size of the plot figure. Default is (12, 2).
    colors (dict, optional): Dictionary specifying colors for each fiber. Keys should be fiber names.

    Returns:
    None: Displays the plot.
    """
    if colors is None:
        colors = {'SCI1': 'blue', 'SCI2': 'green', 'SCI3': 'red'}

    try:
        with fits.open(file_path) as hdul:
            fig, ax = plt.subplots(figsize=figsize)

            for fiber in fibers:
                wave_hdu_name = f'WAVE_INTERP_{fiber}'
                flux_hdu_name = f'FLUX_INTERP_{fiber}'

                if wave_hdu_name in hdul and flux_hdu_name in hdul:
                    wave_data = hdul[wave_hdu_name].data
                    flux_data = hdul[flux_hdu_name].data

                    color = colors.get(fiber, 'black')  # Default color if not specified
                    ax.plot(wave_data, flux_data, label=f'{fiber}', color=color, linewidth=1.5)

            apply_plot_formatting(ax, xlabel='Wavelength (Å)', ylabel='Flux', xlim=xlim, ylim=ylim)
            ax.set_title(f'Interpolated Flux for {file_path}', fontsize=14)
            ax.legend(fontsize=12)

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error plotting interpolated flux for file {file_path}: {e}")
        
def plot_L1_spectrum(L1_path, variance=False, data_over_sqrt_variance=False, 
                     orderlet=None, fig_path=None, show_plot=False):
    """
    Generate a rainbow-colored L1 spectrum plot (adapted from KPF DRP)
    
    Args:
        L1_path (str): Path to the L1 FITS file.
        variance (bool): Plot the variance extensions instead of signal.
        data_over_sqrt_variance (bool): Plot data divided by sqrt(variance) (approximate SNR).
        orderlet (str): One of "CAL", "SCI1", "SCI2", "SCI3", or "SKY".
        fig_path (str): Path to save the figure (optional).
        show_plot (bool): Whether to show the plot interactively.
    
    Returns:
        None. The function either saves the figure or displays it.
    """
    # Open the FITS file
    L1 = fits.open(L1_path)
    
    # Number of orders per panel
    n_orders_per_panel = 8

    # Get lowercase orderlet for comparison
    ord_lower = orderlet.lower()

    # Select wavelength and flux arrays based on orderlet
    if ord_lower == 'sci1':
        wav_green = np.array(L1['GREEN_SCI_WAVE1'].data, dtype='d')
        wav_red   = np.array(L1['RED_SCI_WAVE1'].data, dtype='d')
        if variance:
            flux_green = np.array(L1['GREEN_SCI_VAR1'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_VAR1'].data, dtype='d')
        elif data_over_sqrt_variance:
            flux_green = np.divide(
                np.array(L1['GREEN_SCI_FLUX1'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR1'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['GREEN_SCI_FLUX1'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR1'].data, dtype='d'))) != 0
            )
            flux_red = np.divide(
                np.array(L1['RED_SCI_FLUX1'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['RED_SCI_VAR1'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['RED_SCI_FLUX1'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['RED_SCI_VAR1'].data, dtype='d'))) != 0
            )
        else:
            flux_green = np.array(L1['GREEN_SCI_FLUX1'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_FLUX1'].data, dtype='d')

    elif ord_lower == 'sci2':
        wav_green = np.array(L1['GREEN_SCI_WAVE2'].data, dtype='d')
        wav_red   = np.array(L1['RED_SCI_WAVE2'].data, dtype='d')
        if variance:
            flux_green = np.array(L1['GREEN_SCI_VAR2'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_VAR2'].data, dtype='d')
        elif data_over_sqrt_variance:
            flux_green = np.divide(
                np.array(L1['GREEN_SCI_FLUX2'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR2'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['GREEN_SCI_FLUX2'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR2'].data, dtype='d'))) != 0
            )
            flux_red = np.divide(
                np.array(L1['RED_SCI_FLUX2'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['RED_SCI_VAR2'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['RED_SCI_FLUX2'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['RED_SCI_VAR2'].data, dtype='d'))) != 0
            )
        else:
            flux_green = np.array(L1['GREEN_SCI_FLUX2'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_FLUX2'].data, dtype='d')

    elif ord_lower == 'sci3':
        wav_green = np.array(L1['GREEN_SCI_WAVE3'].data, dtype='d')
        wav_red   = np.array(L1['RED_SCI_WAVE3'].data, dtype='d')
        if variance:
            flux_green = np.array(L1['GREEN_SCI_VAR3'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_VAR3'].data, dtype='d')
        elif data_over_sqrt_variance:
            flux_green = np.divide(
                np.array(L1['GREEN_SCI_FLUX3'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR3'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['GREEN_SCI_FLUX3'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['GREEN_SCI_VAR3'].data, dtype='d'))) != 0
            )
            flux_red = np.divide(
                np.array(L1['RED_SCI_FLUX3'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['RED_SCI_VAR3'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['RED_SCI_FLUX3'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['RED_SCI_VAR3'].data, dtype='d'))) != 0
            )
        else:
            flux_green = np.array(L1['GREEN_SCI_FLUX3'].data, dtype='d')
            flux_red   = np.array(L1['RED_SCI_FLUX3'].data, dtype='d')

    elif ord_lower == 'sky':
        wav_green = np.array(L1['GREEN_SKY_WAVE'].data, dtype='d')
        wav_red   = np.array(L1['RED_SKY_WAVE'].data, dtype='d')
        if variance:
            flux_green = np.array(L1['GREEN_SKY_VAR'].data, dtype='d')
            flux_red   = np.array(L1['RED_SKY_VAR'].data, dtype='d')
        elif data_over_sqrt_variance:
            flux_green = np.divide(
                np.array(L1['GREEN_SKY_FLUX'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['GREEN_SKY_VAR'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['GREEN_SKY_FLUX'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['GREEN_SKY_VAR'].data, dtype='d'))) != 0
            )
            flux_red = np.divide(
                np.array(L1['RED_SKY_FLUX'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['RED_SKY_VAR'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['RED_SKY_FLUX'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['RED_SKY_VAR'].data, dtype='d'))) != 0
            )
        else:
            flux_green = np.array(L1['GREEN_SKY_FLUX'].data, dtype='d')
            flux_red   = np.array(L1['RED_SKY_FLUX'].data, dtype='d')

    elif ord_lower == 'cal':
        wav_green = np.array(L1['GREEN_CAL_WAVE'].data, dtype='d')
        wav_red   = np.array(L1['RED_CAL_WAVE'].data, dtype='d')
        if variance:
            flux_green = np.array(L1['GREEN_CAL_VAR'].data, dtype='d')
            flux_red   = np.array(L1['RED_CAL_VAR'].data, dtype='d')
        elif data_over_sqrt_variance:
            flux_green = np.divide(
                np.array(L1['GREEN_CAL_FLUX'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['GREEN_CAL_VAR'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['GREEN_CAL_FLUX'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['GREEN_CAL_VAR'].data, dtype='d'))) != 0
            )
            flux_red = np.divide(
                np.array(L1['RED_CAL_FLUX'].data, dtype='d'),
                np.sqrt(np.abs(np.array(L1['RED_CAL_VAR'].data, dtype='d'))),
                out=np.zeros_like(np.array(L1['RED_CAL_FLUX'].data, dtype=float)),
                where=np.sqrt(np.abs(np.array(L1['RED_CAL_VAR'].data, dtype='d'))) != 0
            )
        else:
            flux_green = np.array(L1['GREEN_CAL_FLUX'].data, dtype='d')
            flux_red   = np.array(L1['RED_CAL_FLUX'].data, dtype='d')
    else:
        raise ValueError("orderlet not specified properly. Choose from 'SCI1', 'SCI2', 'SCI3', 'SKY', or 'CAL'.")

    # In case of missing data, create placeholders
    if np.shape(flux_green) == (0,):
        flux_green = wav_green * 0.
    if np.shape(flux_red) == (0,):
        flux_red = wav_red * 0.

    # Concatenate the green and red arrays
    wav = np.concatenate((wav_green, wav_red), axis=0)
    flux = np.concatenate((flux_green, flux_red), axis=0)

    # Set up figure and subplots
    cm = plt.cm.get_cmap('rainbow')
    num_panels = int(np.shape(wav)[0] / n_orders_per_panel) + 1
    fig, axes = plt.subplots(num_panels, 1, sharey=False, 
                             figsize=(20, 16), tight_layout=True)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.0)
    
    # Ensure axes is iterable even if there is one panel
    if num_panels == 1:
        axes = [axes]

    # Plot each spectral order
    for i in range(np.shape(wav)[0]):
        # Skip orders with a zero starting wavelength
        if wav[i, 0] == 0:
            continue
        low, high = np.nanpercentile(flux[i, :], [0.1, 99.9])
        # Replace extreme values with NaN
        flux[i, :][(flux[i, :] > high) | (flux[i, :] < low)] = np.nan
        j = int(i / n_orders_per_panel)
        rgba = cm((i % n_orders_per_panel) / n_orders_per_panel)
        axes[j].plot(wav[i, :], flux[i, :], linewidth=0.6, color=rgba)
        # Set x-limits based on orders in the panel
        left = np.min(wav[j * n_orders_per_panel:(j + 1) * n_orders_per_panel, :])
        right = np.max(wav[j * n_orders_per_panel:(j + 1) * n_orders_per_panel, :])
        low_panel, high_panel = np.nanpercentile(flux[j * n_orders_per_panel:(j + 1) * n_orders_per_panel, :], [0.1, 99.9])
        axes[j].set_xlim(left, right)
        axes[j].set_ylim(np.nanmin(flux[j * n_orders_per_panel:(j + 1) * n_orders_per_panel, :]) - high_panel * 0.05,
                         high_panel * 1.15)
        axes[j].tick_params(axis='x', labelsize=19)
        axes[j].tick_params(axis='y', labelsize=19)
        axes[j].axhline(0, color='gray', linestyle='dotted', linewidth=0.5)
        axes[j].grid(False)

    # Add axis labels and overall title
    if variance:
        title = f'L1 Variance Spectrum of {orderlet.upper()}'
        ylabel = f'Variance (e-) in {orderlet.upper()}'
    elif data_over_sqrt_variance:
        title = f'L1 SNR Spectrum of {orderlet.upper()}'
        ylabel = f'SNR (Counts / Variance^(1/2)) in {orderlet.upper()}'
    else:
        title = f'1D Spectrum of {orderlet.upper()}'
        ylabel = 'Counts (e-)'

    mid_panel = int(num_panels / 2)
    axes[mid_panel].set_ylabel(ylabel, fontsize=28)
    plt.xlabel('Wavelength (Å)', fontsize=28)
    fig.suptitle(title, fontsize=28)

    plt.tight_layout()
    
    # Save the figure if a path is provided
    if fig_path is not None:
        t0 = time.process_time()
        plt.savefig(fig_path, dpi=288, facecolor='w')
        print(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
    if show_plot:
        plt.show()
    plt.show()
    plt.close('all')
