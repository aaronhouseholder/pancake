from astropy.io import fits, ascii
from plotting_tools import *
from astropy.time import Time
from astropy.table import Table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate as inter
from wotan import flatten
import time
from tqdm import tqdm
from scipy.interpolate import LSQUnivariateSpline
from tqdm import tqdm
from glob import glob

#functions for cleaning kpf data
def read_kpf_data(directory):
    """
    Reads and extracts KPF data from multiple FITS files in a specified directory.

    Parameters:
    directory (str): Directory containing the FITS files.

    Returns:
    dict: Dictionary with filenames as keys and data as values.
    """
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
            L1_file = os.path.join(directory, filename)
            with fits.open(L1_file) as L1:
                data[filename] = {
                    'GREEN_SCI_WAVE1': np.array(L1['GREEN_SCI_WAVE1'].data),
                    'GREEN_SCI_WAVE2': np.array(L1['GREEN_SCI_WAVE2'].data),
                    'GREEN_SCI_WAVE3': np.array(L1['GREEN_SCI_WAVE3'].data),
                    'GREEN_SCI_FLUX1': np.array(L1['GREEN_SCI_FLUX1'].data),
                    'GREEN_SCI_FLUX2': np.array(L1['GREEN_SCI_FLUX2'].data),
                    'GREEN_SCI_FLUX3': np.array(L1['GREEN_SCI_FLUX3'].data),
                    'RED_SCI_WAVE1': np.array(L1['RED_SCI_WAVE1'].data),
                    'RED_SCI_WAVE2': np.array(L1['RED_SCI_WAVE2'].data),
                    'RED_SCI_WAVE3': np.array(L1['RED_SCI_WAVE3'].data),
                    'RED_SCI_FLUX1': np.array(L1['RED_SCI_FLUX1'].data),
                    'RED_SCI_FLUX2': np.array(L1['RED_SCI_FLUX2'].data),
                    'RED_SCI_FLUX3': np.array(L1['RED_SCI_FLUX3'].data),
                    'Date-Beg': L1[0].header['Date-Beg'],
                    'Date-Mid': L1[0].header['Date-Mid']
                }
    return data

def fit_spline_to_smooth_lamp(fits_file, order, chip, spline_method='rspline', window_length=1, break_tolerance=0.5, plot=False, use_matplotlib=False):
    """
    Fits a spline to a smooth lamp pattern (using wotan) for a given order on a chip.

    Parameters:
    fits_file (str): The path to the FITS file.
    order (int): The spectral order to fit.
    chip (str): The chip ('RED' or 'GREEN').
    window_length (float, optional): The length of the filter window in units of wavelength. Default is 1.
    break_tolerance (float, optional): Split into segments at breaks longer than that. Default is 0.5.
    plot (bool, optional): If True, plots the spline fit for each science fiber. Default is False.
    use_matplotlib (bool, optional): If True, plots using matplotlib. If False, uses plotly. Default is False.

    Returns:
    tuple: Splines for the three science fibers.
    """
    chip = chip.upper()
    # input validation checks
    if chip not in ['RED', 'GREEN']:
        raise ValueError("Invalid chip name. KPF has only two chips: 'RED' and 'GREEN'")
    if chip == 'GREEN' and (order < 0 or order > 34):
        raise ValueError("Invalid order range. KPF has 35 green orders (0-34).")
    if chip == 'RED' and (order < 0 or order > 31):
        raise ValueError("Invalid order range. KPF has 32 red orders (0-31).")

    # reading in FITS file
    try:
        with fits.open(fits_file) as L1:
            wave_keys = [f'{chip}_SCI_WAVE{i}' for i in range(1, 4)]
            flux_keys = [f'{chip}_SCI_FLUX{i}' for i in range(1, 4)]
            waves = [np.array(L1[wave_key].data) for wave_key in wave_keys]
            fluxes = [np.array(L1[flux_key].data) for flux_key in flux_keys]
    except Exception as error:
        print(f"Error opening FITS file: {error}")
        return None

    # spline fitting happens here using wotan
    splines = []
    for wave, flux in zip(waves, fluxes):
        _, spline = flatten(
            np.sort(wave[order, :]),  # wavelength values, sorted b/c kpf is backwards
            flux[order, :],  # Array of flux values
            method=spline_method,
            window_length=window_length,
            break_tolerance=break_tolerance,
            return_trend=True,
        )
        splines.append(spline)

    if plot:
        plot_spline_fit(np.sort(waves[0][order, :]), fluxes[0][order, :], splines[0],
                        np.sort(waves[1][order, :]), fluxes[1][order, :], splines[1],
                        np.sort(waves[2][order, :]), fluxes[2][order, :], splines[2], order, chip, use_matplotlib)

    return tuple(splines)

def fit_spline_many_orders(fits_files, chip, order_range, spline_method='rspline', window_length=1, break_tolerance=0.5, plot_orders=None, use_matplotlib=False, time=False):
    """
    Fits a spline to many spectral orders across a 1D smooth lamp file

    Parameters:
    fits_files (list): List of paths to FITS files.
    chip (str): The chip ('RED' or 'GREEN').
    order_range (range): Range of spectral orders to fit.
    window_length (float, optional): The length of the spline fit filter window in units of wavelength. Default is 1.
    break_tolerance (float, optional): Split spline fit into segments at breaks longer than this value. Default is 0.5.
    plot_orders (list, optional): List of orders to plot. Default is None.
    use_matplotlib (bool, optional): If True, plots using matplotlib. If False, uses plotly. Default is False.
    time (bool, optional): If True, show a progress bar using tqdm. Default is False.

    Returns:
    dict: A dictionary containing splines for each order.
    """
    chip = chip.upper()

    # Input validation checks
    if plot_orders is not None and not isinstance(plot_orders, (list, np.ndarray)):
        raise ValueError("Please make plot_orders a list or an array (even if it's only one order)")

    if plot_orders is not None:
        valid_range = range(0, 35) if chip == 'GREEN' else range(0, 32)
        for order in plot_orders:
            if order not in valid_range:
                raise ValueError(f"Invalid order input: {order}. Order must be within the valid range for the {chip} chip: {valid_range.start}-{valid_range.stop-1}.")

    splines = {'sci1': {}, 'sci2': {}, 'sci3': {}}

    total_iterations = len(order_range) * len(fits_files)

    iterator = tqdm(total=total_iterations, desc="Fitting Spline to Many Orders") if time else range(total_iterations)

    count = 0
    for order in order_range:
        for fits_file in fits_files:
            plot = plot_orders is not None and order in plot_orders
            result = fit_spline_to_smooth_lamp(fits_file, order, chip, spline_method, window_length, break_tolerance, plot, use_matplotlib)
            if result is None:
                if time:
                    iterator.update(1)
                else:
                    count += 1
                continue
            spline_sci1, spline_sci2, spline_sci3 = result
            for sci, spline in zip(['sci1', 'sci2', 'sci3'], [spline_sci1, spline_sci2, spline_sci3]):
                if order not in splines[sci]:
                    splines[sci][order] = []
                splines[sci][order].append(spline)
            if time:
                iterator.update(1)
            else:
                count += 1

    if time:
        iterator.close()

    return splines

def divide_data_by_spline(wave_green, flux_green, spline_fit_smooth_lamp_green, wave_red, flux_red, spline_fit_smooth_lamp_red, order_range_green=range(35), order_range_red=range(32)):
    """
    Divides science data with a spline fit to a 1D smooth lamp file.
    Wrapper function to be used in remove_blaze. 

    Parameters:
    wave_green (numpy.ndarray): 2D array of wavelength data for the green chip.
    flux_green (numpy.ndarray): 2D array of flux data for the green chip.
    spline_fit_smooth_lamp_green (dict): Dictionary containing spline fit to smooth lamp for the green chip.
    wave_red (numpy.ndarray): 2D array of wavelength data for the red chip.
    flux_red (numpy.ndarray): 2D array of flux data for the red chip.
    spline_fit_smooth_lamp_red (dict): Dictionary containing spline fit to smooth lamp for the red chip.
    order_range_green (range): Range of spectral orders to run this function on for the green chip. Default is range(2, 35).
    order_range_red (range): Range of spectral orders to run this function on for the red chip. Default is range(32).

    Returns:
    divided_flux_green (list): List of normalized flux data for the green chip.
    divided_flux_red (list): List of normalized flux data for the red chip.
    """

    divided_flux_green = []
    divided_flux_red = []

    for order in order_range_green:
        flux = flux_green[order, :]
        spline_smooth_lamp = spline_fit_smooth_lamp_green[order][0]
        flux_norm = flux / (spline_smooth_lamp / np.median(spline_smooth_lamp))
        divided_flux_green.append(flux_norm)

    for order in order_range_red:
        flux = flux_red[order, :]
        spline_smooth_lamp = spline_fit_smooth_lamp_red[order][0]
        flux_norm = flux / (spline_smooth_lamp / np.median(spline_smooth_lamp))
        divided_flux_red.append(flux_norm)

    return divided_flux_green, divided_flux_red

def remove_negatives(file_data):
    """
    Remove negative values (usually just a slight oversubtraction issue) in the flux data by setting them to 0.
    Without doing this, you can run into issues w/ wotan fitting.
    
    Parameters:
    file_data (dict): Dictionary containing the FITS file data.

    Returns:
    dict: Updated data with negative flux values removed (set to 0).
    """
    for key in ['GREEN_SCI_FLUX1', 'GREEN_SCI_FLUX2', 'GREEN_SCI_FLUX3',
                'RED_SCI_FLUX1', 'RED_SCI_FLUX2', 'RED_SCI_FLUX3']:
        file_data[key] = np.maximum(file_data[key], 0)
    return file_data

def remove_blaze(directory, splines_green, splines_red):
    """
    Removes the blaze function by dividing by the spline fits to smooth lamp pattern.
    Saves both the divided fluxes and the normalized divided fluxes to FITS files.

    Parameters:
    directory (str): Directory containing the FITS files.
    splines_green (dict): Dictionary containing spline fits for the green chip.
    splines_red (dict): Dictionary containing spline fits for the red chip.

    Returns:
    None
    """
    og_data = read_kpf_data(directory)
    total_files = len(og_data)

    # progress bar b/c i like watching the bar fill up
    with tqdm(total=total_files, desc="Dividing original data by blaze function", unit="file", ncols=100) as pbar:
        for filename, file_data in og_data.items():
            input_file = os.path.join(directory, filename)

            with fits.open(input_file, mode='update') as hdul:
                initial_time = time.time()
                file_data = remove_negatives(file_data)  # Set negative values to 0

                # Divide data by spline fits
                divided_flux_green_sci1, divided_flux_red_sci1 = divide_data_by_spline(
                    file_data['GREEN_SCI_WAVE1'], file_data['GREEN_SCI_FLUX1'], splines_green['sci1'],
                    file_data['RED_SCI_WAVE1'], file_data['RED_SCI_FLUX1'], splines_red['sci1']
                )

                divided_flux_green_sci2, divided_flux_red_sci2 = divide_data_by_spline(
                    file_data['GREEN_SCI_WAVE2'], file_data['GREEN_SCI_FLUX2'], splines_green['sci2'],
                    file_data['RED_SCI_WAVE2'], file_data['RED_SCI_FLUX2'], splines_red['sci2']
                )

                divided_flux_green_sci3, divided_flux_red_sci3 = divide_data_by_spline(
                    file_data['GREEN_SCI_WAVE3'], file_data['GREEN_SCI_FLUX3'], splines_green['sci3'],
                    file_data['RED_SCI_WAVE3'], file_data['RED_SCI_FLUX3'], splines_red['sci3']
                )

                # Save original divided fluxes (before normalization)
                green_flux_divided_sci1 = divided_flux_green_sci1.copy()
                red_flux_divided_sci1 = divided_flux_red_sci1.copy()
                green_flux_divided_sci2 = divided_flux_green_sci2.copy()
                red_flux_divided_sci2 = divided_flux_red_sci2.copy()
                green_flux_divided_sci3 = divided_flux_green_sci3.copy()
                red_flux_divided_sci3 = divided_flux_red_sci3.copy()

                # Calculate and save median values
                green_medians_sci1 = [np.median(flux) for flux in divided_flux_green_sci1]
                red_medians_sci1 = [np.median(flux) for flux in divided_flux_red_sci1]
                green_medians_sci2 = [np.median(flux) for flux in divided_flux_green_sci2]
                red_medians_sci2 = [np.median(flux) for flux in divided_flux_red_sci2]
                green_medians_sci3 = [np.median(flux) for flux in divided_flux_green_sci3]
                red_medians_sci3 = [np.median(flux) for flux in divided_flux_red_sci3]

                # Normalize divided fluxes by dividing by their median
                divided_flux_green_sci1_norm = [flux / median for flux, median in zip(divided_flux_green_sci1, green_medians_sci1)]
                divided_flux_red_sci1_norm = [flux / median for flux, median in zip(divided_flux_red_sci1, red_medians_sci1)]
                divided_flux_green_sci2_norm = [flux / median for flux, median in zip(divided_flux_green_sci2, green_medians_sci2)]
                divided_flux_red_sci2_norm = [flux / median for flux, median in zip(divided_flux_red_sci2, red_medians_sci2)]
                divided_flux_green_sci3_norm = [flux / median for flux, median in zip(divided_flux_green_sci3, green_medians_sci3)]
                divided_flux_red_sci3_norm = [flux / median for flux, median in zip(divided_flux_red_sci3, red_medians_sci3)]

                for i, (divided_green_flux, divided_red_flux, divided_green_flux_norm, divided_red_flux_norm) in enumerate([
                    (green_flux_divided_sci1, red_flux_divided_sci1, divided_flux_green_sci1_norm, divided_flux_red_sci1_norm),
                    (green_flux_divided_sci2, red_flux_divided_sci2, divided_flux_green_sci2_norm, divided_flux_red_sci2_norm),
                    (green_flux_divided_sci3, red_flux_divided_sci3, divided_flux_green_sci3_norm, divided_flux_red_sci3_norm)
                ]):
                    # Save original divided green flux
                    hdu_name = f'SCI{i+1}_GREEN_FLUX_DIV'
                    data_array = np.array(divided_green_flux)
                    if hdu_name in hdul:
                        hdul[hdu_name].data = data_array
                    else:
                        hdul.append(fits.ImageHDU(data=data_array, name=hdu_name))

                    # Save normalized divided green flux
                    hdu_name = f'SCI{i+1}_GREEN_FLUX_DIV_NORM'
                    data_array = np.array(divided_green_flux_norm)
                    if hdu_name in hdul:
                        hdul[hdu_name].data = data_array
                    else:
                        hdul.append(fits.ImageHDU(data=data_array, name=hdu_name))

                    # Save original divided red flux
                    hdu_name = f'SCI{i+1}_RED_FLUX_DIV'
                    data_array = np.array(divided_red_flux)
                    if hdu_name in hdul:
                        hdul[hdu_name].data = data_array
                    else:
                        hdul.append(fits.ImageHDU(data=data_array, name=hdu_name))

                    # Save normalized divided red flux
                    hdu_name = f'SCI{i+1}_RED_FLUX_DIV_NORM'
                    data_array = np.array(divided_red_flux_norm)
                    if hdu_name in hdul:
                        hdul[hdu_name].data = data_array
                    else:
                        hdul.append(fits.ImageHDU(data=data_array, name=hdu_name))

                hdul.flush()

                pbar.update(1)
                elapsed_time = time.time() - initial_time
                pbar.set_postfix({'Elapsed Time': f'{elapsed_time:.2f} sec'})

                hdul.flush()

        print("Data has been divided by the blaze function, normalized, and saved in the directory:", directory)

def continuum_norm(directory, order_range_green=(0, 35), order_range_red=(0, 32),
                   window=5, n_iter=5, ffrac=0.9875, specific_order_windows=None,
                   skip_orders_green=None, skip_orders_red=None, plot=True):
    """
    Continuum normalize the science spectra in FITS files for each fiber after the blaze has been removed.
    The function first creates a "master" median spectrum for each order by combining the science data 
    from all FITS files in the specified directory for both the green and red chips. It then fits a spline 
    to the median flux in each order using an iterative process that filters out non-continuum points based 
    on a specified flux threshold. The final spline is used to divide all of the original spectra, providing a 
    set of continuum-normalized spectra for the science data. Optionally, it also plots the median flux along 
    with its corresponding spline fit for visual inspection (which I like to leave on b/c it's a good check
    to make sure the fitting is working properly)

    Parameters:
    - directory (str): Path to the directory containing input FITS files.
    - order_range_green (tuple of int, optional): Range of orders to process for the green chip, inclusive.
      Defaults to (0, 35).
    - order_range_red (tuple of int, optional): Range of orders to process for the red chip, inclusive.
      Defaults to (0, 32).
    - window (int, optional): Spline fitting window size in Angstroms. Defaults to 5.
    - n_iter (int, optional): Number of iterations for spline refitting with flux filtering. Defaults to 5.
    - ffrac (float, optional): Fractional threshold for selecting continuum points in flux filtering.
      Defaults to 0.9875.
    - specific_order_windows (dict, optional): Custom spline window sizes for specific orders, specified as
      {(chip, order): window_size}. Defaults to None.
    - skip_orders_green, skip_orders_red (iterable of int, optional): Orders to *skip* continuum
      normalization on the GREEN/RED chips. Defaults to None (no skipping).
    - plot (bool, optional): If True, plots median flux and spline fit for each order. Defaults to True.

    Returns:
    - None: All processed FITS files are saved in directory with continuum-normalized flux data appended.
    """

    # convert skips to sets for quick lookup
    skip_orders_green = set(skip_orders_green or [])
    skip_orders_red = set(skip_orders_red or [])

    def median_flux_for_fiber(chip, fiber, order_range):
        print(f"Reading in all spectra and calculating Master median flux spectrum for {chip}_{fiber}...")
        file_paths = glob(os.path.join(directory, "*.fits"))
        median_fluxes, median_wavelengths = [], []

        for order in range(order_range[0], order_range[1]):
            all_fluxes, all_wavelengths = [], []
            for file_path in file_paths:
                with fits.open(file_path) as hdul:
                    wave_key = f'{chip}_SCI_WAVE{fiber[-1]}'
                    flux_key = f'{fiber}_{chip}_FLUX_DIV_NORM'

                    wavelength = hdul[wave_key].data[order]
                    flux = hdul[flux_key].data[order]

                    sorted_indices = np.argsort(wavelength)
                    all_wavelengths.append(wavelength[sorted_indices])
                    all_fluxes.append(flux[sorted_indices])

            median_wavelength = np.median(np.array(all_wavelengths), axis=0)
            median_flux = np.median(np.array(all_fluxes), axis=0)
            median_fluxes.append(median_flux)
            median_wavelengths.append(median_wavelength)

        return median_wavelengths, median_fluxes

    def fit_spline(x, y, window_size):
        num_knots = int((np.max(x) - np.min(x)) / window_size)
        num_knots = max(num_knots, 4)
        breakpoints = np.linspace(np.min(x), np.max(x), num_knots)
        return LSQUnivariateSpline(x, y, breakpoints[1:-1])

    def fit_splines_for_chip(median_wavelengths, median_fluxes, chip):
        splines = []
        for idx, (wavelength, flux) in enumerate(zip(median_wavelengths, median_fluxes)):
            sorted_indices = np.argsort(wavelength)
            wavelength, flux = wavelength[sorted_indices], flux[sorted_indices]

            order_key = (chip, idx)
            order_window = specific_order_windows.get(order_key, window) if specific_order_windows else window
            ss = fit_spline(wavelength, flux, order_window)

            yfit = ss(wavelength)
            for _ in range(n_iter):
                normspec = flux / yfit
                continuum_points = (normspec >= ffrac) & (yfit > 0)
                if np.sum(continuum_points) < 4:
                    break
                ss = fit_spline(wavelength[continuum_points], flux[continuum_points], order_window)
                yfit = ss(wavelength)
            splines.append(ss)
        return splines

    def renormalize(flux_array, percentile=95):
        for i in range(flux_array.shape[0]):
            flux = flux_array[i, :]
            threshold = np.percentile(flux, percentile)
            flux_array[i, :] = flux / threshold
        return flux_array

    fibers = ['SCI1', 'SCI2', 'SCI3']
    for fiber in fibers:
        median_wavelengths_green, median_fluxes_green = median_flux_for_fiber('GREEN', fiber, order_range_green)
        median_wavelengths_red, median_fluxes_red = median_flux_for_fiber('RED', fiber, order_range_red)
        splines_green = fit_splines_for_chip(median_wavelengths_green, median_fluxes_green, "GREEN")
        splines_red = fit_splines_for_chip(median_wavelengths_red, median_fluxes_red, "RED")

        file_paths = glob(os.path.join(directory, "*.fits"))
        with tqdm(total=len(file_paths), desc=f"Continuum Normalizing Individual Spectra for {fiber}") as pbar:
            for file_path in file_paths:
                with fits.open(file_path, mode='update') as hdul:
                    green_norm_data = np.zeros((order_range_green[1] - order_range_green[0], 4080))
                    red_norm_data = np.zeros((order_range_red[1] - order_range_red[0], 4080))

                    for idx, order in enumerate(range(order_range_green[0], order_range_green[1])):
                        wave_key = f'GREEN_SCI_WAVE{fiber[-1]}'
                        flux_key = f'{fiber}_GREEN_FLUX_DIV_NORM'
                        wavelength = hdul[wave_key].data[order]
                        flux = hdul[flux_key].data[order]

                        sorted_indices = np.argsort(wavelength)
                        wavelength, flux = wavelength[sorted_indices], flux[sorted_indices]

                        if order in skip_orders_green:
                            green_norm_data[idx, :] = flux[::-1]
                        else:
                            yfit = splines_green[idx](wavelength)
                            green_norm_data[idx, :] = (flux / yfit)[::-1]
                    
                    green_norm_data = renormalize(green_norm_data, percentile=95)
                    hdu_name = f'{fiber}_GREEN_FLUX_CONT_NORM'
                    if hdu_name in hdul:
                        hdul[hdu_name].data = green_norm_data
                    else:
                        hdul.append(fits.ImageHDU(data=green_norm_data, name=hdu_name))

                    for idx, order in enumerate(range(order_range_red[0], order_range_red[1])):
                        wave_key = f'RED_SCI_WAVE{fiber[-1]}'
                        flux_key = f'{fiber}_RED_FLUX_DIV_NORM'
                        wavelength = hdul[wave_key].data[order]
                        flux = hdul[flux_key].data[order]

                        sorted_indices = np.argsort(wavelength)
                        wavelength, flux = wavelength[sorted_indices], flux[sorted_indices]

                        if order in skip_orders_red:
                            red_norm_data[idx, :] = flux[::-1]
                        else:
                            yfit = splines_red[idx](wavelength)
                            red_norm_data[idx, :] = (flux / yfit)[::-1]
                    red_norm_data = renormalize(red_norm_data, percentile=95)

                    hdu_name = f'{fiber}_RED_FLUX_CONT_NORM'
                    if hdu_name in hdul:
                        hdul[hdu_name].data = red_norm_data
                    else:
                        hdul.append(fits.ImageHDU(data=red_norm_data, name=hdu_name))

                    hdul.flush()
                    pbar.update(1)

        if plot:
            plot_spline_fit_to_median(median_wavelengths_green, median_fluxes_green, splines_green, "GREEN", fiber)
            plot_spline_fit_to_median(median_wavelengths_red, median_fluxes_red, splines_red, "RED", fiber)

    print("All files have been normalized and saved in:", directory)

def fill_flux_gaps(wave_data, flux_data, log_gap_threshold=np.log10(1.06)):
    """
    Finds gaps between the max wavelength of the ith order and the min wavelength of the (i+1)th order
    that exceed a specified threshold (log10(1.06) by default), and adds wavelengths and flux values
    in those gaps, setting flux values to zeros for red and green chips separately (although this
    should only be an issue for the red chip where the FSR is larger).

    Parameters:
    wave_data (list): List containing two elements, each a list of 1D arrays of wavelength values
                      (one for red and one for green).
    flux_data (list): List containing two elements, each a list of 1D arrays of flux values
                      (one for red and one for green).
    log_gap_threshold (float, optional): Logarithmic gap threshold to define a gap. Default is log10(1.06).

    Returns:
    tuple: Updated wavelength arrays and flux arrays for red and green chips with gaps filled and flux values in those gaps set to zeros.
    """
    updated_wave_data = []
    updated_flux_data = []

    for chip_index in range(2):  # 0 for red, 1 for green
        chip_wave_data = wave_data[chip_index]
        chip_flux_data = flux_data[chip_index]

        updated_wave_orders = []
        updated_flux_orders = []

        for order in range(len(chip_wave_data)):
            current_wave = chip_wave_data[order]
            current_flux = chip_flux_data[order]

            updated_wave = list(current_wave)
            updated_flux = list(current_flux)

            if order < len(chip_wave_data) - 1:
                max_wave = np.max(current_wave)
                min_wave_next = np.min(chip_wave_data[order + 1])

                wave_gap = min_wave_next - max_wave
                if np.log10(wave_gap) > log_gap_threshold:
                    gap_wavelengths = np.linspace(max_wave, min_wave_next, num=10, endpoint=False)[1:]  # Exclude max_wave
                    gap_fluxes = np.full_like(gap_wavelengths, 0)

                    updated_wave.extend(gap_wavelengths)
                    updated_flux.extend(gap_fluxes)

            updated_wave_orders.append(np.array(updated_wave))
            updated_flux_orders.append(np.array(updated_flux))

        updated_wave_data.append(updated_wave_orders)
        updated_flux_data.append(updated_flux_orders)

    return updated_wave_data, updated_flux_data

def interpolate_flux(directory, step=np.log(1.06), exclusions=None):
    """
    Interpolates the flux on a new wavelength grid for the green and red chips separately for each FITS file in the given directory,
    with the option to exclude specific wavelength regions.

    Parameters:
    directory (str): Directory containing the FITS files.
    step (float): Step size for the wavelength grid (logarithmic scale). Default is log(1.06).
    exclusions (dict, optional): Dictionary specifying wavelength exclusions for certain orders and chips.
                                 Format: {('GREEN', 'SCI1', 0): (4506, 'max')}
                                 The values can be specific numbers or 'min'/'max' for range boundaries.

    Returns:
    None: The interpolated wavelength and flux data are saved back into the FITS files as new HDUs.
    """
    file_paths = glob(os.path.join(directory, "*.fits"))

    with tqdm(total=len(file_paths), desc="Interpolating Flux for Files") as pbar:
        for file_path in file_paths:
            with fits.open(file_path, mode='update') as hdul:
                fibers = ['SCI1', 'SCI2', 'SCI3']
                wave_grid = None

                for fiber in fibers:
                    wave_keys = [f'RED_SCI_WAVE{fiber[-1]}', f'GREEN_SCI_WAVE{fiber[-1]}']
                    flux_keys = [f'{fiber}_RED_FLUX_CONT_NORM', f'{fiber}_GREEN_FLUX_CONT_NORM']

                    waves = [hdul[wave_key].data for wave_key in wave_keys if wave_key in hdul]
                    fluxes = [hdul[flux_key].data for flux_key in flux_keys if flux_key in hdul]

                    # Fill gaps between orders
                    waves, fluxes = fill_flux_gaps(waves, fluxes)

                    if wave_grid is None:
                        all_waves_flat = np.hstack([np.hstack(wave) for wave in waves])
                        wave_grid = np.arange(np.min(all_waves_flat), np.max(all_waves_flat), step)

                    flux_interpolated = []
                    for wave, flux, wave_key in zip(waves, fluxes, wave_keys):
                        chip = 'GREEN' if 'GREEN' in wave_key else 'RED'
                        for order_index, (order_wave, order_flux) in enumerate(zip(wave, flux)):
                            if exclusions:
                                exclusion_key = (chip, fiber, order_index)
                                if exclusion_key in exclusions:
                                    min_excl, max_excl = exclusions[exclusion_key]
                                    min_excl = np.min(order_wave) if min_excl == 'min' else min_excl
                                    max_excl = np.max(order_wave) if max_excl == 'max' else max_excl
                                    mask = (order_wave < min_excl) | (order_wave > max_excl)
                                    order_wave = order_wave[mask]
                                    order_flux = order_flux[mask]

                            interp_func = inter.interp1d(order_wave, order_flux, fill_value=np.nan, bounds_error=False)
                            interpolated_flux = interp_func(wave_grid)
                            flux_interpolated.append(interpolated_flux)

                    flux_combined = np.nanmean(flux_interpolated, axis=0)
                    nan_mask = ~np.isnan(flux_combined)
                    flux_combined = flux_combined[nan_mask]
                    wave_grid_masked = wave_grid[nan_mask]

                    hdu_wave_name = f'WAVE_INTERP_{fiber}'
                    hdu_flux_name = f'FLUX_INTERP_{fiber}'

                    if hdu_wave_name in hdul:
                        hdul[hdu_wave_name].data = wave_grid_masked
                    else:
                        hdul.append(fits.ImageHDU(data=wave_grid_masked, name=hdu_wave_name))

                    if hdu_flux_name in hdul:
                        hdul[hdu_flux_name].data = flux_combined
                    else:
                        hdul.append(fits.ImageHDU(data=flux_combined, name=hdu_flux_name))

                hdul.flush()
            pbar.update(1)

    print("Interpolation completed and data saved to FITS files in the directory:", directory)
