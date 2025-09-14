from astropy.io import fits, ascii
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scipy.interpolate as inter
from astropy.table import Table
from scipy.ndimage import gaussian_filter1d
import glob
from astropy import time, coordinates as coord, units as u
import os
from pathlib import Path


class ExoplanetSystem:
    """Container for exoplanet system parameters"""
    
    def __init__(self, name, t0, period, kp, k_star, vsys, stellar_radius, 
                 coordinates, observatory, transit_phase_half_width):
        """
        Initialize exoplanet system parameters
        
        Parameters
        ----------
        name : str
            System name
        t0 : float
            Transit ephemeris in BJD_TDB
        period : float
            Orbital period in days
        kp : float
            Planet semi-amplitude in km/s
        k_star : float
            Stellar semi-amplitude in km/s
        vsys : float
            System velocity in km/s
        stellar_radius : float
            Stellar radius in solar radii
        coordinates : astropy.coordinates.SkyCoord
            Star coordinates
        observatory : astropy.coordinates.EarthLocation
            Observatory location
        transit_phase_half_width : float
            Half-width of transit in phase units
        """
        self.name = name
        self.t0 = t0
        self.period = period
        self.kp = kp
        self.k_star = k_star
        self.vsys = vsys
        self.stellar_radius = stellar_radius
        self.coordinates = coordinates
        self.observatory = observatory
        self.transit_phase_half_width = transit_phase_half_width


class Atmosphere:
    def __init__(self, system, file_pattern="latestDRP/*L1.fits"):
        """
        Initialize the analyzer
        
        Parameters
        ----------
        system : ExoplanetSystem
            System parameters
        file_pattern : str
            Glob pattern for data files
        """
        self.system = system
        self.file_pattern = file_pattern
        self.data_table = None
        self.clean_wave_grid = None
        self.clean_flux_grid = None
        self.transit_indices = None
    
    def get_species_label(self, template_name):
        """
        Convert template name to proper species label
        
        Parameters
        ----------
        template_name : str
            Template filename or identifier
            
        Returns
        -------
        label : str
            Formatted species label
        """
        # Remove common suffixes and prefixes
        clean_name = template_name.replace('_custom', '')
        clean_name = clean_name.replace('.dat', '')
        
        template_lower = clean_name.lower()
        
        species_map = {
            'na_lor_cut': 'Na I', 'na_allard': 'Na I', 
            'na_allard_new': 'Na I', 'na_burrows': 'Na I', 'na': 'Na I',
            'ca+': 'Ca II', 'ca': 'Ca II', 
            'fe+': 'Fe+', 'fe': 'Fe I',
            'mg+': 'Mg+', 'mg': 'Mg I',
            'k': 'K I',
            'ti': 'Ti I',
            'v': 'V I',
            'si': 'Si I',
            'tio_48_exomol_mckemmish': 'TiO',
            'vo': 'VO',
            'cah': 'CaH',
            'feh_main_iso': 'FeH'
        }
        
        if template_lower in species_map:
            return species_map[template_lower]
        
        for key, label in species_map.items():
            if key in template_lower:
                return label
        
        return clean_name.replace('_', ' ').title()
    
    def calculate_orbital_phase(self, t, t0=None, period=None):
        """
        Calculate orbital phase from observation time
        
        Parameters
        ----------
        t : float or array
            Observation time in BJD
        t0 : float, optional
            Transit ephemeris (uses system default if None)
        period : float, optional
            Orbital period (uses system default if None)
            
        Returns
        -------
        phases : float or array
            Orbital phases
        """
        if t0 is None:
            t0 = self.system.t0
        if period is None:
            period = self.system.period
            
        if np.isscalar(t):
            phase = (t - t0) / period
            if phase > 1 or phase < -1:
                phase = np.modf(phase)[0]
            if phase > 0.8:
                phase = phase - 1
            if phase < -0.8:
                phase = phase + 1
            return phase
        else:
            phases = []
            for time_val in t:
                phase = (time_val - t0) / period
                if phase > 1 or phase < -1:
                    phase = np.modf(phase)[0]
                if phase > 0.8:
                    phase = phase - 1
                if phase < -0.8:
                    phase = phase + 1
                phases.append(phase)
            return np.array(phases)
    
    def calculate_planet_rv(self, kp, phase):
        """
        Calculate planet radial velocity at given phase
        
        Parameters
        ----------
        kp : float
            Planet semi-amplitude in km/s
        phase : float or array
            Orbital phase
            
        Returns
        -------
        rv : float or array
            Planet radial velocity in km/s
        """
        return kp * np.sin(2*np.pi * phase)
    
    def remove_vertical_outliers(self, flux_array, clip_sigma=3):
        """
        Remove vertical outliers using sigma clipping
        
        Parameters
        ----------
        flux_array : ndarray
            2D flux array (n_obs x n_wavelengths)
        clip_sigma : float
            Sigma threshold for clipping
            
        Returns
        -------
        clipped_flux : ndarray
            Flux array with outliers replaced
        """
        clipped_flux = flux_array.copy()
        
        for i in range(flux_array.shape[1]):
            column_flux = flux_array[:, i]
            median_flux = np.nanmedian(column_flux)
            std_flux = np.nanstd(column_flux)
            outlier_mask = np.abs(column_flux - median_flux) > clip_sigma * std_flux
            
            if np.any(outlier_mask):
                clipped_flux[outlier_mask, i] = np.nan
        
        # Replace NaNs with column means
        for i in range(clipped_flux.shape[1]):
            nan_mask = np.isnan(clipped_flux[:, i])
            if np.any(nan_mask):
                clipped_flux[nan_mask, i] = np.nanmean(clipped_flux[:, i])
        
        return clipped_flux
    
    def apply_pca_cleaning(self, flux_array, n_components=5):
        """
        Apply PCA cleaning to remove systematic effects
        
        Parameters
        ----------
        flux_array : ndarray
            2D flux array
        n_components : int
            Number of PCA components to remove
            
        Returns
        -------
        cleaned_flux : ndarray
            PCA-cleaned flux array
        """
        # Singular value decomposition
        u, s, vt = np.linalg.svd(flux_array, full_matrices=False)
        v = vt.T
        
        # Remove first n components
        s_new = s.copy()
        s_new[:n_components] = 0.0
        
        # Calculate cleaned flux
        cleaned_flux = np.dot(u, np.dot(np.diag(s_new), v.T)) + 1.0
        
        return cleaned_flux
    
    def clean_spectra(self, waves, fluxes, transit_indices, do_pca=True, 
                     pca_components=3, plots=False):
        """
        Complete spectral cleaning pipeline
        
        Parameters
        ----------
        waves : ndarray
            2D wavelength array
        fluxes : ndarray
            2D flux array
        transit_indices : list
            [start, end] indices for transit
        do_pca : bool
            Whether to apply PCA cleaning
        pca_components : int
            Number of PCA components to remove
        plots : bool
            Whether to generate diagnostic plots
            
        Returns
        -------
        wave_grid : ndarray
            Common wavelength grid
        cleaned_flux : ndarray
            Cleaned flux array
        """
        print("Starting spectral cleaning pipeline...")
        
        # Step 1: Remove low flux regions (order gaps)
        avg_flux = np.nanmean(fluxes, axis=0)
        low_flux_mask = avg_flux > 0.05
        waves_clipped = waves[:, low_flux_mask]
        fluxes_clipped = fluxes[:, low_flux_mask]
        
        # Step 2: Normalize spectra
        norm_fluxes = fluxes_clipped / np.nanmean(fluxes_clipped, axis=1, keepdims=True)
        
        # Step 3: Sigma clipping
        sigma_clipped = self.remove_vertical_outliers(norm_fluxes, 3)
        
        # Step 4: Remove broadband variations
        blaze_removed = np.zeros_like(norm_fluxes)
        out_of_transit = np.concatenate((sigma_clipped[:transit_indices[0]], 
                                       sigma_clipped[transit_indices[1]:]), axis=0)
        mean_spec = np.nanmean(out_of_transit, axis=0)
        avg_blaze = gaussian_filter1d(mean_spec, 200)
        cont_rem_spec = norm_fluxes / avg_blaze
        
        for i in range(len(norm_fluxes)):
            blaze = gaussian_filter1d(cont_rem_spec[i], 200)
            blaze_removed[i] = sigma_clipped[i] / blaze
        
        # Step 5: Interpolate onto common grid
        wave_grid = np.nanmean(waves_clipped, axis=0)
        wave_grid = wave_grid[20:-20]  # Avoid extrapolation at edges
        interp_fluxes = np.ones((len(waves_clipped), len(wave_grid)))
        
        for i in range(len(waves_clipped)):
            func = inter.CubicSpline(waves_clipped[i], blaze_removed[i])
            interp_fluxes[i] = func(wave_grid)
        
        # Step 6: PCA (optional)
        if do_pca:
            pca_cleaned = self.apply_pca_cleaning(interp_fluxes, pca_components)
        else:
            pca_cleaned = interp_fluxes
        
        # Step 7: Remove high-variance columns
        std_cols = np.nanstd(pca_cleaned, axis=0)
        mask = std_cols < 1.3 * np.nanmean(std_cols)
        wave_grid_final = wave_grid[mask]
        cleaned_flux = pca_cleaned[:, mask]
        
        # Step 8: Final sigma clipping
        final_cleaned = self.remove_vertical_outliers(cleaned_flux, 3)
        
        print(f"Cleaning complete. Final grid: {len(wave_grid_final)} wavelength points")
        
        return wave_grid_final, final_cleaned
    
    def cross_correlate(self, waves, fluxes, model_wave, model_flux, transit_indices):
        """
        Cross-correlate spectra with atmospheric model
        
        Parameters
        ----------
        waves : ndarray
            Wavelength grid
        fluxes : ndarray
            2D flux array
        model_wave : ndarray
            Model wavelength array
        model_flux : ndarray
            Model flux array
        transit_indices : list
            Transit boundary indices
            
        Returns
        -------
        rv_grid : ndarray
            Radial velocity grid
        cc_grid : list
            Cross-correlation results for each spectrum
        """
        clip_start = waves[0] - 10
        clip_end = waves[-1] + 10
        model_mask = (model_wave > clip_start) & (model_wave < clip_end)
        model_wave_clip = model_wave[model_mask]
        model_flux_clip = model_flux[model_mask]
        
        poly_coeffs = np.polyfit(model_wave_clip, model_flux_clip, 3)
        continuum = np.poly1d(poly_coeffs)(model_wave_clip)
        model_flux_norm = model_flux_clip / continuum
        
        rv_grid = np.arange(-300, 300, 0.5)
        model_flux_grid = []
        
        for rv in rv_grid:
            model_wave_shift = model_wave_clip / (1 - rv / (2.998e5))
            interp_func = inter.CubicSpline(model_wave_shift, model_flux_norm)
            interp_model = interp_func(waves)
            model_flux_grid.append(interp_model / np.sum(interp_model))
        
        cc_grid = []
        for i, flux in enumerate(fluxes):
            cc_row = []
            for j, model_shifted in enumerate(model_flux_grid):
                cc = np.sum(flux * model_shifted)
                cc_row.append(cc)
            cc_grid.append(cc_row)
        
        cc_grid = np.array(cc_grid)
        
        out_of_transit = np.concatenate((cc_grid[:transit_indices[0]], 
                                       cc_grid[transit_indices[1]:]), axis=0)
        mean_cc = np.nanmean(out_of_transit, axis=0)
        
        cc_grid_norm = []
        for cc_row in cc_grid:
            cc_corr = cc_row / mean_cc
            smoothed = gaussian_filter1d(cc_corr, 140)
            residual = cc_corr - smoothed
            cc_grid_norm.append(residual)
        
        print("Cross-correlation complete")
        return rv_grid, cc_grid_norm
    
    def shift_ccf_to_planet_frame(self, rv_grid, cc_grid, planet_rvs):
        """
        Shift CCF to planet rest frame
        """
        new_vel_grid = rv_grid.copy()
        cc_grid_planet_frame = np.zeros_like(cc_grid)

        for i in range(cc_grid.shape[0]):
            shift = planet_rvs[i]
            shifted_rv = rv_grid - shift
            f = inter.interp1d(shifted_rv, cc_grid[i], kind='cubic',
                         bounds_error=False, fill_value=np.nan)
            cc_grid_planet_frame[i] = f(new_vel_grid)

        return new_vel_grid, cc_grid_planet_frame
    
    def find_best_kp_vsys(self, rv_grid, cc_grid, phases, template_name='unknown'):
        """
        Find best Kp and Vsys values using SNR optimization
        
        Parameters
        ----------
        rv_grid : ndarray
            Radial velocity grid
        cc_grid : ndarray
            Cross-correlation grid
        phases : ndarray
            Orbital phases
        template_name : str
            Template identifier for output files
            
        Returns
        -------
        snr_grid : ndarray
            SNR grid
        vsys_grid : ndarray
            System velocity grid
        kp_grid : ndarray
            Planet semi-amplitude grid
        """
        print(f"Making Kp-Vsys plot for {template_name}...")
        
        # Define search grids exactly like original
        kp_grid = np.arange(self.system.kp + 70, self.system.kp - 70, -0.5)
        vsys_grid = np.arange(-70, 70, 0.25)
        shift_grid = np.arange(-100, 100, 1.0)
        
        snr_grid = np.zeros((len(kp_grid), len(vsys_grid)))
        
        for i, kp in enumerate(kp_grid):
            for j, vsys in enumerate(vsys_grid):
                # Shift CCFs to planet rest frame
                shifted_ccs = []
                for k, phase in enumerate(phases):
                    planet_rv = self.calculate_planet_rv(kp, phase) + vsys
                    interp_func = inter.interp1d(rv_grid - planet_rv, cc_grid[k])
                    shifted_cc = interp_func(shift_grid)
                    shifted_ccs.append(shifted_cc)
                
                # Combine and calculate SNR
                combined_cc = np.nanmean(shifted_ccs, axis=0)
                signal = combined_cc[np.where(shift_grid == 0)][0]
                
                # Noise calculation exactly like original
                inds_low = np.where(shift_grid < -54)[0]
                inds_up = np.where(shift_grid > 49)[0]
                nopeak = np.concatenate((combined_cc[inds_low], combined_cc[inds_up]))
                noise_std = np.std(nopeak)
                noise_mean = np.mean(nopeak)
                
                snr = (signal - noise_mean) / noise_std
                snr_grid[i, j] = snr
        
        snr_grid *= -1
        
        max_idx = np.unravel_index(np.argmax(snr_grid), snr_grid.shape)
        best_kp = kp_grid[max_idx[0]]
        best_vsys = vsys_grid[max_idx[1]]
        max_snr = snr_grid[max_idx]
        
        print(f"Best Kp: {best_kp:.2f} km/s")
        print(f"Best Vsys: {best_vsys:.2f} km/s")
        print(f"Maximum SNR: {max_snr:.2f}")
        
        self._plot_kp_vsys_map(snr_grid, vsys_grid, kp_grid, template_name)
        
        return snr_grid, vsys_grid, kp_grid
    
    def load_data(self):
        """
        Load and process observational data
        
        Returns
        -------
        data_table : astropy.table.Table
            Processed data table
        """
        print(f"Loading data from pattern: {self.file_pattern}")
        file_list = glob.glob(self.file_pattern)
        
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {self.file_pattern}")
        
        print(f"Found {len(file_list)} files")
        
        combined_fluxes = []
        rest_waves = []
        phases = []
        planet_rvs = []
        barycentric_corrections = []
        bjds = []
        stellar_rvs = []
        
        for file_path in file_list:
            with fits.open(file_path, memmap=False) as hdul:
                # Read science fibers
                wave_1 = hdul['WAVE_INTERP_SCI1'].data
                flux_1 = hdul['FLUX_INTERP_SCI1'].data
                wave_2 = hdul['WAVE_INTERP_SCI2'].data
                flux_2 = hdul['FLUX_INTERP_SCI2'].data
                wave_3 = hdul['WAVE_INTERP_SCI3'].data
                flux_3 = hdul['FLUX_INTERP_SCI3'].data
                
                # Combine fibers
                interp_func1 = inter.CubicSpline(wave_1, flux_1)
                interp_func2 = inter.CubicSpline(wave_2, flux_2)
                combined_flux = np.nanmean([flux_3, interp_func1(wave_3), 
                                          interp_func2(wave_3)], axis=0)
                
                # Time and velocity corrections
                obs_time = hdul[0].header['DATE-MID']
                t_astropy = time.Time(obs_time, format='isot', scale='utc', 
                                    location=self.system.observatory)
                ltt_bary = t_astropy.light_travel_time(self.system.coordinates)
                bjd = (t_astropy.tdb + ltt_bary).jd
                
                vbar_corr = self.system.coordinates.radial_velocity_correction(obstime=t_astropy)
                vbar = vbar_corr.to(u.km/u.s).value
                
                phase = self.calculate_orbital_phase(bjd)
                stellar_rv = self.calculate_planet_rv(self.system.k_star, phase)
                
                # Apply velocity corrections to wavelengths
                rest_wave = wave_3 / (1 + (-vbar + self.system.vsys - stellar_rv) / (2.99792e5))
                
                combined_fluxes.append(combined_flux)
                rest_waves.append(rest_wave)
                phases.append(phase)
                planet_rvs.append(self.calculate_planet_rv(self.system.kp, phase))
                barycentric_corrections.append(vbar)
                bjds.append(bjd)
                stellar_rvs.append(stellar_rv)
        
        self.data_table = Table({
            'flux': combined_fluxes,
            'waves': rest_waves,
            'phases': phases,
            'planet_rvs': planet_rvs,
            'barycentric_corrections': barycentric_corrections,
            'bjds': bjds,
            'stellar_rvs': stellar_rvs
        })
        
        self.data_table.sort('bjds')
        self.data_table.reverse()
        
        transit_half_width = self.system.transit_phase_half_width
        if self.data_table['phases'][0] > transit_half_width:
            transit_start = np.where(self.data_table['phases'] < transit_half_width)[0][0]
        else:
            transit_start = 0
            
        if self.data_table['phases'][-1] < -transit_half_width:
            transit_end = np.where(self.data_table['phases'] < -transit_half_width)[0][0]
        else:
            transit_end = len(self.data_table)
        
        self.transit_indices = [transit_start, transit_end]
        
        print(f"Data loaded successfully:")
        print(f"  Number of observations: {len(self.data_table)}")
        print(f"  Phase range: {np.min(self.data_table['phases']):.4f} to {np.max(self.data_table['phases']):.4f}")
        print(f"  Transit indices: {self.transit_indices}")
        
        return self.data_table
    
    def _plot_kp_vsys_map(self, snr_grid, vsys_grid, kp_grid, template_name):
        """Generate Kp-Vsys SNR map"""
        fig, ax = plt.subplots(figsize=(7, 6))
        
        im = ax.imshow(snr_grid, cmap=cm.inferno, 
                      extent=[vsys_grid[0], vsys_grid[-1], kp_grid[-1], kp_grid[0]],
                      aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('$v_{sys}$ (km s$^{-1}$)', fontsize=14)
        ax.set_ylabel('$K_p$ (km s$^{-1}$)', fontsize=14)
        
        cb = fig.colorbar(im)
        cb.set_label('SNR', rotation=90, fontsize=14)
        cb.ax.tick_params(labelsize=14)
        cb.ax.get_yaxis().labelpad = 4
        
        ax.plot([vsys_grid[0], vsys_grid[-1]], [self.system.kp, self.system.kp], 'k--')
        ax.plot([0, 0], [kp_grid[0], kp_grid[-1]], 'k--')
        
        species_label = self.get_species_label(template_name)
        ax.text(0.05, 0.95, species_label, 
                transform=ax.transAxes,
                fontsize=24,
                fontweight='bold',
                color='white',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(f'{template_name}_Kp-vsys.png', bbox_inches='tight')
        plt.savefig(f'{template_name}_Kp-vsys.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {template_name}_Kp-vsys.pdf")
        print(f"Saved: {template_name}_Kp-vsys.png")
        plt.show()
    
    def _plot_ccf_map(self, rv_grid, cc_grid, template_name, output_path):
        """Generate 2D CCF map"""
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        
        im = ax.imshow(-cc_grid, cmap=cm.gray_r, 
                      extent=[np.min(rv_grid), np.max(rv_grid), 
                             self.data_table['phases'][-1], self.data_table['phases'][0]])
        
        ax.set_xlabel('Radial Velocity (km s$^{-1}$)', fontsize=14)
        ax.set_ylabel('Orbital Phase', fontsize=14)
        
        # Transit boundaries
        t1, t2 = self.transit_indices
        ax.plot([-200, 200], [self.data_table['phases'][t1], self.data_table['phases'][t1]], 
                color='cyan', linestyle='--', linewidth=3)
        ax.plot([-200, 200], [self.data_table['phases'][t2], self.data_table['phases'][t2]], 
                color='cyan', linestyle='--', linewidth=3)
        
        ax.plot(self.data_table['planet_rvs'][t2:], self.data_table['phases'][t2:], 
                color='lime', linestyle=':', linewidth=6)
        ax.plot(self.data_table['planet_rvs'][:t1], self.data_table['phases'][:t1], 
                color='lime', linestyle=':', linewidth=6)
        
        ax.set_xlim(-150, 150)
        ax.set_aspect(1100)
        
        cb = fig.colorbar(im)
        cb.set_label('CCF Amplitude (ppm)', rotation=270, fontsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.ax.get_yaxis().labelpad = 15
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(f'{template_name}_SRF.png')
        print(f"Saved: {template_name}_SRF.png")
        plt.show()
    
    def _plot_planet_rest_frame(self, rv_grid, cc_grid, template_name, output_path):
        """Generate planet rest frame plot"""
        # Shift to planet rest frame
        new_vel, cc_planet = self.shift_ccf_to_planet_frame(rv_grid, cc_grid, self.data_table['planet_rvs'])
        
        # Sort by phase
        phases_sorted = np.sort(self.data_table['phases'])
        cc_planet_sorted = cc_planet[np.argsort(self.data_table['phases']), :]
        cc_planet_sorted *= -1
        
        extent = [new_vel[0], new_vel[-1], phases_sorted[0], phases_sorted[-1]]
        
        species_label = self.get_species_label(template_name)
        
        color_schemes = [('gray', 'Gray'), ('gray_r', 'Gray Inverted')]
        
        for cmap_name, cmap_label in color_schemes:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            
            im = ax.imshow(cc_planet_sorted,
                           cmap=cmap_name,
                           origin='lower',
                           interpolation='None',
                           rasterized=True,
                           extent=extent,
                           aspect='auto')
            
            ax.set_xlim(-100, 100)
            ax.axvline(0, color='lime', linestyle=':', linewidth=6)
            
            ax.set_xlabel('Radial Velocity in Planet Rest Frame (km s$^{-1}$)', fontsize=18)
            ax.set_ylabel('Orbital Phase', fontsize=18)
            
            t1, t2 = self.transit_indices
            ax.plot([-200, 200], [self.data_table['phases'][t1], self.data_table['phases'][t1]], 
                    color='darkblue', linestyle='--', linewidth=3)
            ax.plot([-200, 200], [self.data_table['phases'][t2], self.data_table['phases'][t2]], 
                    color='magenta', linestyle='--', linewidth=3)
            
            ax.text(0.05, 0.95, species_label, 
                    transform=ax.transAxes,
                    fontsize=24,
                    fontweight='bold',
                    color='white',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            cb = fig.colorbar(im)
            cb.set_label('CCF Amplitude (ppm)', fontsize=18)
            cb.ax.tick_params(labelsize=18)
            cb.ax.get_yaxis().labelpad = 15
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            
            cmap_suffix = cmap_label.lower().replace(' ', '_')
            filename = f'{template_name}_prf_{cmap_suffix}_pm100.pdf'
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        
        for cmap_name, cmap_label in color_schemes:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            
            im = ax.imshow(cc_planet_sorted,
                           cmap=cmap_name,
                           origin='lower',
                           interpolation='None',
                           rasterized=True,
                           extent=extent,
                           aspect='auto')
            
            ax.set_xlim(-100, 100)
            ax.axvline(0, color='lime', linestyle=':', linewidth=6)
            ax.set_xlabel('Radial Velocity in Planet Rest Frame (km s$^{-1}$)', fontsize=18)
            ax.set_ylabel('Orbital Phase', fontsize=18)
            
            # Transit boundary lines
            t1, t2 = self.transit_indices
            ax.plot([-200, 200], [self.data_table['phases'][t1], self.data_table['phases'][t1]], 
                    color='darkblue', linestyle='--', linewidth=3)
            ax.plot([-200, 200], [self.data_table['phases'][t2], self.data_table['phases'][t2]], 
                    color='magenta', linestyle='--', linewidth=3)
            
            # Species label
            ax.text(0.05, 0.95, species_label, 
                    transform=ax.transAxes,
                    fontsize=24,
                    fontweight='bold',
                    color='white',
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            cb = fig.colorbar(im)
            cb.set_label('CCF Amplitude (ppm)', fontsize=18)
            cb.ax.tick_params(labelsize=18)
            cb.ax.get_yaxis().labelpad = 15
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            
            cmap_suffix = cmap_label.lower().replace(' ', '_')
            filename = f'{template_name}_prf_{cmap_suffix}_pm100.png'
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
    
    def process_template(self, template_path, output_dir="."):
        """
        Process a single atmospheric template
        
        Parameters
        ----------
        template_path : str
            Path to template file
        output_dir : str
            Directory for output files
            
        Returns
        -------
        results : dict
            Analysis results
        """
        # Extract template name
        template_name = Path(template_path).stem
        
        print(f"\nProcessing template: {template_name}")
        print(f"Template file: {template_path}")
        
        # Load template
        try:
            template_data = ascii.read(template_path)
            if 'Wavelength' not in template_data.colnames or 'Radius' not in template_data.colnames:
                raise ValueError("Template must have 'Wavelength' and 'Radius' columns")
        except Exception as e:
            print(f"Error loading template: {e}")
            return None
        
        model_wave = template_data['Wavelength']
        model_flux = template_data['Radius']
        
        print(f"Template wavelength range: {model_wave.min():.1f} - {model_wave.max():.1f} Ã…")
        
        if self.data_table is None:
            self.load_data()
        
        if self.clean_wave_grid is None or self.clean_flux_grid is None:
            print("Cleaning spectra...")
            self.clean_wave_grid, self.clean_flux_grid = self.clean_spectra(
                np.array(self.data_table['waves']), 
                np.array(self.data_table['flux']),
                self.transit_indices,
                do_pca=True,
                pca_components=3
            )
        
        rv_grid, cc_grid = self.cross_correlate(
            self.clean_wave_grid, self.clean_flux_grid,
            model_wave, model_flux, self.transit_indices
        )
        cc_grid = np.array(cc_grid)
        
        snr_grid, vsys_grid, kp_grid = self.find_best_kp_vsys(
            rv_grid, cc_grid, self.data_table['phases'], template_name
        )
        
        # Generate plots
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self._plot_ccf_map(rv_grid, cc_grid, template_name, output_path)
        self._plot_planet_rest_frame(rv_grid, cc_grid, template_name, output_path)
        
        results = {
            'template_name': template_name,
            'species_label': self.get_species_label(template_name),
            'rv_grid': rv_grid,
            'cc_grid': cc_grid,
            'snr_grid': snr_grid,
            'vsys_grid': vsys_grid,
            'kp_grid': kp_grid,
            'max_snr': np.max(snr_grid),
            'template_wave_range': (model_wave.min(), model_wave.max())
        }
        
        return results
    
    def process_all_templates(self, template_dir="templates", output_dir="results"):
        """
        Process all templates in a directory
        
        Parameters
        ----------
        template_dir : str
            Directory containing template files
        output_dir : str
            Directory for output files
            
        Returns
        -------
        all_results : list
            List of results dictionaries
        """
        template_pattern = os.path.join(template_dir, "*.dat")
        template_files = glob.glob(template_pattern)
        
        if len(template_files) == 0:
            raise FileNotFoundError(f"No template files found in {template_dir}")
        
        print(f"Found {len(template_files)} template files")
        
        # Ensure data is loaded once
        if self.data_table is None:
            self.load_data()
        
        all_results = []
        for template_file in template_files:
            result = self.process_template(template_file, output_dir)
            if result is not None:
                all_results.append(result)
        
        print(f"\nProcessed {len(all_results)} templates successfully")
        return all_results


if __name__ == "__main__":
    from astropy import coordinates as coord, units as u
    
    star_coords = coord.SkyCoord("01:46:31.90", "+02:42:01.40", 
                               unit=(u.hourangle, u.deg), frame='icrs')
    keck = coord.EarthLocation.of_site('Keck')
    
    wasp76b_system = ExoplanetSystem(
        name="WASP-76b",
        t0=2457273.4191,  # BJD_TDB from Kokori et al.
        period=1.8098806,  # days from Kokori et al.
        kp=196.52,  # km/s from Ehrenreich et al. 2020
        k_star=0.1156,  # km/s from Ehrenreich et al. 2020
        vsys=-1.167,  # km/s from Ehrenreich et al. 2020
        stellar_radius=1.756,  # solar radii from Gaia DR2
        coordinates=star_coords,
        observatory=keck,
        transit_phase_half_width=0.04
    )
    
    
    w76 = Atmosphere(wasp76b_system, "latestDRP/*L1.fits")
    
    fe = w76.process_template("templates/Fe_custom.dat")
    
    if fe:
        print(f"Successfully processed {fe['species_label']}")
        print(f"Max SNR: {fe['max_snr']:.2f}")
    
    print("\nProcessing all templates...")
    all_results = w76.process_all_templates("templates", "results")
