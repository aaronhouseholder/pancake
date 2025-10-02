"""
This code closely follows the methods presented in Kesseli et al. (2021, 2022) for
detecting atmospheric species in high-resolution spectra using cross-correlation
techniques. Much of the code has been adapted from those papers for use with KPF data,
so please cite those papers if you find this useful:
- Kesseli et al. (2021): "Confirmation of Asymmetric Iron Absorption in WASP-76b with HARPS"
- Kesseli et al. (2022): "An Atomic Spectral Survey of WASP-76b: Resolving Chemical Gradients and Asymmetries"
"""

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
from lmfit.models import GaussianModel


class Planet:
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

class DataLoader:
    """Handles data loading and basic preprocessing"""
    
    @staticmethod
    def load_observations(file_pattern, system):
        """Load and process observational data files"""
        print(f"Loading data from pattern: {file_pattern}")
        file_list = glob.glob(file_pattern)
        
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(file_list)} files")
        
        combined_fluxes = []
        rest_waves = []
        phases = []
        planet_rvs = []
        barycentric_corrections = []
        bjds = []
        stellar_rvs = []
        
        for file_path in file_list:
            data = DataLoader._process_single_file(file_path, system)
            combined_fluxes.append(data['flux'])
            rest_waves.append(data['wave'])
            phases.append(data['phase'])
            planet_rvs.append(data['planet_rv'])
            barycentric_corrections.append(data['vbar'])
            bjds.append(data['bjd'])
            stellar_rvs.append(data['stellar_rv'])
        
        data_table = Table({
            'flux': combined_fluxes,
            'waves': rest_waves,
            'phases': phases,
            'planet_rvs': planet_rvs,
            'barycentric_corrections': barycentric_corrections,
            'bjds': bjds,
            'stellar_rvs': stellar_rvs
        })
        
        data_table.sort('bjds')
        data_table.reverse()
        
        # Calculate transit indices
        transit_indices = DataLoader._calculate_transit_indices(
            data_table['phases'], system.transit_phase_half_width
        )
        
        print(f"Data loaded successfully:")
        print(f"  Number of observations: {len(data_table)}")
        print(f"  Phase range: {np.min(data_table['phases']):.4f} to {np.max(data_table['phases']):.4f}")
        print(f"  Transit indices: {transit_indices}")
        
        return data_table, transit_indices
    
    @staticmethod
    def _process_single_file(file_path, system):
        """Process a single FITS file"""
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
                                location=system.observatory)
            ltt_bary = t_astropy.light_travel_time(system.coordinates)
            bjd = (t_astropy.tdb + ltt_bary).jd
            
            vbar_corr = system.coordinates.radial_velocity_correction(obstime=t_astropy)
            vbar = vbar_corr.to(u.km/u.s).value
            
            phase = OrbitalCalculations.calculate_orbital_phase(bjd, system.t0, system.period)
            stellar_rv = OrbitalCalculations.calculate_planet_rv(system.k_star, phase)
            
            # Apply velocity corrections to wavelengths
            rest_wave = wave_3 / (1 + (-vbar + system.vsys - stellar_rv) / (2.99792e5))
            
            return {
                'flux': combined_flux,
                'wave': rest_wave,
                'phase': phase,
                'planet_rv': OrbitalCalculations.calculate_planet_rv(system.kp, phase),
                'vbar': vbar,
                'bjd': bjd,
                'stellar_rv': stellar_rv
            }
    
    @staticmethod
    def _calculate_transit_indices(phases, transit_half_width):
        """Calculate transit start and end indices"""
        if phases[0] > transit_half_width:
            transit_start = np.where(phases < transit_half_width)[0][0]
        else:
            transit_start = 0
            
        if phases[-1] < -transit_half_width:
            transit_end = np.where(phases < -transit_half_width)[0][0]
        else:
            transit_end = len(phases)
        
        return [transit_start, transit_end]


class OrbitalCalculations:
    """Handles orbital mechanics calculations"""
    
    @staticmethod
    def calculate_orbital_phase(t, t0, period):
        """Calculate orbital phase from observation time"""
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
    
    @staticmethod
    def calculate_planet_rv(kp, phase):
        """Calculate planet radial velocity at given phase"""
        return kp * np.sin(2*np.pi * phase)


class SpectralCleaner:
    """Handles additional spectral cleaning operations"""
    
    @staticmethod
    def clean_spectra(waves, fluxes, transit_indices, do_pca=True, 
                     pca_components=5, plots=False):
        print("Starting addition cleaning pipeline...")
        
        # Remove low flux regions (order gaps)
        avg_flux = np.nanmean(fluxes, axis=0)
        low_flux_mask = avg_flux > 0.05
        waves_clipped = waves[:, low_flux_mask]
        fluxes_clipped = fluxes[:, low_flux_mask]
        
        # Normalize spectra
        norm_fluxes = fluxes_clipped / np.nanmean(fluxes_clipped, axis=1, keepdims=True)
        
        # Sigma clipping
        sigma_clipped = SpectralCleaner._remove_vertical_outliers(norm_fluxes, 3)
        
        # Remove broadband variations
        blaze_removed = np.zeros_like(norm_fluxes)
        out_of_transit = np.concatenate((sigma_clipped[:transit_indices[0]], 
                                       sigma_clipped[transit_indices[1]:]), axis=0)
        mean_spec = np.nanmean(out_of_transit, axis=0)
        avg_blaze = gaussian_filter1d(mean_spec, 200)
        cont_rem_spec = norm_fluxes / avg_blaze
        
        for i in range(len(norm_fluxes)):
            blaze = gaussian_filter1d(cont_rem_spec[i], 200)
            blaze_removed[i] = sigma_clipped[i] / blaze
        
        # Interpolate onto common grid
        wave_grid = np.nanmean(waves_clipped, axis=0)
        wave_grid = wave_grid[20:-20]  # Avoid extrapolation at edges
        interp_fluxes = np.ones((len(waves_clipped), len(wave_grid)))
        
        for i in range(len(waves_clipped)):
            func = inter.CubicSpline(waves_clipped[i], blaze_removed[i])
            interp_fluxes[i] = func(wave_grid)
        
        # PCA (optional)
        if do_pca:
            pca_cleaned = SpectralCleaner._apply_pca_cleaning(interp_fluxes, pca_components)
        else:
            pca_cleaned = interp_fluxes
        
        # Remove high-variance columns
        std_cols = np.nanstd(pca_cleaned, axis=0)
        mask = std_cols < 1.3 * np.nanmean(std_cols)
        wave_grid_final = wave_grid[mask]
        cleaned_flux = pca_cleaned[:, mask]
        
        # Final sigma clipping
        final_cleaned = SpectralCleaner._remove_vertical_outliers(cleaned_flux, 3)
        
        print(f"Cleaning complete. Final grid: {len(wave_grid_final)} wavelength points")
        
        return wave_grid_final, final_cleaned
    
    @staticmethod
    def _remove_vertical_outliers(flux_array, clip_sigma=3):
        """Remove vertical outliers using sigma clipping"""
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
    
    @staticmethod
    def _apply_pca_cleaning(flux_array, n_components=5):
        """Apply PCA cleaning to remove systematic effects"""
        u, s, vt = np.linalg.svd(flux_array, full_matrices=False)
        v = vt.T
        
        # Remove first n components
        s_new = s.copy()
        s_new[:n_components] = 0.0
        
        # Calculate cleaned flux
        cleaned_flux = np.dot(u, np.dot(np.diag(s_new), v.T)) + 1.0
        
        return cleaned_flux


class CrossCorrelation:
    """Handles cross-correlation calculations"""
    
    @staticmethod
    def cross_correlate(waves, fluxes, model_wave, model_flux, transit_indices, system):
        """Cross-correlate spectra with atmospheric model"""
        clip_start = waves[0] - 10
        clip_end = waves[-1] + 10
        model_mask = (model_wave > clip_start) & (model_wave < clip_end)
        model_wave_clip = model_wave[model_mask]
        model_flux_clip = model_flux[model_mask]
        
        poly_coeffs = np.polyfit(model_wave_clip, model_flux_clip, 3)
        continuum_rj = np.poly1d(poly_coeffs)(model_wave_clip)
        
        jupiter_to_solar = 0.10049  # Rj/Rs conversion
        rp_atmosphere_rs = model_flux_clip * jupiter_to_solar
        rp_continuum_rs = continuum_rj * jupiter_to_solar
        stellar_radius_rs = system.stellar_radius
        depth_difference_ppm = ((rp_atmosphere_rs**2 - rp_continuum_rs**2) / stellar_radius_rs**2) * 1e6
        
        template_std = np.std(depth_difference_ppm)
        if template_std > 0:
            normalized_template = depth_difference_ppm / template_std
        else:
            normalized_template = depth_difference_ppm
        
        rv_grid = np.arange(-300, 300, 0.5)
        model_depth_grid = []
        
        for rv in rv_grid:
            model_wave_shift = model_wave_clip / (1 - rv / (2.998e5))
            interp_func = inter.CubicSpline(model_wave_shift, normalized_template)
            interp_model = interp_func(waves)
            model_depth_grid.append(interp_model)
        
        flux_residuals_ppm = (1.0 - fluxes) * 1e6
        
        cc_grid = []
        for i, flux_residual in enumerate(flux_residuals_ppm):
            cc_row = []
            for j, template in enumerate(model_depth_grid):
                cc = np.dot(flux_residual, template) * template_std / len(template)
                cc_row.append(cc)
            cc_grid.append(cc_row)
        
        cc_grid = np.array(cc_grid)
        
        out_of_transit = np.concatenate((cc_grid[:transit_indices[0]], 
                                       cc_grid[transit_indices[1]:]), axis=0)
        mean_cc = np.nanmean(out_of_transit, axis=0)
        
        cc_grid_norm = []
        for cc_row in cc_grid:
            cc_systematic = cc_row - mean_cc
            smoothed = gaussian_filter1d(cc_systematic, 140)
            residual = cc_systematic - smoothed
            cc_grid_norm.append(residual)
        
        print("Cross-correlation complete")
        return rv_grid, cc_grid_norm
    
    @staticmethod
    def shift_ccf_to_planet_frame(rv_grid, cc_grid, planet_rvs):
        """Shift CCF to planet rest frame"""
        new_vel_grid = rv_grid.copy()
        cc_grid_planet_frame = np.zeros_like(cc_grid)

        for i in range(cc_grid.shape[0]):
            shift = planet_rvs[i]
            shifted_rv = rv_grid - shift
            f = inter.interp1d(shifted_rv, cc_grid[i], kind='cubic',
                         bounds_error=False, fill_value=np.nan)
            cc_grid_planet_frame[i] = f(new_vel_grid)

        return new_vel_grid, cc_grid_planet_frame

class DetectionAnalysis:
    """Handles Kp-Vsys analysis and blueshift detection"""
    
    @staticmethod
    def find_best_kp_vsys(rv_grid, cc_grid, phases, system, template_name='unknown'):
        """Find best Kp and Vsys values using SNR optimization"""
        print(f"Making Kp-Vsys plot for {template_name}...")
        
        kp_grid = np.arange(system.kp + 80, system.kp - 80, -0.5)
        vsys_grid = np.arange(-70, 70, 0.25)
        shift_grid = np.arange(-100, 100, 1.0)
        
        snr_grid = np.zeros((len(kp_grid), len(vsys_grid)))
        
        for i, kp in enumerate(kp_grid):
            for j, vsys in enumerate(vsys_grid):
                # Shift CCFs to planet rest frame
                shifted_ccs = []
                for k, phase in enumerate(phases):
                    planet_rv = OrbitalCalculations.calculate_planet_rv(kp, phase) + vsys
                    interp_func = inter.interp1d(rv_grid - planet_rv, cc_grid[k])
                    shifted_cc = interp_func(shift_grid)
                    shifted_ccs.append(shifted_cc)
                
                combined_cc = np.nanmean(shifted_ccs, axis=0)
                signal = combined_cc[np.where(shift_grid == 0)][0]
                
                inds_low = np.where(shift_grid < -54)[0]
                inds_up = np.where(shift_grid > 49)[0]
                nopeak = np.concatenate((combined_cc[inds_low], combined_cc[inds_up]))
                noise_std = np.std(nopeak)
                noise_mean = np.mean(nopeak)
                
                snr = (signal - noise_mean) / noise_std
                snr_grid[i, j] = snr
        
        species_label = TemplateUtils.get_species_label(template_name)
        max_idx = np.unravel_index(np.argmax(snr_grid), snr_grid.shape)
        best_kp = kp_grid[max_idx[0]]
        best_vsys = vsys_grid[max_idx[1]]
        max_snr = snr_grid[max_idx]
        
        print(f"Best Kp: {best_kp:.2f} km/s")
        print(f"Best Vsys: {best_vsys:.2f} km/s")
        print(f"Maximum SNR: {max_snr:.2f}")
        
        PlottingUtils.plot_kp_vsys_map(snr_grid, vsys_grid, kp_grid, template_name, system)
        
        return snr_grid, vsys_grid, kp_grid
    
    @staticmethod
    def analyze_blueshift_phase_by_phase(rv_grid, cc_grid, phases, transit_indices, 
                                       kp, vsys, template_name='unknown'):
        """
        Analyze the blueshift of signal by fitting Gaussians
        NOTE: Gaussian fits likely require some tuning based on visual line shapes
        """
        cc_planet_frame = []
        
        for i in range(len(cc_grid)):
            planet_rv = kp * np.sin(2 * np.pi * phases[i]) + vsys
            f_interp = inter.interp1d(rv_grid - planet_rv, cc_grid[i], 
                               bounds_error=False, fill_value=np.nan)
            cc_shifted = f_interp(rv_grid)
            cc_planet_frame.append(cc_shifted)
        
        cc_planet_frame = np.array(cc_planet_frame)
        
        # Define phase ranges for beginning and end of transit
        phase_begin_mask = (phases >= -0.04) & (phases <= -0.02)
        phase_end_mask = (phases >= 0.02) & (phases <= 0.04)
        
        if np.sum(phase_begin_mask) > 0:
            ccf_begin = np.nanmean(cc_planet_frame[phase_begin_mask], axis=0)
        else:
            return None
            
        if np.sum(phase_end_mask) > 0:
            ccf_end = np.nanmean(cc_planet_frame[phase_end_mask], axis=0)
        else:
            return None
        
        in_transit_mask = np.zeros(len(phases), dtype=bool)
        in_transit_mask[transit_indices[0]:transit_indices[1]] = True
        ccf_full = np.nanmean(cc_planet_frame[in_transit_mask], axis=0)
        
        fit_begin = DetectionAnalysis._fit_gaussian_to_ccf(rv_grid, ccf_begin)
        fit_end = DetectionAnalysis._fit_gaussian_to_ccf(rv_grid, ccf_end)
        fit_full = DetectionAnalysis._fit_gaussian_to_ccf(rv_grid, ccf_full)
        
        PlottingUtils.plot_blueshift_analysis(
            rv_grid, ccf_begin, ccf_end, fit_begin, fit_end,
            phases[phase_begin_mask] if np.sum(phase_begin_mask) > 0 else [],
            phases[phase_end_mask] if np.sum(phase_end_mask) > 0 else [],
            template_name
        )
        
        return True
    
    @staticmethod
    def _fit_gaussian_to_ccf(rv, ccf, initial_center=0):
        """Fit a Gaussian to a 1D CCF and return parameters"""
        if np.all(np.isnan(ccf)):
            return None
        
        ccf_scaled = ccf
        
        peak_idx = np.nanargmax(ccf_scaled)
        peak_rv = rv[peak_idx]
        
        model = GaussianModel()
        
        params = model.make_params(
            amplitude=ccf_scaled[peak_idx],
            center=peak_rv,
            sigma=10
        )
        
        params['center'].min = -20
        params['center'].max = 20
        params['sigma'].min = 2
        params['sigma'].max = 30
        params['amplitude'].min = 0
        
        fit_mask = np.abs(rv - peak_rv) < 30
        
        try:
            result = model.fit(ccf_scaled[fit_mask], x=rv[fit_mask], params=params)
            
            noise_mask = (np.abs(rv) > 50) & (~np.isnan(ccf_scaled))
            if np.sum(noise_mask) > 10:
                noise_std = np.std(ccf_scaled[noise_mask])
                snr = result.params['amplitude'].value / noise_std
                center_err = result.params['sigma'].value / snr if snr > 1 else 2.0
            else:
                center_err = result.params['center'].stderr or 2.0
            
            return {
                'center': result.params['center'].value,
                'center_err': center_err,
                'amplitude': result.params['amplitude'].value,
                'sigma': result.params['sigma'].value,
                'fwhm': 2.355 * result.params['sigma'].value,
                'result': result,
                'ccf_scaled': ccf_scaled
            }
        except Exception as e:
            return None


class PlottingUtils:
    """Handles all plotting operations"""
    
    @staticmethod
    def plot_kp_vsys_map(snr_grid, vsys_grid, kp_grid, template_name, system):
        """Generate Kp-Vsys SNR map"""
        fig, ax = plt.subplots(figsize=(7, 6))
        
        im = ax.imshow(snr_grid, cmap=cm.inferno, 
                      extent=[vsys_grid[0], vsys_grid[-1], kp_grid[-1], kp_grid[0]],
                      aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('$v_{sys}$ (km s$^{-1}$)', fontsize=18)
        ax.set_ylabel('$K_p$ (km s$^{-1}$)', fontsize=18)
        
        cb = fig.colorbar(im)
        cb.set_label('SNR', rotation=90, fontsize=18)
        cb.ax.tick_params(labelsize=18)
        cb.ax.get_yaxis().labelpad = 4
        
        ax.plot([vsys_grid[0], vsys_grid[-1]], [system.kp, system.kp], 'k--')
        ax.plot([0, 0], [kp_grid[0], kp_grid[-1]], 'k--')
        
        species_label = TemplateUtils.get_species_label(template_name)
        ax.text(0.05, 0.05, species_label, 
                transform=ax.transAxes,
                fontsize=21.5,
                fontweight='bold',
                color='white',
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        max_snr = np.max(snr_grid)
        if max_snr > 5:
            ax.text(0.975, 0.05, f'SNR = {max_snr:.1f}', 
                transform=ax.transAxes,
                fontsize=21.5,
                fontweight='bold',
                color='white',
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        
        plt.savefig(f'{template_name}_Kp-vsys.png', bbox_inches='tight')
        plt.savefig(f'{template_name}_Kp-vsys.pdf', dpi=300, bbox_inches='tight')
        print(f"Saved: {template_name}_Kp-vsys.pdf")
        print(f"Saved: {template_name}_Kp-vsys.png")
        plt.close()
    
    @staticmethod
    def plot_ccf_map(rv_grid, cc_grid, data_table, transit_indices, template_name):
        """Generate 2D CCF map"""
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        
        im = ax.imshow(-cc_grid, cmap=cm.gray, 
                      extent=[np.min(rv_grid), np.max(rv_grid), 
                             data_table['phases'][-1], data_table['phases'][0]])
        
        ax.set_xlabel('Radial Velocity (km s$^{-1}$)', fontsize=14)
        ax.set_ylabel('Orbital Phase', fontsize=14)
        
        t1, t2 = transit_indices
        ax.plot([-200, 200], [data_table['phases'][t1], data_table['phases'][t1]], 
                color='cyan', linestyle='--', linewidth=3)
        ax.plot([-200, 200], [data_table['phases'][t2], data_table['phases'][t2]], 
                color='cyan', linestyle='--', linewidth=3)
        
        ax.plot(data_table['planet_rvs'][t2:], data_table['phases'][t2:], 
                color='lime', linestyle=':', linewidth=6)
        ax.plot(data_table['planet_rvs'][:t1], data_table['phases'][:t1], 
                color='lime', linestyle=':', linewidth=6)
        
        ax.set_xlim(-150, 150)
        ax.set_aspect(1100)
        
        cb = fig.colorbar(im)
        cb.set_label('Amplitude (ppm)', rotation=270, fontsize=18)
        cb.ax.tick_params(labelsize=14)
        cb.ax.get_yaxis().labelpad = 15
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(f'{template_name}_SRF.png')
        print(f"Saved: {template_name}_SRF.png")
        plt.close()
    
    @staticmethod
    def plot_planet_rest_frame(rv_grid, cc_grid, data_table, transit_indices, template_name):
        """Generate planet rest frame plot"""
        new_vel, cc_planet = CrossCorrelation.shift_ccf_to_planet_frame(
            rv_grid, cc_grid, data_table['planet_rvs']
        )
        
        phases_sorted = np.sort(data_table['phases'])
        cc_planet_sorted = cc_planet[np.argsort(data_table['phases']), :]
        
        extent = [new_vel[0], new_vel[-1], phases_sorted[0], phases_sorted[-1]]
        
        species_label = TemplateUtils.get_species_label(template_name)
        
        color_schemes = [('gray_r', 'Gray Inverted')]
        
        for cmap_name, cmap_label in color_schemes:
            # PDF version
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
            
            t1, t2 = transit_indices
            ax.plot([-200, 200], [data_table['phases'][t1], data_table['phases'][t1]], 
                    color='darkblue', linestyle='--', linewidth=3)
            ax.plot([-200, 200], [data_table['phases'][t2], data_table['phases'][t2]], 
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
            cb.set_label('Amplitude (ppm)', fontsize=18)
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
            
            # PNG version
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
            
            ax.plot([-200, 200], [data_table['phases'][t1], data_table['phases'][t1]], 
                    color='darkblue', linestyle='--', linewidth=3)
            ax.plot([-200, 200], [data_table['phases'][t2], data_table['phases'][t2]], 
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
            cb.set_label('Amplitude (ppm)', fontsize=18)
            cb.ax.tick_params(labelsize=18)
            cb.ax.get_yaxis().labelpad = 15
            
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            
            filename = f'{template_name}_prf_{cmap_suffix}_pm100.png'
            
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
    
    @staticmethod
    def plot_blueshift_analysis(rv_grid, ccf_begin, ccf_end, fit_begin, fit_end,
                               phase_begin_range, phase_end_range, template_name):
        """Create blueshift analysis plot"""
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        
        # Get phase ranges for legend labels
        if len(phase_begin_range) > 0:
            phi_begin_str = f'φ = {np.min(phase_begin_range):.3f} to {np.max(phase_begin_range):.3f}'
        else:
            phi_begin_str = 'φ = early'
            
        if len(phase_end_range) > 0:
            phi_end_str = f'φ = {np.min(phase_end_range):.3f} to {np.max(phase_end_range):.3f}'
        else:
            phi_end_str = 'φ = late'
        
        # Plot 1D CCFs
        if fit_begin is not None:
            ax.plot(rv_grid, ccf_begin, color='magenta', linewidth=2, 
                     label=f'{phi_begin_str}')
            ax.axvline(fit_begin['center'], color='magenta', linestyle='--', 
                       linewidth=1, alpha=0.7)
        else:
            ax.plot(rv_grid, ccf_begin, color='magenta', linewidth=2, 
                     label=f'{phi_begin_str}\n(no fit)')
        
        if fit_end is not None:
            ax.plot(rv_grid, ccf_end, color='darkblue', linewidth=2, 
                     label=f'{phi_end_str}')
            ax.axvline(fit_end['center'], color='darkblue', linestyle='--', 
                       linewidth=1, alpha=0.7)
        else:
            ax.plot(rv_grid, ccf_end, color='darkblue', linewidth=2, 
                     label=f'{phi_end_str}\n(no fit)')
        
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_xlim(-75, 75)
        ax.set_xlabel('Radial Velocity in Planet Rest Frame (km s$^{-1}$)', fontsize=18)
        ax.set_ylabel('Amplitude (ppm)', fontsize=18)
        
        phase_legend = ax.legend(loc='upper right', fontsize=14, frameon=True, fancybox=True, 
                               shadow=True, framealpha=0.9)
        ax.add_artist(phase_legend)
        
        ax.grid(True, alpha=0.3)
        
        if fit_begin is not None and fit_end is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='magenta', linestyle='--', linewidth=2, 
                      label=f'{fit_begin["center"]:.1f} km/s'),
                Line2D([0], [0], color='darkblue', linestyle='--', linewidth=2, 
                      label=f'{fit_end["center"]:.1f} km/s')
            ]
            velocity_legend = ax.legend(handles=legend_elements, loc='upper left', 
                                      fontsize=14, frameon=True, fancybox=True, 
                                      shadow=True, framealpha=0.9)
            ax.add_artist(velocity_legend)
        elif fit_begin is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='magenta', linestyle='--', linewidth=2, 
                      label=f'{fit_begin["center"]:.1f} km/s')
            ]
            velocity_legend = ax.legend(handles=legend_elements, loc='upper left', 
                                      fontsize=14, frameon=True, fancybox=True, 
                                      shadow=True, framealpha=0.9)
            ax.add_artist(velocity_legend)
        elif fit_end is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='darkblue', linestyle='--', linewidth=2, 
                      label=f'{fit_end["center"]:.1f} km/s')
            ]
            velocity_legend = ax.legend(handles=legend_elements, loc='upper left', 
                                      fontsize=14, frameon=True, fancybox=True, 
                                      shadow=True, framealpha=0.9)
            ax.add_artist(velocity_legend)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        plt.savefig(f'{template_name}_blueshift_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{template_name}_blueshift_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

class TemplateUtils:
    """Utilities for template processing"""
    
    @staticmethod
    def get_species_label(template_name):
        """Convert template name to proper species label"""
        clean_name = template_name.replace('_custom', '').replace('.dat', '')
        template_lower = clean_name.lower()
        
        species_map = {
            'na_lor_cut': 'Na I', 'na_allard': 'Na I', 
            'na_allard_new': 'Na I', 'na_burrows': 'Na I', 'na': 'Na I',
            'ca+': 'Ca II', 'ca': 'Ca I', 
            'fe+': 'Fe+', 'fe': 'Fe I',
            'mg+': 'Mg+', 'mg': 'Mg I',
            'k': 'K I',
            'ti': 'Ti I',
            'v': 'V I',
            'si': 'Si I',
            'cr_agss09': 'Cr I', 'cr': 'Cr I',
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
    
    @staticmethod
    def load_template(template_path):
        """Load atmospheric template from file"""
        try:
            template_data = ascii.read(template_path)
            if 'Wavelength' not in template_data.colnames or 'Radius' not in template_data.colnames:
                raise ValueError("Template must have 'Wavelength' and 'Radius' columns")
            
            return template_data['Wavelength'], template_data['Radius']
        except Exception as e:
            print(f"Error loading template: {e}")
            return None, None

class Atmosphere:
    """Main atmospheric analysis class"""
    
    def __init__(self, system, file_pattern="latestDRP/*L1.fits"):
        self.system = system
        self.file_pattern = file_pattern
        self.data_table = None
        self.clean_wave_grid = None
        self.clean_flux_grid = None
        self.transit_indices = None
    
    def load_data(self):
        """Load and process observational data"""
        self.data_table, self.transit_indices = DataLoader.load_observations(
            self.file_pattern, self.system
        )
        return self.data_table
    
    def process_template(self, template_path, output_dir=".", include_blueshift_analysis=True):
        """Process a single atmospheric template"""
        template_name = Path(template_path).stem
        
        print(f"\nProcessing template: {template_name}")
        print(f"Template file: {template_path}")
        
        model_wave, model_flux = TemplateUtils.load_template(template_path)
        if model_wave is None:
            return None
        
        print(f"Template wavelength range: {model_wave.min():.1f} - {model_wave.max():.1f} Å")
        
        if self.data_table is None:
            self.load_data()
        
        if self.clean_wave_grid is None or self.clean_flux_grid is None:
            print("Cleaning spectra...")
            self.clean_wave_grid, self.clean_flux_grid = SpectralCleaner.clean_spectra(
                np.array(self.data_table['waves']), 
                np.array(self.data_table['flux']),
                self.transit_indices,
                do_pca=True,
                pca_components=5
            )
        
        # Cross-correlate
        rv_grid, cc_grid = CrossCorrelation.cross_correlate(
            self.clean_wave_grid, self.clean_flux_grid,
            model_wave, model_flux, self.transit_indices, self.system
        )
        cc_grid = np.array(cc_grid)
        
        snr_grid, vsys_grid, kp_grid = DetectionAnalysis.find_best_kp_vsys(
            rv_grid, cc_grid, self.data_table['phases'], self.system, template_name
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        PlottingUtils.plot_ccf_map(rv_grid, cc_grid, self.data_table, 
                                   self.transit_indices, template_name)
        PlottingUtils.plot_planet_rest_frame(rv_grid, cc_grid, self.data_table,
                                            self.transit_indices, template_name)
        
        if include_blueshift_analysis:
            try:
                DetectionAnalysis.analyze_blueshift_phase_by_phase(
                    rv_grid, cc_grid, self.data_table['phases'], 
                    self.transit_indices, self.system.kp, self.system.vsys, template_name
                )
            except Exception as e:
                pass
        
        results = {
            'template_name': template_name,
            'species_label': TemplateUtils.get_species_label(template_name),
            'rv_grid': rv_grid,
            'cc_grid': cc_grid,
            'snr_grid': snr_grid,
            'vsys_grid': vsys_grid,
            'kp_grid': kp_grid,
            'max_snr': np.max(snr_grid),
            'template_wave_range': (model_wave.min(), model_wave.max())
        }
        
        return results
    
    def process_all_templates(self, template_dir="templates", output_dir="results", 
                            include_blueshift_analysis=True):
        """Process all templates in a directory"""
        template_pattern = os.path.join(template_dir, "*.dat")
        template_files = glob.glob(template_pattern)
        
        if len(template_files) == 0:
            raise FileNotFoundError(f"No template files found in {template_dir}")
        
        print(f"Found {len(template_files)} template files")
        
        if self.data_table is None:
            self.load_data()
        
        all_results = []
        for template_file in template_files:
            result = self.process_template(template_file, output_dir, include_blueshift_analysis)
            if result is not None:
                all_results.append(result)
        
        print(f"\nProcessed {len(all_results)} templates successfully")
        
        print(f"\nSUMMARY:")
        print(f"{'Species':>12} {'Max SNR':>10}")
        print("-" * 25)
        for result in sorted(all_results, key=lambda x: x['max_snr'], reverse=True):
            print(f"{result['species_label']:>12} {result['max_snr']:>10.2f}")
        
        return all_results

if __name__ == "__main__":
    from astropy import coordinates as coord, units as u
    
    star_coords = coord.SkyCoord("01:46:31.90", "+02:42:01.40", 
                               unit=(u.hourangle, u.deg), frame='icrs')
    keck = coord.EarthLocation.of_site('Keck')
    
    wasp76b_system = Planet(
        name="WASP-76b",
        t0=2457273.4191,
        period=1.8098806,
        kp=196.52,
        k_star=0.1156,
        vsys=-1.167,
        stellar_radius=1.756,
        coordinates=star_coords,
        observatory=keck,
        transit_phase_half_width=0.04
    )
    
    run_cc = Atmosphere(wasp76b_system, "latestDRP/*L1.fits")
    
    single_result = run_cc.process_template("templates/Cr_agss09.dat", 
                                            include_blueshift_analysis=True)
    
    if single_result:
        print(f"Max SNR: {single_result['max_snr']:.2f}")
    
    all_results = run_cc.process_all_templates("templates", "results", 
                                                include_blueshift_analysis=True)
