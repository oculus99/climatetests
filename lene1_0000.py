# Physical Units Planet Generator
# Temperature in Celsius, precipitation in mm/year, energy balance from solar constant S

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
STEFAN_BOLTZMANN = 5.67e-8  # W m-2 K-4
SOLAR_CONSTANT = 1361.0     # W m-2 (Earth-Sun distance)
ALBEDO_OCEAN = 0.06
ALBEDO_LAND_BASE = 0.15
ALBEDO_DESERT = 0.35
ALBEDO_FOREST = 0.12
ALBEDO_ICE = 0.7
ALBEDO_SNOW = 0.8

# Atmospheric greenhouse effect parameters
GREENHOUSE_BASE = 0.78      # Base atmospheric absorption (greenhouse)
CO2_EQUIVALENT = 400.0      # ppm CO2 equivalent
WATER_VAPOR_FACTOR = 1.5    # Multiplier for water vapor feedback

# ============================================================================
# NOISE FUNCTIONS (unchanged)
# ============================================================================
@jit(nopython=True)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@jit(nopython=True)
def lerp(t, a, b):
    return a + t * (b - a)

@jit(nopython=True)
def grad(hash_val, x, y, z):
    h = hash_val & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

@jit(nopython=True)
def perlin_noise_3d_at_point(x, y, z, scale, p):
    x_coord = x * scale
    y_coord = y * scale
    z_coord = z * scale
    X = int(np.floor(x_coord)) & 255
    Y = int(np.floor(y_coord)) & 255
    Z = int(np.floor(z_coord)) & 255
    x_coord -= np.floor(x_coord)
    y_coord -= np.floor(y_coord)
    z_coord -= np.floor(z_coord)
    u = fade(x_coord)
    v = fade(y_coord)
    w = fade(z_coord)
    A = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z
    return lerp(w,
                lerp(v,
                     lerp(u, grad(p[AA], x_coord, y_coord, z_coord),
                          grad(p[BA], x_coord-1, y_coord, z_coord)),
                     lerp(u, grad(p[AB], x_coord, y_coord-1, z_coord),
                          grad(p[BB], x_coord-1, y_coord-1, z_coord))),
                lerp(v,
                     lerp(u, grad(p[AA+1], x_coord, y_coord, z_coord-1),
                          grad(p[BA+1], x_coord-1, y_coord, z_coord-1)),
                     lerp(u, grad(p[AB+1], x_coord, y_coord-1, z_coord-1),
                          grad(p[BB+1], x_coord-1, y_coord-1, z_coord-1))))

@jit(nopython=True)
def multifractal_at_point(x, y, z, scale, octaves, persistence, lacunarity, p):
    noise_value = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        octave_noise = perlin_noise_3d_at_point(x, y, z, scale * frequency, p)
        noise_value += amplitude * octave_noise
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= persistence
    if max_amplitude > 0:
        noise_value /= max_amplitude
    return noise_value

@jit(nopython=True)
def ridge_noise_at_point(x, y, z, scale, octaves, persistence, lacunarity, p):
    noise_value = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0
    for _ in range(octaves):
        octave_noise = perlin_noise_3d_at_point(x, y, z, scale * frequency, p)
        ridge_noise = 1.0 - abs(octave_noise)
        noise_value += amplitude * ridge_noise
        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= persistence
    if max_amplitude > 0:
        noise_value /= max_amplitude
    return noise_value

# ============================================================================
# ORBITAL & SEASONAL CALCULATIONS
# ============================================================================
@jit(nopython=True)
def deg2rad(d):
    return d * np.pi / 180.0

@jit(nopython=True)
def orbital_distance_factor(e, true_longitude):
    """Returns (1/r)^2 factor for solar insolation"""
    denom = 1.0 + e * np.cos(true_longitude)
    if denom <= 1e-6:
        denom = 1e-6
    return 1.0 / (denom * denom)

@jit(nopython=True)
def solar_declination(obliquity_rad, true_longitude):
    """Solar declination angle"""
    return np.arcsin(np.sin(obliquity_rad) * np.sin(true_longitude))

@jit(nopython=True)
def month_true_longitude(month_index, mvelp_rad):
    """True longitude for given month"""
    return 2.0 * np.pi * (month_index / 12.0) + mvelp_rad

@jit(nopython=True)
def daily_insolation(latitude, declination, solar_constant, dist_factor):
    """
    Calculate daily averaged insolation at top of atmosphere (W/m²)
    Accounts for day length and solar angle
    """
    lat = latitude
    dec = declination
    
    # Hour angle at sunrise/sunset
    cos_hour_angle = -np.tan(lat) * np.tan(dec)
    
    if cos_hour_angle <= -1.0:
        # Polar day
        hour_angle = np.pi
    elif cos_hour_angle >= 1.0:
        # Polar night
        return 0.0
    else:
        hour_angle = np.arccos(cos_hour_angle)
    
    # Daily average insolation
    insolation = (solar_constant * dist_factor / np.pi) * (
        hour_angle * np.sin(lat) * np.sin(dec) +
        np.cos(lat) * np.cos(dec) * np.sin(hour_angle)
    )
    
    return max(0.0, insolation)

# ============================================================================
# ELEVATION GENERATION
# ============================================================================
@jit(nopython=True, parallel=True)
def create_elevation_map(width, height, radius, p):
    """Generate normalized elevation (0-1 range)"""
    elevation_map = np.zeros((height, width))
    center = width // 2
    
    for i in prange(height):
        for j in range(width):
            theta = 2 * np.pi * j / width
            phi = np.pi * (i / height - 0.5)
            
            x = center + radius * np.cos(phi) * np.cos(theta)
            y = center + radius * np.cos(phi) * np.sin(theta)
            z = center + radius * np.sin(phi)
            
            continental_noise = multifractal_at_point(x, y, z, 0.002, 8, 0.5, 2.0, p)
            mountain_noise = ridge_noise_at_point(x+1000, y+1000, z+1000, 0.005, 6, 0.6, 2.2, p)
            
            elevation = continental_noise * 0.7 + mountain_noise * 0.3
            elevation_map[i, j] = elevation
    
    return elevation_map

# ============================================================================
# ALBEDO CALCULATION
# ============================================================================
def calculate_albedo(elevation_norm, temperature_celsius, ocean_threshold):
    """Calculate surface albedo based on surface type and temperature"""
    h, w = elevation_norm.shape
    albedo = np.zeros((h, w))
    
    for i in range(h):
        for j in range(w):
            elev = elevation_norm[i, j]
            temp = temperature_celsius[i, j]
            
            if elev < ocean_threshold:
                # Ocean albedo
                if temp < -1.8:  # Sea ice
                    albedo[i, j] = ALBEDO_ICE
                else:
                    albedo[i, j] = ALBEDO_OCEAN
            else:
                # Land albedo
                if temp < -5.0:
                    # Ice/snow
                    albedo[i, j] = ALBEDO_SNOW
                elif temp < 10.0:
                    # Mixed snow/vegetation
                    snow_fraction = (10.0 - temp) / 15.0
                    albedo[i, j] = snow_fraction * ALBEDO_SNOW + (1 - snow_fraction) * ALBEDO_LAND_BASE
                else:
                    # Base land albedo (will be refined by vegetation later)
                    albedo[i, j] = ALBEDO_LAND_BASE
    
    return albedo

# ============================================================================
# TEMPERATURE CALCULATION FROM ENERGY BALANCE
# ============================================================================
def calculate_temperature_from_energy_balance(insolation_monthly, albedo, elevation_norm, 
                                             ocean_threshold, greenhouse_factor=GREENHOUSE_BASE):
    """
    Calculate surface temperature from energy balance:
    Incoming solar: S * (1 - albedo)
    Outgoing thermal: σ * T^4 * (1 - greenhouse)
    At equilibrium: S * (1 - α) = σ * T^4 * (1 - g)
    """
    months, h, w = insolation_monthly.shape
    temp_monthly_kelvin = np.zeros((months, h, w))
    
    for m in range(months):
        for i in range(h):
            for j in range(w):
                S = insolation_monthly[m, i, j]
                alpha = albedo[i, j]
                
                # Absorbed solar radiation
                absorbed = S * (1.0 - alpha)
                
                # Effective greenhouse transmissivity (what fraction escapes to space)
                # For Earth: about 0.61 escapes, 0.39 is trapped
                transmissivity = 1.0 - greenhouse_factor
                
                # Energy balance: absorbed solar = emitted thermal / transmissivity
                # S(1-α) = ε·σ·T⁴ where ε is effective emissivity to space
                if absorbed > 0:
                    # Emissivity accounting for atmosphere
                    effective_emissivity = transmissivity * 0.95  # Surface emissivity ~0.95
                    T4 = absorbed / (STEFAN_BOLTZMANN * effective_emissivity)
                    T_kelvin = T4 ** 0.25
                else:
                    T_kelvin = 200.0  # Minimum background temperature
                
                # Apply elevation cooling (lapse rate ~6.5 K/km)
                if elevation_norm[i, j] > ocean_threshold:
                    height_km = (elevation_norm[i, j] - ocean_threshold) * 8.0  # Scale to ~8km max
                    lapse_cooling = height_km * 6.5
                    T_kelvin -= lapse_cooling
                
                temp_monthly_kelvin[m, i, j] = max(0.0, T_kelvin)
    
    # Convert to Celsius
    temp_monthly_celsius = temp_monthly_kelvin - 273.15
    
    return temp_monthly_celsius

# ============================================================================
# PRECIPITATION FROM ATMOSPHERIC CIRCULATION
# ============================================================================
def calculate_precipitation(temperature_celsius, elevation_norm, phis, ocean_threshold):
    """
    Calculate precipitation in mm/year based on:
    - Temperature (warm air holds more moisture)
    - Atmospheric circulation patterns (Hadley, Ferrel, Polar cells)
    - Orographic effects
    - Ocean proximity
    """
    months, h, w = temperature_celsius.shape
    precip_monthly = np.zeros((months, h, w))
    
    for m in range(months):
        for i in range(h):
            phi = phis[i, 0]
            phi_deg = abs(phi * 180.0 / np.pi)
            
            for j in range(w):
                temp = temperature_celsius[m, i, j]
                elev = elevation_norm[i, j]
                
                # Saturation vapor pressure (Clausius-Clapeyron)
                # es = 6.11 * exp(17.27 * T / (T + 237.3)) [hPa]
                if temp > -40:
                    es = 6.11 * np.exp(17.27 * temp / (temp + 237.3))
                else:
                    es = 0.1
                
                # Moisture capacity (mm water equivalent)
                moisture_capacity = es * 10.0  # Rough scaling
                
                # Atmospheric circulation patterns
                if phi_deg < 10:
                    # ITCZ - heavy convective precipitation
                    circulation_factor = 2.5
                elif phi_deg < 30:
                    # Subtropical high - descending dry air
                    circulation_factor = 0.3
                elif phi_deg < 60:
                    # Mid-latitude westerlies - frontal precipitation
                    circulation_factor = 1.5
                else:
                    # Polar - cold, dry
                    circulation_factor = 0.5
                
                # Ocean proximity enhances precipitation
                if elev < ocean_threshold:
                    ocean_factor = 2.0
                else:
                    # Distance from ocean (rough proxy)
                    ocean_dist = abs(elev - ocean_threshold) * 10
                    ocean_factor = 1.0 / (1.0 + ocean_dist)
                
                # Orographic precipitation
                if elev > ocean_threshold and i > 0:
                    slope = (elevation_norm[i, j] - elevation_norm[i-1, j]) * 100
                    oro_factor = 1.0 + max(0, slope * 5.0)
                else:
                    oro_factor = 1.0
                
                # Monthly precipitation (mm)
                precip = moisture_capacity * circulation_factor * ocean_factor * oro_factor * 0.1
                precip_monthly[m, i, j] = max(0, precip)
    
    # Annual total
    precip_annual = np.sum(precip_monthly, axis=0)
    
    return precip_annual, precip_monthly

# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================
class PhysicalPlanetGenerator:
    def __init__(self, width=512, height=256, seed=None, solar_constant=SOLAR_CONSTANT):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.solar_constant = solar_constant
        np.random.seed(self.seed)
        self.p = self._create_permutation_table()
    
    def _create_permutation_table(self):
        p = np.arange(256, dtype=np.int32)
        np.random.shuffle(p)
        return np.tile(p, 2)
    
    def _apply_hypsometric_curve(self, elevation, ocean_threshold):
        """Apply realistic depth/height distribution"""
        h, w = elevation.shape
        transformed = np.zeros_like(elevation)
        
        for i in range(h):
            for j in range(w):
                elev = elevation[i, j]
                if elev < ocean_threshold:
                    # Ocean depth curve
                    depth = elev / ocean_threshold
                    if depth > 0.8:
                        # Shallow continental shelf
                        transformed[i, j] = ocean_threshold - 0.05 * (1 - depth) / 0.2
                    else:
                        # Deep ocean
                        curve = 1.0 - (1.0 - depth / 0.8) ** 2
                        transformed[i, j] = 0.0 + (ocean_threshold - 0.05) * curve
                else:
                    # Land elevation curve (exponential)
                    land_height = (elev - ocean_threshold) / (1.0 - ocean_threshold)
                    exp_height = (np.exp(2.5 * land_height) - 1) / (np.exp(2.5) - 1)
                    transformed[i, j] = ocean_threshold + exp_height * (1.0 - ocean_threshold)
        
        return transformed
    
    def generate_planet(self, radius=200, ocean_threshold=0.35, obliquity_deg=23.5, 
                       eccentricity=0.017, mvelp_deg=0.0, hypsometric=True):
        """
        Generate planet with physical units
        
        Returns:
        - elevation_norm: 0-1 normalized elevation
        - temperature_celsius: monthly temperatures in °C
        - precipitation_mm: annual precipitation in mm/year
        """
        print(f"🌍 Generating planet with physical units...")
        print(f"   Solar constant: {self.solar_constant:.1f} W/m²")
        print(f"   Obliquity: {obliquity_deg}°, Eccentricity: {eccentricity}")
        
        start_time = time.time()
        
        # Generate elevation
        elevation_raw = create_elevation_map(self.width, self.height, radius, self.p)
        elevation_min = np.min(elevation_raw)
        elevation_max = np.max(elevation_raw)
        elevation_norm = (elevation_raw - elevation_min) / (elevation_max - elevation_min)
        
        if hypsometric:
            elevation_norm = self._apply_hypsometric_curve(elevation_norm, ocean_threshold)
        
        # Calculate latitude grid
        phis = np.zeros((self.height, self.width))
        for i in range(self.height):
            phi = np.pi * (i / self.height - 0.5)
            phis[i, :] = phi
        
        # Calculate monthly insolation
        obliquity_rad = deg2rad(obliquity_deg)
        mvelp_rad = deg2rad(mvelp_deg)
        
        insolation_monthly = np.zeros((12, self.height, self.width))
        for m in range(12):
            lambda_sol = month_true_longitude(m, mvelp_rad)
            delta = solar_declination(obliquity_rad, lambda_sol)
            dist_factor = orbital_distance_factor(eccentricity, lambda_sol)
            
            for i in range(self.height):
                lat = phis[i, 0]
                insol = daily_insolation(lat, delta, self.solar_constant, dist_factor)
                insolation_monthly[m, i, :] = insol
        
        # Initial albedo estimate (will iterate)
        albedo = np.full((self.height, self.width), ALBEDO_LAND_BASE)
        albedo[elevation_norm < ocean_threshold] = ALBEDO_OCEAN
        
        # Iterate temperature-albedo-precipitation for convergence (climate feedback)
        print("   Iterating climate feedbacks...")
        for iteration in range(5):
            temp_celsius = calculate_temperature_from_energy_balance(
                insolation_monthly, albedo, elevation_norm, ocean_threshold
            )
            temp_annual = np.mean(temp_celsius, axis=0)
            
            # Update albedo based on temperature
            albedo = calculate_albedo(elevation_norm, temp_annual, ocean_threshold)
            
            if iteration == 4:  # Final iteration - calculate precipitation
                precip_annual, precip_monthly = calculate_precipitation(
                    temp_celsius, elevation_norm, phis, ocean_threshold
                )
        
        generation_time = time.time() - start_time
        
        # Calculate comprehensive statistics
        ocean_mask = elevation_norm < ocean_threshold
        land_mask = ~ocean_mask
        
        ocean_coverage = np.sum(ocean_mask) / (self.height * self.width)
        
        # Temperature stats
        temp_global_mean = np.mean(temp_annual)
        temp_ocean_mean = np.mean(temp_annual[ocean_mask]) if np.any(ocean_mask) else 0
        temp_land_mean = np.mean(temp_annual[land_mask]) if np.any(land_mask) else 0
        temp_equator = np.mean(temp_annual[self.height//2-5:self.height//2+5, :])
        temp_pole_n = np.mean(temp_annual[-10:, :])
        temp_pole_s = np.mean(temp_annual[:10, :])
        
        # Precipitation stats
        precip_global_mean = np.mean(precip_annual)
        precip_ocean_mean = np.mean(precip_annual[ocean_mask]) if np.any(ocean_mask) else 0
        precip_land_mean = np.mean(precip_annual[land_mask]) if np.any(land_mask) else 0
        precip_max = np.max(precip_annual)
        precip_min = np.min(precip_annual)
        
        # Energy budget
        insol_global = np.mean(np.mean(insolation_monthly, axis=0))
        albedo_global = np.mean(albedo)
        absorbed_sw = insol_global * (1 - albedo_global)
        
        print(f"✅ Complete! Time: {generation_time:.2f}s")
        print(f"\n{'='*70}")
        print(f"  PLANET STATISTICS")
        print(f"{'='*70}")
        print(f"\n  SURFACE COVERAGE:")
        print(f"    Ocean:          {ocean_coverage*100:6.2f}%")
        print(f"    Land:           {(1-ocean_coverage)*100:6.2f}%")
        print(f"\n  TEMPERATURE (°C):")
        print(f"    Global mean:    {temp_global_mean:7.1f}")
        print(f"    Ocean mean:     {temp_ocean_mean:7.1f}")
        print(f"    Land mean:      {temp_land_mean:7.1f}")
        print(f"    Equator:        {temp_equator:7.1f}")
        print(f"    North pole:     {temp_pole_n:7.1f}")
        print(f"    South pole:     {temp_pole_s:7.1f}")
        print(f"    Range:          {np.min(temp_annual):7.1f} to {np.max(temp_annual):7.1f}")
        print(f"\n  PRECIPITATION (mm/year):")
        print(f"    Global mean:    {precip_global_mean:7.0f}")
        print(f"    Ocean mean:     {precip_ocean_mean:7.0f}")
        print(f"    Land mean:      {precip_land_mean:7.0f}")
        print(f"    Maximum:        {precip_max:7.0f}")
        print(f"    Minimum:        {precip_min:7.0f}")
        print(f"\n  ENERGY BUDGET (W/m²):")
        print(f"    Solar constant: {self.solar_constant:7.1f}")
        print(f"    TOA insolation: {insol_global:7.1f}")
        print(f"    Global albedo:  {albedo_global:7.3f}")
        print(f"    Absorbed SW:    {absorbed_sw:7.1f}")
        print(f"{'='*70}\n")
        
        return {
            'elevation': elevation_norm,
            'temperature_monthly': temp_celsius,
            'temperature_annual': temp_annual,
            'precipitation_annual': precip_annual,
            'precipitation_monthly': precip_monthly,
            'insolation_monthly': insolation_monthly,
            'albedo': albedo,
            'ocean_area': np.sum(elevation_norm < ocean_threshold) / (self.height * self.width)
        }, generation_time

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_physical_planet(planet_data, generation_time, ocean_threshold):
    """Plot planet data with physical units"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Elevation
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(planet_data['elevation'], cmap='terrain', origin='lower', 
                     extent=[0,360,-90,90])
    ax1.set_title('Elevation (normalized)')
    ax1.contour(planet_data['elevation'], levels=[ocean_threshold], 
                colors='red', linewidths=1.2, origin='lower', extent=[0,360,-90,90])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Annual temperature
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(planet_data['temperature_annual'], cmap='RdBu_r', 
                     origin='lower', extent=[0,360,-90,90], vmin=-40, vmax=40)
    ax2.set_title('Annual Mean Temperature (°C)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='°C')
    
    # Annual precipitation
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(planet_data['precipitation_annual'], cmap='YlGnBu', 
                     origin='lower', extent=[0,360,-90,90], vmin=0, vmax=3000)
    ax3.set_title('Annual Precipitation (mm/year)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='mm/year')
    
    # Albedo
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(planet_data['albedo'], cmap='gray_r', origin='lower', 
                     extent=[0,360,-90,90], vmin=0, vmax=1)
    ax4.set_title('Surface Albedo')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Seasonal temperature variation
    ax5 = fig.add_subplot(gs[1, 1])
    temp_range = np.max(planet_data['temperature_monthly'], axis=0) - \
                 np.min(planet_data['temperature_monthly'], axis=0)
    im5 = ax5.imshow(temp_range, cmap='plasma', origin='lower', 
                     extent=[0,360,-90,90])
    ax5.set_title('Seasonal Temperature Range (°C)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label='°C')
    
    # Insolation (annual mean)
    ax6 = fig.add_subplot(gs[1, 2])
    insol_annual = np.mean(planet_data['insolation_monthly'], axis=0)
    im6 = ax6.imshow(insol_annual, cmap='hot', origin='lower', 
                     extent=[0,360,-90,90])
    ax6.set_title('Annual Mean Insolation (W/m²)')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label='W/m²')
    
    # Zonal mean temperature profile
    ax7 = fig.add_subplot(gs[2, 0])
    lat_deg = np.linspace(-90, 90, planet_data['temperature_annual'].shape[0])
    temp_zonal = np.mean(planet_data['temperature_annual'], axis=1)
    ax7.plot(lat_deg, temp_zonal, 'b-', linewidth=2)
    ax7.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Latitude (°)')
    ax7.set_ylabel('Temperature (°C)')
    ax7.set_title('Zonal Mean Temperature')
    ax7.grid(True, alpha=0.3)
    
    # Zonal mean precipitation profile
    ax8 = fig.add_subplot(gs[2, 1])
    precip_zonal = np.mean(planet_data['precipitation_annual'], axis=1)
    ax8.plot(lat_deg, precip_zonal, 'g-', linewidth=2)
    ax8.axhline(1000, color='k', linestyle='--', alpha=0.3, label='1000 mm/yr')
    ax8.set_xlabel('Latitude (°)')
    ax8.set_ylabel('Precipitation (mm/year)')
    ax8.set_title('Zonal Mean Precipitation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Energy budget info
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    ocean_pct = planet_data['ocean_area']*100
    land_pct = 100 - ocean_pct
    temp_mean = np.mean(planet_data['temperature_annual'])
    precip_mean = np.mean(planet_data['precipitation_annual'])
    albedo_mean = np.mean(planet_data['albedo'])
    insol_mean = np.mean(np.mean(planet_data['insolation_monthly'], axis=0))
    
    info_text = f"""
GENERATION: {generation_time:.1f}s

COVERAGE:
  Ocean: {ocean_pct:.1f}%
  Land:  {land_pct:.1f}%

CLIMATE:
  T_mean:  {temp_mean:+.1f}°C
  T_range: {np.min(planet_data['temperature_annual']):.1f} to {np.max(planet_data['temperature_annual']):.1f}°C
  P_mean:  {precip_mean:.0f} mm/yr
  P_range: {np.min(planet_data['precipitation_annual']):.0f} to {np.max(planet_data['precipitation_annual']):.0f} mm/yr

ENERGY:
  S_0:     {insol_mean:.0f} W/m²
  Albedo:  {albedo_mean:.3f}
  Absorbed: {insol_mean*(1-albedo_mean):.0f} W/m²
    """
    ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Physical Planet Generator - Energy Balance Model', 
                 fontsize=14, fontweight='bold')
    plt.show()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    print('🌍 PHYSICAL PLANET GENERATION')
    print('=' * 60)
    
    # Generate Earth-like planet
    gen = PhysicalPlanetGenerator(width=512, height=256, seed=42, solar_constant=1361.0)
    
    planet_data, gen_time = gen.generate_planet(
        radius=200,
        ocean_threshold=0.35,
        obliquity_deg=23.5,
        eccentricity=0.017,
        mvelp_deg=102.0,  # Earth's current perihelion
        hypsometric=True
    )
    
    plot_physical_planet(planet_data, gen_time, 0.35)
    
    print('\n✅ Generation complete!')
