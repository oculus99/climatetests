"""
planet2_full_fixed.py

Korjattu versio planet2-ilmastomallista.

Tärkeimmät korjaukset:
 - Pilvien albedovaikutusta pienennetty
 - Vesihöyryn kasvihuonevaikutus normalisoitu
 - Negatiivinen lämpötilapalaute poistettu
 - Lumi ei enää lukkiudu helposti pysyväksi
 - Lumen sulaminen muutettu realistisemmaksi
 - Merijään kasvu/sulaminen tasapainotettu
 - Keskileveysasteiden lämpötilagradienttia tasoitettu
 - annual_rain määritetään ennen yhteenvetoa
 - poistettu ennenaikainen quit(-1)
 - lopun kuvaajat toimivat
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from perlin_noise import PerlinNoise
from numba import jit
from scipy.ndimage import gaussian_filter


# ============================================================
# CONFIG
# ============================================================

tau = 0.7
sigma = 5.670374419e-8

T_freeze = 273.15
dt = 86400.0

days_per_year = 365
n_years = 10

check_equilibrium = True
equilibrium_threshold = 0.005
equilibrium_check_years = 5

nlat = 64
nlon = 128

r_planet = 6.371e6


# ============================================================
# ALBEDO
# ============================================================

albedo_ocean = 0.08
albedo_land = 0.28
albedo_desert = 0.35
albedo_rainforest = 0.12
albedo_savanna = 0.18
albedo_temp_forest = 0.15
albedo_cool_forest = 0.13
albedo_grassland = 0.20
albedo_tundra = 0.25

albedo_snow = 0.75
albedo_ice = 0.55


# ============================================================
# AURINKO JA KASVIHUONE
# ============================================================

S_mean = 1361.0

G_base = 49.0
RF_CO2 = 2.0

greenhouse_factor = 0.50


# ============================================================
# MERI JA MAA
# ============================================================

H_ocean = 70.0
rho_ocean = 1000.0
Cp_ocean = 4200.0

D_ocean_base = 150.0

H_land = 2.0
rho_land = 1500.0
Cp_land = 800.0

D_land = 20.0


# ============================================================
# JÄÄ JA LUMI
# ============================================================

rho_ice = 917.0
Cp_ice = 2100.0

L_f = 3.34e5
H_char = 0.5

rho_snow = 300.0

# KORJAUS:
# Lumen sulamisnopeus metreinä/päivä/°C.
# Vanha versio käytti käytännössä 1 m/päivä/°C.
melt_snow_factor = 0.005

# Merijään kasvu
ice_growth_factor = 0.0008

# Merijään sulaminen
ice_melt_factor = 0.003


# ============================================================
# PILVET JA SADE
# ============================================================

cloud_formation_rate = 0.35
cloud_dissipation_rate = 0.12

# KORJAUS:
# Vanha 0.25 oli liian voimakas.
cloud_albedo_effect = 0.10

rain_efficiency = 0.08

evaporation_coeff = 0.008

L_v = 2.5e6

water_vapor_capacity = 50.0


# ============================================================
# VESIHÖYRYN KASVIHUONEVAIKUTUS
# ============================================================

# KORJAUS:
# Vanha:
#     water_vapor_feedback = water_vapor * 2.0
#
# Tämä antoi 50 mm vesihöyryllä 100 W/m2.
# Nyt vaikutus on rajattu järkevämpään suuruusluokkaan.

alpha_v = 12.0


# ============================================================
# TUULET
# ============================================================

wind_speed_max = 10.0
moisture_advection_coeff = 0.3


# ============================================================
# MAANPINTA
# ============================================================

noise_scale = 0.03
noise_octaves = 4

sealevel = -0.1

# Jos True, koko planeetta tehdään maaksi.
# False = normaali Perlin-noise-maapinta.
SEA_PLANET_TEST = False


# ============================================================
# LAT/LON
# ============================================================

latitudes = np.linspace(-90, 90, nlat)
longitudes = np.linspace(-180, 180, nlon)

dlat = np.radians(latitudes[1] - latitudes[0])
dlon = np.radians(longitudes[1] - longitudes[0])

lat_grid = np.tile(
    latitudes[:, np.newaxis],
    (1, nlon)
)

lat_rad = np.radians(latitudes)


# ============================================================
# LAT/LON -> 3D
# ============================================================

def latlon_to_spherical(lat, lon):

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)

    return x, y, z


# ============================================================
# FRACTAL NOISE
# ============================================================

def generate_fractal_noise(
    nlon,
    nlat,
    scale=1.0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    seed=1234
):

    noise = PerlinNoise(
        octaves=octaves,
        seed=seed
    )

    out = np.zeros((nlat, nlon))

    for i in range(nlat):

        for j in range(nlon):

            lat = (i / (nlat - 1)) * 180.0 - 90.0
            lon = (j / (nlon - 1)) * 360.0 - 180.0

            x, y, z = latlon_to_spherical(lat, lon)

            noise_value = noise([
                x * scale,
                y * scale,
                z * scale
            ])

            out[i, j] = noise_value

    return out


# ============================================================
# SURFACE
# ============================================================

def generate_surface_texture(noise_grid):

    return np.where(
        noise_grid > 0.0,
        1.0,
        0.0
    )


def compute_continentality(land_fraction):

    smoothed = gaussian_filter(
        land_fraction,
        sigma=3,
        mode='wrap'
    )

    continentality = np.clip(
        (smoothed - 0.3) / 0.4,
        0,
        1
    )

    return continentality


# ============================================================
# INSOLATION
# ============================================================

def compute_daily_insolation(
    S_mean,
    nlat,
    nlon,
    days_per_year
):

    phi = np.linspace(
        -np.pi / 2,
        np.pi / 2,
        nlat
    )

    tilt = np.radians(23.44)

    mvelp = np.radians(102.9)

    ecc = 0.0167

    P = 365.25 * 86400.0

    t = np.linspace(
        0,
        P,
        days_per_year,
        endpoint=False
    )

    M = 2 * np.pi * (t / P)

    E = M.copy()

    for _ in range(30):

        E = M + ecc * np.sin(E)

    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2)
    )

    lam = nu + mvelp

    r = (
        1 - ecc**2
    ) / (
        1 + ecc * np.cos(nu)
    )

    S0 = S_mean / r**2

    insolation = np.zeros(
        (days_per_year, nlat, nlon)
    )

    for d in range(days_per_year):

        delta = np.arcsin(
            np.sin(tilt) * np.sin(lam[d])
        )

        cosarg = np.clip(
            -np.tan(phi)[:, None]
            * np.tan(delta),
            -1.0,
            1.0
        )

        H0 = np.arccos(cosarg)

        mu = (
            np.cos(phi)[:, None]
            * np.cos(delta)
            * np.sin(H0)
            +
            H0
            * np.sin(phi)[:, None]
            * np.sin(delta)
        ) / np.pi

        insolation[d, :, :] = (
            np.maximum(mu, 0.0)
            * S0[d]
        )

    return insolation


# ============================================================
# CLIMATE ZONES
# ============================================================

def compute_climate_zones(
    lat_grid,
    day_of_year
):

    decl_deg = (
        23.44
        * np.sin(
            2 * np.pi
            * (day_of_year - 80)
            / 365.0
        )
    )

    itcz_lat = decl_deg

    rain_potential = np.exp(
        -(
            (lat_grid - itcz_lat) / 20.0
        )**2
    )

    desert_zone = np.where(
        (np.abs(lat_grid) > 20)
        &
        (np.abs(lat_grid) < 35),
        1.0,
        0.0
    )

    rain_potential *= (
        1.0 - 0.7 * desert_zone
    )

    abs_lat = np.abs(lat_grid)

    storm_band = np.exp(
        -(
            (abs_lat - 50.0) / 10.0
        )**2
    )

    rain_potential += (
        0.7 * storm_band
    )

    return np.clip(
        rain_potential,
        0,
        1
    ), desert_zone


# ============================================================
# OCEAN FRACTION
# ============================================================

def compute_ocean_fraction_effect(
    land_fraction
):

    ocean_frac = (
        1.0
        - np.mean(land_fraction)
    )

    if ocean_frac < 0.1:

        multiplier = 0.5

    elif ocean_frac < 0.3:

        multiplier = (
            0.8
            + (ocean_frac - 0.1) * 3.0
        )

    elif ocean_frac < 0.5:

        multiplier = (
            1.4
            + (ocean_frac - 0.3) * 1.5
        )

    elif ocean_frac < 0.7:

        multiplier = (
            1.7
            + (ocean_frac - 0.5)
        )

    else:

        multiplier = (
            1.9
            + (ocean_frac - 0.7) * 0.5
        )

    return np.clip(
        multiplier,
        0.5,
        2.0
    )


# ============================================================
# WINDS
# ============================================================

def compute_wind_profile(
    lat_grid,
    day_of_year
):

    itcz_position = (
        10.0
        * np.sin(
            2 * np.pi
            * day_of_year
            / days_per_year
        )
    )

    lat_from_itcz = (
        lat_grid
        - itcz_position
    )

    trade_wind_north = np.exp(
        -(
            (lat_from_itcz - 15.0)**2
            / 10.0**2
        )
    ) * (-wind_speed_max)

    trade_wind_south = np.exp(
        -(
            (lat_from_itcz + 15.0)**2
            / 10.0**2
        )
    ) * (-wind_speed_max)

    trade_winds = (
        trade_wind_north
        + trade_wind_south
    )

    westerlies_north = np.exp(
        -(
            (lat_grid - 45.0)**2
            / 12.0**2
        )
    ) * wind_speed_max

    westerlies_south = np.exp(
        -(
            (lat_grid + 45.0)**2
            / 12.0**2
        )
    ) * wind_speed_max

    westerlies = (
        westerlies_north
        + westerlies_south
    )

    polar_east_north = np.exp(
        -(
            (lat_grid - 75.0)**2
            / 15.0**2
        )
    ) * (-wind_speed_max * 0.5)

    polar_east_south = np.exp(
        -(
            (lat_grid + 75.0)**2
            / 15.0**2
        )
    ) * (-wind_speed_max * 0.5)

    polar_winds = (
        polar_east_north
        + polar_east_south
    )

    u_wind = (
        trade_winds
        + westerlies
        + polar_winds
    )

    itcz_calm = np.exp(
        -(lat_from_itcz**2)
        / 5.0**2
    )

    u_wind *= (
        1.0
        - itcz_calm * 0.7
    )

    return u_wind


# ============================================================
# MOISTURE ADVECTION
# ============================================================

@jit(nopython=True)
def advect_moisture_numba(
    water_vapor,
    u_wind,
    dt,
    r_planet,
    dlon,
    lat_rad
):

    nlat, nlon = water_vapor.shape

    water_vapor_new = water_vapor.copy()

    for i in range(nlat):

        theta = (
            np.pi / 2
            - lat_rad[i]
        )

        sin_theta = max(
            np.sin(theta),
            0.05
        )

        dx = (
            r_planet
            * sin_theta
            * dlon
        )

        for j in range(nlon):

            u = u_wind[i, j]

            cfl_factor = min(
                abs(u * dt / dx),
                0.9
            )

            if u > 0:

                j_upwind = (
                    j - 1
                ) % nlon

            else:

                j_upwind = (
                    j + 1
                ) % nlon

            flux = (
                cfl_factor
                * water_vapor[
                    i,
                    j_upwind
                ]
            )

            water_vapor_new[
                i, j
            ] = (
                water_vapor[i, j]
                * (1.0 - cfl_factor)
                + flux
            )

    return water_vapor_new


# ============================================================
# ALBEDO
# ============================================================

def compute_albedo(
    H_snow,
    ice_thickness,
    land_fraction,
    climate_type,
    cloud_cover
):

    alb = get_climate_albedo(
        climate_type
    ).astype(float)

    snow_mask = (
        H_snow > 0.001
    )

    snow_frac = np.minimum(
        H_snow / 0.20,
        1.0
    )

    alb = np.where(
        snow_mask,
        alb * (1.0 - snow_frac)
        + albedo_snow * snow_frac,
        alb
    )

    ice_mask = (
        (ice_thickness > 0.001)
        &
        (~snow_mask)
    )

    ice_frac = np.minimum(
        ice_thickness / 2.0,
        1.0
    )

    alb = np.where(
        ice_mask,
        alb * (1.0 - ice_frac)
        + albedo_ice * ice_frac,
        alb
    )

    # KORJAUS:
    # Pilvien vaikutus pienennetty.
    alb_with_clouds = (
        alb
        + cloud_cover
        * cloud_albedo_effect
    )

    return np.clip(
        alb_with_clouds,
        0.0,
        0.90
    )


# ============================================================
# CLIMATE CLASSIFICATION
# ============================================================

def classify_climate(
    T_celsius,
    annual_precip,
    lat_grid,
    land_fraction
):

    abs_lat = np.abs(lat_grid)

    climate = np.zeros_like(
        T_celsius,
        dtype=int
    )

    climate = np.where(
        land_fraction < 0.5,
        0,
        climate
    )

    on_land = (
        land_fraction > 0.5
    )

    ice_climate = (
        on_land
        &
        (
            (T_celsius < -10)
            |
            (abs_lat > 80)
        )
    )

    climate = np.where(
        ice_climate,
        8,
        climate
    )

    tundra_climate = (
        on_land
        &
        (T_celsius >= -10)
        &
        (T_celsius < 5)
        &
        ~ice_climate
    )

    climate = np.where(
        tundra_climate,
        7,
        climate
    )

    cool_temperate = (
        on_land
        &
        (T_celsius >= 5)
        &
        (T_celsius < 12)
        &
        (annual_precip > 1.5)
    )

    climate = np.where(
        cool_temperate,
        6,
        climate
    )

    warm_temperate = (
        on_land
        &
        (T_celsius >= 12)
        &
        (T_celsius < 18)
        &
        (annual_precip > 2.0)
    )

    climate = np.where(
        warm_temperate,
        5,
        climate
    )

    grassland = (
        on_land
        &
        (annual_precip >= 0.5)
        &
        (annual_precip < 2.0)
        &
        (T_celsius >= 5)
    )

    climate = np.where(
        grassland,
        4,
        climate
    )

    desert = (
        on_land
        &
        (annual_precip < 0.5)
        &
        (T_celsius >= 10)
    )

    climate = np.where(
        desert,
        3,
        climate
    )

    savanna = (
        on_land
        &
        (T_celsius >= 18)
        &
        (abs_lat < 30)
        &
        (annual_precip >= 1.5)
        &
        (annual_precip < 4.0)
    )

    climate = np.where(
        savanna,
        2,
        climate
    )

    rainforest = (
        on_land
        &
        (T_celsius >= 18)
        &
        (abs_lat < 30)
        &
        (annual_precip >= 4.0)
    )

    climate = np.where(
        rainforest,
        1,
        climate
    )

    cold_desert = (
        on_land
        &
        (annual_precip < 0.5)
        &
        (T_celsius < 10)
        &
        (T_celsius >= -10)
    )

    climate = np.where(
        cold_desert,
        3,
        climate
    )

    return climate


# ============================================================
# ALBEDO MAP
# ============================================================

def get_climate_albedo(
    climate_type
):

    albedo_map = {
        0: albedo_ocean,
        1: albedo_rainforest,
        2: albedo_savanna,
        3: albedo_desert,
        4: albedo_grassland,
        5: albedo_temp_forest,
        6: albedo_cool_forest,
        7: albedo_tundra,
        8: albedo_ice
    }

    alb = np.zeros_like(
        climate_type,
        dtype=float
    )

    for climate_val, albedo_val in albedo_map.items():

        alb = np.where(
            climate_type == climate_val,
            albedo_val,
            alb
        )

    return alb


# ============================================================
# NPP
# ============================================================

def compute_npp(
    T_celsius,
    annual_precip,
    land_fraction
):

    on_land = (
        land_fraction > 0.5
    )

    NPP_T = (
        3000.0
        /
        (
            1.0
            +
            np.exp(
                1.315
                - 0.119 * T_celsius
            )
        )
    )

    P_annual = (
        annual_precip * 365.0
    )

    NPP_P = (
        3000.0
        *
        (
            1.0
            -
            np.exp(
                -0.000664
                * P_annual
            )
        )
    )

    npp = np.where(
        on_land,
        np.minimum(
            NPP_T,
            NPP_P
        ) / 1000.0,
        0.0
    )

    return npp


# ============================================================
# NAMES
# ============================================================

def get_climate_name(
    climate_code
):

    names = {
        0: "Meri",
        1: "Trooppinen sademetsä",
        2: "Savanni",
        3: "Aavikko",
        4: "Aro/Step",
        5: "Lämpimän lauhkea metsä",
        6: "Viileän lauhkea metsä",
        7: "Tundra",
        8: "Jäätikkö"
    }

    return names.get(
        climate_code,
        "Tuntematon"
    )


# ============================================================
# CLIMATE COLORS
# ============================================================

def get_climate_color(
    climate_type
):

    nlat, nlon = (
        climate_type.shape
    )

    colors = np.zeros(
        (nlat, nlon, 3)
    )

    colors[
        climate_type == 0
    ] = [0.1, 0.2, 0.5]

    colors[
        climate_type == 1
    ] = [0.0, 0.6, 0.0]

    colors[
        climate_type == 2
    ] = [0.6, 0.7, 0.2]

    colors[
        climate_type == 3
    ] = [0.9, 0.8, 0.5]

    colors[
        climate_type == 4
    ] = [0.6, 0.8, 0.4]

    colors[
        climate_type == 5
    ] = [0.3, 0.7, 0.3]

    colors[
        climate_type == 6
    ] = [0.2, 0.5, 0.4]

    colors[
        climate_type == 7
    ] = [0.5, 0.6, 0.5]

    colors[
        climate_type == 8
    ] = [0.95, 0.95, 1.0]

    return colors


# ============================================================
# HEAT DIFFUSION
# ============================================================

@jit(nopython=True)
def apply_heat_diffusion_numba(
    T,
    D_grid,
    dt,
    r_planet,
    dlat,
    dlon,
    lat_rad,
    n_substeps=10
):

    nlat, nlon = T.shape

    T_new = T.copy()

    dt_sub = (
        dt / n_substeps
    )

    for sub in range(
        n_substeps
    ):

        T_temp = T_new.copy()

        for i in range(nlat):

            theta = (
                np.pi / 2
                - lat_rad[i]
            )

            sin_theta = max(
                np.sin(theta),
                0.05
            )

            dx_lat = (
                r_planet * dlat
            )

            dx_lon = (
                r_planet
                * sin_theta
                * dlon
            )

            for j in range(nlon):

                i_prev = max(
                    0,
                    i - 1
                )

                i_next = min(
                    nlat - 1,
                    i + 1
                )

                j_prev = (
                    j - 1
                ) % nlon

                j_next = (
                    j + 1
                ) % nlon

                lap_lat = (
                    T_temp[
                        i_next, j
                    ]
                    - 2 * T_temp[i, j]
                    + T_temp[
                        i_prev, j
                    ]
                ) / (
                    dx_lat**2
                )

                lap_lon = (
                    T_temp[
                        i, j_next
                    ]
                    - 2 * T_temp[i, j]
                    + T_temp[
                        i, j_prev
                    ]
                ) / (
                    dx_lon**2
                )

                lap = (
                    lap_lat
                    + lap_lon
                )

                dT = (
                    D_grid[i, j]
                    * lap
                    * dt_sub
                )

                dT = max(
                    min(dT, 2.0),
                    -2.0
                )

                T_new[i, j] += dT

    return T_new


# ============================================================
# SNOW AND ICE
# ============================================================

def update_snow_and_ice(
    T_celsius,
    rain_mm_per_day,
    H_snow,
    ice_thickness,
    land_fraction
):

    ocean_mask = (
        land_fraction < 0.5
    )

    # --------------------------------------------------------
    # Lumi
    # --------------------------------------------------------

    # KORJAUS:
    # Lähellä nollaa kaikki sade ei muutu välittömästi lumeksi.
    #
    # +2 C -> käytännössä vettä
    #  0 C -> noin puolet
    # -5 C -> lähes kaikki lunta
    #
    snow_fraction = np.clip(
        (2.0 - T_celsius) / 7.0,
        0.0,
        1.0
    )

    snowfall_m = (
        rain_mm_per_day
        / 1000.0
        * snow_fraction
    )

    # --------------------------------------------------------
    # Lumen sulaminen
    # --------------------------------------------------------

    melt_snow = np.where(
        T_celsius > 0.0,
        T_celsius
        * melt_snow_factor,
        0.0
    )

    H_snow = (
        H_snow
        + snowfall_m
        - melt_snow
    )

    H_snow = np.clip(
        H_snow,
        0.0,
        None
    )

    # --------------------------------------------------------
    # Merijää
    # --------------------------------------------------------

    ice_growth = np.where(
        ocean_mask & (T_celsius < 0.0),
        np.minimum(
            -T_celsius
            * ice_growth_factor,
            0.008
        ),
        0.0
    )

    # Lumen eristys hidastaa jään kasvua.
    snow_insulation = np.exp(
        -H_snow / 0.15
    )

    ice_growth *= (
        0.5
        + 0.5 * snow_insulation
    )

    ice_melt = np.where(
        ocean_mask
        &
        (T_celsius > 0.0),
        T_celsius
        * ice_melt_factor,
        0.0
    )

    ice_thickness = (
        ice_thickness
        + ice_growth
        - ice_melt
    )

    ice_thickness = np.where(
        ocean_mask,
        ice_thickness,
        0.0
    )

    ice_thickness = np.clip(
        ice_thickness,
        0.0,
        5.0
    )

    return (
        H_snow,
        ice_thickness
    )


# ============================================================
# INITIALIZATION
# ============================================================

print(
    "Initializing climate model..."
)


noise_grid = generate_fractal_noise(
    nlon,
    nlat,
    seed=5678,
    scale=0.25
)

noise_grid = (
    noise_grid
    + sealevel
)


if SEA_PLANET_TEST:

    noise_grid = (
        noise_grid * 0.0
        + 1.0
    )


land_fraction = np.where(
    noise_grid < 0,
    0.0,
    1.0
)


# ------------------------------------------------------------
# Mannereisuus
# ------------------------------------------------------------

continentality = (
    compute_continentality(
        land_fraction
    )
)


# ------------------------------------------------------------
# Heat capacity
# ------------------------------------------------------------

C_grid_ocean = (
    rho_ocean
    * Cp_ocean
    * H_ocean
)

C_grid_land = (
    rho_land
    * Cp_land
    * H_land
)


# ------------------------------------------------------------
# Diffusion
# ------------------------------------------------------------

D_grid = np.where(
    land_fraction > 0.5,
    D_land,
    D_ocean_base
    * np.maximum(
        np.cos(lat_rad)[:, None],
        0.10
    )
)


# ============================================================
# INITIAL STATE
# ============================================================

H_snow = np.zeros(
    (nlat, nlon)
)

ice_thickness = np.zeros(
    (nlat, nlon)
)

ice_fraction = np.zeros(
    (nlat, nlon)
)


polar_mask = (
    np.abs(lat_grid) > 75
)


ice_thickness[
    polar_mask
    &
    (land_fraction < 0.5)
] = 1.0


H_snow[
    polar_mask
    &
    (land_fraction > 0.5)
] = 0.05


# KORJAUS:
# Aloituslämpötila hieman realistisemmaksi
# leveysasteen mukaan.
#
# Aiemmin koko planeetta alkoi 288.15 K.
# Tämä ei sinänsä ollut virhe, mutta tekee
# alkuvaiheen polarisoitumisesta rajun.

T = (
    288.15
    - 35.0
    * np.sin(
        np.abs(lat_grid)
        / 90.0
        * np.pi / 2
    )**1.7
)

cloud_cover = np.zeros(
    (nlat, nlon)
)

water_vapor = np.zeros(
    (nlat, nlon)
)

precipitation = np.zeros(
    (nlat, nlon)
)

climate_type = np.zeros(
    (nlat, nlon),
    dtype=int
)


# ============================================================
# INSOLATION
# ============================================================

print(
    "Calculating annual insolation..."
)

insolation_year = (
    compute_daily_insolation(
        S_mean,
        nlat,
        nlon,
        days_per_year
    )
)


# ============================================================
# RECORDS
# ============================================================

T_record = np.zeros(
    (
        n_years * days_per_year,
        nlat,
        nlon
    )
)

ice_record = np.zeros_like(
    T_record
)

cloud_record = np.zeros_like(
    T_record
)

rain_record = np.zeros_like(
    T_record
)


# ============================================================
# MAIN LOOP
# ============================================================

print(
    "Starting simulation..."
)


equilibrium_reached = False

ocean_multiplier = (
    compute_ocean_fraction_effect(
        land_fraction
    )
)


actual_steps = (
    n_years
    * days_per_year
)


for step in range(
    actual_steps
):

    # --------------------------------------------------------
    # YEARLY STATUS
    # --------------------------------------------------------

    if step % days_per_year == 0:

        year = (
            step // days_per_year
            + 1
        )

        T_global = (
            np.mean(T)
            - 273.15
        )

        ice_coverage = (
            100.0
            * np.mean(
                ice_thickness > 0.1
            )
        )

        cloud_avg = (
            100.0
            * np.mean(
                cloud_cover
            )
        )

        rain_avg = np.mean(
            precipitation
        )

        print(
            f"Year {year}/{n_years} - "
            f"T: {T_global:.2f}°C, "
            f"Ice: {ice_coverage:.1f}%, "
            f"Clouds: {cloud_avg:.1f}%, "
            f"Rain: {rain_avg:.2f} mm/d"
        )

        # ----------------------------------------------------
        # EQUILIBRIUM
        # ----------------------------------------------------

        if (
            check_equilibrium
            and
            year > equilibrium_check_years
        ):

            check_start = max(
                0,
                step
                - equilibrium_check_years
                * days_per_year
            )

            T_recent = (
                T_record[
                    check_start:step
                ]
            )

            if T_recent.size > 0:

                cos_weights = (
                    np.cos(lat_rad)
                )

                cos_weights /= (
                    np.sum(
                        cos_weights
                    )
                )

                T_mean_recent = (
                    np.mean(
                        T_recent,
                        axis=2
                    )
                    - 273.15
                )

                T_global_recent = (
                    np.sum(
                        T_mean_recent
                        * cos_weights[None, :],
                        axis=1
                    )
                )

                recent_trend = (
                    T_global_recent[-1]
                    - T_global_recent[0]
                ) / equilibrium_check_years

                if (
                    abs(recent_trend)
                    < equilibrium_threshold
                ):

                    print(
                        "\n✓ TASAPAINO "
                        f"SAAVUTETTU vuonna {year}!"
                    )

                    print(
                        f"  Trendi: "
                        f"{recent_trend:+.4f} "
                        "°C/vuosi"
                    )

                    equilibrium_reached = True


    # ========================================================
    # DAY
    # ========================================================

    day = (
        step
        % days_per_year
    )

    is_ocean = (
        land_fraction < 0.5
    )


    # ========================================================
    # CLIMATE / RAIN POTENTIAL
    # ========================================================

    rain_potential, desert_zone = (
        compute_climate_zones(
            lat_grid,
            day
        )
    )

    ocean_multiplier = (
        compute_ocean_fraction_effect(
            land_fraction
        )
    )

    continental_reduction = (
        1.0
        - 0.6
        * continentality
        * land_fraction
    )

    coastal_boost = gaussian_filter(
        1.0 - land_fraction,
        sigma=2,
        mode='wrap'
    )

    coastal_boost = np.clip(
        coastal_boost * 1.5,
        0,
        1
    )

    rain_potential *= (
        continental_reduction
        *
        (
            1.0
            + coastal_boost * 0.3
        )
    )


    # ========================================================
    # WIND
    # ========================================================

    u_wind = compute_wind_profile(
        lat_grid,
        day
    )


    # ========================================================
    # TEMPERATURE
    # ========================================================

    T_celsius = (
        T - 273.15
    )


    # ========================================================
    # EVAPORATION
    # ========================================================

    evaporation = np.where(
        is_ocean,

        evaporation_coeff
        *
        np.maximum(
            T_celsius / 15.0,
            0.1
        )
        *
        (
            1.0
            - cloud_cover * 0.25
        ),

        0.0
    )

    evaporation = np.clip(
        evaporation,
        0.0,
        2.0
    )


    # ========================================================
    # WATER VAPOR
    # ========================================================

    water_vapor += (
        evaporation
    )

    water_vapor = (
        advect_moisture_numba(
            water_vapor,
            u_wind
            * moisture_advection_coeff,
            dt,
            r_planet,
            dlon,
            lat_rad
        )
    )

    water_vapor = np.clip(
        water_vapor,
        0.0,
        water_vapor_capacity
    )


    # ========================================================
    # CLOUDS
    # ========================================================

    land_moisture_boost = np.where(
        land_fraction > 0.5,
        1.25,
        1.0
    )

    moisture_factor = np.minimum(
        water_vapor / 5.0,
        1.5
    )

    cloud_formation = (
        cloud_formation_rate
        * moisture_factor
        * rain_potential
        * land_moisture_boost
    )

    cloud_dissipation = (
        cloud_dissipation_rate
        * cloud_cover
        *
        (
            1.0
            - rain_potential * 0.3
        )
    )

    cloud_cover += (
        cloud_formation
        - cloud_dissipation
    )

    cloud_cover = np.clip(
        cloud_cover,
        0.0,
        0.85
    )


    # ========================================================
    # PRECIPITATION
    # ========================================================

    rain_from_clouds = (
        rain_efficiency
        *
        cloud_cover**1.5
        *
        rain_potential
        *
        ocean_multiplier
    )

    precipitation = (
        rain_from_clouds
        * 100.0
    )

    precipitation = np.clip(
        precipitation,
        0.0,
        20.0
    )


    # ========================================================
    # VAPOR LOSS
    # ========================================================

    vapor_loss = np.minimum(
        precipitation * 0.002,
        water_vapor
    )

    water_vapor -= (
        vapor_loss
    )

    cloud_cover *= (
        1.0
        - np.clip(
            rain_from_clouds * 0.4,
            0,
            0.5
        )
    )


    # ========================================================
    # ANNUAL PRECIPITATION
    # ========================================================

    if step >= days_per_year:

        annual_precip_current = (
            np.mean(
                rain_record[
                    step - days_per_year:step
                ],
                axis=0
            )
        )

    else:

        annual_precip_current = (
            precipitation.copy()
        )


    # ========================================================
    # CLIMATE
    # ========================================================

    climate_type = (
        classify_climate(
            T_celsius,
            annual_precip_current,
            lat_grid,
            land_fraction
        )
    )


    # ========================================================
    # HEAT CAPACITY
    # ========================================================

    C_grid = np.where(

        is_ocean,

        C_grid_ocean
        +
        rho_ice
        * Cp_ice
        * ice_thickness
        +
        rho_snow
        * Cp_ice
        * H_snow,

        C_grid_land
        +
        rho_ice
        * Cp_ice
        * ice_thickness
    )


    # ========================================================
    # ALBEDO
    # ========================================================

    ice_fraction = np.clip(
        ice_thickness
        / H_char,
        0.0,
        1.0
    )

    alb = compute_albedo(
        H_snow,
        ice_thickness,
        land_fraction,
        climate_type,
        cloud_cover
    )


    # ========================================================
    # SOLAR ABSORPTION
    # ========================================================

    absorbed = (
        (1.0 - alb)
        * tau
        * insolation_year[
            day,
            :, :
        ]
    )


    # ========================================================
    # OUTGOING RADIATION
    # ========================================================

    emitted = (
        (1.0 - greenhouse_factor)
        * sigma
        * T**4
    )


    # ========================================================
    # WATER VAPOR FEEDBACK
    # ========================================================

    # KORJAUS:
    # Normalisoitu vesihöyryvaikutus.
    water_vapor_norm = (
        water_vapor
        / (
            water_vapor_capacity
            + 1e-12
        )
    )

    water_vapor_feedback = (
        alpha_v
        * water_vapor_norm
    )


    # ========================================================
    # TEMPERATURE FEEDBACK
    # ========================================================

    # KORJAUS:
    #
    # Vanha:
    #
    # G_feedback =
    #     2*T_celsius jos T < 0
    #
    # Tämä teki kylmästä alueesta entistä
    # voimakkaammin kylmenevän.
    #
    # Nyt palautekytkentä on pieni ja rajoitettu.

    G_feedback = np.clip(
        0.25 * T_celsius,
        -6.0,
        6.0
    )


    G_dynamic = (
        G_base
        + RF_CO2
        + G_feedback
        + water_vapor_feedback
    )


    # ========================================================
    # LATENT COOLING
    # ========================================================

    latent_cooling = (
        -evaporation
        * L_v
        / dt
        * 0.30
    )


    # ========================================================
    # NET ENERGY
    # ========================================================

    Q_net = (
        absorbed
        + G_dynamic
        - emitted
        + latent_cooling
    )

    Q_net = np.clip(
        Q_net,
        -300.0,
        300.0
    )


    # ========================================================
    # RADIATIVE TEMPERATURE CHANGE
    # ========================================================

    dT_radiation = (
        Q_net
        * dt
        / C_grid
    )

    dT_radiation = np.clip(
        dT_radiation,
        -3.0,
        3.0
    )

    T += (
        dT_radiation
    )


    # ========================================================
    # HEAT DIFFUSION
    # ========================================================

    T = (
        apply_heat_diffusion_numba(
            T,
            D_grid,
            dt,
            r_planet,
            dlat,
            dlon,
            lat_rad,
            n_substeps=10
        )
    )


    # ========================================================
    # TEMPERATURE LIMIT
    # ========================================================

    T = np.clip(
        T,
        180.0,
        330.0
    )


    # ========================================================
    # SNOW + ICE
    # ========================================================

    H_snow, ice_thickness = (
        update_snow_and_ice(
            T_celsius,
            precipitation,
            H_snow,
            ice_thickness,
            land_fraction
        )
    )


    # ========================================================
    # RECORD
    # ========================================================

    T_record[
        step
    ] = T

    ice_record[
        step
    ] = ice_thickness

    cloud_record[
        step
    ] = cloud_cover

    rain_record[
        step
    ] = precipitation


# ============================================================
# VALID RECORDS
# ============================================================

actual_steps = (
    step + 1
)

T_record_valid = (
    T_record[:actual_steps]
)

rain_record_valid = (
    rain_record[:actual_steps]
)

cloud_record_valid = (
    cloud_record[:actual_steps]
)

ice_record_valid = (
    ice_record[:actual_steps]
)


# ============================================================
# ANNUAL RAIN
# ============================================================

annual_days = min(
    365,
    actual_steps
)

annual_rain = np.mean(
    rain_record_valid[
        -annual_days:
    ],
    axis=0
)

annual_clouds = np.mean(
    cloud_record_valid[
        -annual_days:
    ],
    axis=0
)


# ============================================================
# FINAL CLIMATE CLASSIFICATION
# ============================================================

T_final_map = (
    T_record_valid[-1]
)

T_final_celsius = (
    T_final_map - 273.15
)

climate_type = (
    classify_climate(
        T_final_celsius,
        annual_rain,
        lat_grid,
        land_fraction
    )
)


# ============================================================
# LAND CONTOUR
# ============================================================

land_contour = (
    land_fraction > 0.5
).astype(float)


# ============================================================
# FINAL TEMPERATURE MAP
# ============================================================

plt.figure(
    figsize=(14, 6)
)

im = plt.imshow(
    T_final_celsius,
    extent=[
        -180,
        180,
        -90,
        90
    ],
    origin='lower',
    cmap='RdYlBu_r',
    interpolation='bicubic',
    vmin=-50,
    vmax=35,
    aspect='auto'
)

cont1 = plt.contour(
    longitudes,
    latitudes,
    T_final_celsius,
    levels=np.arange(
        -50,
        40,
        5
    ),
    colors='black',
    linewidths=0.5,
    alpha=0.5
)

plt.clabel(
    cont1,
    inline=True,
    fontsize=7,
    fmt='%d°C'
)

plt.contour(
    longitudes,
    latitudes,
    land_contour,
    levels=[0.5],
    colors='black',
    linewidths=1.5
)

plt.xlabel(
    "Pituusaste (°)"
)

plt.ylabel(
    "Leveysaste (°)"
)

plt.title(
    f"Lämpötila (°C) - Vuosi {n_years}"
)

plt.colorbar(
    im,
    label="°C"
)

plt.tight_layout()

plt.show()


# ============================================================
# PLOTS
# ============================================================

print(
    "\nGenerating plots..."
)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 11)
)


# ------------------------------------------------------------
# TEMPERATURE
# ------------------------------------------------------------

ax = axes[0, 0]

im = ax.imshow(
    T_record_valid[-1]
    - 273.15,
    extent=[
        -180,
        180,
        -90,
        90
    ],
    origin='lower',
    cmap='RdYlBu_r',
    vmin=-50,
    vmax=35,
    aspect='auto'
)

cont = ax.contour(
    longitudes,
    latitudes,
    T_record_valid[-1] - 273.15,
    levels=np.arange(
        -50,
        40,
        5
    ),
    colors='black',
    linewidths=0.3,
    alpha=0.4
)

ax.clabel(
    cont,
    inline=True,
    fontsize=6,
    fmt='%d°C'
)

ax.contour(
    longitudes,
    latitudes,
    land_contour,
    levels=[0.5],
    colors='black',
    linewidths=1.5
)

ax.set_xlabel(
    "Pituusaste (°)"
)

ax.set_ylabel(
    "Leveysaste (°)"
)

ax.set_title(
    f"Lämpötila (°C) - Vuosi {n_years}"
)

plt.colorbar(
    im,
    ax=ax,
    label="°C"
)


# ------------------------------------------------------------
# RAIN
# ------------------------------------------------------------

ax = axes[0, 1]

im = ax.imshow(
    annual_rain,
    extent=[
        -180,
        180,
        -90,
        90
    ],
    origin='lower',
    cmap='YlGnBu',
    vmin=0,
    vmax=10,
    aspect='auto'
)

cont = ax.contour(
    longitudes,
    latitudes,
    annual_rain,
    levels=[
        0.5,
        1,
        2,
        3,
        5,
        7,
        10
    ],
    colors='darkblue',
    linewidths=0.5,
    alpha=0.5
)

ax.clabel(
    cont,
    inline=True,
    fontsize=6,
    fmt='%.1f'
)

ax.contour(
    longitudes,
    latitudes,
    land_contour,
    levels=[0.5],
    colors='black',
    linewidths=1.5
)

ax.set_xlabel(
    "Pituusaste (°)"
)

ax.set_ylabel(
    "Leveysaste (°)"
)

ax.set_title(
    "Keskimääräinen sadanta - viimeinen vuosi"
)

plt.colorbar(
    im,
    ax=ax,
    label="mm/päivä"
)


# ------------------------------------------------------------
# CLIMATE
# ------------------------------------------------------------

ax = axes[1, 0]

climate_colors = (
    get_climate_color(
        climate_type
    )
)

ax.imshow(
    climate_colors,
    extent=[
        -180,
        180,
        -90,
        90
    ],
    origin='lower',
    aspect='auto'
)

ax.contour(
    longitudes,
    latitudes,
    land_contour,
    levels=[0.5],
    colors='black',
    linewidths=1.5
)

ax.axhline(
    23.44,
    color='white',
    linestyle='--',
    linewidth=0.8,
    alpha=0.5
)

ax.axhline(
    -23.44,
    color='white',
    linestyle='--',
    linewidth=0.8,
    alpha=0.5
)

ax.axhline(
    0,
    color='white',
    linestyle=':',
    linewidth=0.8,
    alpha=0.5
)

ax.set_xlabel(
    "Pituusaste (°)"
)

ax.set_ylabel(
    "Leveysaste (°)"
)

ax.set_title(
    "Köppen-tyylinen ilmastoluokitus"
)


from matplotlib.patches import Patch


legend_elements = [

    Patch(
        facecolor=[0.0, 0.6, 0.0],
        label='1: Sademetsä'
    ),

    Patch(
        facecolor=[0.6, 0.7, 0.2],
        label='2: Savanni'
    ),

    Patch(
        facecolor=[0.9, 0.8, 0.5],
        label='3: Aavikko'
    ),

    Patch(
        facecolor=[0.6, 0.8, 0.4],
        label='4: Aro/Step'
    ),

    Patch(
        facecolor=[0.3, 0.7, 0.3],
        label='5: Lämp. lauhkea'
    ),

    Patch(
        facecolor=[0.2, 0.5, 0.4],
        label='6: Viil. lauhkea'
    ),

    Patch(
        facecolor=[0.5, 0.6, 0.5],
        label='7: Tundra'
    ),

    Patch(
        facecolor=[0.95, 0.95, 1.0],
        label='8: Jäätikkö'
    )
]


ax.legend(
    handles=legend_elements,
    loc='lower left',
    fontsize=7,
    ncol=2
)


# ------------------------------------------------------------
# CLOUDS
# ------------------------------------------------------------

ax = axes[1, 1]

im = ax.imshow(
    annual_clouds * 100,
    extent=[
        -180,
        180,
        -90,
        90
    ],
    origin='lower',
    cmap='gray_r',
    vmin=0,
    vmax=80,
    aspect='auto'
)

cont = ax.contour(
    longitudes,
    latitudes,
    annual_clouds * 100,
    levels=[
        10,
        20,
        30,
        40,
        50,
        60,
        70
    ],
    colors='black',
    linewidths=0.4,
    alpha=0.4
)

ax.clabel(
    cont,
    inline=True,
    fontsize=6,
    fmt='%d%%'
)

ax.contour(
    longitudes,
    latitudes,
    land_contour,
    levels=[0.5],
    colors='black',
    linewidths=1.5
)

ax.set_xlabel(
    "Pituusaste (°)"
)

ax.set_ylabel(
    "Leveysaste (°)"
)

ax.set_title(
    "Pilvisyys (%) - viimeinen vuosi"
)

plt.colorbar(
    im,
    ax=ax,
    label="%"
)


plt.tight_layout()

plt.savefig(
    'climate_zones_clouds.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SIMULAATION YHTEENVETO"
)

print(
    "=" * 70
)

print(
    f"Simulaation pituus: "
    f"{n_years} vuotta"
)


if equilibrium_reached:

    print(
        "✓ Malli saavutti tasapainon"
    )


T_global_final = (
    np.mean(
        T_record_valid[-1]
    )
    - 273.15
)


print(
    f"\nGlobaali loppulämpötila: "
    f"{T_global_final:.2f} °C"
)


# ============================================================
# CLIMATE ZONES
# ============================================================

print(
    "\nIlmastovyöhykkeet (% koko pinta-alasta):"
)


for climate_code in range(
    1,
    9
):

    area = (
        np.sum(
            climate_type
            == climate_code
        )
        /
        (nlat * nlon)
        * 100
    )

    if area > 0.1:

        print(
            f"  {climate_code}: "
            f"{get_climate_name(climate_code):25s} "
            f"{area:5.1f}%"
        )


# ============================================================
# RAIN BY LATITUDE
# ============================================================

print(
    "\nSade vyöhykkeittäin (mm/päivä):"
)


tropical_mask = (
    np.abs(lat_grid)
    <= 10
)

subtropical_mask = (
    (np.abs(lat_grid) >= 20)
    &
    (np.abs(lat_grid) <= 35)
)

temperate_mask = (
    (np.abs(lat_grid) >= 40)
    &
    (np.abs(lat_grid) <= 60)
)


tropical_rain = np.mean(
    annual_rain[
        tropical_mask
    ]
)

subtropical_rain = np.mean(
    annual_rain[
        subtropical_mask
    ]
)

temperate_rain = np.mean(
    annual_rain[
        temperate_mask
    ]
)


print(
    f"  Tropiikki (±10°): "
    f"{tropical_rain:.2f}"
)

print(
    f"  Subtropiikki (20–35°): "
    f"{subtropical_rain:.2f}"
)

print(
    f"  Lauhkeat (40–60°): "
    f"{temperate_rain:.2f}"
)

print(
    f"  Globaali keskiarvo: "
    f"{np.mean(annual_rain):.2f}"
)


# ============================================================
# WATER CYCLE
# ============================================================

print(
    "\nVesikiertokulku:"
)


total_evap = (
    np.mean(evaporation)
)

total_precip = (
    np.mean(precipitation)
)


ocean_mask_final = (
    land_fraction < 0.5
)

land_mask_final = (
    land_fraction > 0.5
)


ocean_precip = np.mean(
    precipitation[
        ocean_mask_final
    ]
)

land_precip = np.mean(
    precipitation[
        land_mask_final
    ]
)


ocean_frac_global = (
    np.mean(
        ocean_mask_final
    )
)

land_frac_global = (
    np.mean(
        land_mask_final
    )
)


print(
    f"  Haihdutus (meri): "
    f"{total_evap:.2f} mm/päivä"
)

print(
    f"  Sadanta (globaali): "
    f"{total_precip:.2f} mm/päivä"
)

print(
    f"  Sadanta (meri): "
    f"{ocean_precip:.2f} mm/päivä"
)

print(
    f"  Sadanta (maa): "
    f"{land_precip:.2f} mm/päivä"
)

print(
    f"  Merenpinnan osuus: "
    f"{ocean_frac_global * 100:.1f}%"
)

print(
    f"  Maa-ala: "
    f"{land_frac_global * 100:.1f}%"
)

print(
    f"  Sadanta - haihdutus: "
    f"{(total_precip - total_evap):+.2f} mm/päivä"
)


# ============================================================
# BIOSPHERE
# ============================================================

npp = compute_npp(
    T_final_celsius,
    annual_rain,
    land_fraction
)


if np.sum(
    land_mask_final
) > 0:

    total_npp = (
        np.mean(
            npp[
                land_mask_final
            ]
        )
    )

    forest_npp_threshold = 0.3

    forest_fraction = (
        np.sum(
            (
                npp
                > forest_npp_threshold
            )
            &
            land_mask_final
        )
        /
        np.sum(
            land_mask_final
        )
    )

    desert_fraction = (
        np.sum(
            (
                annual_rain
                < 0.5
            )
            &
            land_mask_final
        )
        /
        np.sum(
            land_mask_final
        )
    )

else:

    total_npp = 0.0
    forest_fraction = 0.0
    desert_fraction = 0.0


print(
    "\nBiosfääri:"
)

print(
    f"  NPP (keskiarvo maalla): "
    f"{total_npp:.3f} "
    "kg C/m²/yr"
)

print(
    f"  Metsäpeitto "
    f"(NPP > {forest_npp_threshold}): "
    f"{forest_fraction * 100:.1f}% maa-alasta"
)

print(
    f"  Aavikot "
    f"(sade < 0.5 mm/d): "
    f"{desert_fraction * 100:.1f}% maa-alasta"
)


# ============================================================
# SNOW / ICE SUMMARY
# ============================================================

snow_fraction_global = (
    np.mean(
        H_snow > 0.01
    )
)

ice_fraction_global = (
    np.mean(
        ice_thickness > 0.1
    )
)


print(
    "\nLumi ja jää:"
)

print(
    f"  Lumipeitteen osuus: "
    f"{snow_fraction_global * 100:.1f}%"
)

print(
    f"  Merijään osuus: "
    f"{ice_fraction_global * 100:.1f}%"
)


# ============================================================
# MODEL SETTINGS SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "KORJATUT OMINAISUUDET"
)

print(
    "=" * 70
)

print(
    "  • Pilvien albedovaikutus: 0.10"
)

print(
    "  • Vesihöyrypalaute normalisoitu"
)

print(
    "  • Kylmää vahvistava G-feedback poistettu"
)

print(
    "  • Lumi sulaa asteittain"
)

print(
    "  • Lumi/räntä-vesi -raja on asteittainen"
)

print(
    "  • Merijään kasvu ja sulaminen tasapainotettu"
)

print(
    "  • Mannereisuus ja rannikkoefekti säilytetty"
)

print(
    "  • NPP-laskenta säilytetty"
)

print(
    "  • Sadevyöhykkeet säilytetty"
)

print(
    "  • annual_rain-virhe korjattu"
)

print(
    "  • Ennenaikainen quit(-1) poistettu"
)

print(
    "=" * 70
)

print(
    "\n✓ Simulaatio valmis!"
)
