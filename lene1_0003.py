
# ============================================================================
# 🌍 PHYSICAL PLANET CLIMATE GENERATOR
# ============================================================================
#
# YKSI TIEDOSTO - OSA I
#
# Tärkeimmät fysiikan parametrit:
#
#   mass_earths       = planeetan massa Maan massoina
#   radius_earths     = planeetan säde Maan säteinä
#   patm              = ilmanpaine Pa
#   lapse_rate        = ympäristön lämpötilagradientti K/km
#   ocean_threshold   = merenpinnan korkeus normalisoituna
#
# Tämä versio korjaa aikaisemman liian voimakkaan
# kasvihuone-emissiivisyysparametrin.
#
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

G = 6.67430e-11
STEFAN_BOLTZMANN = 5.670374419e-8

EARTH_MASS = 5.9722e24
EARTH_RADIUS = 6_371_000.0
EARTH_GRAVITY = 9.80665
EARTH_PATM = 101325.0

SOLAR_CONSTANT_EARTH = 1361.0


# ============================================================================
# CLIMATE REFERENCE VALUES
# ============================================================================

EARTH_REFERENCE_TEMP_C = 14.0

# Earth effective radiating temperature is about -18 C.
# Difference to surface temperature is about 32 C.
EARTH_GREENHOUSE_WARMING = 32.0


# Environmental lapse rate.
#
# IMPORTANT:
# This is NOT the dry adiabatic lapse rate (~9.8 K/km).
# For a simple Earth-like climate model we use ~6.5 K/km.
#
DEFAULT_LAPSE_RATE = 6.5


# ============================================================================
# SURFACE ALBEDOS
# ============================================================================

ALBEDO_OCEAN = 0.06
ALBEDO_LAND = 0.15
ALBEDO_DESERT = 0.35
ALBEDO_FOREST = 0.12

ALBEDO_ICE = 0.70
ALBEDO_SNOW = 0.80


# ============================================================================
# ATMOSPHERIC PARAMETERS
# ============================================================================

CO2_REFERENCE_PPM = 280.0
CO2_DEFAULT_PPM = 420.0

# Reference atmospheric pressure
REFERENCE_PATM = 101325.0


# ============================================================================
# PLANET PHYSICS
# ============================================================================

def calculate_gravity(
    mass_earths,
    radius_earths
):
    """
    Calculate planetary surface gravity.

        g = G M / R²

    Inputs:
        mass_earths  : mass in Earth masses
        radius_earths: radius in Earth radii

    Returns:
        gravity in m/s²
    """

    mass_kg = (
        mass_earths
        *
        EARTH_MASS
    )

    radius_m = (
        radius_earths
        *
        EARTH_RADIUS
    )

    if mass_kg <= 0:
        raise ValueError(
            "Planetary mass must be > 0."
        )

    if radius_m <= 0:
        raise ValueError(
            "Planetary radius must be > 0."
        )

    return (
        G
        *
        mass_kg
        /
        radius_m**2
    )


def calculate_lapse_rate(
    gravity,
    lapse_rate_override=None
):
    """
    Environmental lapse rate.

    For a simple planetary climate model:
        Earth default = 6.5 K/km

    Gravity scaling is intentionally weak.
    We do NOT use g/cp directly here because that would
    produce the dry adiabatic lapse rate (~9.8 K/km on Earth).

    """

    if lapse_rate_override is not None:

        return float(
            lapse_rate_override
        )

    gravity_ratio = (
        gravity
        /
        EARTH_GRAVITY
    )

    # Weak gravity dependence.
    #
    # Earth:
    # 9.80665 -> 6.5 K/km
    #
    lapse_rate = (
        DEFAULT_LAPSE_RATE
        *
        gravity_ratio**0.25
    )

    return float(
        np.clip(
            lapse_rate,
            2.0,
            12.0
        )
    )


# ============================================================================
# GREENHOUSE MODEL
# ============================================================================

def calculate_greenhouse_warming(
    temperature_c,
    patm,
    co2_ppm,
    relative_humidity=0.70
):
    """
    Approximate greenhouse warming in degrees C.

    This is intentionally a simple climate-generator parameterization,
    not a line-by-line radiative-transfer model.

    Components:

        pressure effect
        CO2 effect
        water vapor feedback

    Earth-like reference:
        ~32 C greenhouse warming.
    """

    # ------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------

    pressure_ratio = (
        patm
        /
        REFERENCE_PATM
    )

    pressure_ratio = np.clip(
        pressure_ratio,
        0.05,
        20.0
    )

    # Pressure contribution.
    #
    # At Earth pressure -> 1.0
    #
    pressure_factor = (
        pressure_ratio**0.18
    )


    # ------------------------------------------------------------
    # CO2
    # ------------------------------------------------------------

    co2_ratio = max(
        co2_ppm
        /
        CO2_REFERENCE_PPM,
        0.01
    )

    # Logarithmic CO2 forcing.
    #
    # Relative to 280 ppm.
    #
    co2_factor = (
        1.0
        +
        0.12
        *
        np.log2(
            co2_ratio
        )
    )

    co2_factor = np.clip(
        co2_factor,
        0.65,
        1.60
    )


    # ------------------------------------------------------------
    # Water vapour
    # ------------------------------------------------------------

    # Warmer atmosphere -> more water vapour.
    #
    # This is a deliberately soft feedback.
    #
    water_temperature_factor = (
        1.0
        +
        0.015
        *
        max(
            temperature_c
            -
            14.0,
            -30.0
        )
    )

    water_temperature_factor = np.clip(
        water_temperature_factor,
        0.50,
        2.0
    )

    humidity_factor = (
        0.70
        +
        0.30
        *
        np.clip(
            relative_humidity,
            0.0,
            1.0
        )
    )

    water_factor = (
        water_temperature_factor
        *
        humidity_factor
    )

    water_factor = np.clip(
        water_factor,
        0.40,
        2.20
    )


    # ------------------------------------------------------------
    # Total
    # ------------------------------------------------------------

    greenhouse = (
        EARTH_GREENHOUSE_WARMING
        *
        pressure_factor
        *
        co2_factor
        *
        water_factor
        /
        (
            1.0
            +
            0.35
            *
            (
                water_factor
                    -
                1.0
            )
        )
    )

    return float(
        np.clip(
            greenhouse,
            5.0,
            90.0
        )
    )


# ============================================================================
# SIMPLE EFFECTIVE RADIATING TEMPERATURE
# ============================================================================

def effective_radiating_temperature(
    absorbed_solar
):
    """
    Stefan-Boltzmann effective radiating temperature.

        absorbed solar = sigma T_eff^4

    This is the temperature Earth would have without
    the surface greenhouse enhancement.
    """

    if absorbed_solar <= 0:

        return 150.0

    return (
        absorbed_solar
        /
        STEFAN_BOLTZMANN
    ) ** 0.25


# ============================================================================
# ATMOSPHERIC SCALE HEIGHT
# ============================================================================

def calculate_scale_height(
    temperature_k,
    gravity
):

    R_SPECIFIC_AIR = 287.05

    return (
        R_SPECIFIC_AIR
        *
        temperature_k
        /
        max(
            gravity,
            1e-6
        )
    )


# ============================================================================
# NOISE FUNCTIONS
# ============================================================================

@jit(nopython=True)
def fade(t):

    return (
        t
        *
        t
        *
        t
        *
        (
            t
            *
            (
                t * 6.0
                -
                15.0
            )
            +
            10.0
        )
    )


@jit(nopython=True)
def lerp(
    t,
    a,
    b
):

    return (
        a
        +
        t
        *
        (
            b
            -
            a
        )
    )


@jit(nopython=True)
def grad(
    hash_val,
    x,
    y,
    z
):

    h = hash_val & 15

    u = (
        x
        if h < 8
        else y
    )

    v = (
        y
        if h < 4
        else (
            x
            if h == 12
            or h == 14
            else z
        )
    )

    return (
        (
            u
            if (h & 1) == 0
            else -u
        )
        +
        (
            v
            if (h & 2) == 0
            else -v
        )
    )


@jit(nopython=True)
def perlin_noise_3d_at_point(
    x,
    y,
    z,
    scale,
    p
):

    x_coord = x * scale
    y_coord = y * scale
    z_coord = z * scale

    X = (
        int(np.floor(x_coord))
        &
        255
    )

    Y = (
        int(np.floor(y_coord))
        &
        255
    )

    Z = (
        int(np.floor(z_coord))
        &
        255
    )

    x_coord -= np.floor(
        x_coord
    )

    y_coord -= np.floor(
        y_coord
    )

    z_coord -= np.floor(
        z_coord
    )

    u = fade(x_coord)
    v = fade(y_coord)
    w = fade(z_coord)

    A = p[X] + Y

    AA = p[A] + Z
    AB = p[A + 1] + Z

    B = p[X + 1] + Y

    BA = p[B] + Z
    BB = p[B + 1] + Z

    return lerp(
        w,

        lerp(
            v,

            lerp(
                u,

                grad(
                    p[AA],
                    x_coord,
                    y_coord,
                    z_coord
                ),

                grad(
                    p[BA],
                    x_coord - 1.0,
                    y_coord,
                    z_coord
                )
            ),

            lerp(
                u,

                grad(
                    p[AB],
                    x_coord,
                    y_coord - 1.0,
                    z_coord
                ),

                grad(
                    p[BB],
                    x_coord - 1.0,
                    y_coord - 1.0,
                    z_coord
                )
            )
        ),

        lerp(
            v,

            lerp(
                u,

                grad(
                    p[AA + 1],
                    x_coord,
                    y_coord,
                    z_coord - 1.0
                ),

                grad(
                    p[BA + 1],
                    x_coord - 1.0,
                    y_coord,
                    z_coord - 1.0
                )
            ),

            lerp(
                u,

                grad(
                    p[AB + 1],
                    x_coord,
                    y_coord - 1.0,
                    z_coord - 1.0
                ),

                grad(
                    p[BB + 1],
                    x_coord - 1.0,
                    y_coord - 1.0,
                    z_coord - 1.0
                )
            )
        )
    )


@jit(nopython=True)
def multifractal_at_point(
    x,
    y,
    z,
    scale,
    octaves,
    persistence,
    lacunarity,
    p
):

    noise_value = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):

        octave_noise = (
            perlin_noise_3d_at_point(
                x,
                y,
                z,
                scale * frequency,
                p
            )
        )

        noise_value += (
            amplitude
            *
            octave_noise
        )

        max_amplitude += amplitude

        frequency *= lacunarity
        amplitude *= persistence

    if max_amplitude > 0:

        noise_value /= (
            max_amplitude
        )

    return noise_value


@jit(nopython=True)
def ridge_noise_at_point(
    x,
    y,
    z,
    scale,
    octaves,
    persistence,
    lacunarity,
    p
):

    noise_value = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):

        octave_noise = (
            perlin_noise_3d_at_point(
                x,
                y,
                z,
                scale * frequency,
                p
            )
        )

        ridge_noise = (
            1.0
            -
            abs(octave_noise)
        )

        noise_value += (
            amplitude
            *
            ridge_noise
        )

        max_amplitude += amplitude

        frequency *= lacunarity
        amplitude *= persistence

    if max_amplitude > 0:

        noise_value /= (
            max_amplitude
        )

    return noise_value


# ============================================================================
# ORBITAL CALCULATIONS
# ============================================================================

@jit(nopython=True)
def deg2rad(d):

    return (
        d
        *
        np.pi
        /
        180.0
    )


@jit(nopython=True)
def orbital_distance_factor(
    e,
    true_longitude
):

    denom = (
        1.0
        +
        e
        *
        np.cos(
            true_longitude
        )
    )

    if denom <= 1e-6:

        denom = 1e-6

    return (
        1.0
        /
        (
            denom
            *
            denom
        )
    )


@jit(nopython=True)
def solar_declination(
    obliquity_rad,
    true_longitude
):

    return np.arcsin(
        np.sin(
            obliquity_rad
        )
        *
        np.sin(
            true_longitude
        )
    )


@jit(nopython=True)
def month_true_longitude(
    month_index,
    mvelp_rad
):

    return (
        2.0
        *
        np.pi
        *
        (
            month_index
            /
            12.0
        )
        +
        mvelp_rad
    )


@jit(nopython=True)
def daily_insolation(
    latitude,
    declination,
    solar_constant,
    dist_factor
):

    lat = latitude
    dec = declination

    cos_hour_angle = (
        -np.tan(lat)
        *
        np.tan(dec)
    )

    if cos_hour_angle <= -1.0:

        hour_angle = np.pi

    elif cos_hour_angle >= 1.0:

        return 0.0

    else:

        hour_angle = np.arccos(
            cos_hour_angle
        )

    insolation = (
        solar_constant
        *
        dist_factor
        /
        np.pi
    ) * (
        hour_angle
        *
        np.sin(lat)
        *
        np.sin(dec)
        +
        np.cos(lat)
        *
        np.cos(dec)
        *
        np.sin(hour_angle)
    )

    return max(
        0.0,
        insolation
    )


# ============================================================================
# TERRAIN
# ============================================================================

@jit(nopython=True, parallel=True)
def create_elevation_map(
    width,
    height,
    radius_noise,
    p
):

    elevation_map = np.zeros(
        (height, width)
    )

    center = width // 2

    for i in prange(height):

        for j in range(width):

            theta = (
                2.0
                *
                np.pi
                *
                j
                /
                width
            )

            phi = (
                np.pi
                *
                (
                    i / height
                    -
                    0.5
                )
            )

            x = (
                center
                +
                radius_noise
                *
                np.cos(phi)
                *
                np.cos(theta)
            )

            y = (
                center
                +
                radius_noise
                *
                np.cos(phi)
                *
                np.sin(theta)
            )

            z = (
                center
                +
                radius_noise
                *
                np.sin(phi)
            )

            continental_noise = (
                multifractal_at_point(
                    x,
                    y,
                    z,
                    0.002,
                    8,
                    0.5,
                    2.0,
                    p
                )
            )

            mountain_noise = (
                ridge_noise_at_point(
                    x + 1000.0,
                    y + 1000.0,
                    z + 1000.0,
                    0.005,
                    6,
                    0.6,
                    2.2,
                    p
                )
            )

            elevation_map[
                i, j
            ] = (
                continental_noise * 0.7
                +
                mountain_noise * 0.3
            )

    return elevation_map


# ============================================================================
# HYPSOMETRIC CURVE
# ============================================================================

def apply_hypsometric_curve(
    elevation,
    ocean_threshold
):

    h, w = elevation.shape

    transformed = np.zeros_like(
        elevation
    )

    for i in range(h):

        for j in range(w):

            elev = elevation[i, j]

            if elev < ocean_threshold:

                depth = (
                    elev
                    /
                    ocean_threshold
                )

                if depth > 0.8:

                    transformed[
                        i, j
                    ] = (
                        ocean_threshold
                        -
                        0.05
                        *
                        (
                            1.0
                            -
                            depth
                        )
                        /
                        0.2
                    )

                else:

                    curve = (
                        1.0
                        -
                        (
                            1.0
                            -
                            depth / 0.8
                        ) ** 2
                    )

                    transformed[
                        i, j
                    ] = (
                        (
                            ocean_threshold
                            -
                            0.05
                        )
                        *
                        curve
                    )

            else:

                land_height = (
                    elev
                    -
                    ocean_threshold
                ) / (
                    1.0
                    -
                    ocean_threshold
                )

                exp_height = (
                    np.exp(
                        2.5
                        *
                        land_height
                    )
                    -
                    1.0
                ) / (
                    np.exp(2.5)
                    -
                    1.0
                )

                transformed[
                    i, j
                ] = (
                    ocean_threshold
                    +
                    exp_height
                    *
                    (
                        1.0
                        -
                        ocean_threshold
                    )
                )

    return transformed


# ============================================================================
# SURFACE HEIGHT STATISTICS
# ============================================================================

def calculate_surface_height_statistics(
    elevation_norm,
    ocean_threshold,
    max_land_height_m=8000.0,
    max_ocean_depth_m=6000.0
):

    land_mask = (
        elevation_norm
        >
        ocean_threshold
    )

    ocean_mask = ~land_mask

    if np.any(land_mask):

        land_relative = (
            elevation_norm[
                land_mask
            ]
            -
            ocean_threshold
        ) / (
            1.0
            -
            ocean_threshold
        )

        land_height = (
            land_relative
            *
            max_land_height_m
        )

        mean_land_height = (
            np.mean(
                land_height
            )
        )

        median_land_height = (
            np.median(
                land_height
            )
        )

        maximum_land_height = (
            np.max(
                land_height
            )
        )

    else:

        mean_land_height = 0.0
        median_land_height = 0.0
        maximum_land_height = 0.0


    if np.any(ocean_mask):

        ocean_relative = (
            elevation_norm[
                ocean_mask
            ]
            /
            ocean_threshold
        )

        ocean_depth = (
            1.0
            -
            ocean_relative
        ) * max_ocean_depth_m

        mean_ocean_depth = (
            np.mean(
                ocean_depth
            )
        )

    else:

        mean_ocean_depth = 0.0


    return {
        "mean_land_height_m":
            mean_land_height,

        "median_land_height_m":
            median_land_height,

        "max_land_height_m":
            maximum_land_height,

        "mean_ocean_depth_m":
            mean_ocean_depth,

        "land_fraction":
            np.mean(
                land_mask
            ),

        "ocean_fraction":
            np.mean(
                ocean_mask
            )
    }


# ============================================================================
# LATITUDE GRID
# ============================================================================

def create_latitude_grid(
    height,
    width
):

    phis = np.zeros(
        (height, width)
    )

    for i in range(height):

        phi = (
            np.pi
            *
            (
                i / height
                -
                0.5
            )
        )

        phis[i, :] = phi

    return phis



# ============================================================================
# ALBEDO
# ============================================================================

def calculate_albedo(
    elevation_norm,
    temperature_celsius,
    ocean_threshold,
    cloud_fraction=0.60
):
    """
    Surface albedo.

    Pilvet eivät korvaa pinnan albedoa, vaan niiden vaikutus lisätään
    myöhemmin energiabudjetissa.
    """

    h, w = elevation_norm.shape

    albedo = np.zeros(
        (h, w),
        dtype=np.float64
    )

    ocean_mask = (
        elevation_norm
        <
        ocean_threshold
    )

    land_mask = ~ocean_mask

    # ----------------------------------------------------------------
    # Ocean
    # ----------------------------------------------------------------

    ocean_temp = temperature_celsius[
        ocean_mask
    ]

    ocean_albedo = np.full(
        ocean_temp.shape,
        ALBEDO_OCEAN
    )

    sea_ice = (
        ocean_temp
        <
        -1.8
    )

    ocean_albedo[
        sea_ice
    ] = ALBEDO_ICE

    albedo[
        ocean_mask
    ] = ocean_albedo


    # ----------------------------------------------------------------
    # Land
    # ----------------------------------------------------------------

    land_temp = temperature_celsius[
        land_mask
    ]

    land_albedo = np.full(
        land_temp.shape,
        ALBEDO_LAND
    )

    cold = (
        land_temp
        <
        -5.0
    )

    snow = (
        land_temp
        >=
        -5.0
    ) & (
        land_temp
        <
        5.0
    )

    warm = (
        land_temp
        >=
        5.0
    )


    land_albedo[
        cold
    ] = ALBEDO_SNOW


    if np.any(snow):

        snow_fraction = (
            5.0
            -
            land_temp[snow]
        ) / 10.0

        snow_fraction = np.clip(
            snow_fraction,
            0.0,
            1.0
        )

        land_albedo[
            snow
        ] = (
            snow_fraction
            *
            ALBEDO_SNOW
            +
            (
                1.0
                -
                snow_fraction
            )
            *
            ALBEDO_LAND
        )


    # Warm land:
    # modest vegetation/desert approximation.

    if np.any(warm):

        # Very warm and dry surfaces become somewhat brighter.
        desert_fraction = np.clip(
            (
                land_temp[warm]
                -
                20.0
            ) / 35.0,
            0.0,
            1.0
        )

        land_albedo[
            warm
        ] = (
            (
                1.0
                -
                desert_fraction
            )
            *
            ALBEDO_FOREST
            +
            desert_fraction
            *
            ALBEDO_DESERT
        )

    albedo[
        land_mask
    ] = land_albedo

    return np.clip(
        albedo,
        0.02,
        0.90
    )


# ============================================================================
# CLOUD MODEL
# ============================================================================

def calculate_cloud_fraction(
    temperature_celsius,
    precipitation_mm_year=None,
    ocean_mask=None,
    humidity=0.70
):
    """
    Simple cloud climatology.

    Clouds increase with:
      - humidity
      - warm/moist atmosphere
      - ocean fraction
      - precipitation

    But cloud fraction is limited to a realistic 0...0.90 range.
    """

    temp = temperature_celsius

    # Base cloud amount
    clouds = np.full(
        temp.shape,
        0.50
        +
        0.15
        *
        (
            humidity
            -
            0.70
        )
    )

    # Warm moist atmosphere
    clouds += (
        0.003
        *
        np.clip(
            temp,
            -30.0,
            40.0
        )
    )

    if precipitation_mm_year is not None:

        rain_factor = np.clip(
            precipitation_mm_year
            /
            2000.0,
            0.0,
            2.0
        )

        clouds += (
            0.06
            *
            (
                rain_factor
                -
                0.5
            )
        )

    if ocean_mask is not None:

        clouds[
            ocean_mask
        ] += 0.05

    return np.clip(
        clouds,
        0.20,
        0.90
    )


# ============================================================================
# TEMPERATURE CALCULATION
# ============================================================================

def calculate_temperature_field(
    insolation_monthly,
    albedo,
    elevation_norm,
    ocean_threshold,
    gravity,
    patm,
    co2_ppm,
    lapse_rate,
    cloud_fraction,
    ocean_reference_depth_m=3000.0
):
    """
    Calculate monthly surface temperature.

    Energy chain:

        solar radiation
             ↓
        surface albedo
             ↓
        cloud reflection
             ↓
        effective radiating temperature
             ↓
        greenhouse warming
             ↓
        elevation lapse-rate correction

    This is a climate-generator model, not a GCM.
    """

    months, h, w = (
        insolation_monthly.shape
    )

    temperature_monthly = np.zeros(
        (months, h, w),
        dtype=np.float64
    )


    ocean_mask = (
        elevation_norm
        <
        ocean_threshold
    )

    land_mask = ~ocean_mask


    # ----------------------------------------------------------------
    # Global ocean thermal inertia
    # ----------------------------------------------------------------

    # Deep ocean changes temperature more slowly.
    #
    # We implement this as a modest damping toward an ocean mean.
    #
    ocean_inertia = np.clip(
        ocean_reference_depth_m
        /
        4000.0,
        0.20,
        1.00
    )


    for m in range(months):

        S = (
            insolation_monthly[m]
        )

        # Surface absorbed SW
        surface_absorbed = (
            S
            *
            (
                1.0
                -
                albedo
            )
        )


        # ------------------------------------------------------------
        # Clouds
        # ------------------------------------------------------------

        # Clouds reflect a portion of incoming shortwave.
        #
        # Effective cloud optical effect is deliberately modest.
        #
        cloud_reflection = (
            0.25
            *
            cloud_fraction
        )

        absorbed = (
            surface_absorbed
            *
            (
                1.0
                -
                cloud_reflection
            )
        )


        # ------------------------------------------------------------
        # Effective radiating temperature
        # ------------------------------------------------------------

        Teff = np.zeros_like(
            absorbed
        )

        positive = (
            absorbed
            >
            0.0
        )

        Teff[
            positive
        ] = (
            absorbed[
                positive
            ]
            /
            STEFAN_BOLTZMANN
        ) ** 0.25

        Teff[
            ~positive
        ] = 180.0


        Teff_c = (
            Teff
            -
            273.15
        )


        # ------------------------------------------------------------
        # Greenhouse warming
        # ------------------------------------------------------------

        greenhouse = np.zeros_like(
            Teff_c
        )

        for i in range(h):

            # Humidity is slightly latitude dependent.
            latitude = (
                -90.0
                +
                180.0
                *
                (
                    i
                    /
                    max(
                        h - 1,
                        1
                    )
                )
            )

            lat_abs = abs(
                latitude
            )

            local_humidity = (
                0.82
                -
                0.0025
                *
                lat_abs
            )

            local_humidity = np.clip(
                local_humidity,
                0.35,
                0.85
            )

            for j in range(w):

                greenhouse[
                    i, j
                ] = calculate_greenhouse_warming(
                    Teff_c[i, j],
                    patm,
                    co2_ppm,
                    local_humidity
                )


        surface_temp = (
            Teff_c
            +
            greenhouse
        )


        # ------------------------------------------------------------
        # Elevation correction
        # ------------------------------------------------------------

        height_fraction = np.zeros_like(
            elevation_norm
        )

        height_fraction[
            land_mask
        ] = (
            elevation_norm[
                land_mask
            ]
            -
            ocean_threshold
        ) / (
            1.0
            -
            ocean_threshold
        )

        # Actual maximum terrain height = 8 km
        height_km = (
            height_fraction
            *
            8.0
        )

        cooling = (
            height_km
            *
            lapse_rate
        )

        surface_temp[
            land_mask
        ] -= cooling[
            land_mask
        ]


        # ------------------------------------------------------------
        # Ocean thermal damping
        # ------------------------------------------------------------

        if np.any(ocean_mask):

            ocean_mean = np.mean(
                surface_temp[
                    ocean_mask
                ]
            )

            # Damp local ocean extremes.
            surface_temp[
                ocean_mask
            ] = (
                ocean_mean
                *
                (
                    1.0
                    -
                    0.15
                    *
                    ocean_inertia
                )
                +
                surface_temp[
                    ocean_mask
                ]
                *
                (
                    0.15
                    *
                    ocean_inertia
                )
            )


        temperature_monthly[
            m
        ] = surface_temp


    return temperature_monthly


# ============================================================================
# PRECIPITATION
# ============================================================================

def calculate_precipitation(
    temperature_celsius,
    elevation_norm,
    phis,
    ocean_threshold,
    patm,
    gravity,
    cloud_fraction
):
    """
    Approximate global precipitation.

    Result:
        annual precipitation in mm/year

    Unlike the old version, there is no artificial
    "1000 mm/year everywhere" floor.
    """

    months, h, w = (
        temperature_celsius.shape
    )

    precip_monthly = np.zeros(
        (months, h, w),
        dtype=np.float64
    )

    ocean_mask = (
        elevation_norm
        <
        ocean_threshold
    )


    # Atmospheric pressure influence.
    pressure_factor = np.clip(
        (
            patm
            /
            EARTH_PATM
        ) ** 0.25,
        0.30,
        2.50
    )


    # Gravity influence:
    # stronger gravity -> somewhat smaller atmospheric
    # moisture column.
    gravity_factor = np.clip(
        (
            EARTH_GRAVITY
            /
            max(
                gravity,
                0.1
            )
        ) ** 0.20,
        0.60,
        1.50
    )


    for m in range(months):

        for i in range(h):

            phi = phis[i, 0]

            phi_deg = (
                abs(
                    phi
                    *
                    180.0
                    /
                    np.pi
                )
            )


            # --------------------------------------------------------
            # Circulation
            # --------------------------------------------------------

            if phi_deg < 10.0:

                circulation = 2.0

            elif phi_deg < 30.0:

                circulation = 0.45

            elif phi_deg < 60.0:

                circulation = 1.15

            else:

                circulation = 0.40


            for j in range(w):

                temp = (
                    temperature_celsius[
                        m, i, j
                    ]
                )

                elev = (
                    elevation_norm[
                        i, j
                    ]
                )


                # ----------------------------------------------------
                # Saturation vapor pressure
                # ----------------------------------------------------

                if temp > -40.0:

                    denominator = (
                        temp
                        +
                        237.3
                    )

                    if abs(
                        denominator
                    ) < 1e-3:

                        denominator = 1e-3

                    es = (
                        6.11
                        *
                        np.exp(
                            17.27
                            *
                            temp
                            /
                            denominator
                        )
                    )

                else:

                    es = 0.1


                # Convert hPa to rough moisture scale.
                #
                # This is deliberately normalized to produce
                # Earth-like global rainfall rather than
                # tens of thousands of mm/year.
                #
                moisture = (
                    es
                    *
                    0.35
                )


                # Cold air precipitation efficiency
                if temp < 0.0:

                    cold_factor = np.clip(
                        (
                            temp
                            +
                            40.0
                        )
                        /
                        40.0,
                        0.05,
                        1.0
                    )

                else:

                    cold_factor = 1.0


                # ----------------------------------------------------
                # Ocean moisture source
                # ----------------------------------------------------

                if elev < ocean_threshold:

                    ocean_factor = 1.35

                else:

                    # Approximate distance from sea using
                    # topographic position.
                    land_distance = (
                        elev
                        -
                        ocean_threshold
                    ) / (
                        1.0
                        -
                        ocean_threshold
                    )

                    ocean_factor = (
                        1.0
                        /
                        (
                            1.0
                            +
                            2.5
                            *
                            land_distance
                        )
                    )


                # ----------------------------------------------------
                # Clouds
                # ----------------------------------------------------

                cloud_factor = (
                    0.65
                    +
                    0.70
                    *
                    cloud_fraction[
                        i, j
                    ]
                )


                # ----------------------------------------------------
                # Orography
                # ----------------------------------------------------

                if (
                    elev > ocean_threshold
                    and i > 0
                ):

                    slope = (
                        elevation_norm[
                            i, j
                        ]
                        -
                        elevation_norm[
                            i - 1, j
                        ]
                    )

                    oro_factor = (
                        1.0
                        +
                        np.clip(
                            slope
                            *
                            15.0,
                            -0.25,
                            2.0
                        )
                    )

                else:

                    oro_factor = 1.0


                # ----------------------------------------------------
                # Monthly rainfall
                # ----------------------------------------------------

                precip = (
                    moisture
                    *
                    circulation
                    *
                    ocean_factor
                    *
                    cloud_factor
                    *
                    cold_factor
                    *
                    oro_factor
                    *
                    pressure_factor
                    *
                    gravity_factor
                    *
                    0.055
                )


                precip_monthly[
                    m, i, j
                ] = max(
                    0.0,
                    precip
                )


    precip_annual = np.sum(
        precip_monthly,
        axis=0
    )


    return (
        precip_annual,
        precip_monthly
    )


# ============================================================================
# ICE FRACTION
# ============================================================================

def calculate_ice_fraction(
    temperature_annual,
    elevation_norm,
    ocean_threshold
):

    ocean_mask = (
        elevation_norm
        <
        ocean_threshold
    )

    land_mask = ~ocean_mask


    ice = np.zeros(
        temperature_annual.shape
    )


    # Ocean ice
    ocean_temp = (
        temperature_annual[
            ocean_mask
        ]
    )

    ocean_ice = np.clip(
        (
            -ocean_temp
            -
            1.8
        )
        /
        8.0,
        0.0,
        1.0
    )

    ice[
        ocean_mask
    ] = ocean_ice


    # Land snow/ice
    land_temp = (
        temperature_annual[
            land_mask
        ]
    )

    land_ice = np.clip(
        (
            5.0
            -
            land_temp
        )
        /
        15.0,
        0.0,
        1.0
    )

    ice[
        land_mask
    ] = land_ice


    return ice


# ============================================================================
# ENERGY BUDGET
# ============================================================================

def calculate_energy_budget(
    insolation_monthly,
    albedo,
    cloud_fraction
):

    mean_insolation = np.mean(
        insolation_monthly,
        axis=0
    )

    mean_albedo = np.mean(
        albedo
    )

    mean_clouds = np.mean(
        cloud_fraction
    )


    surface_absorbed = (
        mean_insolation
        *
        (
            1.0
            -
            albedo
        )
        *
        (
            1.0
            -
            0.25
            *
            cloud_fraction
        )
    )


    absorbed_global = np.mean(
        surface_absorbed
    )


    return {
        "mean_toa":
            mean_insolation.mean(),

        "mean_albedo":
            mean_albedo,

        "mean_clouds":
            mean_clouds,

        "absorbed_solar":
            absorbed_global
    }


# ============================================================================
# MAIN PLANET GENERATOR
# ============================================================================

class PhysicalPlanetGenerator:

    def __init__(
        self,
        width=360,
        height=180,
        seed=42,
        solar_constant=1361.0,
        mass_earths=1.0,
        radius_earths=1.0,
        patm=101325.0,
        lapse_rate=None,
        co2_ppm=420.0
    ):

        self.width = width
        self.height = height

        self.seed = seed

        self.solar_constant = (
            solar_constant
        )

        self.mass_earths = (
            mass_earths
        )

        self.radius_earths = (
            radius_earths
        )

        self.patm = (
            patm
        )

        self.co2_ppm = (
            co2_ppm
        )


        # ------------------------------------------------------------
        # Planet gravity
        # ------------------------------------------------------------

        self.gravity = (
            calculate_gravity(
                mass_earths,
                radius_earths
            )
        )


        # ------------------------------------------------------------
        # Lapse rate
        # ------------------------------------------------------------

        self.lapse_rate = (
            calculate_lapse_rate(
                self.gravity,
                lapse_rate
            )
        )


        np.random.seed(
            self.seed
        )

        self.p = (
            self._create_permutation_table()
        )


    def _create_permutation_table(
        self
    ):

        p = np.arange(
            256,
            dtype=np.int32
        )

        np.random.shuffle(
            p
        )

        return np.tile(
            p,
            2
        )


    def generate_planet(
        self,
        ocean_threshold=0.75,
        obliquity_deg=23.5,
        eccentricity=0.017,
        mvelp_deg=102.0,
        simulation_years=10,
        radius_noise=200,
        hypsometric=True,
        max_land_height_m=8000.0,
        max_ocean_depth_m=6000.0
    ):

        print()
        print("=" * 72)
        print(
            "🌍 PHYSICAL PLANET CLIMATE GENERATOR"
        )
        print("=" * 72)
        print()

        print(
            f"Mass:           "
            f"{self.mass_earths:.3f} M⊕"
        )

        print(
            f"Radius:         "
            f"{self.radius_earths:.3f} R⊕"
        )

        print(
            f"Radius:         "
            f"{self.radius_earths * EARTH_RADIUS / 1000:,.0f} km"
        )

        print(
            f"Gravity:        "
            f"{self.gravity:.4f} m/s²"
        )

        print(
            f"Atmospheric P:  "
            f"{self.patm:,.0f} Pa"
        )

        print(
            f"Atmospheric P:  "
            f"{self.patm / 100000:.3f} bar"
        )

        print(
            f"Lapse rate:     "
            f"{self.lapse_rate:.3f} K/km"
        )

        print(
            f"CO₂:            "
            f"{self.co2_ppm:.0f} ppm"
        )

        print()

        print(
            f"Solar constant: "
            f"{self.solar_constant:.1f} W/m²"
        )

        print(
            f"Obliquity:      "
            f"{obliquity_deg:.2f}°"
        )

        print(
            f"Eccentricity:   "
            f"{eccentricity:.4f}"
        )

        print(
            f"Ocean threshold:"
            f"{ocean_threshold:.3f}"
        )

        print(
            f"Resolution:     "
            f"{self.width} x {self.height}"
        )

        print(
            f"Simulation:     "
            f"{simulation_years} years"
        )

        start_time = time.time()


        # ============================================================
        # TERRAIN
        # ============================================================

        print()
        print(
            "Generating terrain..."
        )

        elevation_raw = (
            create_elevation_map(
                self.width,
                self.height,
                radius_noise,
                self.p
            )
        )


        elevation_min = (
            np.min(
                elevation_raw
            )
        )

        elevation_max = (
            np.max(
                elevation_raw
            )
        )


        elevation_norm = (
            elevation_raw
            -
            elevation_min
        ) / max(
            elevation_max
            -
            elevation_min,
            1e-12
        )


        if hypsometric:

            elevation_norm = (
                apply_hypsometric_curve(
                    elevation_norm,
                    ocean_threshold
                )
            )


        surface_stats = (
            calculate_surface_height_statistics(
                elevation_norm,
                ocean_threshold,
                max_land_height_m,
                max_ocean_depth_m
            )
        )


        print()
        print(
            f"Ocean coverage: "
            f"{surface_stats['ocean_fraction'] * 100:.1f}%"
        )

        print(
            f"Land coverage:  "
            f"{surface_stats['land_fraction'] * 100:.1f}%"
        )

        print()

        print(
            "SURFACE HEIGHT"
        )

        print(
            f"Mean land height above sea level:"
            f" {surface_stats['mean_land_height_m']:,.0f} m"
        )

        print(
            f"Median land height:"
            f"                {surface_stats['median_land_height_m']:,.0f} m"
        )

        print(
            f"Maximum land height:"
            f"                {surface_stats['max_land_height_m']:,.0f} m"
        )

        print(
            f"Mean ocean depth:"
            f"                   {surface_stats['mean_ocean_depth_m']:,.0f} m"
        )


        # ============================================================
        # LATITUDE
        # ============================================================

        phis = (
            create_latitude_grid(
                self.height,
                self.width
            )
        )


        # ============================================================
        # INSOLATION
        # ============================================================

        print()
        print(
            "Calculating orbital insolation..."
        )


        obliquity_rad = (
            deg2rad(
                obliquity_deg
            )
        )

        mvelp_rad = (
            deg2rad(
                mvelp_deg
            )
        )


        insolation_monthly = np.zeros(
            (
                12,
                self.height,
                self.width
            ),
            dtype=np.float64
        )


        for m in range(12):

            lambda_sol = (
                month_true_longitude(
                    m,
                    mvelp_rad
                )
            )

            delta = (
                solar_declination(
                    obliquity_rad,
                    lambda_sol
                )
            )

            dist_factor = (
                orbital_distance_factor(
                    eccentricity,
                    lambda_sol
                )
            )


            for i in range(
                self.height
            ):

                lat = (
                    phis[
                        i, 0
                    ]
                )

                insol = (
                    daily_insolation(
                        lat,
                        delta,
                        self.solar_constant,
                        dist_factor
                    )
                )

                insolation_monthly[
                    m, i, :
                ] = insol


        # ============================================================
        # INITIAL CLIMATE
        # ============================================================

        print()
        print(
            "Creating initial climate..."
        )


        ocean_mask = (
            elevation_norm
            <
            ocean_threshold
        )


        # Initial Earth-like surface albedo
        albedo = np.full(
            (
                self.height,
                self.width
            ),
            ALBEDO_LAND,
            dtype=np.float64
        )

        albedo[
            ocean_mask
        ] = ALBEDO_OCEAN


        cloud_fraction = np.full(
            (
                self.height,
                self.width
            ),
            0.60,
            dtype=np.float64
        )


        # Initial temperature
        temperature_monthly = (
            calculate_temperature_field(
                insolation_monthly,
                albedo,
                elevation_norm,
                ocean_threshold,
                self.gravity,
                self.patm,
                self.co2_ppm,
                self.lapse_rate,
                cloud_fraction
            )
        )


        # ============================================================
        # CLIMATE SIMULATION
        # ============================================================

        print()
        print("-" * 72)
        print(
            "CLIMATE SIMULATION"
        )
        print("-" * 72)


        previous_global_temp = None

        annual_temperature_history = []

        annual_rain_history = []

        annual_ice_history = []

        annual_cloud_history = []


        for year in range(
            simulation_years
        ):

            # --------------------------------------------------------
            # Annual mean
            # --------------------------------------------------------

            temp_annual = (
                np.mean(
                    temperature_monthly,
                    axis=0
                )
            )


            # --------------------------------------------------------
            # Clouds
            # --------------------------------------------------------

            cloud_fraction = (
                calculate_cloud_fraction(
                    temp_annual,
                    None,
                    ocean_mask,
                    humidity=0.70
                )
            )


            # --------------------------------------------------------
            # Albedo
            # --------------------------------------------------------

            albedo = (
                calculate_albedo(
                    elevation_norm,
                    temp_annual,
                    ocean_threshold,
                    cloud_fraction.mean()
                )
            )


            # --------------------------------------------------------
            # Temperature update
            # --------------------------------------------------------

            new_temperature_monthly = (
                calculate_temperature_field(
                    insolation_monthly,
                    albedo,
                    elevation_norm,
                    ocean_threshold,
                    self.gravity,
                    self.patm,
                    self.co2_ppm,
                    self.lapse_rate,
                    cloud_fraction
                )
            )


            # --------------------------------------------------------
            # Climate relaxation
            # --------------------------------------------------------

            # Prevents instant oscillation.
            #
            # 25% new state
            # 75% previous state
            #
            temperature_monthly = (
                0.75
                *
                temperature_monthly
                +
                0.25
                *
                new_temperature_monthly
            )


            temp_annual = (
                np.mean(
                    temperature_monthly,
                    axis=0
                )
            )


            # --------------------------------------------------------
            # Precipitation
            # --------------------------------------------------------

            precip_annual, precip_monthly = (
                calculate_precipitation(
                    temperature_monthly,
                    elevation_norm,
                    phis,
                    ocean_threshold,
                    self.patm,
                    self.gravity,
                    cloud_fraction
                )
            )


            # --------------------------------------------------------
            # Update clouds using precipitation
            # --------------------------------------------------------

            cloud_fraction = (
                calculate_cloud_fraction(
                    temp_annual,
                    precip_annual,
                    ocean_mask,
                    humidity=0.70
                )
            )


            # --------------------------------------------------------
            # Ice
            # --------------------------------------------------------

            ice_fraction_map = (
                calculate_ice_fraction(
                    temp_annual,
                    elevation_norm,
                    ocean_threshold
                )
            )

            ice_fraction = (
                np.mean(
                    ice_fraction_map
                )
            )


            # --------------------------------------------------------
            # Global values
            # --------------------------------------------------------

            global_temp = (
                np.mean(
                    temp_annual
                )
            )

            global_rain = (
                np.mean(
                    precip_annual
                )
            )

            global_clouds = (
                np.mean(
                    cloud_fraction
                )
            )


            if previous_global_temp is None:

                trend = 0.0

            else:

                trend = (
                    global_temp
                    -
                    previous_global_temp
                )


            previous_global_temp = (
                global_temp
            )


            annual_temperature_history.append(
                global_temp
            )

            annual_rain_history.append(
                global_rain
            )

            annual_ice_history.append(
                ice_fraction
            )

            annual_cloud_history.append(
                global_clouds
            )


            print(
                f"Year {year + 1:02d}/"
                f"{simulation_years} - "
                f"T: {global_temp:7.2f}°C, "
                f"Ice: {ice_fraction * 100:5.1f}%, "
                f"Clouds: {global_clouds * 100:5.1f}%, "
                f"Rain: {global_rain:7.1f} mm/yr, "
                f"Trend: {trend:+.4f}°C"
            )


        # ============================================================
        # FINAL STATISTICS
        # ============================================================

        temp_annual = (
            np.mean(
                temperature_monthly,
                axis=0
            )
        )


        temp_global_mean = (
            np.mean(
                temp_annual
            )
        )


        temp_ocean_mean = (
            np.mean(
                temp_annual[
                    ocean_mask
                ]
            )
            if np.any(ocean_mask)
            else 0.0
        )


        land_mask = ~ocean_mask


        temp_land_mean = (
            np.mean(
                temp_annual[
                    land_mask
                ]
            )
            if np.any(land_mask)
            else 0.0
        )


        center = (
            self.height // 2
        )

        equator_band = (
            temp_annual[
                max(
                    0,
                    center - 5
                ):
                min(
                    self.height,
                    center + 5
                )
            ]
        )


        temp_equator = (
            np.mean(
                equator_band
            )
        )


        temp_north_pole = (
            np.mean(
                temp_annual[
                    -10:,
                    :
                ]
            )
        )


        temp_south_pole = (
            np.mean(
                temp_annual[
                    :10,
                    :
                ]
            )
        )


        # ============================================================
        # ENERGY
        # ============================================================

        energy = (
            calculate_energy_budget(
                insolation_monthly,
                albedo,
                cloud_fraction
            )
        )


        generation_time = (
            time.time()
            -
            start_time
        )


        # ============================================================
        # PRINT STATISTICS
        # ============================================================

        print()
        print("=" * 72)
        print(
            "PLANET STATISTICS"
        )
        print("=" * 72)


        print()
        print(
            "PLANET"
        )

        print(
            f"  Mass:                 "
            f"{self.mass_earths:.3f} M⊕"
        )

        print(
            f"  Radius:               "
            f"{self.radius_earths:.3f} R⊕"
        )

        print(
            f"  Radius:               "
            f"{self.radius_earths * EARTH_RADIUS / 1000:,.0f} km"
        )

        print(
            f"  Gravity:              "
            f"{self.gravity:.4f} m/s²"
        )

        print(
            f"  Atmospheric pressure: "
            f"{self.patm:,.0f} Pa"
        )

        print(
            f"  Lapse rate:           "
            f"{self.lapse_rate:.3f} K/km"
        )

        print(
            f"  CO₂:                  "
            f"{self.co2_ppm:.0f} ppm"
        )


        print()
        print(
            "SURFACE"
        )

        print(
            f"  Ocean:                "
            f"{surface_stats['ocean_fraction'] * 100:.2f}%"
        )

        print(
            f"  Land:                 "
            f"{surface_stats['land_fraction'] * 100:.2f}%"
        )

        print(
            f"  Mean land elevation:  "
            f"{surface_stats['mean_land_height_m']:,.0f} m"
        )

        print(
            f"  Median land elevation:"
            f" {surface_stats['median_land_height_m']:,.0f} m"
        )

        print(
            f"  Maximum land height:  "
            f"{surface_stats['max_land_height_m']:,.0f} m"
        )

        print(
            f"  Mean ocean depth:     "
            f"{surface_stats['mean_ocean_depth_m']:,.0f} m"
        )


        print()
        print(
            "TEMPERATURE"
        )

        print(
            f"  Global mean:          "
            f"{temp_global_mean:8.2f} °C"
        )

        print(
            f"  Ocean mean:           "
            f"{temp_ocean_mean:8.2f} °C"
        )

        print(
            f"  Land mean:            "
            f"{temp_land_mean:8.2f} °C"
        )

        print(
            f"  Equator:              "
            f"{temp_equator:8.2f} °C"
        )

        print(
            f"  North pole:           "
            f"{temp_north_pole:8.2f} °C"
        )

        print(
            f"  South pole:           "
            f"{temp_south_pole:8.2f} °C"
        )

        print(
            f"  Range:                "
            f"{np.min(temp_annual):8.2f}"
            f" to "
            f"{np.max(temp_annual):8.2f} °C"
        )


        print()
        print(
            "PRECIPITATION"
        )

        print(
            f"  Global mean:          "
            f"{np.mean(precip_annual):8.1f} mm/year"
        )

        print(
            f"  Ocean mean:           "
            f"{np.mean(precip_annual[ocean_mask]):8.1f} mm/year"
            if np.any(ocean_mask)
            else
            "  Ocean mean:              0.0 mm/year"
        )

        print(
            f"  Land mean:            "
            f"{np.mean(precip_annual[land_mask]):8.1f} mm/year"
            if np.any(land_mask)
            else
            "  Land mean:                0.0 mm/year"
        )

        print(
            f"  Minimum:              "
            f"{np.min(precip_annual):8.1f} mm/year"
        )

        print(
            f"  Maximum:              "
            f"{np.max(precip_annual):8.1f} mm/year"
        )


        print()
        print(
            "ATMOSPHERE"
        )

        print(
            f"  Cloud fraction:       "
            f"{np.mean(cloud_fraction) * 100:8.2f}%"
        )

        print(
            f"  Ice fraction:         "
            f"{ice_fraction * 100:8.2f}%"
        )


        print()
        print(
            "ENERGY"
        )

        print(
            f"  Solar constant:       "
            f"{self.solar_constant:8.1f} W/m²"
        )

        print(
            f"  Mean TOA solar:       "
            f"{energy['mean_toa']:8.1f} W/m²"
        )

        print(
            f"  Surface albedo:       "
            f"{energy['mean_albedo']:8.3f}"
        )

        print(
            f"  Cloud fraction:       "
            f"{energy['mean_clouds'] * 100:8.2f}%"
        )

        print(
            f"  Absorbed solar:       "
            f"{energy['absorbed_solar']:8.1f} W/m²"
        )


        print()
        print(
            f"Generation time:       "
            f"{generation_time:8.2f} s"
        )

        print(
            "=" * 72
        )


        return {
            "elevation":
                elevation_norm,

            "temperature_monthly":
                temperature_monthly,

            "temperature_annual":
                temp_annual,

            "precipitation_annual":
                precip_annual,

            "precipitation_monthly":
                precip_monthly,

            "insolation_monthly":
                insolation_monthly,

            "albedo":
                albedo,

            "cloud_fraction":
                cloud_fraction,

            "ice_fraction":
                ice_fraction_map,

            "ocean_area":
                surface_stats[
                    "ocean_fraction"
                ],

            "surface_stats":
                surface_stats,

            "gravity":
                self.gravity,

            "lapse_rate":
                self.lapse_rate,

            "mass_earths":
                self.mass_earths,

            "radius_earths":
                self.radius_earths,

            "patm":
                self.patm,

            "co2_ppm":
                self.co2_ppm,

            "temperature_history":
                np.array(
                    annual_temperature_history
                ),

            "rain_history":
                np.array(
                    annual_rain_history
                ),

            "ice_history":
                np.array(
                    annual_ice_history
                ),

            "cloud_history":
                np.array(
                    annual_cloud_history
                )
        }, generation_time


# ============================================================================
# PLOTTING
# ============================================================================

def plot_physical_planet(
    planet_data,
    generation_time,
    ocean_threshold
):

    fig = plt.figure(
        figsize=(18, 13)
    )

    gs = fig.add_gridspec(
        4,
        3,
        hspace=0.35,
        wspace=0.30
    )


    # ================================================================
    # ELEVATION
    # ================================================================

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    im1 = ax1.imshow(
        planet_data[
            "elevation"
        ],
        cmap="terrain",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ]
    )

    ax1.set_title(
        "Elevation"
    )

    ax1.contour(
        planet_data[
            "elevation"
        ],
        levels=[
            ocean_threshold
        ],
        colors="red",
        linewidths=1.0,
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ]
    )

    plt.colorbar(
        im1,
        ax=ax1,
        fraction=0.046
    )


    # ================================================================
    # TEMPERATURE
    # ================================================================

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    im2 = ax2.imshow(
        planet_data[
            "temperature_annual"
        ],
        cmap="RdBu_r",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ],
        vmin=-40,
        vmax=40
    )

    ax2.set_title(
        "Annual Mean Temperature (°C)"
    )

    plt.colorbar(
        im2,
        ax=ax2,
        fraction=0.046,
        label="°C"
    )


    # ================================================================
    # PRECIPITATION
    # ================================================================

    ax3 = fig.add_subplot(
        gs[0, 2]
    )

    im3 = ax3.imshow(
        planet_data[
            "precipitation_annual"
        ],
        cmap="YlGnBu",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ],
        vmin=0,
        vmax=4000
    )

    ax3.set_title(
        "Annual Precipitation (mm/year)"
    )

    plt.colorbar(
        im3,
        ax=ax3,
        fraction=0.046,
        label="mm/year"
    )


    # ================================================================
    # ALBEDO
    # ================================================================

    ax4 = fig.add_subplot(
        gs[1, 0]
    )

    im4 = ax4.imshow(
        planet_data[
            "albedo"
        ],
        cmap="gray_r",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ],
        vmin=0,
        vmax=1
    )

    ax4.set_title(
        "Surface Albedo"
    )

    plt.colorbar(
        im4,
        ax=ax4,
        fraction=0.046
    )


    # ================================================================
    # CLOUDS
    # ================================================================

    ax5 = fig.add_subplot(
        gs[1, 1]
    )

    im5 = ax5.imshow(
        planet_data[
            "cloud_fraction"
        ],
        cmap="Blues",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ],
        vmin=0,
        vmax=1
    )

    ax5.set_title(
        "Cloud Fraction"
    )

    plt.colorbar(
        im5,
        ax=ax5,
        fraction=0.046,
        label="fraction"
    )


    # ================================================================
    # ICE
    # ================================================================

    ax6 = fig.add_subplot(
        gs[1, 2]
    )

    im6 = ax6.imshow(
        planet_data[
            "ice_fraction"
        ],
        cmap="winter",
        origin="lower",
        extent=[
            0,
            360,
            -90,
            90
        ],
        vmin=0,
        vmax=1
    )

    ax6.set_title(
        "Ice / Snow Fraction"
    )

    plt.colorbar(
        im6,
        ax=ax6,
        fraction=0.046,
        label="fraction"
    )


    # ================================================================
    # TEMPERATURE HISTORY
    # ================================================================

    ax7 = fig.add_subplot(
        gs[2, 0]
    )

    years = np.arange(
        1,
        len(
            planet_data[
                "temperature_history"
            ]
        ) + 1
    )

    ax7.plot(
        years,
        planet_data[
            "temperature_history"
        ],
        "r-o",
        linewidth=2
    )

    ax7.set_xlabel(
        "Year"
    )

    ax7.set_ylabel(
        "°C"
    )

    ax7.set_title(
        "Global Temperature History"
    )

    ax7.grid(
        alpha=0.3
    )


    # ================================================================
    # RAIN HISTORY
    # ================================================================

    ax8 = fig.add_subplot(
        gs[2, 1]
    )

    ax8.plot(
        years,
        planet_data[
            "rain_history"
        ],
        "g-o",
        linewidth=2
    )

    ax8.set_xlabel(
        "Year"
    )

    ax8.set_ylabel(
        "mm/year"
    )

    ax8.set_title(
        "Global Precipitation"
    )

    ax8.grid(
        alpha=0.3
    )


    # ================================================================
    # ZONAL TEMPERATURE
    # ================================================================

    ax9 = fig.add_subplot(
        gs[2, 2]
    )

    h = planet_data[
        "temperature_annual"
    ].shape[0]

    lat_deg = np.linspace(
        -90,
        90,
        h
    )

    temp_zonal = np.mean(
        planet_data[
            "temperature_annual"
        ],
        axis=1
    )

    ax9.plot(
        lat_deg,
        temp_zonal,
        "b-",
        linewidth=2
    )

    ax9.axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.3
    )

    ax9.set_xlabel(
        "Latitude"
    )

    ax9.set_ylabel(
        "°C"
    )

    ax9.set_title(
        "Zonal Mean Temperature"
    )

    ax9.grid(
        alpha=0.3
    )


    # ================================================================
    # INFO PANEL
    # ================================================================

    ax10 = fig.add_subplot(
        gs[3, :]
    )

    ax10.axis(
        "off"
    )


    stats = planet_data[
        "surface_stats"
    ]

    mean_temp = np.mean(
        planet_data[
            "temperature_annual"
        ]
    )

    mean_rain = np.mean(
        planet_data[
            "precipitation_annual"
        ]
    )

    mean_albedo = np.mean(
        planet_data[
            "albedo"
        ]
    )

    mean_clouds = np.mean(
        planet_data[
            "cloud_fraction"
        ]
    )

    info = f"""
PHYSICAL PLANET

Mass:                 {planet_data['mass_earths']:.3f} M⊕
Radius:               {planet_data['radius_earths']:.3f} R⊕
Gravity:              {planet_data['gravity']:.4f} m/s²
Atmospheric pressure: {planet_data['patm']:,.0f} Pa
CO₂:                  {planet_data['co2_ppm']:.0f} ppm
Lapse rate:           {planet_data['lapse_rate']:.3f} K/km

SURFACE

Ocean:                {stats['ocean_fraction'] * 100:.1f} %
Land:                 {stats['land_fraction'] * 100:.1f} %
Mean land elevation:  {stats['mean_land_height_m']:,.0f} m
Median land elevation:{stats['median_land_height_m']:,.0f} m
Mean ocean depth:     {stats['mean_ocean_depth_m']:,.0f} m
Maximum land height:  {stats['max_land_height_m']:,.0f} m

CLIMATE

Global temperature:   {mean_temp:+.2f} °C
Global precipitation: {mean_rain:.0f} mm/year
Cloud fraction:       {mean_clouds * 100:.1f} %
Surface albedo:       {mean_albedo:.3f}

Generation time:      {generation_time:.1f} s
"""


    ax10.text(
        0.02,
        0.98,
        info,
        transform=ax10.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="wheat",
            alpha=0.25
        )
    )


    plt.suptitle(
        "Physical Planet Generator",
        fontsize=16,
        fontweight="bold"
    )

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print(
        "🌍 PHYSICAL PLANET GENERATION"
    )

    print(
        "=" * 72
    )


    # ========================================================================
    # PLANET PARAMETERS
    # ========================================================================

    gen = PhysicalPlanetGenerator(

        # ------------------------------------------------------------
        # Planet
        # ------------------------------------------------------------

        mass_earths=1.0,

        radius_earths=1.0,

        # ------------------------------------------------------------
        # Atmosphere
        # ------------------------------------------------------------

        patm=101325.0,

        co2_ppm=420.0,

        # None = calculate from gravity
        #
        # Earth gives approximately 6.5 K/km.
        #
        lapse_rate=None,

        # ------------------------------------------------------------
        # Solar
        # ------------------------------------------------------------

        solar_constant=1361.0,

        # ------------------------------------------------------------
        # Resolution
        # ------------------------------------------------------------

        width=360,

        height=180,

        seed=42
    )


    # ========================================================================
    # GENERATE
    # ========================================================================

    planet_data, generation_time = (
        gen.generate_planet(

            # --------------------------------------------------------
            # IMPORTANT:
            #
            # 0.75 gives approximately Earth-like ocean coverage
            # with this terrain generator.
            #
            # --------------------------------------------------------

            ocean_threshold=0.75,

            # --------------------------------------------------------
            # Orbit
            # --------------------------------------------------------

            obliquity_deg=23.5,

            eccentricity=0.017,

            mvelp_deg=102.0,

            # --------------------------------------------------------
            # First test:
            # ONLY 10 YEARS
            # --------------------------------------------------------

            simulation_years=10,

            # --------------------------------------------------------
            # Terrain noise coordinate scale.
            #
            # This is NOT planetary radius.
            #
            # radius_earths controls physical planet size.
            # radius_noise controls terrain generation.
            # --------------------------------------------------------

            radius_noise=200,

            # --------------------------------------------------------
            # Terrain distribution
            # --------------------------------------------------------

            hypsometric=True,

            # --------------------------------------------------------
            # Physical terrain dimensions
            # --------------------------------------------------------

            max_land_height_m=8000.0,

            max_ocean_depth_m=6000.0
        )
    )


    # ========================================================================
    # PLOT
    # ========================================================================

    print()
    print(
        "Generating plots..."
    )

    plot_physical_planet(
        planet_data,
        generation_time,
        0.75
    )


    print()
    print(
        "✅ Generation complete!"
    )




