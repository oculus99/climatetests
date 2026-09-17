
# -*- coding: utf-8 -*-

## python3 simple 2d seasonal planet energy balance model
## for calculation of temperature
## 16.9.2026 v 0000.0005

import numpy as np

import time
import matplotlib.pyplot as plt


# ============================================================
# CONSTANTS
# ============================================================

SIGMA = 5.670374419e-8
DAY = 86400.0


# ============================================================
# ORBIT
# ============================================================

def orbital_position(day, p_orbit, mvelp, ecc):

    M = 2.0 * np.pi * day / p_orbit

    E = M

    for _ in range(10):
        E -= (
            E - ecc * np.sin(E) - M
        ) / (
            1.0 - ecc * np.cos(E)
        )

    r = 1.0 - ecc * np.cos(E)

    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0)
    )

    solar_longitude = (
        nu + np.deg2rad(mvelp)
    )

    return r, solar_longitude


# ============================================================
# INSOLATION
# ============================================================

def daily_insolation_basic(
        day,
        latrad,
        S,
        p_orbit,
        mvelp,
        tilt,
        ecc):

    r, solar_longitude = orbital_position(
        day,
        p_orbit,
        mvelp,
        ecc
    )

    declination = np.arcsin(
        np.sin(np.deg2rad(tilt))
        * np.sin(solar_longitude)
    )

    solar_flux = S / r**2

    cos_h0 = (
        -np.tan(latrad)
        * np.tan(declination)
    )

    h0 = np.arccos(
        np.clip(cos_h0, -1.0, 1.0)
    )

    h0 = np.where(
        cos_h0 <= -1.0,
        np.pi,
        h0
    )

    h0 = np.where(
        cos_h0 >= 1.0,
        0.0,
        h0
    )

    Q = (
        solar_flux / np.pi
        * (
            h0
            * np.sin(latrad)
            * np.sin(declination)
            +
            np.cos(latrad)
            * np.cos(declination)
            * np.sin(h0)
        )
    )

    return np.maximum(Q, 0.0)


# ============================================================
# ALBEDO
# ============================================================

def calculate_albedo(
        T,
        albedo_warm=0.30,
        albedo_cold=0.60,
        ice_temp=263.15,
        ice_width=8.0):

    x = (
        T - ice_temp
    ) / ice_width

    warm_fraction = (
        0.5
        * (1.0 + np.tanh(x))
    )

    alpha = (
        albedo_cold
        * (1.0 - warm_fraction)
        +
        albedo_warm
        * warm_fraction
    )

    return alpha


# ============================================================
# CO2
# ============================================================

def co2_forcing(
        pco2,
        patm,
        co2_reference=280.0):

    if pco2 <= 0:
        raise ValueError(
            "pco2 must be > 0"
        )

    return (
        5.35
        * np.log(
            (patm * pco2)
            / co2_reference
        )
    )


# ============================================================
# OUTGOING LONGWAVE RADIATION
# ============================================================

def outgoing_longwave(
        T,
        pco2,
        patm,
        A_olr=203.3,
        B_olr=2.09):

    Tc = T - 273.15

    Fco2 = co2_forcing(
        pco2,
        patm
    )

    return (
        A_olr
        + B_olr * Tc
        - Fco2
    )


# ============================================================
# PERMANENT ICE CAP
# ============================================================

def permanent_ice_cap(
        lat,
        Tannual,
        ice_limit_C=-10.0):

    ice_limit = (
        273.15
        + ice_limit_C
    )

    # --------------------------------------------------------
    # Southern hemisphere
    # --------------------------------------------------------

    south = lat < 0

    south_lat = lat[south]
    south_T = Tannual[south]

    south_warm = np.where(
        south_T >= ice_limit
    )[0]

    if len(south_warm) == 0:

        ice_south = -90.0

    else:

        i = south_warm[0]

        if i == 0:

            ice_south = south_lat[i]

        else:

            T1 = south_T[i - 1]
            T2 = south_T[i]

            lat1 = south_lat[i - 1]
            lat2 = south_lat[i]

            ice_south = (
                lat1
                +
                (ice_limit - T1)
                * (lat2 - lat1)
                / (T2 - T1)
            )

    # --------------------------------------------------------
    # Northern hemisphere
    # --------------------------------------------------------

    north = lat > 0

    north_lat = lat[north]
    north_T = Tannual[north]

    north_warm = np.where(
        north_T >= ice_limit
    )[0]

    if len(north_warm) == 0:

        ice_north = 90.0

    else:

        i = north_warm[-1]

        if i == len(north_lat) - 1:

            ice_north = north_lat[i]

        else:

            T1 = north_T[i]
            T2 = north_T[i + 1]

            lat1 = north_lat[i]
            lat2 = north_lat[i + 1]

            ice_north = (
                lat1
                +
                (ice_limit - T1)
                * (lat2 - lat1)
                / (T2 - T1)
            )

    return ice_south, ice_north


# ============================================================
# SPHERICAL DIFFUSION
# ============================================================

def diffusion_spherical_2(
        T,
        latrad,
        D,
        R):
    """
    1D spherical diffusion operator.

    T      : temperature [K]
    latrad : latitude [rad]
    D      : diffusion coefficient
    R      : relative/physical planetary radius

    NOTE:
    np.gradient() must receive latrad as the coordinate
    array, not np.gradient(latrad).
    """

    coslat = np.cos(latrad)

    # --------------------------------------------------------
    # Temperature gradient
    # --------------------------------------------------------

    dT_dlat = np.gradient(
        T,
        latrad
    )

    # --------------------------------------------------------
    # Meridional heat flux
    # --------------------------------------------------------

    flux = (
        -D
        * dT_dlat
        / R
    )

    # --------------------------------------------------------
    # Spherical geometry
    # --------------------------------------------------------

    flux_spherical = (
        flux
        * coslat
    )

    # --------------------------------------------------------
    # Minus divergence of heat flux
    # --------------------------------------------------------

    diffusion = (
        -1.0
        / (R * coslat)
        * np.gradient(
            flux_spherical,
            latrad
        )
    )

    return diffusion









from numba import njit


# ============================================================
# NUMBA EBM CORE
#
# Tämä on kehitysversio.
#
# Tärkeää:
# - ei fastmath
# - ei parallel
# - debug mukana
# - diffuusio kirjoitettu eksplisiittisesti
# ============================================================

@njit
def ebm_core_numba(
        T,
        Q_year,
        C,
        latrad,
        D_scaled,
        transport_kerroin,
        radius_rel,
        pco2,
        patm,
        dt,
        n_years,
        debug):

    steps_per_year = Q_year.shape[0]
    nlat = Q_year.shape[1]

    T_year = np.zeros(
        (steps_per_year, nlat)
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    dlat = (
        latrad[1]
        - latrad[0]
    )

    coslat = np.cos(
        latrad
    )

    dt_seconds = (
        dt * 86400.0
    )

    # --------------------------------------------------------
    # CO2 forcing
    #
    # vakio koko integraation ajan
    # --------------------------------------------------------

    Fco2 = (
        5.35
        * np.log(
            (
                patm * pco2
            )
            / 280.0
        )
    )

    # --------------------------------------------------------
    # Temporary arrays
    # --------------------------------------------------------

    alpha = np.empty(
        nlat,
        dtype=np.float64
    )

    dT_dlat = np.empty(
        nlat,
        dtype=np.float64
    )

    flux = np.empty(
        nlat,
        dtype=np.float64
    )

    dflux_dlat = np.empty(
        nlat,
        dtype=np.float64
    )

    transport_array = np.empty(
        nlat,
        dtype=np.float64
    )

    dTdt = np.empty(
        nlat,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Debug information
    #
    # debug_code
    #
    # 0 = OK
    # 1 = alpha
    # 2 = dT/dlat
    # 3 = flux
    # 4 = dflux/dlat
    # 5 = transport
    # 6 = dTdt
    # 7 = temperature
    # --------------------------------------------------------

    debug_code = 0
    debug_year = -1
    debug_step = -1
    debug_lat = -1
    debug_value = 0.0

    max_abs_transport = 0.0
    max_abs_dTdt = 0.0
    max_abs_T = 0.0

    # ========================================================
    # YEARS
    # ========================================================

    for year in range(
        n_years
    ):

        # ====================================================
        # TIME STEPS
        # ====================================================

        for step in range(
            steps_per_year
        ):

            Q = Q_year[
                step
            ]

            # =================================================
            # ALBEDO
            # =================================================

            for i in range(
                nlat
            ):

                x = (
                    T[i]
                    - 263.15
                ) / 8.0

                warm_fraction = (
                    0.5
                    * (
                        1.0
                        + np.tanh(x)
                    )
                )

                alpha[i] = (
                    0.60
                    * (
                        1.0
                        - warm_fraction
                    )
                    +
                    0.30
                    * warm_fraction
                )

                if debug:

                    if not np.isfinite(
                        alpha[i]
                    ):

                        return (
                            T,
                            T_year,
                            1,
                            year,
                            step,
                            i,
                            alpha[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # TEMPERATURE GRADIENT
            #
            # latitude grid is uniform
            #
            # dT/dlat
            # =================================================

            dT_dlat[0] = (
                T[1]
                - T[0]
            ) / dlat

            for i in range(
                1,
                nlat - 1
            ):

                dT_dlat[i] = (
                    T[i + 1]
                    - T[i - 1]
                ) / (
                    2.0 * dlat
                )

            dT_dlat[nlat - 1] = (
                T[nlat - 1]
                - T[nlat - 2]
            ) / dlat

            if debug:

                for i in range(
                    nlat
                ):

                    if not np.isfinite(
                        dT_dlat[i]
                    ):

                        return (
                            T,
                            T_year,
                            2,
                            year,
                            step,
                            i,
                            dT_dlat[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # SPHERICAL DIFFUSION FLUX
            #
            # Tämä vastaa:
            #
            # flux = -D * (dT/dlat / R) * cos(lat)
            #
            # =================================================

            for i in range(
                nlat
            ):

                flux[i] = (
                    -D_scaled
                    * (
                        dT_dlat[i]
                        / radius_rel
                    )
                    * coslat[i]
                )

            if debug:

                for i in range(
                    nlat
                ):

                    if not np.isfinite(
                        flux[i]
                    ):

                        return (
                            T,
                            T_year,
                            3,
                            year,
                            step,
                            i,
                            flux[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # FLUX GRADIENT
            # =================================================

            dflux_dlat[0] = (
                flux[1]
                - flux[0]
            ) / dlat

            for i in range(
                1,
                nlat - 1
            ):

                dflux_dlat[i] = (
                    flux[i + 1]
                    - flux[i - 1]
                ) / (
                    2.0 * dlat
                )

            dflux_dlat[nlat - 1] = (
                flux[nlat - 1]
                - flux[nlat - 2]
            ) / dlat

            if debug:

                for i in range(
                    nlat
                ):

                    if not np.isfinite(
                        dflux_dlat[i]
                    ):

                        return (
                            T,
                            T_year,
                            4,
                            year,
                            step,
                            i,
                            dflux_dlat[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # TRANSPORT + ENERGY BALANCE
            # =================================================

            for i in range(
                nlat
            ):

                # ---------------------------------------------
                # Original diffusion_spherical_2:
                #
                # divergence =
                #   -(1 / (R*coslat))
                #   * d(flux_spherical)/dlat
                #
                # ------------------------------------------------

                transport_array[i] = (
                    transport_kerroin
                    *
                    (
                        -dflux_dlat[i]
                        /
                        (
                            radius_rel
                            *
                            max(
                                coslat[i],
                                0.05
                            )
                        )
                    )
                )

                if (
                    np.abs(
                        transport_array[i]
                    )
                    >
                    max_abs_transport
                ):

                    max_abs_transport = np.abs(
                        transport_array[i]
                    )

                if debug:

                    if not np.isfinite(
                        transport_array[i]
                    ):

                        return (
                            T,
                            T_year,
                            5,
                            year,
                            step,
                            i,
                            transport_array[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

                # ---------------------------------------------
                # OLR
                # ---------------------------------------------

                Tc = (
                    T[i]
                    - 273.15
                )

                OLR = (
                    203.3
                    + 2.09 * Tc
                    - Fco2
                )

                # ---------------------------------------------
                # Energy balance
                # ---------------------------------------------

                dTdt[i] = (
                    Q[i]
                    * (
                        1.0
                        - alpha[i]
                    )
                    - OLR
                    + transport_array[i]
                ) / C

                if (
                    np.abs(
                        dTdt[i]
                    )
                    >
                    max_abs_dTdt
                ):

                    max_abs_dTdt = np.abs(
                        dTdt[i]
                    )

                if debug:

                    if not np.isfinite(
                        dTdt[i]
                    ):

                        return (
                            T,
                            T_year,
                            6,
                            year,
                            step,
                            i,
                            dTdt[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # TEMPERATURE UPDATE
            # =================================================

            for i in range(
                nlat
            ):

                T[i] += (
                    dTdt[i]
                    * dt_seconds
                )

                if (
                    np.abs(
                        T[i]
                    )
                    >
                    max_abs_T
                ):

                    max_abs_T = np.abs(
                        T[i]
                    )

                if debug:

                    if not np.isfinite(
                        T[i]
                    ):

                        return (
                            T,
                            T_year,
                            7,
                            year,
                            step,
                            i,
                            T[i],
                            max_abs_transport,
                            max_abs_dTdt,
                            max_abs_T
                        )

            # =================================================
            # STORE FINAL YEAR
            # =================================================

            if year == (
                n_years - 1
            ):

                for i in range(
                    nlat
                ):

                    T_year[
                        step,
                        i
                    ] = T[i]

    # ========================================================
    # SUCCESS
    # ========================================================

    return (
        T,
        T_year,
        debug_code,
        debug_year,
        debug_step,
        debug_lat,
        debug_value,
        max_abs_transport,
        max_abs_dTdt,
        max_abs_T
    )



# ============================================================
# MAIN EBM
# ============================================================

def main_ebm_old(
        mplanet=1.0,
        rplanet=1.0,
        S=1361.0,
        p_orbit=365.25,
        prot=24,
        mvelp=102.0,
        tilt=23.44,
        ecc=0.0167,
        patm=1.0,
        pco2=420.0,
        nlat=180,
        n_years=100,
        dt=0.25,
        D=0.55,
        ocean_fraction=0.70):

    gee_earths = (
        mplanet
        / (rplanet * rplanet)
    )

    omega_rel = (
        24.0 / prot
    )

    radius_rel = rplanet

    # --------------------------------------------------------
    # Latitude grid
    # --------------------------------------------------------

    lat = np.linspace(
        -89.5,
        89.5,
        nlat
    )

    latrad = np.deg2rad(lat)

    # --------------------------------------------------------
    # Heat capacity
    # --------------------------------------------------------

    C_land = 1.0e6
    C_ocean = 4.0e7

    C = (
        C_land
        * (1.0 - ocean_fraction)
        +
        C_ocean
        * ocean_fraction
    )

    # --------------------------------------------------------
    # Time grid
    # --------------------------------------------------------

    steps_per_year = int(
        np.round(
            p_orbit / dt
        )
    )

    total_steps = (
        n_years
        * steps_per_year
    )

    # --------------------------------------------------------
    # Initial temperature
    # --------------------------------------------------------

    T = np.full(
        nlat,
        288.0
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    T_year = np.zeros(
        (steps_per_year, nlat)
    )

    Q_year = np.zeros(
        (steps_per_year, nlat)
    )

    time_year = np.zeros(
        steps_per_year
    )

    # --------------------------------------------------------
    # Time integration
    # --------------------------------------------------------

    for year in range(n_years):

        for step in range(
            steps_per_year
        ):

            day = (
                step * dt
            )

            # ------------------------------------------------
            # Solar radiation
            # ------------------------------------------------

            Q = daily_insolation_basic(
                day,
                latrad,
                S,
                p_orbit,
                mvelp,
                tilt,
                ecc
            )

            # ------------------------------------------------
            # Albedo
            # ------------------------------------------------

            alpha = calculate_albedo(
                T
            )

            absorbed = (
                Q
                * (1.0 - alpha)
            )

            # ------------------------------------------------
            # OLR
            # ------------------------------------------------

            OLR = outgoing_longwave(
                T,
                pco2,
                patm
            )

            # ------------------------------------------------
            # Diffusion scaling
            # ------------------------------------------------

            D_scaled = (
                D
                / omega_rel**2
            )

            # ------------------------------------------------
            # Planet scaling
            #
            # R^2 scaling is kept here.
            # diffusion_spherical_2 also contains
            # geometrical R factors, so use R carefully.
            # ------------------------------------------------

            transport_kerroin = (
                patm
                / np.sqrt(gee_earths)
            )

            transport = (
                transport_kerroin
                * diffusion_spherical_2(
                    T,
                    latrad,
                    D_scaled,
                    radius_rel
                )
            )

            # ------------------------------------------------
            # Energy balance
            # ------------------------------------------------

            dTdt = (
                absorbed
                - OLR
                + transport
            ) / C

            # ------------------------------------------------
            # Update
            # ------------------------------------------------

            T += (
                dTdt
                * dt
                * DAY
            )

            # ------------------------------------------------
            # Store final year
            # ------------------------------------------------

            if year == n_years - 1:

                T_year[
                    step, :
                ] = T

                Q_year[
                    step, :
                ] = Q

                time_year[
                    step
                ] = day

    # ========================================================
    # ANNUAL DIAGNOSTICS
    # ========================================================

    Tannual = np.mean(
        T_year,
        axis=0
    )

    Tmax = np.max(
        T_year,
        axis=0
    )

    Tmin = np.min(
        T_year,
        axis=0
    )

    # --------------------------------------------------------
    # Latitude weighted global mean
    # --------------------------------------------------------

    weights = np.cos(
        latrad
    )

    Tglobal = (
        np.sum(
            Tannual
            * weights
        )
        /
        np.sum(weights)
    )

    # --------------------------------------------------------
    # Permanent ice cap
    # --------------------------------------------------------

    ice_south, ice_north = (
        permanent_ice_cap(
            lat,
            Tannual,
            -10.0
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================
    results = {
        "lat": lat,
        "latrad": latrad,

        "T": T,
        "T_year": T_year,
        "Tannual": Tannual,
        "Tmax": Tmax,
        "Tmin": Tmin,

        "Q_year": Q_year,
        "time_year": time_year,

        "Tglobal": Tglobal,

        "ice_south": ice_south,
        "ice_north": ice_north,

        # ====================================================
        # MODEL / PLANET PARAMETERS
        # ====================================================

        "mplanet": mplanet,
        "rplanet": rplanet,

        "S": S,
        "p_orbit": p_orbit,
        "prot": prot,

        "mvelp": mvelp,
        "tilt": tilt,
        "ecc": ecc,

        "patm": patm,
        "pco2": pco2,

        "D": D,
        "ocean_fraction": ocean_fraction
    }

    return results



# ============================================================
# MAIN EBM NUMBA
# ============================================================

def main_ebm_numba(
        mplanet=1.0,
        rplanet=1.0,
        S=1361.0,
        p_orbit=365.25,
        prot=24,
        mvelp=102.0,
        tilt=23.44,
        ecc=0.0167,
        patm=1.0,
        pco2=420.0,
        nlat=180,
        n_years=100,
        dt=0.1,
        D=0.55,
        ocean_fraction=0.70,
        debug_numba=False):

    atmosphere_factor=np.pow(patm,1/3)
    tau = -np.log(atmosphere_factor)

    # ========================================================
    # PLANET
    # ========================================================
    land_fraction=1.0-ocean_fraction
    gee_earths = (
        mplanet
        / (
            rplanet
            * rplanet
        )
    )

    omega_rel = (
        24.0 / prot
    )

    radius_rel = rplanet

    # ========================================================
    # LATITUDE GRID
    # ========================================================

    lat = np.linspace(
        -89.5,
        89.5,
        nlat
    )

    latrad = np.deg2rad(
        lat
    )

    # ========================================================
    # HEAT CAPACITY
    # ========================================================

    C_land = 1.0e6
    C_ocean = 4.0e7

    C = (
        C_land
        * (
            1.0
            - ocean_fraction
        )
        +
        C_ocean
        * ocean_fraction
    )

    # ========================================================
    # TIME GRID
    # ========================================================

    steps_per_year = int(
        np.round(
            p_orbit / dt
        )
    )

    time_year = (
        np.arange(
            steps_per_year
        )
        * dt
    )

    # ========================================================
    # INITIAL TEMPERATURE
    # ========================================================

    T = np.full(
        nlat,
        288.0,
        dtype=np.float64
    )

    # ========================================================
    # PRECOMPUTE INSOLATION
    #
    # Orbit is identical every simulated year,
    # so Q only needs to be calculated once.
    # ========================================================

    Q_year = np.zeros(
        (
            steps_per_year,
            nlat
        ),
        dtype=np.float64
    )
    cosz_mean = np.zeros(
        (
            steps_per_year,
            nlat
        ),
        dtype=np.float64
    )
    cosz = np.zeros(
        (
            steps_per_year,
            nlat
        ),
        dtype=np.float64
    )
    Q_attenuated_0 = np.zeros(
        (
            steps_per_year,
            nlat
        ),
        dtype=np.float64
    )

    for step in range(
        steps_per_year
    ):

        day = (
            step * dt
        )

        Q_year[
          step,
            :
        ]  = daily_insolation_basic(
            day,
            latrad,
            S,
            p_orbit,
            mvelp,
            tilt,
            ecc
        )
        #Q_year = daily_insolation_basic(
        #    day,
        #    latrad,
        #    S,
        #    p_orbit,
        #    mvelp,
        #    tilt,
        #    ecc
        # )

    # ========================================================
    # DIFFUSION SCALING
    # ========================================================

    D_scaled = (
        D
        / (
            omega_rel
            * omega_rel
        )
    )
    D_ocean_efficiency = 1.0
    D_land_efficiency = 0.4  
    Df_landfrac = (1 - land_fraction) * D_ocean_efficiency + land_fraction * D_land_efficiency

    D_scaled = D_scaled * Df_landfrac
    transport_kerroin = (
        (
            patm
            / np.sqrt(
                gee_earths
            )
        )
        /
        (
            radius_rel
            * radius_rel
        )
    )

    # ========================================================
    # NUMBA CORE
    # ========================================================

    (
        T,
        T_year,
        debug_code,
        debug_year,
        debug_step,
        debug_lat,
        debug_value,
        max_abs_transport,
        max_abs_dTdt,
        max_abs_T
    ) = ebm_core_numba(
        T,
        Q_year,
        C,
        latrad,
        D_scaled,
        transport_kerroin,
        radius_rel,
        pco2,
        patm,
        dt,
        n_years,
        debug_numba
    )

    # ========================================================
    # DEBUG REPORT
    # ========================================================

    if debug_numba:

        print()
        print("=" * 72)
        print("NUMBA DEBUG")
        print("=" * 72)

        if debug_code == 0:

            print(
                "Status: OK"
            )

        else:

            print(
                "Status: INSTABILITY"
            )

            if debug_code == 1:
                print(
                    "Problem: albedo"
                )

            elif debug_code == 2:
                print(
                    "Problem: dT/dlat"
                )

            elif debug_code == 3:
                print(
                    "Problem: flux"
                )

            elif debug_code == 4:
                print(
                    "Problem: dflux/dlat"
                )

            elif debug_code == 5:
                print(
                    "Problem: transport"
                )

            elif debug_code == 6:
                print(
                    "Problem: dTdt"
                )

            elif debug_code == 7:
                print(
                    "Problem: temperature"
                )

            print(
                "Year       :",
                debug_year
            )

            print(
                "Step       :",
                debug_step
            )

            print(
                "Day        :",
                debug_step * dt
            )

            print(
                "Latitude   :",
                lat[debug_lat]
            )

            print(
                "Bad value  :",
                debug_value
            )

        print()
        print(
            "Max |transport| :",
            max_abs_transport
        )

        print(
            "Max |dT/dt|     :",
            max_abs_dTdt
        )

        print(
            "Max |T|         :",
            max_abs_T
        )

        print("=" * 72)

    # ========================================================
    # ANNUAL DIAGNOSTICS
    # ========================================================

    Tannual = np.mean(
        T_year,
        axis=0
    )

    Tmax = np.max(
        T_year,
        axis=0
    )

    Tmin = np.min(
        T_year,
        axis=0
    )

    # ========================================================
    # GLOBAL COSINE WEIGHTED MEAN
    # ========================================================

    weights = np.cos(
        latrad
    )

    Tglobal = (
        np.sum(
            Tannual
            * weights
        )
        /
        np.sum(
            weights
        )
    )

    # ========================================================
    # PERMANENT ICE
    # ========================================================

    ice_south, ice_north = (
        permanent_ice_cap(
            lat,
            Tannual,
            -10.0
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "lat":
            lat,

        "latrad":
            latrad,

        "T":
            T,

        "T_year":
            T_year,

        "Tannual":
            Tannual,

        "Tmax":
            Tmax,

        "Tmin":
            Tmin,

        "Q_year":
            Q_year,

        "time_year":
            time_year,

        "Tglobal":
            Tglobal,

        "ice_south":
            ice_south,

        "ice_north":
            ice_north,

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        "mplanet":
            mplanet,

        "rplanet":
            rplanet,

        "S":
            S,

        "p_orbit":
            p_orbit,

        "prot":
            prot,

        "mvelp":
            mvelp,

        "tilt":
            tilt,

        "ecc":
            ecc,

        "patm":
            patm,

        "pco2":
            pco2,

        "D":
            D,

        "ocean_fraction":
            ocean_fraction,

        # ----------------------------------------------------
        # Numba debug
        # ----------------------------------------------------

        "numba_debug_code":
            debug_code,

        "numba_debug_year":
            debug_year,

        "numba_debug_step":
            debug_step,

        "numba_debug_lat":
            debug_lat,

        "numba_debug_value":
            debug_value,

        "numba_max_abs_transport":
            max_abs_transport,

        "numba_max_abs_dTdt":
            max_abs_dTdt,

        "numba_max_abs_T":
            max_abs_T
    }

    return results





# ============================================================
# DEBUG: LATITUDE TEMPERATURE TABLE
# ============================================================

def print_temperature_debug(
        results,
        every=1):

    lat = results["lat"]

    Tannual_C = (
        results["Tannual"]
        - 273.15
    )

    Tmin_C = (
        results["Tmin"]
        - 273.15
    )

    Tmax_C = (
        results["Tmax"]
        - 273.15
    )

    print()
    print("=" * 72)
    print("DEBUG: LATITUDE TEMPERATURES")
    print("=" * 72)

    print(
        f"{'Lat [deg]':>10}"
        f"{'Mean [C]':>15}"
        f"{'Min [C]':>15}"
        f"{'Max [C]':>15}"
    )

    print("-" * 72)

    for i in range(
        0,
        len(lat),
        every
    ):

        print(
            f"{lat[i]:10.1f}"
            f"{Tannual_C[i]:15.2f}"
            f"{Tmin_C[i]:15.2f}"
            f"{Tmax_C[i]:15.2f}"
        )

    print("-" * 72)

    print(
        f"{'GLOBAL MEAN':>10}"
        f"{results['Tglobal'] - 273.15:15.2f}"
        f"{'':15}"
        f"{'':15}"
    )

    print("=" * 72)


# ============================================================
# PLOT: TEMPERATURE CURVES AT MULTIPLE LATITUDES
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def plot_temperature_curves(
        results,
        latitudes,
        object_name="Unknown object"):

    model_lat = results["lat"]
    time = results["time_year"]
    T_year = results["T_year"]

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    p_orbit = results["p_orbit"]
    S = results["S"]
    tilt = results["tilt"]
    ecc = results["ecc"]
    mvelp = results["mvelp"]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    for lat in latitudes:
        # Lasketaan värin voimakkuus leveysasteen itseisarvon mukaan:
        # Päiväntasaaja (0°) -> 1.0 (punainen), Navat (90°) -> 0.0 (sininen)
        color_val = 1.0 - (np.abs(lat) / 90.0)
        
        # Rajataan arvo varmuuden vuoksi välille [0.05, 0.95], jotta värit erottuvat hyvin
        color_val = np.clip(color_val, 0.05, 0.95)
        color = plt.cm.coolwarm(color_val)

        # Find nearest model latitude
        i = np.argmin(
            np.abs(
                model_lat - lat
            )
        )

        actual_lat = model_lat[i]

        temperature_C = (
            T_year[:, i]
            - 273.15
        )

        mean_T = np.mean(
            temperature_C
        )

        min_T = np.min(
            temperature_C
        )

        max_T = np.max(
            temperature_C
        )

        # ----------------------------------------------------
        # Temperature curve
        # ----------------------------------------------------

        plt.plot(
            time,
            temperature_C,
            color=color,
            lw=2,
            label=(
                f"{actual_lat:+.1f}° "
                f"mean={mean_T:.1f}°C "
                f"min={min_T:.1f}°C "
                f"max={max_T:.1f}°C"
            )
        )

    # --------------------------------------------------------
    # Freezing point & Plot formatting
    # --------------------------------------------------------

    plt.axhline(
        0,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="Jäätymispiste (0°C)"
    )
    
    plt.title(f"Lämpötilakäyrät - {object_name}")
    plt.xlabel("Aika (vuosi)")
    plt.ylabel("Lämpötila (°C)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

# ============================================================
# PLOT: TEMPERATURE CURVE AT SELECTED LATITUDE
# ============================================================

def plot_temperature_curve(
        results,
        lat,
        object_name="Unknown object"):

    latitudes = results["lat"]
    time = results["time_year"]
    T_year = results["T_year"]

    # --------------------------------------------------------
    # Find nearest model latitude
    # --------------------------------------------------------

    i = np.argmin(
        np.abs(
            latitudes - lat
        )
    )

    actual_lat = latitudes[i]

    temperature_C = (
        T_year[:, i]
        - 273.15
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    p_orbit = results["p_orbit"]
    S = results["S"]
    tilt = results["tilt"]
    ecc = results["ecc"]
    mvelp = results["mvelp"]

    # --------------------------------------------------------
    # Temperature statistics
    # --------------------------------------------------------

    mean_T = np.mean(
        temperature_C
    )

    min_T = np.min(
        temperature_C
    )

    max_T = np.max(
        temperature_C
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        time,
        temperature_C,
        color="orange",
        lw=2.5,
        label=f"T at {actual_lat:.1f}°"
    )

    # Annual mean
    plt.axhline(
        mean_T,
        color="black",
        ls="--",
        lw=1.5,
        label=f"Mean = {mean_T:.2f} °C"
    )

    # Freezing point
    plt.axhline(
        0,
        color="blue",
        ls=":",
        lw=1.5,
        label="0 °C"
    )

    # --------------------------------------------------------
    # Min / max markers
    # --------------------------------------------------------

    imin = np.argmin(
        temperature_C
    )

    imax = np.argmax(
        temperature_C
    )

    plt.scatter(
        time[imin],
        min_T,
        color="blue",
        zorder=5
    )

    plt.scatter(
        time[imax],
        max_T,
        color="red",
        zorder=5
    )

    plt.annotate(
        f"Min {min_T:.2f} °C",
        (
            time[imin],
            min_T
        ),
        xytext=(8, 8),
        textcoords="offset points",
        color="blue"
    )

    plt.annotate(
        f"Max {max_T:.2f} °C",
        (
            time[imax],
            max_T
        ),
        xytext=(8, -18),
        textcoords="offset points",
        color="red"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "Day of orbital year"
    )

    plt.ylabel(
        "Temperature [°C]"
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = (
        f"{object_name} — Temperature at "
        f"{actual_lat:.1f}° latitude\n"
        f"P = {p_orbit:.2f} d   |   "
        f"S = {S:.2f} W/m²   |   "
        f"tilt = {tilt:.2f}°   |   "
        f"ecc = {ecc:.5f}   |   "
        f"mvelp = {mvelp:.2f}°"
    )

    plt.title(
        title,
        fontsize=13
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()



# ============================================================
# OPTIONAL: TEMPERATURE CURVE
# ============================================================





# ============================================================
# PLOT: ANNUAL TEMPERATURE
# ============================================================

def plot_temperature(
        results,
        title="1D EBM"):

    lat = results["lat"]

    Tannual = (
        results["Tannual"]
        - 273.15
    )

    Tmax = (
        results["Tmax"]
        - 273.15
    )

    Tmin = (
        results["Tmin"]
        - 273.15
    )

    ice_south = results[
        "ice_south"
    ]

    ice_north = results[
        "ice_north"
    ]

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        lat,
        Tannual,
        color="orange",
        lw=4,
        label="Tannual"
    )

    plt.plot(
        lat,
        Tmax,
        color="red",
        lw=1.5,
        label="Tmax"
    )

    plt.plot(
        lat,
        Tmin,
        color="blue",
        lw=1.5,
        label="Tmin"
    )

    plt.axhline(
        0,
        color="black",
        ls=":",
        lw=2,
        label="0 C"
    )

    plt.axhline(
        -10,
        color="cyan",
        ls="--",
        lw=2,
        label="Permanent ice: -10 C"
    )

    plt.axvline(
        ice_south,
        color="cyan",
        ls="--",
        lw=2
    )

    plt.axvline(
        ice_north,
        color="cyan",
        ls="--",
        lw=2
    )

    plt.xlabel(
        "Latitude [deg]"
    )

    plt.ylabel(
        "Temperature [C]"
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# PLOT: SEASONAL TEMPERATURE
# ============================================================

# ============================================================
# PLOT: SEASONAL TEMPERATURE
# ============================================================

def plot_seasonal_temperature(
        results,
        object_name="Unknown object"):

    lat = results["lat"]
    latrad = results["latrad"]

    time = results["time_year"]
    T_year = results["T_year"]

    # ========================================================
    # PARAMETERS
    # ========================================================

    p_orbit = results["p_orbit"]
    S = results["S"]
    tilt = results["tilt"]
    ecc = results["ecc"]
    mvelp = results["mvelp"]

    # ========================================================
    # TEMPERATURE IN CELSIUS
    # ========================================================

    T_year_C = (
        T_year - 273.15
    )

    # ========================================================
    # COS(LATITUDE) WEIGHTS
    # ========================================================

    weights = np.cos(
        latrad
    )

    # ========================================================
    # LATITUDE MEAN / MIN / MAX
    # ========================================================

    Tannual = np.mean(
        T_year_C,
        axis=0
    )

    Tmin = np.min(
        T_year_C,
        axis=0
    )

    Tmax = np.max(
        T_year_C,
        axis=0
    )

    # ========================================================
    # COS(LATITUDE) WEIGHTED GLOBAL VALUES
    # ========================================================

    Tmean_global = (
        np.sum(
            Tannual * weights
        )
        /
        np.sum(weights)
    )

    Tmin_global_weighted = (
        np.sum(
            Tmin * weights
        )
        /
        np.sum(weights)
    )

    Tmax_global_weighted = (
        np.sum(
            Tmax * weights
        )
        /
        np.sum(weights)
    )

    # ========================================================
    # ABSOLUTE GLOBAL MINIMUM
    #
    # Lowest individual temperature anywhere on the planet
    # during the whole simulated orbital year.
    # ========================================================

    min_index = np.unravel_index(
        np.argmin(T_year_C),
        T_year_C.shape
    )

    min_time_index = min_index[0]
    min_lat_index = min_index[1]

    absolute_temp_min = (
        T_year_C[
            min_time_index,
            min_lat_index
        ]
    )

    absolute_temp_min_day = (
        time[min_time_index]
    )

    absolute_temp_min_lat = (
        lat[min_lat_index]
    )

    # ========================================================
    # ABSOLUTE GLOBAL MAXIMUM
    #
    # Highest individual temperature anywhere on the planet
    # during the whole simulated orbital year.
    # ========================================================

    max_index = np.unravel_index(
        np.argmax(T_year_C),
        T_year_C.shape
    )

    max_time_index = max_index[0]
    max_lat_index = max_index[1]

    absolute_temp_max = (
        T_year_C[
            max_time_index,
            max_lat_index
        ]
    )

    absolute_temp_max_day = (
        time[max_time_index]
    )

    absolute_temp_max_lat = (
        lat[max_lat_index]
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    # ========================================================
    # TEMPERATURE FIELD
    # ========================================================

    im = ax.imshow(
        T_year_C.T,
        aspect="auto",
        origin="lower",
        extent=[
            time[0],
            time[-1],
            lat[0],
            lat[-1]
        ],
        cmap="RdBu_r"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Temperature [°C]"
    )

    # ========================================================
    # TEMPERATURE CONTOURS
    # ========================================================

    levels = [
        -100,
        -75,
        -50,
        -40,
        -30,
        -20,
        -10,
        0,
        10,
        20,
        30,
        40,
        50,
        75,
        100
    ]

    cnt = ax.contour(
        time,
        lat,
        T_year_C.T,
        levels=levels,
        colors="black",
        linewidths=0.5
    )

    ax.clabel(
        cnt,
        fontsize=9,
        fmt="%g °C"
    )

    # ========================================================
    # MARK ABSOLUTE MINIMUM
    # ========================================================

    ax.scatter(
        absolute_temp_min_day,
        absolute_temp_min_lat,
        color="blue",
        s=70,
        edgecolor="white",
        linewidth=1.5,
        zorder=10
    )

    ax.annotate(
        f"ABS MIN\n"
        f" T {absolute_temp_min:.1f} °C\n"
        f" lat. {absolute_temp_min_lat:+.1f}°",
        (
            absolute_temp_min_day,
            absolute_temp_min_lat
        ),
        xytext=(10, -35),
        textcoords="offset points",
        color="blue",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85
        )
    )

    # ========================================================
    # MARK ABSOLUTE MAXIMUM
    # ========================================================

    ax.scatter(
        absolute_temp_max_day,
        absolute_temp_max_lat,
        color="red",
        s=70,
        edgecolor="white",
        linewidth=1.5,
        zorder=10
    )

    ax.annotate(
        f"ABS MAX\n"
        f" T {absolute_temp_max:.1f} °C\n"
        f" lat. {absolute_temp_max_lat:+.1f}°",
        (
            absolute_temp_max_day,
            absolute_temp_max_lat
        ),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85
        )
    )

    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(
        "Day of orbital year"
    )

    ax.set_ylabel(
        "Latitude [deg]"
    )

    # ========================================================
    # TITLE
    # ========================================================

    title = (
        f"{object_name} — Seasonal Temperature\n"
        f"P = {p_orbit:.2f} d   |   "
        f"S = {S:.2f} W/m²   |   "
        f"tilt = {tilt:.2f}°   |   "
        f"ecc = {ecc:.5f}   |   "
        f"mvelp = {mvelp:.2f}°"
    )

    ax.set_title(
        title,
        fontsize=13
    )

    # ========================================================
    # SUMMARY BELOW PLOT
    # ========================================================

    summary = (
        f"Cos(lat) weighted:  "
        f"Tmean = {Tmean_global:.2f} °C   |   "
        f"Tmin(mean of lat minima) = "
        f"{Tmin_global_weighted:.2f} °C   |   "
        f"Tmax(mean of lat maxima) = "
        f"{Tmax_global_weighted:.2f} °C\n"
        f"ABSOLUTE MIN = "
        f"{absolute_temp_min:.2f} °C "
        f"at {absolute_temp_min_lat:+.1f}° "
        f"on day {absolute_temp_min_day:.1f}"
        f"   |   "
        f"ABSOLUTE MAX = "
        f"{absolute_temp_max:.2f} °C "
        f"at {absolute_temp_max_lat:+.1f}° "
        f"on day {absolute_temp_max_day:.1f}"
    )

    fig.text(
        0.5,
        0.018,
        summary,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95
        )
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[
            0,
            0.105,
            1,
            1
        ]
    )

    plt.show()

    # ========================================================
    # DEBUG PRINT
    # ========================================================

    print()
    print("=" * 72)
    print("SEASONAL TEMPERATURE EXTREMES")
    print("=" * 72)

    print(
        f"Absolute minimum : "
        f"{absolute_temp_min:.2f} °C"
    )

    print(
        f"  latitude       : "
        f"{absolute_temp_min_lat:+.2f}°"
    )

    print(
        f"  orbital day    : "
        f"{absolute_temp_min_day:.2f}"
    )

    print()

    print(
        f"Absolute maximum : "
        f"{absolute_temp_max:.2f} °C"
    )

    print(
        f"  latitude       : "
        f"{absolute_temp_max_lat:+.2f}°"
    )

    print(
        f"  orbital day    : "
        f"{absolute_temp_max_day:.2f}"
    )

    print()

    print(
        f"Cos(lat) Tmean   : "
        f"{Tmean_global:.2f} °C"
    )

    print(
        f"Weighted Tmin    : "
        f"{Tmin_global_weighted:.2f} °C"
    )

    print(
        f"Weighted Tmax    : "
        f"{Tmax_global_weighted:.2f} °C"
    )

    print("=" * 72)




# ============================================================
# PLOT: INSOLATION
# ============================================================

def plot_insolation(
        results):

    lat = results["lat"]
    time = results["time_year"]
    Q_year = results["Q_year"]

    plt.figure(
        figsize=(11, 7)
    )

    plt.imshow(
        Q_year.T,
        aspect="auto",
        origin="lower",
        extent=[
            time[0],
            time[-1],
            lat[0],
            lat[-1]
        ],
        cmap="inferno"
    )

    plt.colorbar(
        label="Insolation [W/m2]"
    )

    plt.xlabel(
        "Day of orbital year"
    )

    plt.ylabel(
        "Latitude [deg]"
    )

    plt.title(
        "Seasonal insolation"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # NUMBA WARM-UP
    # ============================================================
    print()
    print("Numba warm-up...")
    #planetname="Test"
    planetname="KOI 4878.01"

    #_ = main_ebm_numba(D=0.5)
    # KOI 4878.01
    _ = main_ebm_numba(
    mplanet=1.14,
    rplanet=np.pow(1.14, 0.27),
    S=1361.0 * 0.92,
    p_orbit=449,
    prot=24,
    mvelp=102.0,
    tilt=23.44,
    ecc=0.0167,
    patm=1.0,
    pco2=280,
    n_years=100, D=0.5
    )
    # ============================================================
    # NUMBA
    # ============================================================
    t0 = time.perf_counter()
    #results_numba = main_ebm_numba(D=0.5)
    # KOI 4878.01
    #results = main_ebm_old(
    results = main_ebm_numba(
    mplanet=1.14,
    rplanet=np.pow(1.14, 0.27),
    S=1361.0 * 0.92,
    p_orbit=449,
    prot=24,
    mvelp=102.0,
    tilt=23.44,
    ecc=0.0167,
    patm=1.0,
    pco2=280,
    n_years=100, D=0.5
    )
    numba_time = (
    time.perf_counter()
    - t0
    )
    # ============================================================
    # BENCHMARK
    # ============================================================
    print()
    print("=" * 60)
    print("EBM BENCHMARK")
    print("=" * 60)
    print(
    f"Numba : {numba_time:.3f} s"
    )
    print("=" * 60)
    # ============================================================
    # RESULT COMPARISON
    # ============================================================
    Tglobal_numba = (
    results["Tglobal"]
    - 273.15
    )
    print()
    print("RESULT COMPARISON")
    print("=" * 60)
    print(
    f"Tglobal Numba : "
    f"{Tglobal_numba:.6f} °C"
    )
    print("=" * 60)
    print("=" * 60)

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    print()

    print(
        "Global annual mean:",
        round(
            results["Tglobal"]
            - 273.15,
            2
        ),
        "C"
    )

    print(
        "Permanent ice south:",
        round(
            results["ice_south"],
            2
        ),
        "deg"
    )

    print(
        "Permanent ice north:",
        round(
            results["ice_north"],
            2
        ),
        "deg"
    )


    # ========================================================
    # DEBUG TABLE
    #
    # every=1  -> kaikki leveysasteet
    # every=2  -> joka toinen
    # every=5  -> noin 5 asteen välein
    # every=10 -> noin 10 asteen välein
    # ========================================================

    print_temperature_debug(
        results,
        every=5
    )


    # ========================================================
    # PLOTS
    # ========================================================


    plot_seasonal_temperature(
    results,
    object_name=planetname
    )
    plot_temperature_curves(
    results,
    latitudes=[-60, -30, 0, 30, 60],
    object_name=planetname
    )


    #plot_seasonal_temperature(
    #    results
    #)
    
    #plot_temperature_curve(
    #results,
    #lat=0,
    #object_name=planetname
    #)

    plot_temperature(
        results
    )
    plot_insolation(
        results
    )
