
# -*- coding: utf-8 -*-


## python3 simple 2d seasonal planet energy balance model
## for calculation of temperature
## 15.9.2026 v 0000.0002


import numpy as np
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
    """
    Calculate orbital distance and solar longitude.

    day      : day of orbital year
    p_orbit  : orbital period [days]
    mvelp    : longitude of perihelion [deg]
    ecc      : eccentricity
    """

    M = 2.0 * np.pi * day / p_orbit

    # Solve Kepler equation
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

def daily_insolation(
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

    # Solar declination
    declination = np.arcsin(
        np.sin(np.deg2rad(tilt))
        * np.sin(solar_longitude)
    )

    # Solar flux at planet
    solar_flux = S / r**2

    # Sunset hour angle
    cos_h0 = (
        -np.tan(latrad)
        * np.tan(declination)
    )

    h0 = np.arccos(
        np.clip(cos_h0, -1.0, 1.0)
    )

    # Polar day
    h0 = np.where(
        cos_h0 <= -1.0,
        np.pi,
        h0
    )

    # Polar night
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
        5.35* np.log(((patm*pco2) / co2_reference))
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
# DIFFUSION
# ============================================================

def diffusion_spherical_1(
        T,
        latrad,
        D):

    dlat = (
        latrad[1]
        - latrad[0]
    )

    coslat = np.cos(latrad)

    dT_dlat = np.gradient(
        T,
        dlat
    )

    flux = (
        coslat
        * dT_dlat
    )

    divergence = np.gradient(
        flux,
        dlat
    )

    return (
        D
        * divergence
        / np.maximum(
            coslat,
            0.05
        )
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

    # Southern hemisphere
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

    # Northern hemisphere
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

def diffusion_spherical_2(T, latrad, D, R):
    """
    1D-diffuusio pallon pinnalla (leveysasteittain).
    R: Planeetan säde metreinä
    """
    dlat = np.gradient(latrad)
    
    # 1. Gradientti: dT / d_metrit. 
    # Koska etäisyys metreinä on d_metrit = R * dlat
    dT_dlat = np.gradient(T, dlat)
    flux = -D * (dT_dlat / R) 
    
    # Pallogeometrian korjaus vuolle
    flux_spherical = flux * np.cos(latrad)
    
    # 2. Divergenssi: d(vuo) / d_metrit
    # Tässä jaetaan taas R:llä, jolloin R^2 päätyy automaattisesti jakajaksi
    divergence = - (1 / (R * np.cos(latrad))) * np.gradient(flux_spherical, dlat)
    
    return divergence

# ============================================================
# MAIN EBM
# ============================================================

def main_ebm(mplanet=1.0, rplanet=1.0,
        S=1361.0,
        p_orbit=365.25, prot=24,
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

    gee_earths=mplanet/(rplanet*rplanet)
    omega_rel=24/prot
    radius_rel=rplanet
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
    # Storage for final year
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

            # Solar radiation
            Q = daily_insolation(
                day,
                latrad,
                S,
                p_orbit,
                mvelp,
                tilt,
                ecc
            )

            # Albedo
            alpha = calculate_albedo(
                T
            )

            absorbed = (
                Q * (1.0 - alpha)
            )

            # OLR
            OLR = outgoing_longwave(
                T,
                pco2,
                patm
            )

            D_scaled = D / (omega_rel**2)

            transport_kerroin = (patm / np.sqrt(gee_earths)) / (radius_rel**2)
            transport=transport_kerroin*diffusion_spherical_1(T,latrad,D)
            #transport = transport_kerroin * diffusion_spherical(T,latrad,D_scaled, radius_rel)

            # Energy balance
            dTdt = (
                absorbed
                - OLR
                + transport
            ) / C

            # Update
            T += (
                dTdt
                * dt
                * DAY
            )

            # Store final year
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

    # --------------------------------------------------------
    # Annual diagnostics
    # --------------------------------------------------------

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

    # Latitude weighted global mean
    weights = np.cos(latrad)

    Tglobal = (
        np.sum(
            Tannual * weights
        )
        /
        np.sum(weights)
    )

    # Permanent ice cap
    ice_south, ice_north = (
        permanent_ice_cap(
            lat,
            Tannual,
            -10.0
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

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
        "ice_north": ice_north
    }

    return results


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

    # Water freezing
    plt.axhline(
        0,
        color="black",
        ls=":",
        lw=2,
        label="0 C"
    )

    # Permanent ice criterion
    plt.axhline(
        -10,
        color="cyan",
        ls="--",
        lw=2,
        label="Permanent ice: -10 C"
    )

    # Ice cap boundaries
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

def plot_seasonal_temperature(
        results):

    lat = results["lat"]
    time = results["time_year"]
    T_year = results["T_year"]

    plt.figure(
        figsize=(11, 7)
    )

    plt.imshow(
        T_year.T - 273.15,
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
    plt.colorbar(
        label="Temperature [C]"
    )
    cnt=plt.contour(
        T_year.T - 273.15,
        levels=[-100, -75,-50,-40,-30,-20,-10,0,10,20,30,40,50,75,100],
        aspect="auto",
        origin="lower",
        extent=[
            time[0],
            time[-1],
            lat[0],
            lat[-1]
        ]
    )
    plt.clabel(cnt, fontsize=16)    


    plt.xlabel(
        "Day of orbital year"
    )

    plt.ylabel(
        "Latitude [deg]"
    )

    plt.title(
        "Seasonal temperature"
    )

    plt.tight_layout()

    plt.show()


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
    ## earth
    #results = main_ebm(
    #    mplanet=1.0, rplanet=1.0,
    #    S=1361.0,
    #    p_orbit=365.25,
    #    prot=24,
    #    mvelp=102.0,
    #    tilt=23.44,
    #    ecc=0.0167,
    #    patm=1.0,
    #    pco2=420.0
    #)
    ##KOI 4878.01
    results = main_ebm(
        mplanet=1.14, rplanet=np.pow(1.14, 0.27),
        S=1361.0*0.92,
        p_orbit=449,
        prot=24,
        mvelp=102.0,
        tilt=23.44,
        ecc=0.0167,
        patm=1.0,
        pco2=280
    )
    print()
    print(
        "Global annual mean:",
        round(
            results["Tglobal"] - 273.15,
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

    plot_temperature(
        results
    )

    plot_seasonal_temperature(
        results
    )

    plot_insolation(
        results
    )
