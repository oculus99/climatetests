
import numpy as np
import xarray as xr
from dataclasses import dataclass


# ============================================================
# VAKIOT
# ============================================================

SIGMA = 5.670374419e-8
SECONDS_PER_DAY = 86400.0
WATER_DENSITY = 1000.0


# ============================================================
# PLANEETAN PARAMETRIT
# ============================================================

@dataclass
class Planet:

    # --------------------------------------------------------
    # TÄHTI
    # --------------------------------------------------------

    S0: float = 1361.0

    # --------------------------------------------------------
    # RATA
    # --------------------------------------------------------

    ecc: float = 0.0167

    year_length: float = 365.25

    mvelp: float = 102.9

    # --------------------------------------------------------
    # PYÖRIMINEN
    # --------------------------------------------------------

    day_length: float = 24.0

    tilt: float = 23.44

    # --------------------------------------------------------
    # ILMAKEHÄ
    # --------------------------------------------------------

    co2: float = 420.0

    reference_co2: float = 280.0

    # W/m2 per ln(CO2/reference)
    climate_sensitivity: float = 5.35

    # --------------------------------------------------------
    # KONVEKTIO
    # --------------------------------------------------------

    convection_cells: int = 3

    # --------------------------------------------------------
    # NUMERIIKKA
    # --------------------------------------------------------

    nlat: int = 90

    nlon: int = 180

    steps_per_year: int = 120

    # --------------------------------------------------------
    # PINTA
    # --------------------------------------------------------

    # Lumettoman / jäättömän pinnan albedo
    surface_albedo: float = 0.30

    # Jään albedo
    ice_albedo: float = 0.65

    # --------------------------------------------------------
    # LÄMPÖKAPASITEETTI
    # --------------------------------------------------------

    # J/m2/K
    ocean_heat_capacity: float = 4.0e8

    # --------------------------------------------------------
    # LÄMMÖNKULJETUS
    # --------------------------------------------------------

    # W/m2/K
    heat_transport: float = 2.0

    # --------------------------------------------------------
    # VESI
    # --------------------------------------------------------

    initial_water: float = 25.0

    initial_humidity: float = 0.75

    # --------------------------------------------------------
    # DEM
    # --------------------------------------------------------

    dem: np.ndarray = None


# ============================================================
# ILMASTOMALLI
# ============================================================

class ClimateModel:

    def __init__(self, planet):

        self.p = planet

        # ----------------------------------------------------
        # TARKISTUKSET
        # ----------------------------------------------------

        if self.p.convection_cells < 1:
            raise ValueError(
                "convection_cells pitää olla vähintään 1"
            )

        if self.p.steps_per_year < 12:
            raise ValueError(
                "steps_per_year pitää olla vähintään 12"
            )

        # ----------------------------------------------------
        # LEVEYSASTEET
        # ----------------------------------------------------

        self.lat = np.linspace(
            -90.0 + 90.0 / self.p.nlat,
            90.0 - 90.0 / self.p.nlat,
            self.p.nlat
        )

        # ----------------------------------------------------
        # PITUUSASTEET
        # ----------------------------------------------------

        self.lon = np.linspace(
            0.0,
            360.0,
            self.p.nlon,
            endpoint=False
        )

        # ----------------------------------------------------
        # 2D-HILA
        # ----------------------------------------------------

        self.lat2d, self.lon2d = np.meshgrid(
            self.lat,
            self.lon,
            indexing="ij"
        )

        self.lat_rad = np.deg2rad(
            self.lat2d
        )

        self.lon_rad = np.deg2rad(
            self.lon2d
        )

        # ----------------------------------------------------
        # DEM
        # ----------------------------------------------------

        if self.p.dem is None:

            self.dem = np.zeros(
                (
                    self.p.nlat,
                    self.p.nlon
                ),
                dtype=float
            )

        else:

            self.dem = np.asarray(
                self.p.dem,
                dtype=float
            )

            if self.dem.shape != (
                self.p.nlat,
                self.p.nlon
            ):

                raise ValueError(
                    "DEM:n koko ei vastaa hilaa"
                )

        # ----------------------------------------------------
        # AIKA
        # ----------------------------------------------------

        self.year_seconds = (
            self.p.year_length *
            SECONDS_PER_DAY
        )

        self.dt = (
            self.year_seconds /
            self.p.steps_per_year
        )

        # ----------------------------------------------------
        # PYÖRIMISNOPEUS
        # ----------------------------------------------------

        self.omega = (
            2.0 *
            np.pi /
            (
                self.p.day_length *
                3600.0
            )
        )

        # ----------------------------------------------------
        # LÄMPÖKAPASITEETTI
        # ----------------------------------------------------

        lat_factor = np.cos(
            np.deg2rad(self.lat)
        )

        heat_1d = (
            self.p.ocean_heat_capacity *
            (
                0.70 +
                0.30 * lat_factor
            )
        )

        self.heat_capacity = np.repeat(
            heat_1d[:, None],
            self.p.nlon,
            axis=1
        )

        # ----------------------------------------------------
        # ALKULÄMPÖTILA
        #
        # Maa-tyyppinen lähtöjakauma.
        # ----------------------------------------------------

        lat_temperature = (
            288.0
            -
            35.0 *
            np.sin(
                np.deg2rad(
                    self.lat
                )
            ) ** 2
        )

        self.temperature = np.repeat(
            lat_temperature[:, None],
            self.p.nlon,
            axis=1
        )

        # ----------------------------------------------------
        # KOSTEUS
        # ----------------------------------------------------

        self.humidity = np.full(
            (
                self.p.nlat,
                self.p.nlon
            ),
            self.p.initial_humidity,
            dtype=float
        )

        # ----------------------------------------------------
        # PINTAVESI
        #
        # kg/m2
        # ----------------------------------------------------

        self.water = np.full(
            (
                self.p.nlat,
                self.p.nlon
            ),
            self.p.initial_water,
            dtype=float
        )

        # ----------------------------------------------------
        # JÄÄ
        #
        # 0 = ei jäätä
        # 1 = täysin jäässä
        #
        # Aloitetaan ilman jäätä.
        # Malli muodostaa jään itse.
        # ----------------------------------------------------

        self.ice_fraction = np.zeros(
            (
                self.p.nlat,
                self.p.nlon
            ),
            dtype=float
        )


    # ========================================================
    # ORBITA
    # ========================================================

    def orbital_position(self, fraction):

        e = self.p.ecc

        M = (
            2.0 *
            np.pi *
            fraction
        )

        E = M

        for _ in range(10):

            E -= (
                E -
                e * np.sin(E) -
                M
            ) / (
                1.0 -
                e * np.cos(E)
            )

        nu = (
            2.0 *
            np.arctan2(

                np.sqrt(
                    1.0 + e
                ) *
                np.sin(
                    E / 2.0
                ),

                np.sqrt(
                    1.0 - e
                ) *
                np.cos(
                    E / 2.0
                )
            )
        )

        r = (
            1.0 -
            e * np.cos(E)
        )

        longitude = (
            nu +
            np.deg2rad(
                self.p.mvelp
            )
        )

        return r, longitude


    # ========================================================
    # AURINGON SÄTEILY
    # ========================================================

    def solar_flux(self, fraction):

        r, solar_longitude = (
            self.orbital_position(
                fraction
            )
        )

        # Aurinkovakio etäisyyden funktiona

        S = (
            self.p.S0 /
            (r * r)
        )

        tilt = np.deg2rad(
            self.p.tilt
        )

        # Auringon deklinaatio

        declination = np.arcsin(

            np.sin(tilt) *
            np.sin(
                solar_longitude
            )
        )

        lat = np.deg2rad(
            self.lat
        )

        x = (
            -np.tan(lat) *
            np.tan(declination)
        )

        polar_day = (
            x <= -1.0
        )

        polar_night = (
            x >= 1.0
        )

        x = np.clip(
            x,
            -1.0,
            1.0
        )

        H0 = np.arccos(x)

        Q = (

            S /
            np.pi

        ) * (

            H0 *
            np.sin(lat) *
            np.sin(declination)

            +

            np.cos(lat) *
            np.cos(declination) *
            np.sin(H0)
        )

        # Napapäivä

        Q[polar_day] = (

            S *
            np.sin(
                lat[polar_day]
            ) *
            np.sin(
                declination
            )
        )

        # Polaarinen yö

        Q[polar_night] = 0.0

        Q = np.maximum(
            Q,
            0.0
        )

        return np.repeat(
            Q[:, None],
            self.p.nlon,
            axis=1
        )


    # ========================================================
    # CO2-FORCING
    # ========================================================

    def co2_forcing(self):

        return (

            self.p.climate_sensitivity *

            np.log(
                self.p.co2 /
                self.p.reference_co2
            )
        )


    # ========================================================
    # ILMAKEHÄN KASVIHUONEVAIKUTUS
    # ========================================================

    def atmospheric_emissivity(self):

        """
        Yksinkertaistettu harmaan ilmakehän emissiivisyys.

        Suurempi CO2 -> pienempi tehokas emissiivisyys
        -> vähemmän pitkäaaltoista säteilyä karkaa avaruuteen.
        """

        forcing = (
            self.co2_forcing()
        )

        # Referenssissä noin 0.61

        epsilon = (
            0.61 *
            np.exp(
                -0.018 *
                (
                    forcing -
                    2.0
                )
            )
        )

        return np.clip(
            epsilon,
            0.45,
            0.70
        )


    # ========================================================
    # PILVI / VESIHÖYRY
    # ========================================================

    def cloud_factor(self):

        """
        Hyvin yksinkertainen pilvivaikutus.

        Kostea ilma kasvattaa hieman albedoa.
        """

        humidity_effect = (
            self.humidity -
            0.50
        )

        return np.clip(
            0.02 *
            humidity_effect,
            -0.01,
            0.02
        )


    # ========================================================
    # ALBEDO
    # ========================================================

    def albedo_field(self):

        # Jään vaikutus

        ice_albedo = (

            self.p.surface_albedo *
            (
                1.0 -
                self.ice_fraction
            )

            +

            self.p.ice_albedo *
            self.ice_fraction
        )

        # Pilvien pieni vaikutus

        albedo = (
            ice_albedo +
            self.cloud_factor()
        )

        return np.clip(
            albedo,
            0.05,
            0.90
        )


    # ========================================================
    # JÄÄN TERMODYNAAMIIKKA
    # ========================================================

    def update_ice(self):

        T = self.temperature

        freezing_point = 273.15

        delta = (
            T -
            freezing_point
        )

        # ----------------------------------------------------
        # JÄÄTYMINEN
        # ----------------------------------------------------

        freezing_strength = np.clip(
            -delta / 12.0,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # SULAMINEN
        # ----------------------------------------------------

        melting_strength = np.clip(
            delta / 8.0,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # JÄÄN KASVU
        #
        # Kasvu hidastuu, kun pinta on jo lähes täysin jäässä.
        # ----------------------------------------------------

        growth = (

            0.020 *

            freezing_strength *

            (
                1.0 -
                self.ice_fraction
            )
        )

        # ----------------------------------------------------
        # JÄÄN SULAMINEN
        # ----------------------------------------------------

        melt = (

            0.035 *

            melting_strength *

            self.ice_fraction
        )

        # ----------------------------------------------------
        # NETTOMUUTOS
        # ----------------------------------------------------

        self.ice_fraction += (
            growth -
            melt
        )

        self.ice_fraction = np.clip(
            self.ice_fraction,
            0.0,
            1.0
        )

        return self.ice_fraction.copy()


    # ========================================================
    # VESIHÖYRYN KYLLÄSTYSHÖYRY
    # ========================================================

    def saturation_vapour_pressure(
        self,
        temperature
    ):

        Tc = (
            temperature -
            273.15
        )

        Tc = np.clip(
            Tc,
            -80.0,
            60.0
        )

        return (

            610.94 *

            np.exp(

                17.625 *
                Tc /
                (
                    Tc +
                    243.04
                )
            )
        )


    # ========================================================
    # HAIHTUMINEN
    # ========================================================

    def evaporation(self):

        es = (
            self.saturation_vapour_pressure(
                self.temperature
            )
        )

        # ----------------------------------------------------
        # Suhteellinen kuivuus
        # ----------------------------------------------------

        dry = np.clip(

            1.0 -
            self.humidity,

            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Lämpötilakerroin
        # ----------------------------------------------------

        warm = np.clip(

            (
                self.temperature -
                250.0
            ) / 45.0,

            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Jää vähentää haihtumista
        # ----------------------------------------------------

        surface_factor = (

            1.0 -
            0.85 *
            self.ice_fraction
        )

        # ----------------------------------------------------
        # Haihtuminen kg/m2/s
        # ----------------------------------------------------

        evap = (

            1.5e-6 *

            warm *

            dry *

            np.sqrt(
                es /
                1000.0
            ) *

            surface_factor
        )

        # Ei voi haihduttaa enempää kuin vettä on.

        max_evap = (
            self.water /
            self.dt
        )

        evap = np.minimum(
            evap,
            max_evap
        )

        return np.maximum(
            evap,
            0.0
        )


    # ========================================================
    # KONVEKTIOSOLUT
    # ========================================================

    def convection_pattern(self):

        n = self.p.convection_cells

        lat = self.lat

        abs_lat = np.abs(
            lat
        )

        x = (
            abs_lat /
            90.0
        )

        cell = np.floor(
            x * n
        ).astype(int)

        cell = np.minimum(
            cell,
            n - 1
        )

        local = (
            x * n -
            cell
        )

        direction = np.where(
            cell % 2 == 0,
            1.0,
            -1.0
        )

        hemisphere = np.where(
            lat >= 0.0,
            1.0,
            -1.0
        )

        circulation = (

            np.sin(
                np.pi *
                local
            )

            *

            direction

            *

            hemisphere
        )

        vy_1d = (
            7.0 *
            circulation
        )

        vz_1d = (

            0.02 *

            np.cos(
                np.pi *
                local
            )

            *

            direction

            *

            hemisphere
        )

        vy = np.repeat(

            vy_1d[:, None],

            self.p.nlon,

            axis=1
        )

        vz = np.repeat(

            vz_1d[:, None],

            self.p.nlon,

            axis=1
        )

        return vy, vz


    # ========================================================
    # TUULIKENTTÄ
    # ========================================================

    def wind_field(self):

        vy, vz = (
            self.convection_pattern()
        )

        # ----------------------------------------------------
        # LÄMPÖTILAN LEVEYSASTEGRADIENTTI
        # ----------------------------------------------------

        Tmean = np.mean(
            self.temperature,
            axis=1
        )

        dlat = np.deg2rad(
            self.lat[1] -
            self.lat[0]
        )

        gradient = np.gradient(
            Tmean,
            dlat
        )

        # ----------------------------------------------------
        # ZONAALINEN TUULI
        # ----------------------------------------------------

        rotation_factor = np.clip(

            24.0 /
            self.p.day_length,

            0.25,
            4.0
        )

        vx_1d = (

            8.0 *

            np.sin(
                2.0 *
                np.deg2rad(
                    self.lat
                )
            )

            *

            rotation_factor
        )

        vx_1d += (
            -0.5 *
            gradient
        )

        vx_1d = np.clip(
            vx_1d,
            -40.0,
            40.0
        )

        vx = np.repeat(

            vx_1d[:, None],

            self.p.nlon,

            axis=1
        )

        # ----------------------------------------------------
        # DEM
        # ----------------------------------------------------

        if np.any(
            self.dem != 0.0
        ):

            dz = np.gradient(
                self.dem,
                axis=0
            )

            vz += (
                -0.00002 *
                dz
            )

        vy = np.clip(
            vy,
            -25.0,
            25.0
        )

        vz = np.clip(
            vz,
            -2.0,
            2.0
        )

        wind = np.empty(

            (
                self.p.nlat,
                self.p.nlon,
                3
            )
        )

        wind[..., 0] = vx
        wind[..., 1] = vy
        wind[..., 2] = vz

        return wind


    # ========================================================
    # MERIDIONAALINEN LÄMMÖNKULJETUS
    # ========================================================

    def heat_transport(self):

        T = self.temperature

        # ----------------------------------------------------
        # Leveysastegradientti
        # ----------------------------------------------------

        dT = np.gradient(
            T,
            axis=0
        )

        # ----------------------------------------------------
        # Lämmönkuljetus lämpimästä kylmään.
        # ----------------------------------------------------

        flux = (
            -self.p.heat_transport *
            dT
        )

        # Divergenssi

        divergence = np.gradient(
            flux,
            axis=0
        )

        return -divergence


    # ========================================================
    # SADE
    # ========================================================

    def precipitation(
        self,
        evaporation,
        wind
    ):

        speed = np.sqrt(

            wind[..., 0] ** 2 +

            wind[..., 1] ** 2 +

            wind[..., 2] ** 2
        )

        # ----------------------------------------------------
        # Lämpötilakerroin
        # ----------------------------------------------------

        warm = np.clip(

            (
                self.temperature -
                250.0
            ) / 35.0,

            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Konvektio
        # ----------------------------------------------------

        convection = (

            0.30 +
            0.70 *
            warm
        )

        # ----------------------------------------------------
        # Tuulen vaikutus
        # ----------------------------------------------------

        transport = (

            1.0 +
            0.008 *
            speed
        )

        # ----------------------------------------------------
        # Sade kg/m2/s
        # ----------------------------------------------------

        rain = (

            evaporation *

            convection *

            transport
        )

        return np.maximum(
            rain,
            0.0
        )


    # ========================================================
    # YKSI AIKA-ASKEL
    # ========================================================

    def step(self, fraction):

        # ----------------------------------------------------
        # AURINKO
        # ----------------------------------------------------

        solar = (
            self.solar_flux(
                fraction
            )
        )

        # ----------------------------------------------------
        # JÄÄ
        # ----------------------------------------------------

        self.update_ice()

        # ----------------------------------------------------
        # ALBEDO
        # ----------------------------------------------------

        albedo = (
            self.albedo_field()
        )

        # ----------------------------------------------------
        # ABSORBOITU AURINGON SÄTEILY
        # ----------------------------------------------------

        absorbed = (

            solar *

            (
                1.0 -
                albedo
            )
        )

        # ----------------------------------------------------
        # KASVIHUONEILMIÖ
        # ----------------------------------------------------

        epsilon = (
            self.atmospheric_emissivity()
        )

        # ----------------------------------------------------
        # ULOSMENEVA PITKÄAALTOSÄTEILY
        # ----------------------------------------------------

        outgoing = (

            epsilon *

            SIGMA *

            self.temperature ** 4
        )

        # ----------------------------------------------------
        # CO2-FORCING
        # ----------------------------------------------------

        co2_forcing = (
            self.co2_forcing()
        )

        # ----------------------------------------------------
        # LÄMMÖNKULJETUS
        # ----------------------------------------------------

        transport = (
            self.heat_transport()
        )

        # ----------------------------------------------------
        # ENERGIATASE
        # ----------------------------------------------------

        net_flux = (

            absorbed

            -

            outgoing

            +

            0.35 *
            co2_forcing

            +

            transport
        )

        # ----------------------------------------------------
        # LÄMPÖTILAN MUUTOS
        # ----------------------------------------------------

        dT = (

            net_flux *

            self.dt /

            self.heat_capacity
        )

        # ----------------------------------------------------
        # NUMEERINEN RAJOITUS
        # ----------------------------------------------------

        dT = np.clip(
            dT,
            -1.0,
            1.0
        )

        self.temperature += dT

        self.temperature = np.clip(

            self.temperature,

            150.0,
            400.0
        )

        # ----------------------------------------------------
        # TUULI
        # ----------------------------------------------------

        wind = (
            self.wind_field()
        )

        # ----------------------------------------------------
        # HAIHTUMINEN
        # ----------------------------------------------------

        evaporation = (
            self.evaporation()
        )

        # ----------------------------------------------------
        # SADE
        # ----------------------------------------------------

        rain = (
            self.precipitation(
                evaporation,
                wind
            )
        )

        # ----------------------------------------------------
        # VESIBUDJETTI
        # ----------------------------------------------------

        evaporation_amount = (

            evaporation *
            self.dt
        )

        rain_amount = (

            rain *
            self.dt
        )

        self.water += (

            evaporation_amount
            -
            rain_amount
        )

        self.water = np.maximum(
            self.water,
            0.0
        )

        # ----------------------------------------------------
        # KOSTEUS
        # ----------------------------------------------------

        # Haihtuminen lisää kosteutta.
        # Sade poistaa kosteutta.

        humidity_change = (

            0.0005 *
            evaporation_amount

            -

            0.0010 *
            rain_amount
        )

        self.humidity += (
            humidity_change
        )

        # Pieni palautuminen kohti 70 % RH:ta.

        self.humidity += (

            0.001 *

            (
                0.70 -
                self.humidity
            )
        )

        self.humidity = np.clip(

            self.humidity,

            0.02,
            1.0
        )

        # ----------------------------------------------------
        # PALAUTA
        # ----------------------------------------------------

        return {

            "temperature":
                self.temperature.copy(),

            "precipitation":
                rain.copy(),

            "evaporation":
                evaporation.copy(),

            "humidity":
                self.humidity.copy(),

            "wind":
                wind.copy(),

            "solar":
                solar.copy(),

            "albedo":
                albedo.copy(),

            "ice":
                self.ice_fraction.copy(),

            "outgoing":
                outgoing.copy(),

            "absorbed":
                absorbed.copy(),

            "water":
                self.water.copy()
        }


    # ========================================================
    # KOKO VUOSI
    # ========================================================

    def run(self):

        steps = (
            self.p.steps_per_year
        )

        T_month = [
            [] for _ in range(12)
        ]

        P_month = [
            [] for _ in range(12)
        ]

        W_month = [
            [] for _ in range(12)
        ]

        S_month = [
            [] for _ in range(12)
        ]

        I_month = [
            [] for _ in range(12)
        ]

        A_month = [
            [] for _ in range(12)
        ]

        O_month = [
            [] for _ in range(12)
        ]

        E_month = [
            [] for _ in range(12)
        ]

        H_month = [
            [] for _ in range(12)
        ]

        # ----------------------------------------------------
        # KUUKAUSI
        # ----------------------------------------------------

        months = np.floor(

            np.arange(steps) *
            12 /
            steps

        ).astype(int)

        # ----------------------------------------------------
        # AIKA-AJON
        # ----------------------------------------------------

        for step in range(
            steps
        ):

            fraction = (
                step /
                steps
            )

            result = self.step(
                fraction
            )

            month = months[step]

            T_month[month].append(
                result[
                    "temperature"
                ]
            )

            P_month[month].append(
                result[
                    "precipitation"
                ]
            )

            W_month[month].append(
                result[
                    "wind"
                ]
            )

            S_month[month].append(
                result[
                    "solar"
                ]
            )

            I_month[month].append(
                result[
                    "ice"
                ]
            )

            A_month[month].append(
                result[
                    "albedo"
                ]
            )

            O_month[month].append(
                result[
                    "outgoing"
                ]
            )

            E_month[month].append(
                result[
                    "evaporation"
                ]
            )

            H_month[month].append(
                result[
                    "humidity"
                ]
            )

        # ----------------------------------------------------
        # AGGREGOI
        # ----------------------------------------------------

        temperature = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in T_month

        ])

        # ----------------------------------------------------
        # SADE
        #
        # Muutetaan kg/m2 -> mm.
        #
        # 1 kg/m2 vettä = 1 mm.
        # ----------------------------------------------------

        precipitation = np.asarray([

            np.sum(
                x,
                axis=0
            ) *
            1000.0

            for x in P_month

        ])

        # ----------------------------------------------------
        # TUULI
        # ----------------------------------------------------

        wind = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in W_month

        ])

        # ----------------------------------------------------
        # AURINKO
        # ----------------------------------------------------

        solar = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in S_month

        ])

        # ----------------------------------------------------
        # JÄÄ
        # ----------------------------------------------------

        ice = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in I_month

        ])

        # ----------------------------------------------------
        # ALBEDO
        # ----------------------------------------------------

        albedo = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in A_month

        ])

        # ----------------------------------------------------
        # ULOSMENEVA SÄTEILY
        # ----------------------------------------------------

        outgoing = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in O_month

        ])

        # ----------------------------------------------------
        # HAIHTUMINEN
        #
        # mm/kk
        # ----------------------------------------------------

        evaporation = np.asarray([

            np.sum(
                x,
                axis=0
            ) *
            1000.0

            for x in E_month

        ])

        # ----------------------------------------------------
        # KOSTEUS
        # ----------------------------------------------------

        humidity = np.asarray([

            np.mean(
                x,
                axis=0
            )

            for x in H_month

        ])

        return {

            "temperature":
                temperature,

            "precipitation":
                precipitation,

            "wind":
                wind,

            "solar":
                solar,

            "ice":
                ice,

            "albedo":
                albedo,

            "outgoing":
                outgoing,

            "evaporation":
                evaporation,

            "humidity":
                humidity,

            "latitude":
                self.lat.copy(),

            "longitude":
                self.lon.copy()
        }


# ============================================================
# GLOBAALI KESKIARVO
# ============================================================

def global_mean(
    field,
    latitude
):

    weights = np.cos(
        np.deg2rad(
            latitude
        )
    )

    weights = (
        weights /
        np.sum(weights)
    )

    zonal = np.mean(
        field,
        axis=-1
    )

    return np.sum(
        zonal *
        weights
    )


# ============================================================
# VUOSIDIAGNOSTIIKKA
# ============================================================

def annual_diagnostics(
    result
):

    T = result[
        "temperature"
    ]

    P = result[
        "precipitation"
    ]

    I = result[
        "ice"
    ]

    A = result[
        "albedo"
    ]

    lat = result[
        "latitude"
    ]

    # --------------------------------------------------------
    # LÄMPÖTILA
    # --------------------------------------------------------

    temperature = global_mean(

        np.mean(
            T,
            axis=0
        ),

        lat
    )

    # --------------------------------------------------------
    # JÄÄ
    # --------------------------------------------------------

    ice = (

        100.0 *

        global_mean(

            np.mean(
                I,
                axis=0
            ),

            lat
        )
    )

    # --------------------------------------------------------
    # ALBEDO
    # --------------------------------------------------------

    albedo = global_mean(

        np.mean(
            A,
            axis=0
        ),

        lat
    )

    # --------------------------------------------------------
    # VUOSISADE
    # --------------------------------------------------------

    rain = global_mean(

        np.sum(
            P,
            axis=0
        ),

        lat
    )

    return {

        "temperature":
            temperature,

        "ice":
            ice,

        "albedo":
            albedo,

        "rain":
            rain
    }


# ============================================================
# KUUKAUSITULOSTUS
# ============================================================

def print_summary(
    result
):

    T = result[
        "temperature"
    ]

    P = result[
        "precipitation"
    ]

    W = result[
        "wind"
    ]

    I = result[
        "ice"
    ]

    A = result[
        "albedo"
    ]

    lat = result[
        "latitude"
    ]

    print()

    print(
        "Month | Temp [C] | Rain [mm] | "
        "Wind [m/s] | Ice [%] | Albedo"
    )

    print(
        "----------------------------------------------------------------"
    )

    for month in range(12):

        temp = global_mean(

            T[month],

            lat
        )

        rain = global_mean(

            P[month],

            lat
        )

        speed = np.sqrt(

            W[
                month,
                ...,
                0
            ] ** 2

            +

            W[
                month,
                ...,
                1
            ] ** 2

            +

            W[
                month,
                ...,
                2
            ] ** 2
        )

        wind = global_mean(

            speed,

            lat
        )

        ice = (

            100.0 *

            global_mean(

                I[month],

                lat
            )
        )

        albedo = global_mean(

            A[month],

            lat
        )

        print(

            f"{month + 1:5d} | "

            f"{temp - 273.15:8.2f} | "

            f"{rain:9.2f} | "

            f"{wind:10.2f} | "

            f"{ice:7.2f} | "

            f"{albedo:6.3f}"
        )


# ============================================================
# NETCDF-TALLENNUS
# ============================================================

def save_netcdf(
    result,
    filename="planet_year.nc"
):

    ds = xr.Dataset(

        data_vars={

            "temperature": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "temperature"
                ]
            ),

            "precipitation": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "precipitation"
                ]
            ),

            "evaporation": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "evaporation"
                ]
            ),

            "humidity": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "humidity"
                ]
            ),

            "ice": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "ice"
                ]
            ),

            "albedo": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "albedo"
                ]
            ),

            "solar": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "solar"
                ]
            ),

            "outgoing_longwave": (

                (
                    "month",
                    "latitude",
                    "longitude"
                ),

                result[
                    "outgoing"
                ]
            ),

            "wind": (

                (
                    "month",
                    "latitude",
                    "longitude",
                    "component"
                ),

                result[
                    "wind"
                ]
            )
        },

        coords={

            "month": np.arange(
                1,
                13
            ),

            "latitude":
                result[
                    "latitude"
                ],

            "longitude":
                result[
                    "longitude"
                ],

            "component": [
                "zonal",
                "meridional",
                "vertical"
            ]
        }
    )

    # --------------------------------------------------------
    # YKSIKÖT
    # --------------------------------------------------------

    ds[
        "temperature"
    ].attrs[
        "units"
    ] = "K"

    ds[
        "precipitation"
    ].attrs[
        "units"
    ] = "mm/month"

    ds[
        "evaporation"
    ].attrs[
        "units"
    ] = "mm/month"

    ds[
        "humidity"
    ].attrs[
        "units"
    ] = "1"

    ds[
        "ice"
    ].attrs[
        "units"
    ] = "fraction"

    ds[
        "albedo"
    ].attrs[
        "units"
    ] = "1"

    ds[
        "solar"
    ].attrs[
        "units"
    ] = "W/m2"

    ds[
        "outgoing_longwave"
    ].attrs[
        "units"
    ] = "W/m2"

    ds[
        "wind"
    ].attrs[
        "units"
    ] = "m/s"

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    ds.attrs[
        "description"
    ] = (
        "Simplified physical planetary climate model"
    )

    ds.attrs[
        "CO2_ppm"
    ] = result.get(
        "co2",
        np.nan
    )

    ds.to_netcdf(
        filename
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # HILA
    # --------------------------------------------------------

    nlat = 90
    nlon = 180

    # --------------------------------------------------------
    # VESIPALLO
    # --------------------------------------------------------

    dem = np.zeros(
        (
            nlat,
            nlon
        ),
        dtype=float
    )

    # --------------------------------------------------------
    # PLANEETTA
    # --------------------------------------------------------

    planet = Planet(

        S0=1361.0,

        ecc=0.0167,

        year_length=365.25,

        mvelp=102.9,

        day_length=24.0,

        tilt=23.44,

        co2=420.0,

        reference_co2=280.0,

        climate_sensitivity=5.35,

        convection_cells=3,

        nlat=nlat,

        nlon=nlon,

        steps_per_year=120,

        surface_albedo=0.30,

        ice_albedo=0.65,

        ocean_heat_capacity=4.0e8,

        heat_transport=2.0,

        initial_water=25.0,

        initial_humidity=0.75,

        dem=dem
    )

    # --------------------------------------------------------
    # MALLI
    # --------------------------------------------------------

    model = ClimateModel(
        planet
    )

    print(
        "Running climate model..."
    )

    print(
        f"Grid: "
        f"{nlat} x {nlon}"
    )

    print(
        f"Steps/year: "
        f"{planet.steps_per_year}"
    )

    print(
        f"Convection cells: "
        f"{planet.convection_cells}"
    )

    print(
        f"CO2: "
        f"{planet.co2:.0f} ppm"
    )

    # --------------------------------------------------------
    # PITKÄ ILMASTOAJO
    # --------------------------------------------------------

    years = 200

    print()

    print(
        "Long-term climate evolution"
    )

    print(
        "------------------------------------------------------"
    )

    print(
        "Year | Temp [C] | Ice [%] | Albedo | Rain [mm]"
    )

    print(
        "------------------------------------------------------"
    )

    for year in range(
        years
    ):

        result = model.run()

        diag = annual_diagnostics(
            result
        )

        print(

            f"{year + 1:4d} | "

            f"{diag['temperature'] - 273.15:8.2f} | "

            f"{diag['ice']:7.2f} | "

            f"{diag['albedo']:6.3f} | "

            f"{diag['rain']:8.1f}"
        )

    # --------------------------------------------------------
    # LOPULLINEN VUOSI
    # --------------------------------------------------------

    print()

    print(
        "Final year"
    )

    print(
        "==========="
    )

    print_summary(
        result
    )

    # --------------------------------------------------------
    # VUOSIDIAGNOSTIIKKA
    # --------------------------------------------------------

    diag = annual_diagnostics(
        result
    )

    print()

    print(
        "Annual diagnostics"
    )

    print(
        "------------------"
    )

    print(
        f"Global mean temperature : "
        f"{diag['temperature'] - 273.15:.2f} C"
    )

    print(
        f"Global mean ice cover   : "
        f"{diag['ice']:.2f} %"
    )

    print(
        f"Planetary albedo        : "
        f"{diag['albedo']:.3f}"
    )

    print(
        f"Annual precipitation    : "
        f"{diag['rain']:.1f} mm"
    )

    # --------------------------------------------------------
    # NETCDF
    # --------------------------------------------------------

    save_netcdf(
        result,
        "planet_year.nc"
    )

    print()

    print(
        "Results saved to planet_year.nc"
    )


# ============================================================
# KÄYNNISTYS
# ============================================================

if __name__ == "__main__":

    main()
