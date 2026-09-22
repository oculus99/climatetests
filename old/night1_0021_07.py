
#############################
#
## simple planet climate 
#
## 21.09.2026 0000.0021.07
#
#####################

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import shift

from pygam import LinearGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor ## good
from sklearn.ensemble import HistGradientBoostingRegressor
import cupy as cp

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from sklearn.neighbors import BallTree
# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus = 180*2
leveys = 360*2

#seed1=9 ##ok
#seed1 = 53 ## hyva
#seed1=3333
#seed1=44
seed1=77


#maapallon_sade_km = 6371.0
#syvin_kohta=-11000
#korkein_kohta=8848
#manner_osuus=0.30

maapallon_sade_km = 6371.0
syvin_kohta=-6000
korkein_kohta=4000
#manner_osuus=0.30
manner_osuus=0.3



#tilt_planet_axis=23.44
#ecc_planet=0.013
#mvelp_planet=102.0
# Parametrit (esimerkkinä Maan arvot)
tilt_planet_axis = 23.44*1
ecc_planet = 0.013*1
mvelp_planet = 102.0  # Perihelin pituus asteina (Maa saavuttaa perihelin tammikuun alussa)
tmax_planet=57
global_moisture_coeff=1




###########################################
#######################################
### kartta

import cupy as cp


class KartanTekijaCuPy:
    """
    GPU-versio pallomaisen maailman proseduraalisesta korkeuskartan
    generaattorista.

    Tärkeimmät ominaisuudet:

    - pallokoordinaatit -> ei napojen vääristymää
    - deterministinen seed
    - sama maailma säilyy zoomattaessa
    - globaali vedenpinta
    - erilliset meri-, tasanko-, ylänkö- ja vuoristokerrokset
    - vuoristoalue on rajattu erillisellä maskilla
    - korkeimmat huiput ovat pieniä ja paikallisia
    - CuPy/GPU

    kartan_alue:

        (lon_min, lon_max, lat_min, lat_max)

    Esimerkiksi:

        kartta(
            kartan_alue=(10, 15, 50, 55),
            leveys=2000,
            korkeus=2000,
        )
    """

    def __init__(
        self,

        leveys=360,
        korkeus=180,

        kartan_alue=(
            -180.0,
            180.0,
            -90.0,
            90.0,
        ),

        syvin_kohta=-4000,
        korkein_kohta=2000,

        manner_osuus=0.30,

        suurten_mantereiden_osuus=0.99,

        mantereen_koko=0.55,
        mantereen_muoto=0.5,

        saaren_koko=2.5,
        saaren_maara=1.0,

        # Tasangot
        tasangon_koko=0.65,
        tasangon_voima=0.45,

        # Ylängöt
        ylangon_koko=0.90,
        ylangon_voima=0.20,

        # Vuoristot
        vuoriston_koko=0.55,
        vuoriston_voima=0.35,

        # UUSI:
        # vuoristoalueiden määrä/koko
        vuoristoalueen_koko=0.75,
        vuoristoalueen_voima=0.85,

        # Korkeimpien huippujen rajaus
        vuoriston_huipun_teravyys=2.8,

        # Pieni maasto
        pieni_maasto_voima=0.035,

        # Merenpohja
        merenpohjan_koko=0.8,
        merenpohjan_voima=0.25,

        rannikon_loivuus=0.20,

        seed=42,

        octaves=64,

        debug=False,
    ):

        self.leveys = int(leveys)
        self.korkeus = int(korkeus)

        self.kartan_alue = tuple(
            float(v)
            for v in kartan_alue
        )

        self.syvin_kohta = float(
            syvin_kohta
        )

        self.korkein_kohta = float(
            korkein_kohta
        )

        self.manner_osuus = float(
            manner_osuus
        )

        self.suurten_mantereiden_osuus = float(
            suurten_mantereiden_osuus
        )

        self.mantereen_koko = float(
            mantereen_koko
        )

        self.mantereen_muoto = float(
            mantereen_muoto
        )

        self.saaren_koko = float(
            saaren_koko
        )

        self.saaren_maara = float(
            saaren_maara
        )

        self.tasangon_koko = float(
            tasangon_koko
        )

        self.tasangon_voima = float(
            tasangon_voima
        )

        self.ylangon_koko = float(
            ylangon_koko
        )

        self.ylangon_voima = float(
            ylangon_voima
        )

        self.vuoriston_koko = float(
            vuoriston_koko
        )

        self.vuoriston_voima = float(
            vuoriston_voima
        )

        self.vuoristoalueen_koko = float(
            vuoristoalueen_koko
        )

        self.vuoristoalueen_voima = float(
            vuoristoalueen_voima
        )

        self.vuoriston_huipun_teravyys = float(
            vuoriston_huipun_teravyys
        )

        self.pieni_maasto_voima = float(
            pieni_maasto_voima
        )

        self.merenpohjan_koko = float(
            merenpohjan_koko
        )

        self.merenpohjan_voima = float(
            merenpohjan_voima
        )

        self.rannikon_loivuus = float(
            rannikon_loivuus
        )

        self.seed = int(seed)

        self.octaves = int(octaves)

        self.debug = bool(debug)

        self.korkeus_syvyyskartta = None
        self.korkeuskartta = None
        self.raakakorkeuskartta = None

        self.maa_maski = None
        self.meri_maski = None
        self.manner_maski = None
        self.saari_maski = None

        self.vedenpinta = None

        self.debug_data = {}

    # ==========================================================
    # Kartan alue
    # ==========================================================

    def _aseta_kartan_alue(
        self,
        kartan_alue,
    ):

        if kartan_alue is None:
            kartan_alue = (
                -180.0,
                180.0,
                -90.0,
                90.0,
            )

        if len(kartan_alue) != 4:
            raise ValueError(
                "kartan_alue pitää olla "
                "(lon_min, lon_max, lat_min, lat_max)"
            )

        (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ) = map(
            float,
            kartan_alue,
        )

        lat_min = max(
            -90.0,
            lat_min,
        )

        lat_max = min(
            90.0,
            lat_max,
        )

        if lon_max <= lon_min:
            raise ValueError(
                "lon_max pitää olla suurempi "
                "kuin lon_min"
            )

        if lat_max <= lat_min:
            raise ValueError(
                "lat_max pitää olla suurempi "
                "kuin lat_min"
            )

        return (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        )

    # ==========================================================
    # Pallokoordinaatit
    # ==========================================================

    def _pallokoordinaatit(
        self,
        kartan_alue=None,
    ):

        (
            lon_min,
            lon_max,
            lat_min,
            lat_max,
        ) = self._aseta_kartan_alue(
            kartan_alue
        )

        lat = cp.radians(
            cp.linspace(
                lat_min,
                lat_max,
                self.korkeus,
                dtype=cp.float32,
            )
        )

        lon = cp.radians(
            cp.linspace(
                lon_min,
                lon_max,
                self.leveys,
                endpoint=False,
                dtype=cp.float32,
            )
        )

        lon_grid, lat_grid = cp.meshgrid(
            lon,
            lat,
        )

        cos_lat = cp.cos(
            lat_grid
        )

        x = (
            cos_lat
            * cp.cos(lon_grid)
        )

        y = (
            cos_lat
            * cp.sin(lon_grid)
        )

        z = cp.sin(
            lat_grid
        )

        return x, y, z

    # ==========================================================
    # Value noise 3D
    # ==========================================================

    def _value_noise_3d(
        self,
        x,
        y,
        z,
        seed,
        grid_size=4,
    ):

        px = (
            x
            * cp.float32(grid_size)
        )

        py = (
            y
            * cp.float32(grid_size)
        )

        pz = (
            z
            * cp.float32(grid_size)
        )

        x0 = cp.floor(px).astype(
            cp.int32
        )

        y0 = cp.floor(py).astype(
            cp.int32
        )

        z0 = cp.floor(pz).astype(
            cp.int32
        )

        fx = px - x0
        fy = py - y0
        fz = pz - z0

        fx = (
            fx
            * fx
            * (3.0 - 2.0 * fx)
        )

        fy = (
            fy
            * fy
            * (3.0 - 2.0 * fy)
        )

        fz = (
            fz
            * fz
            * (3.0 - 2.0 * fz)
        )

        def random_value(
            ix,
            iy,
            iz,
        ):

            h = (
                ix.astype(cp.int64)
                * 374761393
                + iy.astype(cp.int64)
                * 668265263
                + iz.astype(cp.int64)
                * 2147483647
                + cp.int64(seed)
                * 1274126177
            )

            h ^= h >> 13

            h *= 1274126177

            h ^= h >> 16

            h &= cp.int64(
                0xffffffff
            )

            return (
                h.astype(cp.float32)
                / cp.float32(
                    2147483647.5
                )
                - cp.float32(1.0)
            )

        c000 = random_value(
            x0,
            y0,
            z0,
        )

        c100 = random_value(
            x0 + 1,
            y0,
            z0,
        )

        c010 = random_value(
            x0,
            y0 + 1,
            z0,
        )

        c110 = random_value(
            x0 + 1,
            y0 + 1,
            z0,
        )

        c001 = random_value(
            x0,
            y0,
            z0 + 1,
        )

        c101 = random_value(
            x0 + 1,
            y0,
            z0 + 1,
        )

        c011 = random_value(
            x0,
            y0 + 1,
            z0 + 1,
        )

        c111 = random_value(
            x0 + 1,
            y0 + 1,
            z0 + 1,
        )

        nx00 = (
            c000 * (1.0 - fx)
            + c100 * fx
        )

        nx10 = (
            c010 * (1.0 - fx)
            + c110 * fx
        )

        nx01 = (
            c001 * (1.0 - fx)
            + c101 * fx
        )

        nx11 = (
            c011 * (1.0 - fx)
            + c111 * fx
        )

        nxy0 = (
            nx00 * (1.0 - fy)
            + nx10 * fy
        )

        nxy1 = (
            nx01 * (1.0 - fy)
            + nx11 * fy
        )

        return (
            nxy0 * (1.0 - fz)
            + nxy1 * fz
        )

    # ==========================================================
    # FBM
    # ==========================================================

    def _fbm(
        self,
        x,
        y,
        z,
        seed,
        octaves=None,
        base_grid=4,
    ):

        if octaves is None:
            octaves = self.octaves

        value = cp.zeros_like(
            x,
            dtype=cp.float32,
        )

        amplitude = cp.float32(
            1.0
        )

        amplitude_sum = cp.float32(
            0.0
        )

        frequency = cp.float32(
            1.0
        )

        for octave in range(
            int(octaves)
        ):

            value += (
                amplitude
                * self._value_noise_3d(
                    x * frequency,
                    y * frequency,
                    z * frequency,
                    seed=(
                        seed
                        + octave * 101
                    ),
                    grid_size=base_grid,
                )
            )

            amplitude_sum += amplitude

            frequency *= cp.float32(
                2.0
            )

            amplitude *= cp.float32(
                0.5
            )

        return (
            value
            / amplitude_sum
        )

    # ==========================================================
    # Ridged noise
    # ==========================================================

    def _ridged_noise(
        self,
        x,
        y,
        z,
        seed,
        octaves=None,
        base_grid=4,
    ):

        noise = self._fbm(
            x,
            y,
            z,
            seed,
            octaves=octaves,
            base_grid=base_grid,
        )

        ridge = (
            1.0
            - cp.abs(noise)
        )

        ridge = cp.clip(
            ridge,
            0.0,
            1.0,
        )

        return ridge * ridge

    # ==========================================================
    # Normalisointi
    # ==========================================================

    def _normalisoi_01(
        self,
        value,
    ):

        value = value.astype(
            cp.float32,
            copy=False,
        )

        minimum = cp.min(
            value
        )

        maximum = cp.max(
            value
        )

        return cp.where(
            maximum > minimum,
            (
                value - minimum
            )
            / (
                maximum - minimum
            ),
            cp.zeros_like(value),
        )

    # ==========================================================
    # Smoothstep
    # ==========================================================

    def _smoothstep(
        self,
        value,
    ):

        value = cp.clip(
            value,
            0.0,
            1.0,
        )

        return (
            value
            * value
            * (
                3.0
                - 2.0 * value
            )
        )

    # ==========================================================
    # Perusmantereet
    # ==========================================================

    def _generoi_mantereet(
        self,
        x,
        y,
        z,
    ):

        frequency = (
            1.35
            * self.mantereen_koko
        )

        large = self._fbm(
            x * frequency,
            y * frequency,
            z * frequency,
            seed=self.seed + 100,
            octaves=self.octaves,
            base_grid=2,
        )

        large = self._normalisoi_01(
            large
        )

        if (
            self.mantereen_muoto
            > 0.0
        ):

            detail = self._fbm(
                x * frequency * 2.0,
                y * frequency * 2.0,
                z * frequency * 2.0,
                seed=self.seed + 200,
                octaves=min(
                    self.octaves,
                    4,
                ),
                base_grid=3,
            )

            detail = self._normalisoi_01(
                detail
            )

            large = (
                (
                    1.0
                    - self.mantereen_muoto
                )
                * large
                +
                self.mantereen_muoto
                * (
                    0.78 * large
                    + 0.22 * detail
                )
            )

        return large

    # ==========================================================
    # Saaret
    # ==========================================================

    def _generoi_saaret(
        self,
        x,
        y,
        z,
    ):

        islands = self._fbm(
            x * self.saaren_koko,
            y * self.saaren_koko,
            z * self.saaren_koko,
            seed=self.seed + 300,
            octaves=self.octaves,
            base_grid=3,
        )

        return self._normalisoi_01(
            islands
        )

    # ==========================================================
    # Maa
    # ==========================================================

    def _generoi_maakentta(
        self,
        x,
        y,
        z,
    ):

        large = self._generoi_mantereet(
            x,
            y,
            z,
        )

        saari_arvot = self._generoi_saaret(
            x,
            y,
            z,
        )

        tavoite_maaosuus = cp.float32(
            min(
                max(
                    self.manner_osuus,
                    0.001,
                ),
                0.999,
            )
        )

        suuri_osuus = cp.float32(
            min(
                max(
                    self.suurten_mantereiden_osuus,
                    0.0,
                ),
                1.0,
            )
        )

        tavoite_manner_osuus = (
            tavoite_maaosuus
            * suuri_osuus
        )

        manner_threshold = cp.quantile(
            large,
            1.0
            - tavoite_manner_osuus,
        )

        manner_maski = (
            large
            >= manner_threshold
        )

        maara = cp.float32(
            min(
                max(
                    self.saaren_maara,
                    0.0,
                ),
                2.0,
            )
        )

        saari_threshold = cp.clip(
            cp.float32(0.72)
            - cp.float32(0.20)
            * maara,
            cp.float32(0.30),
            cp.float32(0.85),
        )

        saari_maski = (
            saari_arvot
            >= saari_threshold
        )

        saari_maski &= (
            ~manner_maski
        )

        nykyinen_manner_osuus = cp.mean(
            manner_maski
        )

        sallittu_saarialue = cp.maximum(
            tavoite_maaosuus
            - nykyinen_manner_osuus,
            cp.float32(0.0),
        )

        nykyinen_saariosuus = cp.mean(
            saari_maski
        )

        if (
            float(
                sallittu_saarialue
            )
            <= 0.0
        ):

            saari_maski = (
                cp.zeros_like(
                    manner_maski,
                    dtype=cp.bool_,
                )
            )

        elif (
            float(
                nykyinen_saariosuus
            )
            > float(
                sallittu_saarialue
            )
        ):

            saarien_arvot = (
                saari_arvot[
                    saari_maski
                ]
            )

            if saarien_arvot.size > 0:

                pidettava_osuus = cp.clip(
                    sallittu_saarialue
                    / nykyinen_saariosuus,
                    cp.float32(0.0),
                    cp.float32(1.0),
                )

                saarien_kynnys = (
                    cp.quantile(
                        saarien_arvot,
                        1.0
                        - pidettava_osuus,
                    )
                )

                saari_maski &= (
                    saari_arvot
                    >= saarien_kynnys
                )

        maa_maski = (
            manner_maski
            | saari_maski
        )

        return (
            maa_maski,
            manner_maski,
            saari_maski,
        )

    # ==========================================================
    # Tasangot
    #
    # Ei enää:
    #
    #     1 - abs(terrain * 2 - 1)
    #
    # koska se tuotti valtavan keskialueen,
    # joka käyttäytyi kuin keinotekoinen mesa.
    # ==========================================================

    def _generoi_tasangot(
        self,
        x,
        y,
        z,
    ):

        terrain = self._fbm(
            x * self.tasangon_koko,
            y * self.tasangon_koko,
            z * self.tasangon_koko,
            seed=self.seed + 1000,
            octaves=min(
                self.octaves,
                6,
            ),
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        # Tasanko alkaa vasta ylemmällä alueella.
        plateau = cp.clip(
            (
                terrain
                - cp.float32(0.58)
            )
            / cp.float32(0.24),
            0.0,
            1.0,
        )

        plateau = self._smoothstep(
            plateau
        )

        # Rikkoo tasangon reunan.
        detail = self._fbm(
            x
            * self.tasangon_koko
            * 2.5,
            y
            * self.tasangon_koko
            * 2.5,
            z
            * self.tasangon_koko
            * 2.5,
            seed=self.seed + 1100,
            octaves=min(
                self.octaves,
                3,
            ),
            base_grid=3,
        )

        detail = self._normalisoi_01(
            detail
        )

        plateau *= (
            cp.float32(0.65)
            + cp.float32(0.35)
            * detail
        )

        return cp.clip(
            plateau,
            0.0,
            1.0,
        )

    # ==========================================================
    # Ylängöt
    # ==========================================================

    def _generoi_ylangot(
        self,
        x,
        y,
        z,
    ):

        terrain = self._fbm(
            x * self.ylangon_koko,
            y * self.ylangon_koko,
            z * self.ylangon_koko,
            seed=self.seed + 2000,
            octaves=min(
                self.octaves,
                5,
            ),
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        ylanko = cp.clip(
            (
                terrain
                - cp.float32(0.46)
            )
            / cp.float32(0.54),
            0.0,
            1.0,
        )

        return self._smoothstep(
            ylanko
        )

    # ==========================================================
    # VUORISTOALUEEN MASKI
    #
    # TÄMÄ on tärkein muutos.
    #
    # Vuoristo ei ole enää kaikkialla missä ridged noise
    # sattuu olemaan korkea.
    #
    # Ensin muodostetaan harva, suurialainen vuoristokenttä.
    # Vain sen korkeimmat osat aktivoituvat.
    # ==========================================================

    def _generoi_vuoristoalueet(
        self,
        x,
        y,
        z,
    ):

        region = self._fbm(
            x * self.vuoristoalueen_koko,
            y * self.vuoristoalueen_koko,
            z * self.vuoristoalueen_koko,
            seed=self.seed + 2500,
            octaves=min(
                self.octaves,
                4,
            ),
            base_grid=2,
        )

        region = self._normalisoi_01(
            region
        )

        # Vain korkein ~20-30 % noise-kentästä
        # muuttuu varsinaiseksi vuoristoalueeksi.
        #
        # Tämä estää vuoristoa leviämästä koko mantereelle.

        mountain_region = cp.clip(
            (
                region
                - cp.float32(0.67)
            )
            / cp.float32(0.25),
            0.0,
            1.0,
        )

        mountain_region = (
            self._smoothstep(
                mountain_region
            )
        )

        # Vielä pehmeämpi reuna.
        mountain_region = (
            mountain_region
            * mountain_region
        )

        return mountain_region

    # ==========================================================
    # Vuoristot
    #
    # Ridged noise antaa terävät harjanteet.
    # Sen jälkeen voimakas eksponentti tekee korkeimmista
    # kohdista pieniä -> Everest ei täytä koko vuoristoa.
    # ==========================================================

    def _generoi_vuoristot(
        self,
        x,
        y,
        z,
    ):

        mountains = self._ridged_noise(
            x * self.vuoriston_koko,
            y * self.vuoriston_koko,
            z * self.vuoriston_koko,
            seed=self.seed + 3000,
            octaves=min(
                self.octaves,
                8,
            ),
            base_grid=3,
        )

        mountains = self._normalisoi_01(
            mountains
        )

        # Poistetaan matalat ridgesignaalit.
        mountains = cp.clip(
            (
                mountains
                - cp.float32(0.38)
            )
            / cp.float32(0.62),
            0.0,
            1.0,
        )

        # Terävöitys.
        #
        # 2.8 tarkoittaa, että vain pieni osa
        # vuoriston korkeimmasta osasta pääsee
        # todella korkealle.
        mountains = (
            mountains
            ** cp.float32(
                self.vuoriston_huipun_teravyys
            )
        )

        return cp.clip(
            mountains,
            0.0,
            1.0,
        )

    # ==========================================================
    # Pieni maasto
    # ==========================================================

    def _generoi_pieni_maasto(
        self,
        x,
        y,
        z,
    ):

        small = self._fbm(
            x * 5.0,
            y * 5.0,
            z * 5.0,
            seed=self.seed + 4000,
            octaves=min(
                self.octaves,
                6,
            ),
            base_grid=4,
        )

        return self._normalisoi_01(
            small
        )

    # ==========================================================
    # Merenpohja
    # ==========================================================

    def _generoi_merenpohja(
        self,
        x,
        y,
        z,
    ):

        ocean = self._fbm(
            x * self.merenpohjan_koko,
            y * self.merenpohjan_koko,
            z * self.merenpohjan_koko,
            seed=self.seed + 5000,
            octaves=min(
                self.octaves,
                8,
            ),
            base_grid=3,
        )

        return self._normalisoi_01(
            ocean
        )

    # ==========================================================
    # RAAKAKORKEUS
    # ==========================================================

    def _generoi_raakakorkeus(
        self,
        x,
        y,
        z,
        maa_maski,
        manner_maski,
        saari_maski,
    ):

        tasanko = self._generoi_tasangot(
            x,
            y,
            z,
        )

        ylanko = self._generoi_ylangot(
            x,
            y,
            z,
        )

        mountain_region = (
            self._generoi_vuoristoalueet(
                x,
                y,
                z,
            )
        )

        mountains = (
            self._generoi_vuoristot(
                x,
                y,
                z,
            )
        )

        pieni = (
            self._generoi_pieni_maasto(
                x,
                y,
                z,
            )
        )

        merenpohja = (
            self._generoi_merenpohja(
                x,
                y,
                z,
            )
        )

        # ======================================================
        # MAAN PERUSKORKEUS
        # ======================================================

        maan_korkeus = cp.full_like(
            tasanko,
            cp.float32(0.20),
            dtype=cp.float32,
        )

        # ======================================================
        # YLÄNKÖ
        # ======================================================

        maan_korkeus += (
            cp.float32(
                self.ylangon_voima
            )
            * ylanko
        )

        # ======================================================
        # TASANGOT
        #
        # Ei lisätä raakaa korkeutta.
        # Vedetään maastoa hieman kohti tasankotasoa.
        # ======================================================

        tasanko_taso = cp.float32(
            0.52
        )

        tasanko_voima = (
            cp.float32(
                self.tasangon_voima
            )
            * cp.float32(
                0.42
            )
        )

        maan_korkeus += (
            tasanko
            * tasanko_voima
            * (
                tasanko_taso
                - maan_korkeus
            )
        )

        # ======================================================
        # VUORISTO
        #
        # mountain_region kertoo missä vuoret saavat esiintyä.
        #
        # mountains kertoo vuoren sisäisen rakenteen.
        #
        # Näitä EI lisätä koko maailmaan.
        # ======================================================

        mountain_height = (
            mountains
            * mountain_region
        )

        maan_korkeus += (
            mountain_height
            * cp.float32(
                self.vuoriston_voima
            )
        )

        # ======================================================
        # VUORISTON KESKIALUEIDEN HIENO SULAMINEN
        #
        # Estää tasangon ja vuoriston jyrkän keinotekoisen
        # rajapinnan.
        # ======================================================

        mountain_blend = (
            self._smoothstep(
                mountain_region
            )
        )

        maan_korkeus += (
            mountain_blend
            * mountain_height
            * cp.float32(0.12)
        )

        # ======================================================
        # PIENI MAANPINNAN VARIAATIO
        # ======================================================

        maan_korkeus += (
            cp.float32(
                self.pieni_maasto_voima
            )
            * pieni
        )

        # ======================================================
        # SAARET
        # ======================================================

        maan_korkeus -= (
            saari_maski
            * cp.float32(0.06)
        )

        # ======================================================
        # MERENPOHJA
        # ======================================================

        meren_korkeus = (
            cp.float32(-0.30)
            - cp.float32(
                self.merenpohjan_voima
            )
            * merenpohja
        )

        # ======================================================
        # RANNIKKO
        # ======================================================

        rannikko_noise = self._fbm(
            x * 1.5,
            y * 1.5,
            z * 1.5,
            seed=self.seed + 6000,
            octaves=min(
                self.octaves,
                8,
            ),
            base_grid=3,
        )

        rannikko_noise = (
            rannikko_noise
            + cp.float32(1.0)
        ) * cp.float32(0.5)

        rannikko_noise = (
            self._smoothstep(
                rannikko_noise
            )
        )

        maan_korkeus -= (
            rannikko_noise
            * cp.float32(
                self.rannikon_loivuus
            )
            * cp.float32(0.06)
        )

        return cp.where(
            maa_maski,
            maan_korkeus,
            meren_korkeus,
        )

    # ==========================================================
    # Raakakorkeus -> metrit
    # ==========================================================

    def _raakakorkeus_metreiksi(
        self,
        raakakorkeus,
    ):

        raw = cp.clip(
            raakakorkeus,
            cp.float32(-1.0),
            cp.float32(1.0),
        )

        positiivinen = cp.maximum(
            raw,
            cp.float32(0.0),
        )

        negatiivinen = cp.minimum(
            raw,
            cp.float32(0.0),
        )

        maa_metrit = (
            positiivinen
            * cp.float32(
                self.korkein_kohta
            )
        )

        meri_metrit = (
            negatiivinen
            * cp.float32(
                -self.syvin_kohta
            )
        )

        return cp.where(
            raw >= 0.0,
            maa_metrit,
            meri_metrit,
        ).astype(
            cp.float32,
            copy=False,
        )

    # ==========================================================
    # Globaali vedenpinta
    # ==========================================================

    def _laske_globaali_vedenpinta(
        self,
        raakakorkeuskartta,
        maa_maski,
    ):

        raw = raakakorkeuskartta

        threshold = cp.quantile(
            raw,
            cp.float32(
                1.0
                - self.manner_osuus
            ),
        )

        vedenpinta = (
            self._raakakorkeus_metreiksi(
                threshold
            )
        )

        return float(
            vedenpinta.get()
        )

    # ==========================================================
    # Absoluuttinen korkeus
    # ==========================================================

    def _skaalaa_korkeudet(
        self,
        raakakorkeuskartta,
        sealevel,
    ):

        absoluuttinen = (
            self._raakakorkeus_metreiksi(
                raakakorkeuskartta
            )
        )

        maa_maski = (
            absoluuttinen
            >= cp.float32(
                sealevel
            )
        )

        meri_maski = (
            ~maa_maski
        )

        return (
            absoluuttinen,
            maa_maski,
            meri_maski,
        )

    # ==========================================================
    # GENEROI
    # ==========================================================

    def generoi(
        self,
        sealevel=None,
    ):

        x, y, z = (
            self._pallokoordinaatit(
                self.kartan_alue
            )
        )

        (
            rakenne_maa_maski,
            manner_maski,
            saari_maski,
        ) = self._generoi_maakentta(
            x,
            y,
            z,
        )

        raakakorkeuskartta = (
            self._generoi_raakakorkeus(
                x,
                y,
                z,
                rakenne_maa_maski,
                manner_maski,
                saari_maski,
            )
        )

        self.raakakorkeuskartta = (
            raakakorkeuskartta.astype(
                cp.float32,
                copy=False,
            )
        )

        # ======================================================
        # VEDENPINTA
        # ======================================================

        if sealevel is None:

            if self.vedenpinta is None:

                self.vedenpinta = (
                    self._laske_globaali_vedenpinta(
                        raakakorkeuskartta,
                        rakenne_maa_maski,
                    )
                )

        else:

            self.vedenpinta = float(
                sealevel
            )

        # ======================================================
        # ABSOLUUTTINEN KORKEUS
        # ======================================================

        (
            korkeus_syvyyskartta,
            maa_maski,
            meri_maski,
        ) = self._skaalaa_korkeudet(
            raakakorkeuskartta,
            self.vedenpinta,
        )

        self.korkeus_syvyyskartta = (
            korkeus_syvyyskartta.astype(
                cp.float32,
                copy=False,
            )
        )

        self.korkeuskartta = cp.where(
            maa_maski,
            self.korkeus_syvyyskartta,
            cp.float32(0.0),
        )

        self.maa_maski = maa_maski
        self.meri_maski = meri_maski

        self.manner_maski = (
            manner_maski
            & maa_maski
        )

        self.saari_maski = (
            saari_maski
            & maa_maski
        )

        # ======================================================
        # DEBUG
        # ======================================================

        if self.debug:

            self.debug_data = {

                "maa_osuus":
                    float(
                        cp.mean(
                            maa_maski
                        )
                    ),

                "meri_osuus":
                    float(
                        cp.mean(
                            meri_maski
                        )
                    ),

                "manner_osuus":
                    float(
                        cp.mean(
                            self.manner_maski
                        )
                    ),

                "saari_osuus":
                    float(
                        cp.mean(
                            self.saari_maski
                        )
                    ),

                "vedenpinta":
                    float(
                        self.vedenpinta
                    ),

                "maa_min":
                    float(
                        cp.min(
                            self.korkeuskartta[
                                maa_maski
                            ]
                        )
                    )
                    if bool(
                        cp.any(
                            maa_maski
                        )
                    )
                    else 0.0,

                "maa_max":
                    float(
                        cp.max(
                            self.korkeuskartta[
                                maa_maski
                            ]
                        )
                    )
                    if bool(
                        cp.any(
                            maa_maski
                        )
                    )
                    else 0.0,

                "meri_min":
                    float(
                        cp.min(
                            self.korkeus_syvyyskartta[
                                meri_maski
                            ]
                        )
                    )
                    if bool(
                        cp.any(
                            meri_maski
                        )
                    )
                    else 0.0,

                "meri_max":
                    float(
                        cp.max(
                            self.korkeus_syvyyskartta[
                                meri_maski
                            ]
                        )
                    )
                    if bool(
                        cp.any(
                            meri_maski
                        )
                    )
                    else 0.0,
            }

        return self

    # ==========================================================
    # ZOOMAUS
    # ==========================================================

    def kartta(
        self,
        kartan_alue=None,
        leveys=None,
        korkeus=None,
        sealevel=None,
    ):

        if kartan_alue is not None:

            self.kartan_alue = (
                self._aseta_kartan_alue(
                    kartan_alue
                )
            )

        if leveys is not None:
            self.leveys = int(
                leveys
            )

        if korkeus is not None:
            self.korkeus = int(
                korkeus
            )

        if sealevel is None:

            if self.vedenpinta is not None:

                sealevel = (
                    self.vedenpinta
                )

        self.generoi(
            sealevel=sealevel
        )

        return self

    # ==========================================================
    # TULOKSET
    # ==========================================================

    def tulokset(
        self,
        cpu=False,
    ):

        if (
            self.korkeus_syvyyskartta
            is None
        ):

            self.generoi()

        tulos = {

            "korkeus_syvyyskartta":
                self.korkeus_syvyyskartta,

            "korkeuskartta":
                self.korkeuskartta,

            "maa_maski":
                self.maa_maski,

            "meri_maski":
                self.meri_maski,

            "manner_maski":
                self.manner_maski,

            "saari_maski":
                self.saari_maski,

            "vedenpinta":
                self.vedenpinta,

            "raakakorkeuskartta":
                self.raakakorkeuskartta,

            "kartta_nelio":
                self.kartan_alue,

            "leveys":
                self.leveys,

            "korkeus":
                self.korkeus,

            "debug_data":
                self.debug_data,
        }

        if cpu:

            return {
                key: (
                    value.get()
                    if isinstance(
                        value,
                        cp.ndarray,
                    )
                    else value
                )
                for key, value
                in tulos.items()
            }

        return tulos








######################################################
####################################################



def histrgb_im(
    im,
    variables,
    variable_names,
    max_iter=100,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    train_samples=100_000,
    random_state=42,
):
    """
    Sovittaa HistGradientBoosting-regressiomallit
    imshow-rasterin RGB-arvoihin.

    Opetus tehdään tarvittaessa satunnaisella pikseliotoksella,
    ennuste koko rasterille.

    Palauttaa H x W x 3 uint8 RGB-rasterin.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen imshow-data
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D-rasteri (H, W), "
            f"mutta shape on {data.shape}"
        )

    H, W = data.shape
    N = H * W

    # --------------------------------------------------
    # 2. Imshown todelliset RGB-värit
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))

    rgb = (
        rgba[..., :3]
        .astype(np.float32)
        * 255.0
    )

    rgb_flat = rgb.reshape(N, 3)

    # --------------------------------------------------
    # 3. Selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(
            variables[name],
            dtype=np.float32
        ).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != N:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"imshow={H}x{W}, X={X.shape}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(rgb_flat), axis=1)

    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) == 0:
        raise ValueError(
            "Yhtään validia pikseliä ei löytynyt."
        )

    # --------------------------------------------------
    # 5. Opetusotos
    # --------------------------------------------------

    rng = np.random.default_rng(random_state)

    if (
        train_samples is not None
        and len(valid_idx) > train_samples
    ):
        train_idx = rng.choice(
            valid_idx,
            size=train_samples,
            replace=False
        )
    else:
        train_idx = valid_idx

    X_train = X[train_idx]
    y_train = rgb_flat[train_idx]

    X_pred = X[valid_idx]

    # --------------------------------------------------
    # 6. Kolme RGB-regressiota
    # --------------------------------------------------

    result = np.full(
        (N, 3),
        np.nan,
        dtype=np.float32
    )

    for channel in range(3):

        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

        model.fit(
            X_train,
            y_train[:, channel]
        )

        result[valid_idx, channel] = model.predict(
            X_pred
        )

    # --------------------------------------------------
    # 7. Takaisin RGB-rasteriksi
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

def et_im(
    im,
    variables,
    variable_names,
    n_estimators=30,
    max_depth=None,
    min_samples_leaf=5,
    max_samples=None,
    train_samples=1000,
    random_state=42,
):
    """
    Sovittaa ExtraTrees-regressiomallit imshow-rasterin RGB-arvoihin.

    Opetus voidaan tehdä satunnaisella pikseliotoksella, mutta
    ennuste tehdään kaikille valideille pikseleille.

    Palauttaa
    ---------
    rgb : np.ndarray
        H x W x 3 uint8 RGB-rasteri.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen imshow-data
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D-rasteri (H, W), "
            f"mutta shape on {data.shape}"
        )

    H, W = data.shape
    N = H * W

    # --------------------------------------------------
    # 2. Imshown todelliset RGB-värit
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))

    rgb = (
        rgba[..., :3]
        .astype(np.float32)
        * 255.0
    )

    rgb_flat = rgb.reshape(N, 3)

    # --------------------------------------------------
    # 3. Selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(
            variables[name],
            dtype=np.float32
        ).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != N:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"imshow={H}x{W}, X={X.shape}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(rgb_flat), axis=1)

    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) == 0:
        raise ValueError(
            "Yhtään validia pikseliä ei löytynyt."
        )

    # --------------------------------------------------
    # 5. Opetusotos
    # --------------------------------------------------

    rng = np.random.default_rng(random_state)

    if (
        train_samples is not None
        and len(valid_idx) > train_samples
    ):
        train_idx = rng.choice(
            valid_idx,
            size=train_samples,
            replace=False
        )
    else:
        train_idx = valid_idx

    X_train = X[train_idx]
    y_train = rgb_flat[train_idx]

    X_pred = X[valid_idx]

    # --------------------------------------------------
    # 6. Kolme RGB-regressiota
    # --------------------------------------------------

    result = np.full(
        (N, 3),
        np.nan,
        dtype=np.float32
    )

    for channel in range(3):

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=-1,
            random_state=random_state,
        )

        model.fit(
            X_train,
            y_train[:, channel]
        )

        result[valid_idx, channel] = model.predict(
            X_pred
        )

    # --------------------------------------------------
    # 7. Takaisin RGB-rasteriksi
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

def lm_im(
    im,
    variables,
    variable_names,
):
    """
    Sovittaa lineaarisen regression suoraan imshow-rasterin
    RGB-arvoihin.

    Palauttaa H x W x 3 uint8 RGB-rasterin.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen imshow-data
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D-rasteri (H, W), "
            f"mutta shape on {data.shape}"
        )

    H, W = data.shape
    N = H * W

    # --------------------------------------------------
    # 2. Imshown todelliset RGB-värit
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))

    rgb = (
        rgba[..., :3]
        .astype(np.float32)
        * 255.0
    )

    rgb_flat = rgb.reshape(N, 3)

    # --------------------------------------------------
    # 3. Selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(
            variables[name],
            dtype=np.float32
        ).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != N:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"{X.shape[0]} != {N}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(rgb_flat), axis=1)

    X_valid = X[valid]
    y_rgb = rgb_flat[valid]

    if len(X_valid) == 0:
        raise ValueError("Yhtään validia pikseliä ei löytynyt.")

    # --------------------------------------------------
    # 5. Yksi LM, kolme outputtia
    # --------------------------------------------------

    model = LinearRegression(n_jobs=-1)

    model.fit(X_valid, y_rgb)

    # --------------------------------------------------
    # 6. Ennuste
    # --------------------------------------------------

    result = np.full(
        (N, 3),
        np.nan,
        dtype=np.float32
    )

    result[valid] = model.predict(X_valid)

    # --------------------------------------------------
    # 7. RGB-rasteri
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)




def svm_im_fast(
    im,
    variables,
    variable_names,
    C=10.0,
    gamma="scale",
    epsilon=0.5,
    train_samples=5000,
    random_state=42,
):
    """
    RBF-SVM RGB-rasterin mallintamiseen.

    Opetus tehdään satunnaisella pikseliotoksella,
    mutta ennuste tehdään koko rasterille.

    Palauttaa H x W x 3 uint8 RGB-rasterin.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen rasteri
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D, shape={data.shape}"
        )

    H, W = data.shape
    N = H * W

    # --------------------------------------------------
    # 2. Alkuperäiset RGB-arvot
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))

    rgb = (
        rgba[..., :3]
        .astype(np.float32)
        * 255.0
    )

    rgb_flat = rgb.reshape(N, 3)

    # --------------------------------------------------
    # 3. Selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(
            variables[name],
            dtype=np.float32
        ).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != N:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"{X.shape[0]} != {N}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(rgb_flat), axis=1)

    valid_idx = np.flatnonzero(valid)

    if len(valid_idx) == 0:
        raise ValueError("Yhtään validia pikseliä ei löytynyt.")

    # --------------------------------------------------
    # 5. Otos opetukseen
    # --------------------------------------------------

    rng = np.random.default_rng(random_state)

    if len(valid_idx) > train_samples:
        train_idx = rng.choice(
            valid_idx,
            size=train_samples,
            replace=False
        )
    else:
        train_idx = valid_idx

    X_train = X[train_idx]
    y_train = rgb_flat[train_idx]

    X_pred = X[valid_idx]

    # --------------------------------------------------
    # 6. Kolme SVM:ää
    # --------------------------------------------------

    result = np.full(
        (N, 3),
        np.nan,
        dtype=np.float32
    )

    for channel in range(3):

        model = SVR(
            kernel="rbf",
            C=C,
            gamma=gamma,
            epsilon=epsilon,
        )

        model.fit(
            X_train,
            y_train[:, channel]
        )

        result[valid_idx, channel] = model.predict(X_pred)

    # --------------------------------------------------
    # 7. RGB-rasteri
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)



def rf_im(
    im,
    variables,
    variable_names,
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    max_samples=None,
    random_state=42,
):
    """
    Sovittaa Random Forest -mallit suoraan matplotlibin
    AxesImage-olion näyttämään rasteriin.

    Palauttaa H x W x 3 uint8 RGB-rasterin.

    Nopeuteen optimoitu versio.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen imshow-data
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D-rasteri (H, W), "
            f"mutta shape on {data.shape}"
        )

    H, W = data.shape
    N = H * W

    # --------------------------------------------------
    # 2. Imshown todelliset RGB-värit
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))
    rgb = rgba[..., :3].astype(np.float32) * 255.0

    rgb_flat = rgb.reshape(N, 3)

    # --------------------------------------------------
    # 3. Selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(variables[name], dtype=np.float32).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != N:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"imshow={H}x{W}, X={X.shape}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)
    valid &= np.all(np.isfinite(rgb_flat), axis=1)

    X_valid = X[valid]

    if X_valid.shape[0] == 0:
        raise ValueError("Yhtään validia pikseliä ei löytynyt.")

    # --------------------------------------------------
    # 5. Random Forest RGB-kanaville
    # --------------------------------------------------

    result = np.full(
        (N, 3),
        np.nan,
        dtype=np.float32
    )

    for channel in range(3):

        y = rgb_flat[valid, channel]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=-1,
            random_state=random_state,
        )

        rf.fit(X_valid, y)

        result[valid, channel] = rf.predict(X_valid)

    # --------------------------------------------------
    # 6. Takaisin rasteriksi
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)

def gam_im(
    im,
    variables,
    variable_names,
    lam=0.6,
    n_splines=20,
):
    """
    Sovittaa GAM-mallit suoraan matplotlibin AxesImage-olion
    näyttämään Köppen-rasteriin.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        esim. ax.imshow(koppen_data, cmap=...)

    variables : dict[str, np.ndarray]
        H x W -rasterit.

    variable_names : list[str]
        GAMin käyttämät muuttujat.

    Returns
    -------
    rgb : np.ndarray
        H x W x 3 uint8 RGB-rasteri.
    """

    # --------------------------------------------------
    # 1. Alkuperäinen imshow-data
    # --------------------------------------------------

    data = np.asarray(im.get_array())

    if data.ndim != 2:
        raise ValueError(
            f"imshow-datan pitää olla 2D-rasteri (H, W), "
            f"mutta shape on {data.shape}"
        )

    H, W = data.shape

    # --------------------------------------------------
    # 2. Haetaan imshown käyttämät todelliset RGB-värit
    # --------------------------------------------------

    rgba = im.cmap(im.norm(data))
    rgb = rgba[..., :3].astype(float)

    # 0..1 -> 0..255
    rgb *= 255.0

    # --------------------------------------------------
    # 3. Rakennetaan selittävät muuttujat
    # --------------------------------------------------

    X = np.column_stack([
        np.asarray(variables[name]).reshape(-1)
        for name in variable_names
    ])

    if X.shape[0] != H * W:
        raise ValueError(
            f"Muuttujarasterien koko ei vastaa imshow-rasteria: "
            f"imshow={H}x{W}, X={X.shape}"
        )

    # --------------------------------------------------
    # 4. Validit pikselit
    # --------------------------------------------------

    valid = np.all(np.isfinite(X), axis=1)

    # myös RGB:n pitää olla validi
    valid &= np.all(
        np.isfinite(rgb.reshape(-1, 3)),
        axis=1
    )

    X_valid = X[valid]
    y_rgb = rgb.reshape(-1, 3)

    # --------------------------------------------------
    # 5. GAM-termit
    # --------------------------------------------------

    terms = s(0)

    for i in range(1, len(variable_names)):
        terms += s(i)

    # --------------------------------------------------
    # 6. Kolme GAMia: R, G ja B
    # --------------------------------------------------

    result = np.full(
        (H * W, 3),
        np.nan,
        dtype=float
    )

    for channel in range(3):

        y = y_rgb[valid, channel]

        gam = LinearGAM(
            terms,
            lam=lam,
            n_splines=n_splines
        )

        gam.fit(X_valid, y)

        result[valid, channel] = gam.predict(X_valid)

    # --------------------------------------------------
    # 7. Takaisin rasteriksi
    # --------------------------------------------------

    result = result.reshape(H, W, 3)

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


def plot_grid_streamplot(u, v, alue=[-180, 180, -90, 90]):
    """
    Piirtää streamplot-kuvaajan suoraan valmiista 2D U- ja V-matriiseista.
    
    Parametrit:
    - u: 2D numpy-taulukko X-suuntaisista nopeuksista (koko: [rivit, sarakkeet])
    - v: 2D numpy-taulukko Y-suuntaisista nopeuksista (koko: [rivit, sarakkeet])
    - alue: Lista tai tuple [x_min, x_max, y_min, y_max] (oletus koko maapallo)
    """
    # Haetaan matriisin ulottuvuudet (rivit vastaavat Y-akselia, sarakkeet X-akseliota)
    naytteet_y, naytteet_x = u.shape
    
    x_min, x_max, y_min, y_max = alue
    
    # Luodaan 1D-akselit, jotka vastaavat matriisin kokoa
    x = np.linspace(x_min, x_max, naytteet_x)
    y = np.linspace(y_min, y_max, naytteet_y)
    
    # Muutetaan tarvittaessa mahdolliset puuttuvat arvot (NaN) nolliksi, jotta streamplot toimii
    u_puhdas = np.nan_to_num(u)
    v_puhdas = np.nan_to_num(v)
    
    # Lasketaan virtauksen voimakkuus (nopeus) viivojen väritystä varten
    nopeus = np.sqrt(u_puhdas**2 + v_puhdas**2)
    
    # Luodaan kuvaaja
    plt.figure(figsize=(12, 6))
    
    # Piirretään virtausviivat
    # tiheys (density) säätää viivojen määrää kuvaajassa (oletus on 1)
    strm = plt.streamplot(x, y, u_puhdas, v_puhdas, color=nopeus, cmap='jet', density=4, linewidth=1)
    
    # Lisätään väripalkki
    plt.colorbar(strm.lines, label='Virtausnopeus')
    
    # Muotoillaan kuvaaja
    plt.title('Globaali virtauskenttä (Streamplot)')
    plt.xlabel('Pituusaste (Longitude)')
    plt.ylabel('Leveysaste (Latitude)')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()



def plot_precip_raster(dem, data, title="Vuotuinen sademäärä"):
    """Plottaa sademäärä-rasteridatan imshow- ja contour-toiminnoilla.

    Parametrit:
    data (2D numpy array): Piirrettävä sademäärädata (mm), muotoa (y, x)
    """
    # Määritetään maantieteellinen laajuus [xmin, xmax, ymin, ymax]
    extent = [-180, 180, -90, 90]

    # Luodaan kuva ja akselit
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Piirretään rasteri (imshow)
    # Käytetään kelta-vihreä-sinistä värikarttaa, asteikko lukittu 0 - 2000 mm
    im = ax.imshow(
        data,
        extent=extent,
        cmap="YlGnBu",
        vmin=0,
        vmax=2000,
        origin="lower",
        aspect="auto",
    )

    # 2. Lisätään määritetyt epätasaiset kontuuriviivat
    # [100, 200, 400, 800, 1200, 1600]
    levels = [100, 200, 400, 800, 1200, 1600]

    # Piirretään viivat (tummansininen erottuu hyvin vaalealta taustalta)
    contours = ax.contour(
        data, levels=levels, extent=extent, colors="#002266", linewidths=0.75
    )

    # Lisätään tekstiarvot kontuuriviivoille
    ax.clabel(contours, inline=True, fmt="%d mm", fontsize=8, colors="#002266")

    # 3. Muotoilu ja karttaelementit
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Lisätään väriskaalapalkki (colorbar) oikeaan reunaan
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
    cbar.set_label("Precipitation (mm)", fontsize=11)

    # Ruudukko taustalle
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_tmean_raster(dem, data, title="Mean temp (tmean) Celsius"):
    """Plottaa tmean-rasteridatan imshow- ja contour-toiminnoilla.

    Parametrit:
    data (2D numpy array): Piirrettävä lämpötiladata, muotoa (y, x)
    """
    # Määritetään maantieteellinen laajuus [xmin, xmax, ymin, ymax]
    extent = [-180, 180, -90, 90]

    # Luodaan kuva ja akselit
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Piirretään rasteri (imshow)
    # origin='lower' varmistaa, että etelä (tai ymin) on alhaalla
    im = ax.imshow(
        data,
        extent=extent,
        cmap="RdYlBu_r",
        vmin=-60,
        vmax=60,
        origin="lower",
        aspect="auto",
    )

    # 2. Lisätään kontuuriviivat (contour)
    # Luodaan tasot esim. 20 asteen välein välille -80...+180
    levels = np.arange(-60, 60, 10)
    contours = ax.contour(
        data, levels=levels, extent=extent, colors="black", linewidths=0.5
    )

    # Lisätään tekstiarvot kontuuriviivoille
    ax.clabel(contours, inline=True, fmt="%d", fontsize=8)

    # 3. Muotoilu ja karttaelementit
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Lisätään väriskaalapalkki (colorbar)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8)
    cbar.set_label("Tmean (°C)", fontsize=11)

    # Ruudukko taustalle (valinnainen, asetettu kontuurien alle)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


import cupy as cp
from cupyx.scipy import ndimage


def laske_etaisyys_rannikosta_gpu(
    maa_maski,
    maapallon_sade_km=6371.0088,
):
    """
    Laskee maapikselien etäisyyden lähimpään meripikseliin GPU:lla.

    Parametrit
    ----------
    maa_maski : cupy.ndarray tai numpy.ndarray
        True  = maa
        False = meri

    maapallon_sade_km : float
        Maapallon säde kilometreinä.

    Palauttaa
    ----------
    cupy.ndarray
        float32-CuPy-array kilometreinä.

        Maa  = etäisyys lähimpään mereen
        Meri = 0
    """

    # ----------------------------------------------------------
    # Varmistetaan CuPy-array
    # ----------------------------------------------------------

    maa_maski = cp.asarray(
        maa_maski,
        dtype=cp.bool_,
    )

    korkeus, leveys = maa_maski.shape

    # ----------------------------------------------------------
    # Latitude
    # ----------------------------------------------------------

    lat = cp.linspace(
        90.0,
        -90.0,
        korkeus,
        dtype=cp.float32,
    )

    lat_rad = cp.deg2rad(lat)

    # ----------------------------------------------------------
    # Maapallon rasterin pystysuuntainen pikselikoko
    # ----------------------------------------------------------

    if korkeus > 1:

        dlat_rad = (
            cp.pi
            / cp.float32(korkeus - 1)
        )

    else:

        dlat_rad = cp.float32(0.0)

    dy_km = (
        cp.float32(maapallon_sade_km)
        * dlat_rad
    )

    # ----------------------------------------------------------
    # Vaakasuuntainen pikselikoko päiväntasaajalla
    # ----------------------------------------------------------

    dx_equator_km = (
        cp.float32(maapallon_sade_km)
        * (
            cp.float32(2.0)
            * cp.pi
            / cp.float32(leveys)
        )
    )

    # ----------------------------------------------------------
    # EDT
    #
    # Maa = 1
    # Meri = 0
    #
    # distance_transform_edt palauttaa etäisyyden
    # lähimpään nollaan eli lähimpään meripikseliin.
    # ----------------------------------------------------------

    rasteri = maa_maski.astype(
        cp.float32
    )

    etaisyys_pixeleina = (
        ndimage.distance_transform_edt(
            rasteri
        )
    )

    # ----------------------------------------------------------
    # Leveysasteen aiheuttama vaakapikselin koon muutos
    # ----------------------------------------------------------

    cos_lat = cp.cos(
        lat_rad
    )

    dx_km = (
        dx_equator_km
        * cos_lat
    )

    # ----------------------------------------------------------
    # Muutetaan pikselietäisyys kilometreiksi.
    #
    # Käytetään pysty- ja vaakasuuntaisen pikselikoon
    # keskiarvoa.
    # ----------------------------------------------------------

    pixel_size_km = (
        dy_km + dx_km
    ) * cp.float32(0.5)

    etaisyys_km = (
        etaisyys_pixeleina
        * pixel_size_km[:, None]
    )

    # ----------------------------------------------------------
    # Merelle etäisyys = 0
    # ----------------------------------------------------------

    etaisyys_km = cp.where(
        maa_maski,
        etaisyys_km,
        cp.float32(0.0),
    )

    return etaisyys_km.astype(
        cp.float32,
        copy=False,
    )



import numpy as np


def laske_koppen(
    kuukausi_lampotilat,
    kuukausi_sateet,
    maa_maski,
    etaisyys_meresta_km
):
    """
    Laskee Köppen-ilmastoluokituksen kuukausittaisista NumPy-rastereista.

    Parametrit
    ----------
    kuukausi_lampotilat : np.ndarray
        Muoto (12, y, x), kuukausien keskilämpötilat °C.

    kuukausi_sateet : np.ndarray
        Muoto (12, y, x), kuukausien sademäärät mm.

    maa_maski : np.ndarray
        Muoto (y, x), True = maa, False = meri.

    etaisyys_meresta_km : np.ndarray
        Muoto (y, x), etäisyys merestä kilometreinä.

    Palauttaa
    ----------
    ilmastovyohyke_tarkka : np.ndarray
        Köppen-luokkarasteri kokonaislukuna.

    lampotila_ka : np.ndarray
        Vuoden keskilämpötila °C.

    lampotila_min : np.ndarray
        Kylmimmän kuukauden lämpötila °C.

    lampotila_max : np.ndarray
        Lämpimimmän kuukauden lämpötila °C.

    vuosi_sademaarä_kartta : np.ndarray
        Vuosisademäärä mm.

    kuivimman_kuukauden_sade : np.ndarray
        Kuivimman kuukauden sademäärä mm.

    sateisimman_kuukauden_sade : np.ndarray
        Sateisimman kuukauden sademäärä mm.

    kylmimman_kuukauden_sade : np.ndarray
        Kylmimmän kuukauden sademäärä mm.

    lampimimman_kuukauden_sade : np.ndarray
        Lämpimimmän kuukauden sademäärä mm.

    kuivimman_kuukauden_lampotila : np.ndarray
        Kuivimman kuukauden lämpötila °C.

    sateisimman_kuukauden_lampotila : np.ndarray
        Sateisimman kuukauden lämpötila °C.
    """

    # ================================================================
    # 1. Vuositason rasterit
    # ================================================================

    lampotila_ka = np.mean(kuukausi_lampotilat, axis=0)
    lampotila_min = np.min(kuukausi_lampotilat, axis=0)
    lampotila_max = np.max(kuukausi_lampotilat, axis=0)

    vuosi_sademaarä_kartta = np.sum(kuukausi_sateet, axis=0)

    kuivimman_kuukauden_sade = np.min(kuukausi_sateet, axis=0)
    sateisimman_kuukauden_sade = np.max(kuukausi_sateet, axis=0)

    # ================================================================
    # 2. Kylmimmän ja lämpimimmän kuukauden sateet
    # ================================================================

    kylmin_kk_indeksi = np.argmin(kuukausi_lampotilat, axis=0)
    lampimin_kk_indeksi = np.argmax(kuukausi_lampotilat, axis=0)

    kylmimman_kuukauden_sade = np.take_along_axis(
        kuukausi_sateet,
        kylmin_kk_indeksi[np.newaxis, ...],
        axis=0
    )[0]

    lampimimman_kuukauden_sade = np.take_along_axis(
        kuukausi_sateet,
        lampimin_kk_indeksi[np.newaxis, ...],
        axis=0
    )[0]

    # ================================================================
    # 3. Kuivimman ja sateisimman kuukauden lämpötilat
    # ================================================================

    kuivin_kk_indeksi = np.argmin(kuukausi_sateet, axis=0)
    sateisin_kk_indeksi = np.argmax(kuukausi_sateet, axis=0)

    kuivimman_kuukauden_lampotila = np.take_along_axis(
        kuukausi_lampotilat,
        kuivin_kk_indeksi[np.newaxis, ...],
        axis=0
    )[0]

    sateisimman_kuukauden_lampotila = np.take_along_axis(
        kuukausi_lampotilat,
        sateisin_kk_indeksi[np.newaxis, ...],
        axis=0
    )[0]

    # ================================================================
    # 4. Merialueet pois
    # ================================================================

    vuosi_sademaarä_kartta = vuosi_sademaarä_kartta.copy()
    vuosi_sademaarä_kartta[~maa_maski] = 0

    # ================================================================
    # 5. Köppen-alavyöhykkeet
    # ================================================================

    ilmastovyohyke_tarkka = np.zeros_like(
        lampotila_ka,
        dtype=int
    )

    # ------------------------------------------------
    # E - POLAARISET
    # ------------------------------------------------

    maski_E = maa_maski & (lampotila_max < 10.0)
    maski_ET = maa_maski & (lampotila_max < 10.0)
    maski_EF = maa_maski & (lampotila_max < 0.0)

    # ------------------------------------------------
    # B - KUIVAT
    # ------------------------------------------------

    suojattu_temp = np.maximum(lampotila_ka, 0.0)

    aro_raja = 20.0 * suojattu_temp
    aavikko_raja = 10.0 * suojattu_temp

    maski_BWh = (
        maa_maski
        & (~maski_E)
        & (vuosi_sademaarä_kartta < aavikko_raja)
        & (lampotila_ka >= 18.0)
    )

    maski_BWk = (
        maa_maski
        & (~maski_E)
        & (vuosi_sademaarä_kartta < aavikko_raja)
        & (lampotila_ka < 18.0)
    )

    maski_BSh = (
        maa_maski
        & (~maski_E)
        & (vuosi_sademaarä_kartta >= aavikko_raja)
        & (vuosi_sademaarä_kartta < aro_raja)
        & (lampotila_ka >= 18.0)
    )

    maski_BSk = (
        maa_maski
        & (~maski_E)
        & (vuosi_sademaarä_kartta >= aavikko_raja)
        & (vuosi_sademaarä_kartta < aro_raja)
        & (lampotila_ka < 18.0)
    )

    maski_B_kaikki = (
        maski_BWh
        | maski_BWk
        | maski_BSh
        | maski_BSk
    )

    # ------------------------------------------------
    # A - TROPIIKKI
    # ------------------------------------------------

    pohja_A = (
        maa_maski
        & (~maski_E)
        & (~maski_B_kaikki)
        & (lampotila_min >= 18.0)
    )

    maski_Af = (
        pohja_A
        & (kuivimman_kuukauden_sade >= 60.0)
    )

    maski_Aw = (
        pohja_A
        & (kuivimman_kuukauden_sade < 60.0)
        & (
            kuivimman_kuukauden_sade
            < (100.0 - vuosi_sademaarä_kartta / 25.0)
        )
    )

    maski_Am = (
        pohja_A
        & (~maski_Af)
        & (~maski_Aw)
    )

    # ------------------------------------------------
    # C - LAUHKEAT
    # ------------------------------------------------

    pohja_C = (
        maa_maski
        & (~maski_E)
        & (~maski_B_kaikki)
        & (~pohja_A)
        & (lampotila_min >= -3.0)
        & (lampotila_min < 18.0)
    )

    maski_Csa_Csb = (
        pohja_C
        & (kuivimman_kuukauden_sade < 40.0)
        & (
            kuivimman_kuukauden_sade
            < sateisimman_kuukauden_sade / 3.0
        )
    )

    maski_Cfb = (
        pohja_C
        & (~maski_Csa_Csb)
        & (etaisyys_meresta_km <= 400.0)
    )

    maski_Cfa_Cwa = (
        pohja_C
        & (~maski_Csa_Csb)
        & (~maski_Cfb)
    )

    # ------------------------------------------------
    # D - MANNERILMASTOT
    # ------------------------------------------------

    pohja_D = (
        maa_maski
        & (~maski_E)
        & (~maski_B_kaikki)
        & (~pohja_A)
        & (~pohja_C)
        & (lampotila_min < -3.0)
        & (lampotila_max >= 10.0)
    )

    maski_Dfa_Dfb = (
        pohja_D
        & (lampotila_max >= 22.0)
    )

    maski_Dfc_Dfd = (
        pohja_D
        & (lampotila_max < 22.0)
        & (~maski_E)
    )

    # ================================================================
    # 6. Numerointi
    # ================================================================

    ilmastovyohyke_tarkka[maski_Af] = 1
    ilmastovyohyke_tarkka[maski_Am] = 2
    ilmastovyohyke_tarkka[maski_Aw] = 3

    ilmastovyohyke_tarkka[maski_BWh] = 4
    ilmastovyohyke_tarkka[maski_BWk] = 5
    ilmastovyohyke_tarkka[maski_BSh] = 6
    ilmastovyohyke_tarkka[maski_BSk] = 7

    ilmastovyohyke_tarkka[maski_Csa_Csb] = 8
    ilmastovyohyke_tarkka[maski_Cfb] = 9
    ilmastovyohyke_tarkka[maski_Cfa_Cwa] = 10

    ilmastovyohyke_tarkka[maski_Dfa_Dfb] = 11
    ilmastovyohyke_tarkka[maski_Dfc_Dfd] = 12

    ilmastovyohyke_tarkka[maski_ET] = 13
    ilmastovyohyke_tarkka[maski_EF] = 14

    # ================================================================
    # 7. Palautetaan kaikki NumPy-rasterit
    # ================================================================

    return (
        ilmastovyohyke_tarkka,
        lampotila_ka,
        lampotila_min,
        lampotila_max,
        vuosi_sademaarä_kartta,
        kuivimman_kuukauden_sade,
        sateisimman_kuukauden_sade,
        kylmimman_kuukauden_sade,
        lampimimman_kuukauden_sade,
        kuivimman_kuukauden_lampotila,
        sateisimman_kuukauden_lampotila
    )


##################################
## ilmasto


def laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    korkeus_gradientti_y,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,

    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    tmax_planet_param=50,
    global_moisture_coeff=1.0,

    palauta_cpu=True,
    debug=False,
):
    """
    KOKO ILMASTOMALLI GPU:LLE / CuPy.

    Suunniteltu NVIDIA RTX 5060 -näytönohjaimelle.

    ------------------------------------------------------------
    SYÖTTEET
    ------------------------------------------------------------

    lat_grid_deg
        2D-latitudikartta asteina.
        Odotettu alue noin -90 ... +90.

    korkeuskartta
        2D-korkeuskartta metreinä.

    korkeus_gradientti_x
        Korkeuskartan gradientti X-suunnassa.

    korkeus_gradientti_y
        Korkeuskartan gradientti Y-suunnassa.

    maa_maski
        Maa-alueen maski.
        Arvot voivat olla esimerkiksi 0/1 tai bool.

    meri_maski
        Merialueen maski.
        Arvot voivat olla esimerkiksi 0/1 tai bool.
        Maski normalisoidaan sisäisesti binääriseksi.

    etaisyys_meresta_km
        Etäisyys merestä kilometreinä.

    tilt_planet_axis
        Akselikallistuma ASTEINA.

        Esimerkiksi Maan kaltainen:
            23.5

    ecc_planet
        Radan eksentrisyys.

        Esimerkiksi Maan kaltainen:
            0.0167

        Pitää olla:
            0 <= ecc_planet < 1

    mvelp_planet
        Perihelin argumentti / radan vaihe ASTEINA.

    global_moisture_coeff
        Maailmanlaajuinen sademäärän kosteuskerroin.

    palauta_cpu
        True  -> NumPy-taulukot
        False -> CuPy-taulukot GPU-muistissa

    debug
        True -> tulostetaan diagnostisia tietoja.

    ------------------------------------------------------------
    PALAUTUS
    ------------------------------------------------------------

    (
        kuukausi_lampotilat,
        kuukausi_sateet,
        kuukausi_tuuli_suunta,
        kuukausi_tuuli_voima,
        kuukausi_merivirta,
    )

    Kaikkien muoto:
        (12, ny, nx)

    ------------------------------------------------------------
    HUOM
    ------------------------------------------------------------

    Tämä funktio käyttää:

        gradientti_x
        gradientti_y

    erikseen.

    Tuulesta muodostetaan myöhemmin 2D-vektori, jota voidaan
    käyttää oikeaan maaston gradienttiin ja orografiaan.
    """

    # ============================================================
    # 1. PARAMETRITARKISTUKSET
    # ============================================================

    ecc_planet = float(
        ecc_planet
    )

    tilt_planet_axis = float(
        tilt_planet_axis
    )

    mvelp_planet = float(
        mvelp_planet
    )

    global_moisture_coeff = float(
        global_moisture_coeff
    )

    if not np.isfinite(ecc_planet):
        raise ValueError(
            "ecc_planet ei ole kelvollinen luku."
        )

    if ecc_planet < 0.0 or ecc_planet >= 1.0:
        raise ValueError(
            "ecc_planet pitää olla välillä 0 <= e < 1."
        )

    if not np.isfinite(tilt_planet_axis):
        raise ValueError(
            "tilt_planet_axis ei ole kelvollinen luku."
        )

    if not np.isfinite(mvelp_planet):
        raise ValueError(
            "mvelp_planet ei ole kelvollinen luku."
        )

    if global_moisture_coeff < 0.0:
        raise ValueError(
            "global_moisture_coeff ei voi olla negatiivinen."
        )

    # ============================================================
    # 2. DATA GPU:LLE
    # ============================================================

    lat = cp.asarray(
        lat_grid_deg,
        dtype=cp.float32
    )

    korkeus = cp.asarray(
        korkeuskartta,
        dtype=cp.float32
    )

    gradientti_x = cp.asarray(
        korkeus_gradientti_x,
        dtype=cp.float32
    )

    gradientti_y = cp.asarray(
        korkeus_gradientti_y,
        dtype=cp.float32
    )

    maa = cp.asarray(
        maa_maski
    )

    meri = cp.asarray(
        meri_maski
    )

    etaisyys_meresta = cp.asarray(
        etaisyys_meresta_km,
        dtype=cp.float32
    )

    # ============================================================
    # 3. MUOTOJEN TARKISTUS
    # ============================================================

    if lat.ndim != 2:
        raise ValueError(
            "lat_grid_deg pitää olla 2D-taulukko."
        )

    odotettu_shape = lat.shape

    tarkistettavat = {
        "korkeuskartta": korkeus,
        "gradientti_x": gradientti_x,
        "gradientti_y": gradientti_y,
        "maa_maski": maa,
        "meri_maski": meri,
        "etaisyys_meresta_km": etaisyys_meresta,
    }

    for nimi, data in tarkistettavat.items():

        if data.shape != odotettu_shape:
            raise ValueError(
                f"{nimi} väärän kokoinen: "
                f"{data.shape}, odotettu "
                f"{odotettu_shape}."
            )

    # ============================================================
    # 4. LATITUDIN TARKISTUS
    # ============================================================

    lat_min = float(
        cp.nanmin(lat)
    )

    lat_max = float(
        cp.nanmax(lat)
    )

    if debug:

        print(
            "LAT min/max:",
            lat_min,
            lat_max
        )

    if (
        not np.isfinite(lat_min)
        or
        not np.isfinite(lat_max)
    ):
        raise ValueError(
            "Latitudikartassa on NaN/Inf-arvoja."
        )

    if lat_min < -90.1 or lat_max > 90.1:
        raise ValueError(
            "lat_grid_deg näyttää olevan väärässä "
            "yksikössä. Odotetaan asteita -90 ... +90."
        )

    # ============================================================
    # 5. KORKEUS / GRADIENTTI - DIAGNOSTIIKKA
    # ============================================================

    if debug:

        print(
            "KORKEUS min/max:",
            float(cp.nanmin(korkeus)),
            float(cp.nanmax(korkeus))
        )

        print(
            "GRADIENTTI X min/max:",
            float(cp.nanmin(gradientti_x)),
            float(cp.nanmax(gradientti_x))
        )

        print(
            "GRADIENTTI Y min/max:",
            float(cp.nanmin(gradientti_y)),
            float(cp.nanmax(gradientti_y))
        )

    # ============================================================
    # 6. MASKIT BINÄÄRISEKSI
    # ============================================================
    #
    # Tärkeä korjaus aiemmasta versiosta:
    #
    # meri_maski voi sisältää muutakin kuin 0/1-arvoja.
    #
    # Esimerkiksi jos merimaskissa on vahingossa suurempia
    # kokonaislukuja, niitä EI saa käyttää suoraan
    # meri * 10.0 -tyyppisessä laskennassa.
    #
    # Siksi tehdään aina binäärinen maski.
    #
    # ============================================================

    maa_bin = (
        maa > 0
    ).astype(cp.float32)

    meri_bin = (
        meri > 0
    ).astype(cp.float32)

    # ============================================================
    # 7. MASKIDIAGNOSTIIKKA
    # ============================================================

    if debug:

        print(
            "MAA maski:",
            float(cp.min(maa_bin)),
            float(cp.max(maa_bin))
        )

        print(
            "MERI maski:",
            float(cp.min(meri_bin)),
            float(cp.max(meri_bin))
        )

    # ============================================================
    # 8. KOKO
    # ============================================================

    ny, nx = lat.shape

    # ============================================================
    # 9. TULOSARRAYT
    # ============================================================

    kuukausi_lampotilat = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )

    kuukausi_sateet = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )

    kuukausi_tuuli_suunta = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )

    kuukausi_tuuli_voima = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )

    kuukausi_merivirta_x = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )
    kuukausi_merivirta_y = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )
    kuukausi_merivirta = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )
    kuukausi_merivirta_suunta = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )
    # ============================================================
    # 10. KIINTEÄT KARTTATERMIT
    # ============================================================
    lapse_rate=6.5
    korkeus_vaikutus = (
        korkeus
        / cp.float32(1000.0)
    ) * cp.float32(lapse_rate)

    # ============================================================
    # 11. LÄMPÖTILAN PERUSPARAMETRI
    # ============================================================
    #
    # tmax_planet on CELSIUSASTEINA.
    #
    # Esimerkiksi:
    #
    #     50.0
    #
    # tarkoittaa +50 °C.
    #
    # Muunnos Kelvin-asteikolle tehdään vasta laskennassa.
    #
    # ============================================================

    tmax_planet = cp.float32(
        tmax_planet_param
    )

    # ============================================================
    # 12. KUUKAUSISILMUKKA
    # ============================================================

    for kk in range(12):

        # ========================================================
        # 12.1 RATAVAIHE
        # ========================================================

        ratakulma_deg = (
            kk
            * 30.0
        )

        ratakulma_rad = np.radians(
            ratakulma_deg
        )

        # ========================================================
        # 12.2 TRUE ANOMALY
        # ========================================================

        true_anomaly = np.radians(
            ratakulma_deg
            - mvelp_planet
        )

        # ========================================================
        # 12.3 ELLIPTISEN RADAN ETÄISYYS
        # ========================================================
        #
        # r = (1 - e²) / (1 + e cos(v))
        #
        # Tässä ei käytetä enää:
        #
        #     etaisyys_au = max(..., 0.01)
        #
        # koska se voi peittää oikean parametrivirheen.
        #
        # ========================================================

        nimittaja = (
            1.0
            +
            ecc_planet
            *
            np.cos(
                true_anomaly
            )
        )

        etaisyys_au = (
            1.0
            -
            ecc_planet ** 2
        ) / nimittaja

        if (
            not np.isfinite(
                etaisyys_au
            )
            or
            etaisyys_au <= 0.0
        ):
            raise ValueError(
                "Virheellinen planeetan rataetäisyys: "
                f"{etaisyys_au} AU "
                f"(kk={kk}, "
                f"ecc={ecc_planet}, "
                f"mvelp={mvelp_planet})"
            )

        # ========================================================
        # 12.4 SÄTEILYKORJAUS
        # ========================================================
        #
        # Säteilyn intensiteetti ~ 1 / sqrt(r)
        #
        # r = 1 AU -> 1.0
        #
        # ========================================================

        sateily_kerroin = (
            1.0
            /
            np.sqrt(
                etaisyys_au
            )
        )

        if debug:

            print(
                f"kk={kk:02d} "
                f"rata={etaisyys_au:.6f} AU "
                f"sateily={sateily_kerroin:.6f} "
                f"true_anomaly="
                f"{np.degrees(true_anomaly):.2f}°"
            )

        # ========================================================
        # 12.5 AURINGON DEKLINAATIO
        # ========================================================
        #
        # tilt_planet_axis oletetaan ASTEIKSI.
        #
        # ========================================================

        deklinaatio = (
            tilt_planet_axis
            *
            np.sin(
                ratakulma_rad
                -
                np.radians(
                    60.0
                )
            )
        )

        # ========================================================
        # 12.6 PERUSLÄMPÖTILAN TASAPAINOTERMI
        # ========================================================
        #
        # Säteilyn vaikutus tehdään Kelvin-asteikolla.
        #
        # ========================================================

        tasapaino_temp = (
            (
                tmax_planet
                +
                cp.float32(
                    273.15
                )
            )
            *
            cp.float32(
                sateily_kerroin
            )
            -
            cp.float32(
                273.15
            )
        )

        # ========================================================
        # 12.7 ZENIITTIKAUDEN ETÄISYYS
        # ========================================================

        zeniitti_etaisyys = cp.abs(
            lat
            -
            cp.float32(
                deklinaatio
            )
        )

        # ========================================================
        # 12.8 ZENIITTIVAIKUTUS
        # ========================================================

        K = cp.float32(
            0.73*3
        )

        zeniitti_vaikutus = (
            K
            *
            zeniitti_etaisyys
        )

        # ========================================================
        # 12.9 MERIPILVIEN JÄÄHDYTYS
        # ========================================================
        #
        # Tärkeä korjaus:
        #
        # käytetään meri_bin-muuttujaa, ei alkuperäistä
        # mahdollisesti ei-binaarista merimaskia.
        #
        # ========================================================

        meri_pilvisyys = cp.float32(
            1.0
        )

        meri_pilvi_max_viilennys = cp.float32(
            10.0
        )

        meri_pilvi_vaikutus = (
            meri_bin
            *
            meri_pilvisyys
            *
            meri_pilvi_max_viilennys
        )

        # ========================================================
        # 12.10 PERUSLÄMPÖTILA
        # ========================================================

        perus_temp = (
            tasapaino_temp
            -
            zeniitti_vaikutus
            -
            meri_pilvi_vaikutus
        )

        if debug:

            print(
                f"KK {kk} "
                f"tasapaino min/max:",
                float(
                    cp.min(
                        tasapaino_temp
                    )
                ),
                float(
                    cp.max(
                        tasapaino_temp
                    )
                )
            )

            print(
                f"KK {kk} "
                f"zeniitti min/max:",
                float(
                    cp.min(
                        zeniitti_vaikutus
                    )
                ),
                float(
                    cp.max(
                        zeniitti_vaikutus
                    )
                )
            )

            print(
                f"KK {kk} "
                f"meri-pilvi min/max:",
                float(
                    cp.min(
                        meri_pilvi_vaikutus
                    )
                ),
                float(
                    cp.max(
                        meri_pilvi_vaikutus
                    )
                )
            )

    # ============================================================
    # OSA 2 JATKUU TÄSTÄ
    # ============================================================


        # ========================================================
        # 13. ILMASTOLLINEN PÄIVÄNTASAaja / ITCZ
        # ========================================================
        #
        # Akselikallistuma siirtää vuodenaikojen mukana
        # termistä päiväntasaajaa.
        #
        # 0.65 tekee liikkeestä hieman vaimeamman kuin
        # todellinen aurinkodeklinaatio.
        #
        # ========================================================

        akselin_kallistuma = (
            tilt_planet_axis
            *
            np.cos(
                ratakulma_rad
            )
        )

        ilmastollinen_paivantaasaaja = (
            akselin_kallistuma
            *
            0.65
        )

        # ========================================================
        # 14. RELATIIVINEN LATITUDI
        # ========================================================

        relatiivinen_lat = (
            lat
            -
            cp.float32(
                ilmastollinen_paivantaasaaja
            )
        )

        abs_rel_lat = cp.abs(
            relatiivinen_lat
        )

        # ========================================================
        # 15. TUULIVYÖHYKKEET
        # ========================================================
        #
        # Nyt erotetaan:
        #
        #   tuulen suunta
        #   tuulen voimakkuus
        #
        # Tuulisuunta on kulma-tyyppinen suure:
        #
        #       0 ... 360 astetta
        #
        # Tuulivektori muodostetaan myöhemmin.
        #
        # ========================================================

        # --------------------------------------------------------
        # Peruspainegradientin kaltaista vyöhykeprofiilia
        # --------------------------------------------------------

        perus_tuuli = cp.sin(
            cp.deg2rad(
                relatiivinen_lat
                * 2.0
            )
        )

        # ========================================================
        # 16. HADLEY / FERREL / POLAR
        # ========================================================

        # Hadley:
        # tropiikin pääasiallinen pintatuuli

        hadley_maski = (
            abs_rel_lat < 30.0
        )

        # Ferrel:
        # keskileveyksien vastakkainen vyöhyke

        ferrel_maski = (
            (abs_rel_lat >= 30.0)
            &
            (abs_rel_lat < 62.0)
        )

        # Polar:
        # korkeiden leveysasteiden vyöhyke

        polar_maski = (
            abs_rel_lat >= 62.0
        )

        # --------------------------------------------------------
        # Tuulen X-komponentti
        # --------------------------------------------------------
        #
        # Tämä ei vielä ole lopullinen maantieteellinen tuulen
        # suunta, vaan yksinkertaistettu planetaarinen
        # itä-länsisuuntainen komponentti.
        #
        # --------------------------------------------------------

        tuuli_x = cp.zeros_like(
            lat,
            dtype=cp.float32
        )

        tuuli_x = cp.where(
            hadley_maski,
            -cp.abs(
                perus_tuuli
            ),
            tuuli_x
        )

        tuuli_x = cp.where(
            ferrel_maski,
            cp.abs(
                perus_tuuli
            ),
            tuuli_x
        )

        tuuli_x = cp.where(
            polar_maski,
            -cp.abs(
                perus_tuuli
            ),
            tuuli_x
        )

        # ========================================================
        # 17. MERIDIONAALINEN TUULIKOMPONENTTI
        # ========================================================
        #
        # Todellisessa ilmastossa tuuli ei ole täydellisesti
        # itä-länsisuuntainen.
        #
        # Lisätään pieni vuodenaikainen komponentti kohti/
        # poispäin ilmastollisesta päiväntasaajasta.
        #
        # Pieni kerroin on tarkoituksellinen:
        # emme halua tuhota pääasiallista vyöhykekiertoa.
        #
        # ========================================================

        vuodenaika_sign = np.sign(
            akselin_kallistuma
        )

        tuuli_y = (
            cp.float32(
                0.12
                * vuodenaika_sign
            )
            *
            cp.tanh(
                relatiivinen_lat
                /
                cp.float32(
                    25.0
                )
            )
        )

        tuuli_y = cp.asarray(
            tuuli_y,
            dtype=cp.float32
        )

        # ========================================================
        # 18. TUULIVEKTORIN VOIMAKKUUS
        # ========================================================

        tuuli_voima_raaka = cp.sqrt(
            tuuli_x
            *
            tuuli_x
            +
            tuuli_y
            *
            tuuli_y
        )

        # --------------------------------------------------------
        # Alaraja estää lähes nollatuulen aiheuttamat
        # epävakaat kosteus- ja orografiavaikutukset.
        # --------------------------------------------------------

        tuuli_voima = cp.clip(
            tuuli_voima_raaka
            * cp.float32(
                3.0
            ),
            cp.float32(
                0.3
            ),
            cp.float32(
                3.0
            )
        )

        # ========================================================
        # 19. TUULEN KOMPONENTTIEN NORMALISOINTI
        # ========================================================

        tuuli_normi = cp.sqrt(
            tuuli_x
            *
            tuuli_x
            +
            tuuli_y
            *
            tuuli_y
            +
            cp.float32(
                1.0e-8
            )
        )

        tuuli_x_yksikko = (
            tuuli_x
            /
            tuuli_normi
        )

        tuuli_y_yksikko = (
            tuuli_y
            /
            tuuli_normi
        )

        # ========================================================
        # 20. TUULEN KULMA
        # ========================================================
        #
        # Tallennetaan tulokseen asteina.
        #
        # 0   = +X
        # 90  = +Y
        # 180 = -X
        # 270 = -Y
        #
        # Tämä on matemaattinen suunta, ei vielä meteorologinen
        # "mistä tuuli tulee" -suunta.
        #
        # ========================================================

        tuuli_suunta = (
            cp.degrees(
                cp.arctan2(
                    tuuli_y_yksikko,
                    tuuli_x_yksikko
                )
            )
            % cp.float32(
                360.0
            )
        )

        # ========================================================
        # 21. TALLENNUS
        # ========================================================

        kuukausi_tuuli_suunta[
            kk
        ] = tuuli_suunta

        kuukausi_tuuli_voima[
            kk
        ] = tuuli_voima

        # ========================================================
        # 22. TUULEN SUUNTAINEN MAASTOGRADIENTTI
        # ========================================================
        #
        # TÄMÄ ON YKSI TÄRKEIMMISTÄ UUDISTUKSISTA.
        #
        # Aikaisemmin käytettiin käytännössä vain:
        #
        #     gradientti_x * tuuli_suunta
        #
        # Nyt käytetään molempia gradientteja.
        #
        # --------------------------------------------------------
        #
        # Jos gradientti on positiivinen tuulen suunnassa:
        #
        #     ilma nousee -> enemmän orografista sadetta
        #
        # Jos negatiivinen:
        #
        #     ilma laskee -> sadevarjon suunta
        #
        # ========================================================

        maasto_gradientti_tuulen_suunnassa = (
            gradientti_x
            *
            tuuli_x_yksikko
            +
            gradientti_y
            *
            tuuli_y_yksikko
        )

        # ========================================================
        # 23. OROGRAFINEN PAKOTE
        # ========================================================

        orografinen_pakote = (
            maasto_gradientti_tuulen_suunnassa
            *
            tuuli_voima
        )

        # ========================================================
        # 24. VUORISTON NOUSU / LASKU
        # ========================================================
        #
        # Pelkkä gradientti voi olla hyvin pieni.
        # Käytetään sitä yhdessä absoluuttisen korkeuden kanssa,
        # mutta pidetään vaikutus rajattuna.
        #
        # ========================================================

        korkeus_orografia = cp.clip(
            korkeus
            /
            cp.float32(
                150.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                80.0
            )
        )

        dynaaminen_orografia = cp.clip(
            korkeus_orografia
            +
            orografinen_pakote
            *
            cp.float32(
                2.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                250.0
            )
        )

        # ========================================================
        # 25. MERIVIRRAN PERUSVAIKUTUS
        # ========================================================
        #
        # Merivirta riippuu:
        #
        #   leveysasteesta
        #   tuulen X-suunnasta
        #   etäisyydestä merestä
        #
        # Vaikutus vaimenee rannasta sisämaahan.
        #
        # ========================================================

        # ========================================================
        # 25. MERIVIRRAN PERUSVAIKUTUS
        # ========================================================
        #
        # Merivirta riippuu:
        #
        #   leveysasteesta
        #   tuulen X- ja Y-suunnista
        #   etäisyydestä merestä
        #
        # Vaikutus vaimenee rannasta sisämaahan.
        #
        # ========================================================

        virta_lat = cp.sin(
            cp.deg2rad(
                lat
                *
                cp.float32(
                    2.0
                )
            )
        )

        virta_suunta_x = (
            virta_lat
            *
            tuuli_x_yksikko
        )

        virta_suunta_y = (
            virta_lat
            *
            tuuli_y_yksikko
        )

        meri_etaisyys_paino = cp.exp(
            -etaisyys_meresta
            /
            cp.float32(
                200.0
            )
        )

        meri_anomalia_x = (
            virta_suunta_x
            *
            meri_etaisyys_paino
            *
            cp.float32(
                4.0
            )
            *
            tuuli_voima
        )

        meri_anomalia_y = (
            virta_suunta_y
            *
            meri_etaisyys_paino
            *
            cp.float32(
                4.0
            )
            *
            tuuli_voima
        )

        # Tallennetaan X- ja Y-anomaliat omiin kuukausitaulukoihinsa
        kuukausi_merivirta_x[
            kk
        ] = cp.where(
            maa_bin < 1.0,
            virta_suunta_x*tuuli_voima,
            cp.float32(
                0.0
            )
        )

        kuukausi_merivirta_y[
            kk
        ] = cp.where(
            maa_bin < 1.0,
            virta_suunta_y*tuuli_voima,
            cp.float32(
                0.0
            )
        )
        # ========================================================
        # 27. MERIVIRRAN VOIMAKKUUDEN LASKENTA
        # ========================================================

        # Käytetään valmiita paikallisia anomalia-muuttujia hypot-funktiolla
        meri_voima = cp.hypot(
            #meri_anomalia_x,
            #meri_anomalia_y
        kuukausi_merivirta_x[
            kk
        ],
        kuukausi_merivirta_y[
            kk
        ]
        )

        kuukausi_merivirta[
            kk
        ] = cp.where(
            maa_bin < 1.0,
            meri_voima,
            cp.float32(
                0.0
            )
        )
        # ========================================================
        # 26. MERIVIRRAN SUUNNAN LASKENTA
        # ========================================================
        #
        # Lasketaan merivirran suunta asteina (0 - 360°)
        # 0° = Pohjoinen, 90° = Itä, 180° = Etelä, 270° = Länsi
        #
        # ========================================================

        meri_radiaanit = cp.arctan2(
            meri_anomalia_y,
            meri_anomalia_x
        )

        meri_asteet_matemaattinen = cp.degrees(
            meri_radiaanit
        )

        # Muunnetaan kompassisuunnaksi (0 = Pohjoinen, myötäpäivään)
        meri_suunta_kompassi = (
            cp.float32(
                90.0
            )
            -
            meri_asteet_matemaattinen
        )

        # Normalisoidaan välille 0 - 360 astetta
        meri_suunta_normalisoitu = (
            meri_suunta_kompassi
            +
            cp.float32(
                360.0
            )
        ) % cp.float32(
            360.0
        )

        kuukausi_merivirta_suunta[
            kk
        ] = cp.where(
            maa_bin > 0.0,
            meri_suunta_normalisoitu,
            cp.float32(
                0.0
            )
        )

        # ========================================================
        # 26. ITCZ
        # ========================================================
        #
        # ITCZ:n maksimi seuraa ilmastollista päiväntasaajaa.
        #
        # Leveämpi jakauma kuin vanhassa versiossa tekee
        # tropiikista vähemmän keinotekoisen kapean.
        #
        # ========================================================

        itcz_etaisyys = (
            lat
            -
            cp.float32(
                ilmastollinen_paivantaasaaja
            )
        )

        itcz_vaikutus = (
            cp.exp(
                -(
                    itcz_etaisyys
                    *
                    itcz_etaisyys
                )
                /
                cp.float32(
                    55.0
                )
            )
            *
            cp.float32(
                520.0
            )
        )

        # ========================================================
        # 27. POLAARIFRONTIT
        # ========================================================

        polar_pohjoinen = (
            cp.float32(
                52.0
            )
            +
            cp.float32(
                ilmastollinen_paivantaasaaja
                *
                0.5
            )
        )

        polar_etela = (
            cp.float32(
                -52.0
            )
            +
            cp.float32(
                ilmastollinen_paivantaasaaja
                *
                0.5
            )
        )

        polar_n_etaisyys = (
            lat
            -
            polar_pohjoinen
        )

        polar_s_etaisyys = (
            lat
            -
            polar_etela
        )

        polar_front_vaikutus = (
            cp.exp(
                -(
                    polar_n_etaisyys
                    *
                    polar_n_etaisyys
                )
                /
                cp.float32(
                    180.0
                )
            )
            *
            cp.float32(
                180.0
            )
            +
            cp.exp(
                -(
                    polar_s_etaisyys
                    *
                    polar_s_etaisyys
                )
                /
                cp.float32(
                    180.0
                )
            )
            *
            cp.float32(
                180.0
            )
        )

        # ========================================================
        # 28. SUBTROPIIKIN KUIVUUS
        # ========================================================
        #
        # Vanhan version kuivavyöhyke oli liian jyrkkä.
        #
        # Nyt käytetään leveämpää ja pehmeämpää profiilia.
        #
        # Tämä auttaa erityisesti:
        #
        #     sademetsä
        #          ↓
        #     kostea savanni
        #          ↓
        #     kuiva savanni
        #          ↓
        #     puoliaavikko
        #          ↓
        #     aavikko
        #
        # ========================================================

        kuiva_pohjoinen = (
            cp.float32(
                ilmastollinen_paivantaasaaja
            )
            +
            cp.float32(
                28.0
            )
        )

        kuiva_etela = (
            cp.float32(
                ilmastollinen_paivantaasaaja
            )
            -
            cp.float32(
                28.0
            )
        )

        kuivuus_n = cp.exp(
            -(
                (
                    lat
                    -
                    kuiva_pohjoinen
                )
                *
                (
                    lat
                    -
                    kuiva_pohjoinen
                )
            )
            /
            cp.float32(
                95.0
            )
        )

        kuivuus_s = cp.exp(
            -(
                (
                    lat
                    -
                    kuiva_etela
                )
                *
                (
                    lat
                    -
                    kuiva_etela
                )
            )
            /
            cp.float32(
                95.0
            )
        )

        subtropiikki_kuivuus = (
            cp.float32(
                1.0
            )
            -
            cp.float32(
                0.58
            )
            *
            (
                kuivuus_n
                +
                kuivuus_s
            )
        )

        subtropiikki_kuivuus = cp.clip(
            subtropiikki_kuivuus,
            cp.float32(
                0.22
            ),
            cp.float32(
                1.0
            )
        )

        # ========================================================
        # 29. KOSTEUDEN PERUSKERROIN
        # ========================================================
        #
        # Etäisyys merestä ei yksin määrää sadetta.
        #
        # Tropiikissa kosteus voi kulkeutua pitkälle sisämaahan,
        # etenkin pasaattien mukana.
        #
        # ========================================================

        abs_lat = cp.abs(
            lat
        )

        pasaati_paino = cp.exp(
            -(
                abs_lat
                /
                cp.float32(
                    18.0
                )
            )
        )

        # ========================================================
        # 30. KOSTEUDEN KANTAVUUS
        # ========================================================

        kantavuus_matka = (
            cp.float32(
                900.0
            )
            *
            (
                cp.float32(
                    1.0
                )
                +
                tuuli_voima
                *
                cp.float32(
                    0.7
                )
            )
            *
            (
                cp.float32(
                    1.0
                )
                +
                pasaati_paino
                *
                cp.float32(
                    1.5
                )
            )
        )

        # ========================================================
        # 31. TEHOKAS ETÄISYYS MERELTÄ
        # ========================================================
        #
        # Korkea vuoristo muodostaa ilmastollisen esteen.
        #
        # Mutta vaikutusta ei tehdä yhtä rajuna kuin vanhassa
        # versiossa.
        #
        # ========================================================

        korkeuseste = cp.where(
            korkeus > cp.float32(
                1200.0
            ),
            cp.float32(
                500.0
            )
            *
            (
                korkeus
                /
                cp.float32(
                    2000.0
                )
            ),
            cp.float32(
                0.0
            )
        )

        efektiivinen_etaisyys = (
            etaisyys_meresta
            +
            korkeuseste
        )

        # ========================================================
        # 32. MERELTÄ TULEVAN KOSTEUDEN PERUSPAINO
        # ========================================================

        rannikko_kerroin = cp.exp(
            -efektiivinen_etaisyys
            /
            cp.maximum(
                kantavuus_matka,
                cp.float32(
                    100.0
                )
            )
        )

        # ========================================================
        # 33. TROPPIKINEN KOSTEUDEN TEHOSTUS
        # ========================================================

        trooppinen_kosteus = (
            rannikko_kerroin
            +
            cp.float32(
                0.35
            )
            *
            pasaati_paino
        )

        trooppinen_kosteus = cp.clip(
            trooppinen_kosteus,
            cp.float32(
                0.0
            ),
            cp.float32(
                1.5
            )
        )

        # ========================================================
        # 34. ONSHORE / OFFSHORE
        # ========================================================
        #
        # Maaston gradientin ja tuulivektorin pistetulo kertoo,
        # kulkeeko ilma ylöspäin vai alaspäin.
        #
        # Positiivinen:
        #     kostea ilma nousee maastoa vasten
        #
        # Negatiivinen:
        #     ilma kulkee laskevaan maastoon
        #
        # ========================================================

        onshore_pakote = (
            maasto_gradientti_tuulen_suunnassa
        )

        onshore_paino = cp.clip(
            onshore_pakote
            /
            cp.float32(
                500.0
            ),
            cp.float32(
                -1.0
            ),
            cp.float32(
                1.0
            )
        )

        # ========================================================
        # 35. KOSTEUDEN OROGRAFINEN MUOKKAUS
        # ========================================================

        orografinen_kosteus = (
            cp.float32(
                1.0
            )
            +
            onshore_paino
            *
            cp.float32(
                0.45
            )
        )

        orografinen_kosteus = cp.clip(
            orografinen_kosteus,
            cp.float32(
                0.45
            ),
            cp.float32(
                1.6
            )
        )

        # ========================================================
        # OSA 3 JATKUU TÄSTÄ
        # ========================================================
        # ========================================================
        # 36. KOSTEUDEN KANTAVUUS / SISÄMAAN KOSTEUS
        # ========================================================
        #
        # Rannikon ei pidä olla ainoa paikka, jossa tropiikki
        # voi olla erittäin kostea.
        #
        # Erityisesti päiväntasaajan ympärillä kosteus säilyy
        # pitkälle sisämaahan.
        #
        # ========================================================

        sisamaan_kosteus = cp.exp(
            -efektiivinen_etaisyys
            /
            cp.maximum(
                kantavuus_matka
                *
                (
                    cp.float32(1.0)
                    +
                    pasaati_paino
                    *
                    cp.float32(1.5)
                ),
                cp.float32(150.0)
            )
        )

        # ========================================================
        # 37. TROOPPINEN KOSTEUS
        # ========================================================
        #
        # ITCZ:n kohdalla kosteuden saatavuus on korkea.
        #
        # rannikko + sisämaan kosteus + ITCZ
        #
        # yhdistetään pehmeästi.
        #
        # ========================================================

        itcz_kosteus = cp.exp(
            -(
                itcz_etaisyys
                *
                itcz_etaisyys
            )
            /
            cp.float32(
                100.0
            )
        )

        trooppinen_kosteus = (
            cp.float32(
                0.55
            )
            *
            rannikko_kerroin
            +
            cp.float32(
                0.65
            )
            *
            sisamaan_kosteus
            +
            cp.float32(
                0.55
            )
            *
            itcz_kosteus
        )

        # ========================================================
        # 38. PASAATIEN KOSTEUSTEHOSTUS
        # ========================================================
        #
        # Tropiikissa kosteus kulkeutuu tehokkaasti sisämaahan.
        #
        # Tätä ei tehdä pelkällä kertolaskulla alkuperäiseen
        # kosteuteen, jotta CuPy:n broadcasting-ongelmaa ei synny.
        #
        # ========================================================

        pasaati_tehostus = (
            cp.float32(
                1.0
            )
            +
            cp.float32(
                0.28
            )
            *
            pasaati_paino
        )

        trooppinen_kosteus = (
            trooppinen_kosteus
            *
            pasaati_tehostus
        )

        # ========================================================
        # 39. ONSHORE-VAIKUTUS
        # ========================================================
        #
        # Tuulen tuodessa ilmaa nousevaan maastoon kosteus
        # säilyy paremmin.
        #
        # Laskevassa ilmassa kosteus pienenee.
        #
        # ========================================================

        trooppinen_kosteus = (
            trooppinen_kosteus
            *
            orografinen_kosteus
        )

        # ========================================================
        # 40. KOSTEUDEN ABSOLUUTTINEN RAJOITUS
        # ========================================================

        trooppinen_kosteus = cp.clip(
            trooppinen_kosteus,
            cp.float32(
                0.0
            ),
            cp.float32(
                1.8
            )
        )

        # ========================================================
        # 41. TROOPPISEN SADEMÄÄRÄN PERUSOSA
        # ========================================================
        #
        # ITCZ tuottaa voimakkaan sateen.
        #
        # Kosteuskerroin tekee rannikko -> sisämaa -siirtymästä
        # asteittaisen.
        #
        # ========================================================

        tropiikki_sade = (
            itcz_vaikutus
            *
            cp.clip(
                trooppinen_kosteus
                +
                cp.float32(
                    0.15
                ),
                cp.float32(
                    0.0
                ),
                cp.float32(
                    1.5
                )
            )
        )

        # ========================================================
        # 42. POLAARIRINTAMAN SADE
        # ========================================================
        #
        # Keski- ja korkeilla leveysasteilla sade ei seuraa
        # pelkästään etäisyyttä merestä.
        #
        # ========================================================

        lauhkea_kosteus = (
            cp.float32(
                0.35
            )
            +
            cp.float32(
                0.65
            )
            *
            sisamaan_kosteus
        )

        lauhkea_kosteus = (
            lauhkea_kosteus
            *
            orografinen_kosteus
        )

        lauhkea_kosteus = cp.clip(
            lauhkea_kosteus,
            cp.float32(
                0.15
            ),
            cp.float32(
                1.4
            )
        )

        polar_sade = (
            polar_front_vaikutus
            *
            lauhkea_kosteus
        )

        # ========================================================
        # 43. PERUSSADE ENNEN KUIVUUSVYÖHYKETTÄ
        # ========================================================

        perus_sade = (
            tropiikki_sade
            +
            polar_sade
            *
            cp.float32(
                0.75
            )
        )

        # ========================================================
        # 44. SUBTROPIIKIN KUIVUUS
        # ========================================================
        #
        # Tärkeä ero vanhaan:
        #
        # emme kerro koko sadetta suoraan erittäin pienellä
        # kuivuuskertoimella.
        #
        # Sen sijaan kuivuus vaikuttaa vain osaan sateesta.
        #
        # Tämä tekee:
        #
        #     sademetsä
        #       ->
        #     kostea savanni
        #       ->
        #     kuiva savanni
        #       ->
        #     puoliaavikko
        #       ->
        #     aavikko
        #
        # paljon pehmeämmin.
        #
        # ========================================================

        kuivuus_painotus = cp.clip(
            subtropiikki_kuivuus,
            cp.float32(
                0.22
            ),
            cp.float32(
                1.0
            )
        )

        subtrooppinen_sade = (
            perus_sade
            *
            (
                cp.float32(
                    0.30
                )
                +
                cp.float32(
                    0.70
                )
                *
                kuivuus_painotus
            )
        )

        # ========================================================
        # 45. TROPPIIKIN SUOJAAMINEN LIIALTA KUIVUUSEFEKTILTÄ
        # ========================================================
        #
        # Päiväntasaajan lähellä subtrooppinen kuivuus ei saa
        # leikata sademetsää.
        #
        # ========================================================

        tropiikin_suoja = cp.clip(
            (
                cp.float32(
                    12.0
                )
                -
                abs_rel_lat
            )
            /
            cp.float32(
                8.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        sade_ennen_vuoristoa = (
            subtrooppinen_sade
            *
            (
                cp.float32(
                    1.0
                )
                +
                tropiikin_suoja
                *
                cp.float32(
                    0.25
                )
            )
        )

        # ========================================================
        # 46. KORKEUSSADE
        # ========================================================
        #
        # Korkeus itsessään lisää sateen mahdollisuutta vain
        # kohtuullisesti.
        #
        # Vanhan version:
        #
        #     korkeus / 150
        #
        # tuotti helposti valtavan lisäsateen.
        #
        # Nyt vaikutus on loivempi.
        #
        # ========================================================

        korkeussade = cp.clip(
            korkeus
            /
            cp.float32(
                250.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                35.0
            )
        )

        # ========================================================
        # 47. DYNAMIIKAN OROGRAFINEN SADE
        # ========================================================
        #
        # Positiivinen gradientti:
        #     kostea ilma nousee
        #
        # Negatiivinen:
        #     laskeva ilma
        #
        # Käytetään erikseen nousu- ja laskuosaa.
        #
        # ========================================================

        nousu = cp.maximum(
            orografinen_pakote,
            cp.float32(
                0.0
            )
        )

        lasku = cp.maximum(
            -orografinen_pakote,
            cp.float32(
                0.0
            )
        )

        noususade = cp.clip(
            nousu
            *
            cp.float32(
                1.2
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                180.0
            )
        )

        sadevarjo = cp.clip(
            lasku
            *
            cp.float32(
                0.45
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                80.0
            )
        )

        # ========================================================
        # 48. VUORISTON SADE
        # ========================================================

        vuoristosade = (
            korkeussade
            *
            cp.float32(
                1.5
            )
            +
            noususade
        )

        # ========================================================
        # 49. VUORISTON KOSTEUSRAJOITUS
        # ========================================================
        #
        # Vuoristo ei saa luoda sateeseen energiaa tyhjästä.
        #
        # Siksi orografinen sade sidotaan saatavilla olevaan
        # kosteuteen.
        #
        # ========================================================

        kosteuden_saatavuus = cp.clip(
            (
                trooppinen_kosteus
                +
                sisamaan_kosteus
            )
            /
            cp.float32(
                1.5
            ),
            cp.float32(
                0.15
            ),
            cp.float32(
                1.5
            )
        )

        vuoristosade = (
            vuoristosade
            *
            kosteuden_saatavuus
        )

        # ========================================================
        # 50. SADEVARJON VAIKUTUS
        # ========================================================
        #
        # Sadevarjo vähentää jo muodostunutta sadetta, mutta ei
        # saa tehdä siitä negatiivista.
        #
        # ========================================================

        sade = (
            sade_ennen_vuoristoa
            +
            vuoristosade
            -
            sadevarjo
        )

        # ========================================================
        # 51. RANNIKON VÄHIMMÄISKOSTEUS
        # ========================================================
        #
        # Tämä estää epärealistisen "kuiva rannikko ->
        # muutama kilometri -> aavikko" -katkoksen.
        #
        # Vaikutus on voimakas vain lähellä merta ja tropiikissa.
        #
        # ========================================================

        rannikko_suoja = cp.exp(
            -etaisyys_meresta
            /
            cp.float32(
                180.0
            )
        )

        rannikko_suoja *= (
            cp.float32(
                0.35
            )
            +
            cp.float32(
                0.65
            )
            *
            itcz_kosteus
        )

        sade = (
            sade
            +
            rannikko_suoja
            *
            cp.float32(
                25.0
            )
        )

        # ========================================================
        # 52. MEREN JA MAAN ERO
        # ========================================================
        #
        # Merellä sade saa olla hieman tasaisempaa.
        # Maalla vuodenaikaisuus ja orografia saavat näkyä
        # enemmän.
        #
        # ========================================================

        meri_sadekerroin = (
            cp.float32(
                0.95
            )
        )

        maa_sadekerroin = (
            cp.float32(
                1.0
            )
        )

        sade = cp.where(
            meri_bin > 0.0,
            sade * meri_sadekerroin,
            sade * maa_sadekerroin
        )

        # ========================================================
        # 53. LOPULLINEN KOSTEUSKERROIN
        # ========================================================

        sade = (
            sade
            *
            cp.float32(
                global_moisture_coeff
            )
        )

        # ========================================================
        # 54. SADEMÄÄRÄN TURVARAJAT
        # ========================================================
        #
        # Negatiivinen sade ei ole mahdollinen.
        #
        # Yläraja suojaa myös vahingossa syntyviltä numeerisilta
        # piikeiltä.
        #
        # ========================================================

        sade = cp.nan_to_num(
            sade,
            nan=0.0,
            posinf=5000.0,
            neginf=0.0
        )

        sade = cp.clip(
            sade,
            cp.float32(
                0.0
            ),
            cp.float32(
                5000.0
            )
        )

        # ========================================================
        # 55. TALLENNA SADE
        # ========================================================

        kuukausi_sateet[
            kk
        ] = sade

        # ========================================================
        # 56. SADE-DIAGNOSTIIKKA
        # ========================================================

        if debug:

            print(
                f"KK {kk} "
                f"SADE min/max:",
                float(
                    cp.min(
                        sade
                    )
                ),
                float(
                    cp.max(
                        sade
                    )
                )
            )

            print(
                f"KK {kk} "
                f"KOSTEUS min/max:",
                float(
                    cp.min(
                        trooppinen_kosteus
                    )
                ),
                float(
                    cp.max(
                        trooppinen_kosteus
                    )
                )
            )

            print(
                f"KK {kk} "
                f"OROGRAFIA min/max:",
                float(
                    cp.min(
                        orografinen_pakote
                    )
                ),
                float(
                    cp.max(
                        orografinen_pakote
                    )
                )
            )

        # ========================================================
        # 57. PILVISYYDEN JÄÄHDYTYS
        # ========================================================
        #
        # Sade vaikuttaa pilvisyyteen, mutta vaikutus pidetään
        # lämpötilassa maltillisena.
        #
        # ========================================================

        pilvisyys_dt = cp.clip(
            sade
            /
            cp.float32(
                100.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                4.0
            )
        )

        # ========================================================
        # OSA 4 JATKUU TÄSTÄ
        # ========================================================
        # ========================================================
        # 58. MANNERMAISUUS
        # ========================================================
        #
        # Mannermaisuus ei saa yksinään tehdä lämpötilasta
        # valtavan kylmää.
        #
        # Vaikutus kasvaa:
        #
        #   - etäisyyden merestä kasvaessa
        #   - leveysasteen kasvaessa
        #   - kuivuuden kasvaessa
        #
        # Mutta vaikutus on pehmeä.
        #
        # ========================================================

        mannermaisuus_etaisyys = cp.clip(
            etaisyys_meresta
            /
            cp.float32(
                1800.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        mannermaisuus_etaisyys = (
            cp.float32(
                0.35
            )
            +
            cp.float32(
                0.65
            )
            *
            mannermaisuus_etaisyys
        )

        # ========================================================
        # 59. LEVEYSASTEPAINOTUS
        # ========================================================

        leveysaste_painotus = cp.clip(
            cp.abs(
                lat
            )
            /
            cp.float32(
                90.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        # ========================================================
        # 60. KUIVUUDEN VAIKUTUS
        # ========================================================
        #
        # Kuiva manner saa suuremman lämpötilan vuodenaikais-
        # vaihtelun.
        #
        # Tätä ei käytetä suoraan lämpötilan vähentämiseen.
        #
        # ========================================================

        kuivuus_anomalia = cp.clip(
            cp.float32(
                1.0
            )
            -
            sade
            /
            cp.float32(
                180.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        mannermaisuus_indeksi = (
            mannermaisuus_etaisyys
            *
            (
                cp.float32(
                    0.35
                )
                +
                cp.float32(
                    0.65
                )
                *
                kuivuus_anomalia
            )
            *
            leveysaste_painotus
        )

        mannermaisuus_indeksi = cp.clip(
            mannermaisuus_indeksi,
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        # ========================================================
        # 61. VUODENAIKAINEN MANNERILMASTON VAIHTELU
        # ========================================================
        #
        # Tärkeä muutos:
        #
        # Ei käytetä vanhaa:
        #
        #     +12 / -22
        #
        # suoraan merkin perusteella.
        #
        # Nyt käytetään jatkuvaa vuodenaikafunktiota.
        #
        # Tämä poistaa äkilliset kuukausihypyt.
        #
        # ========================================================

        vuodenaika = np.cos(
            ratakulma_rad
            -
            np.pi
        )

        vuodenaika = cp.float32(
            vuodenaika
        )

        # ========================================================
        # 62. MANNERILMASTON AMPLITUDI
        # ========================================================

        manner_amplitudi = (
            cp.float32(
                18.0
            )
            *
            mannermaisuus_indeksi
        )

        # ========================================================
        # 63. VUODENAIKAINEN LÄMPÖTILAVAIKUTUS
        # ========================================================

        manner_lampotila_muutos = (
            manner_amplitudi
            *
            vuodenaika
        )

        # ========================================================
        # 64. TROPPIKINEN VAIMENNUS
        # ========================================================
        #
        # Päiväntasaajan lähellä vuodenaikainen lämpötilavaihtelu
        # on pienempi.
        #
        # ========================================================

        tropiikin_lampotila_vaimennus = cp.clip(
            cp.abs(
                lat
            )
            /
            cp.float32(
                30.0
            ),
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        manner_lampotila_muutos *= (
            cp.float32(
                0.35
            )
            +
            cp.float32(
                0.65
            )
            *
            tropiikin_lampotila_vaimennus
        )

        # ========================================================
        # 65. MEREN LÄMPÖTILAN VUODENAIKAINEN VAIMENNUS
        # ========================================================
        #
        # Meri tasaa vuodenaikaisvaihtelua.
        #
        # ========================================================

        meri_vaimennus = cp.exp(
            -etaisyys_meresta
            /
            cp.float32(
                250.0
            )
        )

        meri_vaimennus = cp.clip(
            meri_vaimennus,
            cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            )
        )

        manner_lampotila_muutos *= (
            cp.float32(
                1.0
            )
            -
            meri_vaimennus
            *
            cp.float32(
                0.70
            )
        )

        # ========================================================
        # 66. KORKEUDEN LÄMPÖTILAVAIKUTUS
        # ========================================================
        #
        # Standardi lapse rate noin 6.5 °C / km.
        #
        # Pidetään tämä erillisenä ja lineaarisena.
        #
        # ========================================================
        korkeus_vaikutus = (
        korkeus / cp.float32(1000.0)
        ) * cp.float32(6.5)

        # ========================================================
        # 67. KORKEUDEN SUOJARAJA
        # ========================================================

        korkeus_vaikutus = cp.clip(
            korkeus_vaikutus,
            cp.float32(
                0.0
            ),
            cp.float32(
                35.0
            )
        )

        # ========================================================
        # 68. PERUSLÄMPÖTILA
        # ========================================================
        #
        # TÄRKEÄ:
        #
        # tmax_planet on Celsius-asteina.
        #
        # Siksi emme lisää siihen 273.15:tä ja jätä sitä
        # skaalaamaan väärässä yksikössä.
        #
        # Planeetan säteilykerroin vaikuttaa maltillisesti
        # lämpötilaan.
        #
        # ========================================================

        # --------------------------------------------------------
        # Säteilyn suhteellinen poikkeama
        # --------------------------------------------------------

        sateily_poikkeama = (
            sateily_kerroin
            -
            cp.float32(
                1.0
            )
        )

        # --------------------------------------------------------
        # Celsius-asteinen säteilykorjaus
        # --------------------------------------------------------
        #
        # tmax_planet toimii planeetan referenssilämpötilana.
        #
        # --------------------------------------------------------

        sateily_lampotila_muutos = (
            cp.float32(
                0.50
            )
            *
            cp.float32(
                tmax_planet
            )
            *
            sateily_poikkeama
        )

        tasapaino_temp = (
            cp.float32(
                tmax_planet
            )
            +
            sateily_lampotila_muutos
        )

        # ========================================================
        # 69. AURINGON KULMAN LÄMPÖVAIKUTUS
        # ========================================================
        #
        # Vanhan version:
        #
        #     K * zenith_distance
        #
        # oli hyvin suuri verrattuna itse lämpötilaskaalaan.
        #
        # Nyt vaikutus on loivempi.
        #
        # ========================================================

        zeniitti_etaisyys = cp.abs(
            lat
            -
            cp.float32(
                deklinaatio
            )
        )

        zeniitti_etaisyys = cp.clip(
            zeniitti_etaisyys,
            cp.float32(
                0.0
            ),
            cp.float32(
                90.0
            )
        )

        # ========================================================
        # 70. LEVEYSASTEEN PERUSVIILENNYS
        # ========================================================

        leveysaste_viilennys = (
            cp.float32(
             ##   0.34*2.5
             0.73
            )
            *
            zeniitti_etaisyys
        )

        # ========================================================
        # 71. TALVEN POLAARINEN LISÄVIILENNYS
        # ========================================================
        #
        # Korkeilla leveysasteilla talvi saa olla kylmä, mutta
        # vaikutus kasvaa asteittain.
        #
        # ========================================================

        talvi_paino = cp.maximum(
            -(
                cp.sin(
                    cp.deg2rad(
                        lat
                    )
                )
                *
                cp.float32(
                    vuodenaika
                )
            ),
            cp.float32(
                0.0
            )
        )

        polaarinen_viilennys = (
            cp.clip(
                cp.abs(
                    lat
                )
                /
                cp.float32(
                    90.0
                ),
                cp.float32(
                    0.0
                ),
                cp.float32(
                    1.0
                )
            )
            *
            talvi_paino
            *
            cp.float32(
                10.0
            )
        )

        # ========================================================
        # 72. MEREN PILVIJÄÄHDYTYS
        # ========================================================
        #
        # VAIN MERELLÄ.
        #
        # Tämä on tärkeä korjaus aikaisempaan tilanteeseen,
        # jossa meri-pilvikerroin saattoi vahingossa saada
        # valtavia arvoja.
        #
        # ========================================================

        meri_pilvi_max_viilennys = cp.float32(
            10.0
        )

        meri_pilvisyys = cp.where(
            meri_bin > cp.float32(
                0.0
            ),
            cp.float32(
                1.0
            ),
            cp.float32(
                0.0
            )
        )

        meri_pilvi_vaikutus = (
            meri_pilvisyys
            *
            meri_pilvi_max_viilennys
        )

        # ========================================================
        # 73. LOPULLINEN LÄMPÖTILA
        # ========================================================

        lampotila = (
            tasapaino_temp
            -
            leveysaste_viilennys
            -
            polaarinen_viilennys
            -
            korkeus_vaikutus
            +
            manner_lampotila_muutos
            -
            meri_pilvi_vaikutus
            -
            pilvisyys_dt
        )

        # ========================================================
        # 74. LÄMPÖTILAN NUMEERINEN TURVARAJA
        # ========================================================
        #
        # Tämä EI ole fysiikan korvaaminen.
        #
        # Se estää ohjelmointivirheen tai epäkelvon syötteen
        # muuttamasta koko karttaa esimerkiksi -23000 °C:een.
        #
        # Jos halutaan aidosti alle -100 °C lämpötiloja,
        # rajaa voidaan myöhemmin muuttaa.
        #
        # ========================================================

        lampotila = cp.nan_to_num(
            lampotila,
            nan=-50.0,
            posinf=100.0,
            neginf=-100.0
        )

        lampotila = cp.clip(
            lampotila,
            cp.float32(
                -100.0
            ),
            cp.float32(
                80.0
            )
        )

        # ========================================================
        # 75. TALLENNA LÄMPÖTILA
        # ========================================================

        kuukausi_lampotilat[
            kk
        ] = lampotila

        # ========================================================
        # 76. DIAGNOSTIIKKA
        # ========================================================

        if debug:

            print(
                f"KK {kk} "
                f"LÄMPÖ min/max:",
                float(
                    cp.min(
                        lampotila
                    )
                ),
                float(
                    cp.max(
                        lampotila
                    )
                )
            )

            print(
                f"KK {kk} "
                f"SADE min/max:",
                float(
                    cp.min(
                        sade
                    )
                ),
                float(
                    cp.max(
                        sade
                    )
                )
            )

            print(
                f"KK {kk} "
                f"MANNER min/max:",
                float(
                    cp.min(
                        manner_lampotila_muutos
                    )
                ),
                float(
                    cp.max(
                        manner_lampotila_muutos
                    )
                )
            )

            print(
                f"KK {kk} "
                f"KORKEUSVAIKUTUS min/max:",
                float(
                    cp.min(
                        korkeus_vaikutus
                    )
                ),
                float(
                    cp.max(
                        korkeus_vaikutus
                    )
                )
            )

            print(
                f"KK {kk} "
                f"PILVI min/max:",
                float(
                    cp.min(
                        pilvisyys_dt
                    )
                ),
                float(
                    cp.max(
                        pilvisyys_dt
                    )
                )
            )

        # ========================================================
        # 77. SEURAAVA KUUKAUSI
        # ========================================================


    # ============================================================
    # 78. GPU-SYNKRONOINTI
    # ============================================================

    cp.cuda.Stream.null.synchronize()

    # ============================================================
    # 79. LOPULLINEN NUMEERINEN TARKISTUS
    # ============================================================

    kuukausi_lampotilat = cp.nan_to_num(
        kuukausi_lampotilat,
        nan=-50.0,
        posinf=80.0,
        neginf=-100.0
    )

    kuukausi_sateet = cp.nan_to_num(
        kuukausi_sateet,
        nan=0.0,
        posinf=5000.0,
        neginf=0.0
    )

    kuukausi_tuuli_suunta = cp.nan_to_num(
        kuukausi_tuuli_suunta,
        nan=0.0,
        posinf=360.0,
        neginf=0.0
    )

    kuukausi_tuuli_voima = cp.nan_to_num(
        kuukausi_tuuli_voima,
        nan=0.3,
        posinf=3.0,
        neginf=0.3
    )

    kuukausi_merivirta = cp.nan_to_num(
        kuukausi_merivirta,
        nan=0.0,
        posinf=100.0,
        neginf=-100.0
    )

    # ============================================================
    # 80. GPU -> CPU
    # ============================================================

    if palauta_cpu:

        return (
            cp.asnumpy(
                kuukausi_lampotilat
            ),

            cp.asnumpy(
                kuukausi_sateet
            ),

            cp.asnumpy(
                kuukausi_tuuli_suunta
            ),

            cp.asnumpy(
                kuukausi_tuuli_voima
            ),

            cp.asnumpy(
                kuukausi_merivirta_x
            ),
            cp.asnumpy(
                kuukausi_merivirta_y
            )
        )

    # ============================================================
    # 81. JÄTETÄÄN GPU-MUISTIIN
    # ============================================================

    return (
        kuukausi_lampotilat,
        kuukausi_sateet,
        kuukausi_tuuli_suunta,
        kuukausi_tuuli_voima,
        kuukausi_merivirta_x,
        kuukausi_merivirta_y
    )








## .... ilmasto
####################






def shift_gpu(arr, shift=(0, 0), cval=0.0):
    """
    2D-arrayn siirto CuPyllä.
    Vastaa tässä käyttötapauksessa scipy.ndimage.shift(..., order=0).
    Solut, jotka siirtyvät reunan yli, täytetään cval-arvolla.
    """
    dy, dx = shift
    dy = int(dy)
    dx = int(dx)

    out = cp.full_like(arr, cval)

    y_src_start = max(0, -dy)
    y_src_end = arr.shape[0] - max(0, dy)

    x_src_start = max(0, -dx)
    x_src_end = arr.shape[1] - max(0, dx)

    y_dst_start = max(0, dy)
    y_dst_end = arr.shape[0] - max(0, -dy)

    x_dst_start = max(0, dx)
    x_dst_end = arr.shape[1] - max(0, -dx)

    if y_src_end > y_src_start and x_src_end > x_src_start:
        out[
            y_dst_start:y_dst_end,
            x_dst_start:x_dst_end
        ] = arr[
            y_src_start:y_src_end,
            x_src_start:x_src_end
        ]

    return out


def mallinna_merijaa(
    korkeuskartta,
    maa_maski,
    kuukausi_lampotilat,
    kuukausi_sateet,
    tuuli_suunta_base,
    meri_anomalia,
    lat_grid_deg
):
    """
    CuPy/GPU-versio merijäämallista.

    Kaikki suuret NumPy-taulukot kannattaa siirtää GPU:lle
    ennen tämän funktion kutsumista.
    """

    # Varmistetaan, että data on GPU:lla
    korkeuskartta = cp.asarray(korkeuskartta)
    maa_maski = cp.asarray(maa_maski)
    kuukausi_lampotilat = cp.asarray(kuukausi_lampotilat)
    kuukausi_sateet = cp.asarray(kuukausi_sateet)
    tuuli_suunta_base = cp.asarray(tuuli_suunta_base)
    meri_anomalia = cp.asarray(meri_anomalia)
    lat_grid_deg = cp.asarray(lat_grid_deg)

    meri_maski = korkeuskartta <= 0

    shape_3d = kuukausi_lampotilat.shape

    jaa_paksuus_vuosi = cp.zeros(
        shape_3d,
        dtype=cp.float32
    )

    jaa_peittavyys_vuosi = cp.zeros(
        shape_3d,
        dtype=cp.float32
    )

    nykyinen_paksuus = cp.zeros_like(
        korkeuskartta,
        dtype=cp.float32
    )

    stefan_vakio = 2.0

    paivat_kk = cp.asarray(
        [31, 28, 31, 30, 31, 30,
         31, 31, 30, 31, 30, 31],
        dtype=cp.float32
    )

    # Tuulen komponentit
    tuuli_u = tuuli_suunta_base * 1.5

    tuuli_v = (
        cp.sin(cp.radians(lat_grid_deg))
        * tuuli_u
        * 0.5
    )

    for kk in range(12):

        T_ilma = kuukausi_lampotilat[kk]
        Sade = kuukausi_sateet[kk]

        T_meri_eff = T_ilma + meri_anomalia

        # --------------------------------------------------
        # 1. TERMODYNAMIIKKA
        # --------------------------------------------------

        pakkasaste_paivat = cp.zeros_like(
            korkeuskartta,
            dtype=cp.float32
        )

        sulamis_paivat = cp.zeros_like(
            korkeuskartta,
            dtype=cp.float32
        )

        T_jaatyminen = -1.8

        jaatymismaski = T_meri_eff < T_jaatyminen

        pakkasaste_paivat = cp.where(
            jaatymismaski,
            (T_jaatyminen - T_meri_eff) * paivat_kk[kk],
            0.0
        )

        sulamis_paivat = cp.where(
            ~jaatymismaski,
            (T_meri_eff - T_jaatyminen) * paivat_kk[kk],
            0.0
        )

        lumi_eriste = 1.0 + Sade * 0.005

        kasvu = (
            cp.sqrt(
                nykyinen_paksuus ** 2
                + (stefan_vakio * pakkasaste_paivat)
                / lumi_eriste
            )
            - nykyinen_paksuus
        )

        sulaminen = sulamis_paivat * 0.015

        nykyinen_paksuus = cp.clip(
            nykyinen_paksuus + kasvu - sulaminen,
            0.0,
            10.0
        )

        # --------------------------------------------------
        # 2. DYNAMIIKKA / AJOJÄÄ
        # --------------------------------------------------

        liike_u = tuuli_u + meri_anomalia * 0.2
        liike_v = tuuli_v

        # Tämä on ainoa kohta, jossa tarvitaan scalar-arvo GPU:lta.
        # Koska shift vaatii kokonaisluvun, siirto on joka tapauksessa
        # diskretoitu pikseleiksi.
        shift_x = int(
            cp.clip(
                cp.mean(liike_u),
                -2,
                2
            ).get()
        )

        shift_y = int(
            cp.clip(
                cp.mean(liike_v),
                -2,
                2
            ).get()
        )

        if shift_x != 0 or shift_y != 0:

            liikutettu_paksuus = shift_gpu(
                nykyinen_paksuus,
                shift=(shift_y, shift_x),
                cval=0.0
            )

            # Törmäys maa-alueeseen
            rannikko_tormays = (
                liikutettu_paksuus * maa_maski
            )

            # Huom:
            # cp.any(...).get() aiheuttaa GPU -> CPU synkronoinnin.
            # Se voidaan myöhemmin optimoida kokonaan pois.
            if bool(cp.any(rannikko_tormays > 0).get()):

                nykyinen_paksuus = (
                    shift_gpu(
                        liikutettu_paksuus * meri_maski,
                        shift=(-shift_y, -shift_x),
                        cval=0.0
                    )
                )

                nykyinen_paksuus += (
                    shift_gpu(
                        rannikko_tormays,
                        shift=(-shift_y, -shift_x),
                        cval=0.0
                    )
                    * 1.5
                )

            else:

                nykyinen_paksuus = (
                    liikutettu_paksuus * meri_maski
                )

        # --------------------------------------------------
        # 3. PEITTÄVYYS
        # --------------------------------------------------

        peittavyys = cp.clip(
            nykyinen_paksuus / 0.4,
            0.0,
            1.0
        )

        tuuli_rasitus = (
            cp.abs(tuuli_suunta_base) * 0.15
        )

        peittavyys = cp.clip(
            peittavyys - tuuli_rasitus,
            0.0,
            1.0
        )

        # Maa-alueet nollaksi
        nykyinen_paksuus = cp.where(
            meri_maski,
            nykyinen_paksuus,
            0.0
        )

        peittavyys = cp.where(
            meri_maski,
            peittavyys,
            0.0
        )

        jaa_paksuus_vuosi[kk] = nykyinen_paksuus
        jaa_peittavyys_vuosi[kk] = peittavyys

    return jaa_paksuus_vuosi, jaa_peittavyys_vuosi


def aja_jaatikko_malli(
    korkeus_metreina,
    kuukausi_lampotilat,
    kuukausi_sateet,
    vuodet=1
):
    """
    CuPy/GPU-versio mannerjäätikkömallista.
    """

    # GPU:lle
    korkeus_metreina = cp.asarray(korkeus_metreina)
    kuukausi_lampotilat = cp.asarray(kuukausi_lampotilat)
    kuukausi_sateet = cp.asarray(kuukausi_sateet)

    Y, X = korkeus_metreina.shape

    # --------------------------------------------------
    # Alustus
    # --------------------------------------------------

    lumi_paksuus = cp.zeros(
        (Y, X),
        dtype=cp.float32
    )

    jaa_paksuus = cp.zeros(
        (Y, X),
        dtype=cp.float32
    )

    # --------------------------------------------------
    # Parametrit
    # --------------------------------------------------

    LUMEN_TIHEYS = 300.0
    JAAN_TIHEYS = 917.0

    LUMI_KYNNYS = 2.0
    SULAMIS_KERROIN = 0.005

    KRIITTINEN_PAKSUUS = 40.0
    VIRTAUS_NOPEUS = 0.02

    # --------------------------------------------------
    # Simulointi
    # --------------------------------------------------

    for vuosi in range(vuodet):

        for kk in range(12):

            T = kuukausi_lampotilat[kk]
            Sade = kuukausi_sateet[kk]

            # --------------------------------------------------
            # 1. KERTYMÄ
            # --------------------------------------------------

            sade_lumena_m = cp.where(
                T < 0,
                Sade / LUMEN_TIHEYS,
                0.0
            )

            lumi_paksuus += sade_lumena_m

            # --------------------------------------------------
            # 2. SULAMINEN
            # --------------------------------------------------

            potentiaalinen_sulaminen = cp.where(
                T > 0,
                T * SULAMIS_KERROIN * 30,
                0.0
            )

            sulava_lumi = cp.minimum(
                lumi_paksuus,
                potentiaalinen_sulaminen
            )

            lumi_paksuus -= sulava_lumi

            jaljella_sulamista = (
                potentiaalinen_sulaminen
                - sulava_lumi
            )

            sulava_jaa = cp.minimum(
                jaa_paksuus,
                jaljella_sulamista
            )

            jaa_paksuus -= sulava_jaa

            # --------------------------------------------------
            # 3. FIRNIFIKAATIO
            # --------------------------------------------------

            ylimaara_lumi = cp.maximum(
                0.0,
                lumi_paksuus - LUMI_KYNNYS
            )

            lumi_paksuus = cp.minimum(
                lumi_paksuus,
                LUMI_KYNNYS
            )

            uusi_jaa = (
                ylimaara_lumi
                * LUMEN_TIHEYS
                / JAAN_TIHEYS
            )

            jaa_paksuus += uusi_jaa

            # --------------------------------------------------
            # 4. JÄÄN VIRTAUS
            # --------------------------------------------------

            pinnan_korkeus = (
                korkeus_metreina
                + jaa_paksuus
                + lumi_paksuus
            )

            uusi_jaa_paksuus = jaa_paksuus.copy()

            virtaava_jaa_maski = (
                jaa_paksuus > KRIITTINEN_PAKSUUS
            )

            # Ei tarvitse siirtyä CPU:lle:
            # cp.any palauttaa GPU-skalaarin, joka voidaan
            # tarvittaessa muuttaa booliksi.
            if bool(cp.any(virtaava_jaa_maski).get()):

                maksimi_valuma = (
                    jaa_paksuus
                    - KRIITTINEN_PAKSUUS
                ) / 4.0

                # 4 naapuria
                for dy, dx in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1)
                ]:

                    naapurin_korkeus = cp.roll(
                        pinnan_korkeus,
                        shift=(-dy, -dx),
                        axis=(0, 1)
                    )

                    korkeusero = (
                        pinnan_korkeus
                        - naapurin_korkeus
                    )

                    valuva_massa = cp.where(
                        (
                            (korkeusero > 0)
                            & virtaava_jaa_maski
                        ),
                        korkeusero * VIRTAUS_NOPEUS,
                        0.0
                    )

                    valuva_massa = cp.minimum(
                        valuva_massa,
                        maksimi_valuma
                    )

                    uusi_jaa_paksuus -= (
                        valuva_massa
                    )

                    uusi_jaa_paksuus += cp.roll(
                        valuva_massa,
                        shift=(dy, dx),
                        axis=(0, 1)
                    )

                jaa_paksuus = cp.maximum(
                    0.0,
                    uusi_jaa_paksuus
                )

    # --------------------------------------------------
    # 5. LOPPUTULOS
    # --------------------------------------------------

    jaan_huippu_metreina = (
        korkeus_metreina
        + jaa_paksuus
        + lumi_paksuus
    )

    return (
        lumi_paksuus,
        jaa_paksuus,
        jaan_huippu_metreina
    )












def f_precip_relto_earthlike(oceanfrac):
    """
    Laskee planeetan kokonaissadannan suhteessa maapalloon (Earth = 1.0)
    perustuen ExoPlaSim-simulaatiodataan.
    """
    x = oceanfrac
    # 2. asteen polynomikaavan kertoimet
    return -0.5794 * x**2 + 1.8299 * x - 0.0151



def laske_hillshade(korkeuskartta, azimuth=315, altitude=45):
    """Laskee varjostuskartan valon suunnan (azimuth) ja korkeuden (altitude) mukaan."""
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Lasketaan gradientit (pinnan suunnat)
    x, y = np.gradient(korkeuskartta)
    
    # Rinteen jyrkkyys ja suunta
    slope = np.pi/2. - np.arctan(np.sqrt(x**2 + y**2))
    aspect = np.arctan2(-x, y)
    
    # Hillshade-yhtälö (palauttaa arvon -1 ja 1 väliltä, siirretään välille 0-1)
    shaded = np.sin(altitude_rad) * np.sin(slope) + \
             np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
             
    # Normalisoidaan välille 0.0 - 1.0 (0.5 on tasainen maa)
    return (shaded + 1.0) / 2.0

def sekoita_soft_light(kuva, varjo):
    """
    Sekoittaa varjon kuvan päälle Pegmeä valo (Soft Light) -menetelmällä.
    Kuva ja varjo molemmat float-muodossa (0.0 - 1.0).
    """
    # Varmistetaan, että varjokartta laajenee kuvan värikanaviin jos kuva on RGB
    #if len(kuva.shape) == 3 and len(varjo.shape) == 2:
    #    varjo = np.expand_dims(varjo, axis=2)
        
    # Standardi Soft Light -kaava (Pegasi-tyyppi)
    # Pitää keskialueet ennallaan, kirkastaa valoisia ja tummentaa varjoja
    #tulos = (1 - 2 * varjo) * (kuva ** 2) + 2 * varjo * kuva
    tulos=kuva
    return np.clip(tulos, 0.0, 1.0)





########################################################################
## main program #####################
###########################################



#kartantekija = KartanTekijaCuPy(
#    leveys=leveys,
#    korkeus=korkeus,
#    syvin_kohta=syvin_kohta,
#    korkein_kohta=korkein_kohta,
#    manner_osuus=manner_osuus,
#    seed=seed1,
#)

kartantekija = KartanTekijaCuPy(
    leveys=leveys,
    korkeus=korkeus,
    syvin_kohta=syvin_kohta,
    korkein_kohta=korkein_kohta,
    manner_osuus=manner_osuus,
    seed=seed1,
    octaves=24,
)

tulokset = kartantekija.tulokset(
    cpu=True
)

print("globaali vedenpinta:",
      tulokset["vedenpinta"])

zoomaaja = KartanTekijaCuPy(
    leveys=1000,
    korkeus=1000,
    syvin_kohta=syvin_kohta,
    korkein_kohta=korkein_kohta,
    manner_osuus=manner_osuus,
    seed=seed1,
    octaves=24,
)

zoomattu = zoomaaja.kartta(
    kartan_alue=(0, 90, 0, 90),
    leveys=400,
    korkeus=400,
    sealevel=tulokset["vedenpinta"],
).tulokset(
    cpu=True
)




tulokset = kartantekija.tulokset(cpu=True)


korkeus_syvyyskartta = kartantekija.korkeus_syvyyskartta.get()
korkeuskartta = kartantekija.korkeuskartta.get()
maa_maski = kartantekija.maa_maski.get()
meri_maski = kartantekija.meri_maski.get()
zoomattu_nelio=zoomattu["kartta_nelio"]
zoomattu_korkeus_syvyyskartta =  zoomattu["korkeus_syvyyskartta"]
zoomattu_korkeuskartta = zoomattu["korkeuskartta"]
zoomattu_maa_maski = zoomattu["maa_maski"]
zoomattu_meri_maski = zoomattu["meri_maski"]

print(korkeus_syvyyskartta.shape)
print(korkeuskartta.shape)
print(maa_maski.shape)
print(meri_maski.shape)
print("zoomattu")
print(zoomattu["kartta_nelio"])

#plt.imshow(korkeuskartta )
#plt.show()
#plt.imshow(zoomattu_korkeuskartta )
#plt.show()
#quit(-1)


# KÄYTTÖESIMERKKI:
# oletetaan että 'korkeus' on 2D-numpy-taulukko ja 'vari_kuva' on RGB-kuva (0-1 float)

# valmis_kuva = sekoita_soft_light(vari_kuva, varjo_maski)

# =====================================================================
# 3. MERIETÄISYYSRASATERIN LASKENTA
# =====================================================================


leveysasteet = np.linspace(90, -90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)
lon_grid_deg, lat_grid_deg = np.meshgrid(pituusasteet, leveysasteet)

korkeus, leveys = maa_maski.shape
lats = np.linspace(90, -90, korkeus)   # Y-akseli (pohjoisesta etelään)
lons = np.linspace(-180, 180, leveys) # X-akseli (lännestä itään)
lon_grid, lat_grid = np.meshgrid(lons, lats)





korkeus, leveys = maa_maski.shape

lats = np.linspace(
    90.0,
    -90.0,
    korkeus,
)

lons = np.linspace(
    -180.0,
    180.0,
    leveys,
    endpoint=False,
)

# ------------------------------------------------------------
# Meripikselit
# ------------------------------------------------------------

# ------------------------------------------------------------
# Vain maapikselit
# ------------------------------------------------------------


maa_maski_gpu = kartantekija.maa_maski

# GPU-laskenta
rannikko_etaisyys_gpu = (
    laske_etaisyys_rannikosta_gpu(
        maa_maski_gpu
    )
)

# GPU -> CPU / NumPy
etaisyys_meresta_km = (
    rannikko_etaisyys_gpu.get()
)



meri_y, meri_x = np.where(
    ~maa_maski
)

meri_coords = np.radians(
    np.column_stack(
        (
            lats[meri_y],
            lons[meri_x],
        )
    )
)


maa_y, maa_x = np.where(
    maa_maski
)

maa_coords = np.radians(
    np.column_stack(
        (
            lats[maa_y],
            lons[maa_x],
        )
    )
)


varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)
korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)
korkeus_gradientti_y = np.gradient(korkeuskartta, axis=0)

n_lats, n_lons = korkeuskartta.shape

lats = np.linspace(-90, 90, n_lats)

cos_weights = np.cos(np.radians(lats))

weight_grid = cos_weights[:, np.newaxis] * np.ones(n_lons)

# 6. Lasketaan painotetut pinta-alat (summaten kosinipainot maskatuilta alueilta)
maa_painotettu = np.sum(weight_grid[maa_maski])
meri_painotettu = np.sum(weight_grid[meri_maski])
kokonais_paino = np.sum(weight_grid)

# 7. Lasketaan prosenttiosuudet
maa_prosentti = (maa_painotettu / kokonais_paino) * 100
meri_prosentti = (meri_painotettu / kokonais_paino) * 100

maan_keskikorkeus = np.sum(korkeuskartta[maa_maski] * weight_grid[maa_maski]) /maa_painotettu

print(f"Maan osuus: {maa_prosentti:.2f} %")
print(f"Meren osuus: {meri_prosentti:.2f} %")
print(f"Maan kosinipainotettu keskikorkeus: {maan_keskikorkeus:.1f} metriä")


# =====================================================================
# 4. KUUKAUSITTAINEN LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS (12 KUUKAUTTA)
# =====================================================================
kuukausi_lämpötilat = np.zeros((12, korkeus, leveys))
kuukausi_sateet = np.zeros((12, korkeus, leveys))

global_moisture_coeff=global_moisture_coeff*f_precip_relto_earthlike(meri_prosentti/71.0)
 
    


import time

print ("GPU warming up ...")

# Ensimmäinen ajo = warm-up
laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    korkeus_gradientti_y,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet, tmax_planet,
    global_moisture_coeff,
)


print ("GPU calculation ...")
# Varsinainen mittaus
cp.cuda.Stream.null.synchronize()

t0 = time.perf_counter()


kuukausi_lämpötilat_gpu, kuukausi_sateet_gpu, \
kuukausi_tuuli_suunta_gpu, kuukausi_tuuli_voima_gpu, \
kuukausi_merivirta_x_gpu, kuukausi_merivirta_y_gpu = laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    korkeus_gradientti_y,
    maa_maski=maa_maski,
    meri_maski=meri_maski,
    etaisyys_meresta_km=etaisyys_meresta_km,
    tilt_planet_axis=tilt_planet_axis,
    ecc_planet=ecc_planet,
    mvelp_planet=mvelp_planet,
    global_moisture_coeff=global_moisture_coeff,
    tmax_planet_param=tmax_planet
)

cp.cuda.Stream.null.synchronize()

dt = time.perf_counter() - t0

print(f"GPU-laskenta: {dt:.4f} s")

kuukausi_lämpötilat=kuukausi_lämpötilat_gpu
kuukausi_sateet=kuukausi_sateet_gpu

kuukausi_tuuli_suunta=kuukausi_tuuli_suunta_gpu
kuukausi_tuuli_voima=kuukausi_tuuli_voima_gpu
kuukausi_merivirta_x=kuukausi_merivirta_x_gpu
kuukausi_merivirta_y=kuukausi_merivirta_y_gpu
kuukausi_merivirta=np.sqrt(kuukausi_merivirta_x*kuukausi_merivirta_x+kuukausi_merivirta_y*kuukausi_merivirta_y)

##Y, X = np.mgrid[-90:90:complex(0, korkeus), -180:180:complex(0, leveys)]
    

#piirra_hila_streamplot(u_esimerkki, v_esimerkki, alue=[-180, 180, -90, 90])
plot_grid_streamplot(kuukausi_merivirta_x[0], kuukausi_merivirta_y[0], alue=[-180, 180, -90, 90])

#meri_anomalia=np.mean(kk_meri_anomalia, axis=0)
meri_anomalia=np.mean(kuukausi_sateet, axis=0)/1000
tuuli_suunta_base=np.mean(kuukausi_tuuli_suunta, axis=0)/1000


korkeuskartta_gpu = cp.asarray(korkeuskartta)
maa_maski_gpu = cp.asarray(maa_maski)

lampotilat_gpu = cp.asarray(kuukausi_lämpötilat)
sateet_gpu = cp.asarray(kuukausi_sateet)

tuuli_gpu = cp.asarray(tuuli_suunta_base)
anomalia_gpu = cp.asarray(meri_anomalia)
lat_gpu = cp.asarray(lat_grid_deg)








#lumi, jaa, kokonais_korkeus = aja_jaatikko_malli(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuodet=200)

#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia*0+1,lat_grid_deg)

#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, maa_maski, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg)

#merijaa_paksuus_gpu, merijaa_peittavyys_gpu = mallinna_merijaa(
#    korkeuskartta_gpu,
#    maa_maski_gpu,
#    lampotilat_gpu,
#    sateet_gpu,
#    tuuli_gpu,
#    anomalia_gpu,
#    lat_gpu
#)



#lumi, jaa, kokonais_korkeus = aja_jaatikko_malli(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuodet=200)
gpu_lumi, gpu_jaa, gpu_kokonais_korkeus = aja_jaatikko_malli(korkeuskartta_gpu, lampotilat_gpu, sateet_gpu, vuodet=200)

lumi = cp.asnumpy(gpu_lumi)
jaa = cp.asnumpy(gpu_jaa)
kokonais_korkeus = cp.asnumpy(gpu_kokonais_korkeus)
#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia*0+1,lat_grid_deg)

#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, maa_maski, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg)

merijaa_paksuus_gpu, merijaa_peittavyys_gpu = mallinna_merijaa(
    korkeuskartta_gpu,
    maa_maski_gpu,
    lampotilat_gpu,
    sateet_gpu,
    tuuli_gpu,
    anomalia_gpu,
    lat_gpu
)

merijaa_peittavyys_vuosi = cp.asnumpy(merijaa_peittavyys_gpu)
merijaa_paksuus = cp.asnumpy(merijaa_paksuus_gpu)


jaa_peittavyys = cp.asnumpy(merijaa_peittavyys_gpu)
merijaa_paksuus = cp.asnumpy(merijaa_paksuus_gpu)
#merijaa_laajin=np.sum(merijaa_peittavyys_vuosi, axis=1)
merijaa_aina = np.all(merijaa_peittavyys_vuosi > 0, axis=0)
merijaa_joskus_tai_aina = np.any(merijaa_peittavyys_vuosi > 0, axis=0)

# Vähennetään tästä ne alueet, joissa jäätä on aina
merijaa_vain_joskus = merijaa_joskus_tai_aina & ~merijaa_aina

(
    koppen,
    tmean_annual,
    tmin_annual,
    tmax_annual,
    precip_annual,
    precip_dry,
    precip_wet,
    precip_cold,
    precip_warm,
    temp_dry,
    temp_wet
) = laske_koppen(
    kuukausi_lämpötilat,
    kuukausi_sateet,
    maa_maski,
    etaisyys_meresta_km
)


plot_tmean_raster(korkeuskartta, tmean_annual, title="Annual mean temperature degC")

plot_precip_raster(korkeuskartta, precip_annual, title="Annual precipitation mm")


#plt.imshow(merijaa_peittavyys_vuosi[6])
#plt.imshow(merijaa_paksuus_vuosi[1])

#plt.contour(maa_maski, levels=[0,1,2], colors=["red"], lw=2)
#plt.show()

#quit(-1)


# =====================================================================
# 8. VISUALISOINTI (UUSI ERITELTY IKKUNA JA VALKOINEN NAPAVYÖHYKE)
# =====================================================================
# Päivitetty väri: Viimeinen indeksi (13: Napavyöhyke) muutettu harmaasta valkoiseksi
koppen_tarkat_varit = [
    "#1a233a",  # 0: Meri
    "#0000fe",  # 1: Af (Sademetsä - tummansininen)
    "#0077fe",  # 2: Am (Monsuuni - keskisininen)
    "#41c2ff",  # 3: Aw (Savanni - vaaleansininen)
    "#fe0000",  # 4: BWh (Kuuma aavikko - punainen)
    "#fe9999",  # 5: BWk (Kylmä aavikko - vaaleanpunainen)
    "#fec200",  # 6: BSh (Kuuma aro - oranssinkeltainen)
    "#fffe00",  # 7: BSk (Kylmä aro - keltainen)
    "#c6ff4e",  # 8: Csa/Csb (Välimerenilmasto - oliivinvihreä)
    "#007e00",  # 9: Cfb (Meri-ilmasto - tummanvihreä)
    "#96ff96",  # 10: Cfa/Cwa (Kostea subtrooppinen - vaaleanvihreä)
    "#327e65",  # 11: Dfa/Dfb (Lämmin mannerilmasto - sinivihreä)
    "#00465f",  # 12: Dfc/Dfd (Taiga / Havumetsä - tumma syaaninsininen)
    "#7f7f7f" , # 13: ET (Tundra harmaa)
    "#ffffff"   # 14: EF (Napavyöhyke - PUHDAS VALKOINEN)
]

koppen_luonnolliset_varit_000 = [
    "#1a233a",  # 0: Meri (Syvä laivastonsininen)
    "#0b5345",  # 1: Af (Sademetsä - tumma, tiheä vihreä)
    "#117a65",  # 2: Am (Monsuuni - viidakkohkon vihreä)
    "#7dcea0",  # 3: Aw (Savanni - vaaleanvihreä / kuivahtava ruoho)
    "#edbb99",  # 4: BWh (Kuuma aavikko - vaalea hiekanruskea)
    "#e59866",  # 5: BWk (Kylmä aavikko - kuiva savimaa)
    "#f8c471",  # 6: BSh (Kuuma aro - kellertävä kuiva ruoho)
    "#f7dc6f",  # 7: BSk (Kylmä aro - oljenkeltainen)
    "#a2d9ce",  # 8: Csa/Csb (Välimerenilmasto - oliivinvihreä/haalea välimeren kasvillisuus)
    "#1e8449",  # 9: Cfb (Meri-ilmasto - raikas lehtimetsän vihreä)
    "#2ecc71",  # 10: Cfa/Cwa (Kostea subtrooppinen - kirkas vihreä)
    "#273746",  # 11: Dfa/Dfb (Lämmin mannerilmasto - tumma havupuun vihreä)
    "#1c2833",  # 12: Dfc/Dfd (Taiga / Havumetsä - erittäin tumma vihreä/harmahtava taiga)
    "#a6acaf",  # 13: ET (Tundra - kelonharmaa / sammalenharmaa)
    "#ffffff"   # 14: EF (Napavyöhyke - PUHDAS VALKOINEN jää ja lumi)
]



koppen_luonnolliset_varit = [
    "#0f1e38",  # 0: Meri (Koppenpastan syvä valtameri)
    "#2a410f",  # 1: Af (Sademetsä - Koppenpasta true oliivinvihreä)
    "#385714",  # 2: Am (Monsuuni - Koppenpasta true vihreä)
    "#98a94f",  # 3: Aw (Savanni - Koppenpasta kuivahtava ruohovihreä)
    "#d5b785",  # 4: BWh (Kuuma aavikko - Koppenpasta vaalea hiekka)
    "#b89c6c",  # 5: BWk (Kylmä aavikko - Koppenpasta haalea savi)
    "#cbb27a",  # 6: BSh (Kuuma aro - Koppenpasta arovihreä/keltainen)
    "#ddcb94",  # 7: BSk (Kylmä aro - Koppenpasta oljenkeltainen)
    "#717b43",  # 8: Csa/Csb (Välimerenilmasto - Koppenpasta kuiva välimerellinen vihreä)
    "#4c6a21",  # 9: Cfb (Meri-ilmasto - Koppenpasta lauhkea metsä)
    "#557525",  # 10: Cfa/Cwa (Kostea subtrooppinen - Koppenpasta rehevä vihreä)
    "#2f421f",  # 11: Dfa/Dfb (Lämmin mannerilmasto - Koppenpasta tummempi havumetsä)
    "#1d2a13",  # 12: Dfc/Dfd (Taiga / Havumetsä - Koppenpasta erittäin tumma pohjoinen metsä)
    "#828675",  # 13: ET (Tundra - Koppenpasta sammalen- ja kivenharmaa)
    "#cfcfcf"   # 14: EF (Napavyöhyke jäätikkö tai polaariaavikko)
]

tarkat_nimet = [
    "Sea", "Rain forest (Af)", "Monsoon (Am)", "Savanna (Aw)", 
    "Hot desert (BWh)", "Cold desert (BWk)", "Hot steppe (BSh)", "cold steppe (BSk)",
    "Meciterranean climate (Csa/Csb)", "Oceanic climate (Cfb)", "Subtropical moist (Cfa)",
    "Continental (Dfa/Dfb)", "Taiga/Boreal (Dfc/Dfd)", "Tundra (ET)", "Polar desert / ice (EF)"
]

#cmap_tarkka = ListedColormap(koppen_tarkat_varit)
cmap_tarkka = ListedColormap(koppen_luonnolliset_varit)

rajat_tarkka = np.arange(-0.5, 15.5, 1.0)
norm_tarkka = BoundaryNorm(rajat_tarkka, cmap_tarkka.N)

# PAKOTETAAN MATPLOTLIB AVAAMAAN UUSI IKKUNA (Estää koodisolun sisäisen lukituksen)
fig = plt.figure("Köppen-ilmastokartta", figsize=(14, 7), dpi=100)

im = plt.imshow(koppen, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])

variables = {
    "keski_lampotila": tmean_annual,
    "keski_sademaara": precip_annual,
    "maaston_korkeus_m": korkeuskartta,
    "lampotila_min": tmin_annual,
    "lampotila_max": tmax_annual,
    "kuivimman_kuukauden_sade": precip_dry,
    "sateisimman_kuukauden_sade": precip_wet,
}
variable_names=[
        "keski_lampotila",
        "keski_sademaara",
        "maaston_korkeus_m",
        "lampotila_min",
        "lampotila_max",        
        "kuivimman_kuukauden_sade",
        "sateisimman_kuukauden_sade",
    ]
    
## et_im ok gam svm_im_fast histrgb_im
new_rgb = et_im(
    im,variables, variable_names,
)



im2 = plt.imshow(new_rgb, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])

#im2 = plt.imshow(rgb, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])
varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)

#valmis_kuva = sekoita_soft_light(im, varjo_maski)

plt.imshow(np.exp(varjo_maski), cmap="gray", alpha=0.2,extent=[-180, 180, -90, 90])

#plt.contour(merijaa_aina, levels=[0.25,1],extent=[-180, 180, 90, -90], lw=1, alpha=0.7, colors=["lightblue"])
plt.contourf(merijaa_aina, levels=[0.25,1],extent=[-180, 180, 90, -90], lw=1, alpha=0.7, colors=["lightblue"])

#plt.contour(merijaa_vain_joskus, levels=[0.5,1],extent=[-180, 180, 90, -90], lw=1,linestyle=":", colors=["blue"])

#plt.contour(jaa, levels=[20,50,100,200],extent=[-180, 180, 90, -90], lw=2, colors=["white"])
plt.contourf(jaa, levels=[20,50,100,200],extent=[-180, 180, 90, -90], lw=2, colors=["white"])


cbar = plt.colorbar(im, ticks=list(range(15)), orientation='horizontal', pad=0.15, shrink=0.9)
cbar.ax.set_xticklabels(tarkat_nimet, rotation=25, ha='right')
cbar.ax.tick_params(labelsize=8)

plt.title("Planet map", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Lon")
plt.ylabel("Lat")
plt.grid(color='black', linestyle='--', alpha=0.1) # Vaihdettu valkoinen ruudukko mustaksi, jotta se näkyy navoilla
plt.tight_layout()

# Näytetään ikkuna
plt.show()

quit(-1)

#plt.imshow(lumi)
#plt.imshow(jaa)

#plt.imshow(huippu)
  
    
#plt.imshow(korkeuskartta)
# ==========================================
# RASTERIKUVAN PIIRTÄMINEN (MATPLOTLIB)
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Vasen kuva: Kallioperän muoto (korkeuskartta)
im1 = ax[0].imshow(korkeuskartta, cmap='terrain', origin='lower')
ax[0].set_title('Kallioperän korkeus (m)')
fig.colorbar(im1, ax=ax[0], label='Metriä merenpinnasta')

# Oikea kuva: Jään paksuus rasterina
# Käytetään 'Blues'-värikarttaa, jotta jää erottuu selkeästi
im2 = ax[1].imshow(jaa, cmap='Blues', origin='lower')
ax[1].set_title('Jään paksuus n vuoden jälkeen (m)')
fig.colorbar(im2, ax=ax[1], label='Jään paksuus metreinä')

plt.tight_layout()
plt.show()
