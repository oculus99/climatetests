
#############################
#
## simple planet climate 
#
## 20.09.2026 0000.00021.01
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


# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus = 180*12
leveys = 360*12

seed1=9 ##ok
seed1 = 15

#maapallon_sade_km = 6371.0
#syvin_kohta=-11000
#korkein_kohta=8848
#manner_osuus=0.30


maapallon_sade_km = 6371.0
syvin_kohta=-4000
korkein_kohta=4000
manner_osuus=0.30

leveysasteet = np.linspace(90, -90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)


#tilt_planet_axis=23.44
#ecc_planet=0.01
#mvelp_planet=90
# Parametrit (esimerkkinä Maan arvot)
tilt_planet_axis = 23.44*1
ecc_planet = 0.013*1
mvelp_planet = 102.0  # Perihelin pituus asteina (Maa saavuttaa perihelin tammikuun alussa)
tmax_planet=47
global_moisture_coeff=1.0


###########################################
#######################################
### kartta


import cupy as cp


class KartanTekijaCuPy:
    """
    GPU-versio pallomaisen maailman proseduraalisesta korkeuskartan
    generaattorista.

    Vaatii esimerkiksi:
        pip install cupy-cuda12x

    Käytä CUDA-versiosi mukaista CuPy-pakettia.
    """

    def __init__(
        self,
        leveys=360,
        korkeus=180,

        syvin_kohta=-3000,
        korkein_kohta=1000,
        manner_osuus=0.30,

        suurten_mantereiden_osuus=0.99,
        mantereen_koko=0.55,
        mantereen_muoto=0.5,
        saaren_koko=2.5,
        saaren_maara=1.0,

        tasangon_koko=0.65,
        tasangon_voima=0.65,

        ylangon_koko=0.90,
        ylangon_voima=0.20,

        vuoriston_koko=0.55,
        vuoriston_voima=0.35,

        pieni_maasto_voima=0.05,

        merenpohjan_koko=0.8,
        merenpohjan_voima=0.25,

        rannikon_loivuus=0.20,

        seed=42,
        debug=False,
    ):

        self.leveys = int(leveys)
        self.korkeus = int(korkeus)

        self.syvin_kohta = float(syvin_kohta)
        self.korkein_kohta = float(korkein_kohta)

        self.manner_osuus = float(manner_osuus)

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
        self.debug = bool(debug)

        self.korkeus_syvyyskartta = None
        self.korkeuskartta = None
        self.raakakorkeuskartta = None

        self.maa_maski = None
        self.meri_maski = None
        self.manner_maski = None
        self.saari_maski = None

        self.tasanko_maski = None
        self.ylanko_maski = None
        self.vuoristo_maski = None

        self.vedenpinta = 0.0

        self.debug_data = {}

    # ==========================================================
    # Pallokoordinaatit
    # ==========================================================

    def _pallokoordinaatit(self):

        lat = cp.radians(
            cp.linspace(
                -90.0,
                90.0,
                self.korkeus,
                dtype=cp.float32,
            )
        )

        lon = cp.radians(
            cp.linspace(
                -180.0,
                180.0,
                self.leveys,
                endpoint=False,
                dtype=cp.float32,
            )
        )

        lon_grid, lat_grid = cp.meshgrid(
            lon,
            lat,
        )

        cos_lat = cp.cos(lat_grid)

        x = cos_lat * cp.cos(lon_grid)
        y = cos_lat * cp.sin(lon_grid)
        z = cp.sin(lat_grid)

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

        px = x * cp.float32(grid_size)
        py = y * cp.float32(grid_size)
        pz = z * cp.float32(grid_size)

        x0 = cp.floor(px).astype(cp.int32)
        y0 = cp.floor(py).astype(cp.int32)
        z0 = cp.floor(pz).astype(cp.int32)

        fx = px - x0
        fy = py - y0
        fz = pz - z0

        # Smoothstep
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        fz = fz * fz * (3.0 - 2.0 * fz)

        def random_value(ix, iy, iz):

            h = (
                ix.astype(cp.int64) * 374761393
                + iy.astype(cp.int64) * 668265263
                + iz.astype(cp.int64) * 2147483647
                + cp.int64(seed) * 1274126177
            )

            h ^= h >> 13
            h *= 1274126177
            h ^= h >> 16

            h &= cp.int64(0xffffffff)

            # Muutetaan [0, 2^32-1] -> [-1, 1]
            return (
                h.astype(cp.float32)
                / cp.float32(2147483647.5)
                - cp.float32(1.0)
            )

        c000 = random_value(x0,     y0,     z0)
        c100 = random_value(x0 + 1, y0,     z0)
        c010 = random_value(x0,     y0 + 1, z0)
        c110 = random_value(x0 + 1, y0 + 1, z0)

        c001 = random_value(x0,     y0,     z0 + 1)
        c101 = random_value(x0 + 1, y0,     z0 + 1)
        c011 = random_value(x0,     y0 + 1, z0 + 1)
        c111 = random_value(x0 + 1, y0 + 1, z0 + 1)

        nx00 = c000 * (1.0 - fx) + c100 * fx
        nx10 = c010 * (1.0 - fx) + c110 * fx

        nx01 = c001 * (1.0 - fx) + c101 * fx
        nx11 = c011 * (1.0 - fx) + c111 * fx

        nxy0 = nx00 * (1.0 - fy) + nx10 * fy
        nxy1 = nx01 * (1.0 - fy) + nx11 * fy

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
        octaves=16,
        base_grid=4,
    ):

        value = cp.zeros_like(
            x,
            dtype=cp.float32,
        )

        amplitude = cp.float32(1.0)
        amplitude_sum = cp.float32(0.0)
        frequency = cp.float32(1.0)

        for octave in range(octaves):

            value += (
                amplitude
                * self._value_noise_3d(
                    x * frequency,
                    y * frequency,
                    z * frequency,
                    seed=seed + octave * 101,
                    grid_size=base_grid,
                )
            )

            amplitude_sum += amplitude

            frequency *= cp.float32(2.0)
            amplitude *= cp.float32(0.5)

        return value / amplitude_sum

    # ==========================================================
    # Ridged noise
    # ==========================================================

    def _ridged_noise(
        self,
        x,
        y,
        z,
        seed,
        octaves=15,
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

        ridge = 1.0 - cp.abs(noise)

        return ridge * ridge

    # ==========================================================
    # Normalisointi 0...1
    # ==========================================================

    def _normalisoi_01(self, value):

        value = value.astype(
            cp.float32,
            copy=False,
        )

        minimum = cp.min(value)
        maximum = cp.max(value)

        return cp.where(
            maximum > minimum,
            (value - minimum)
            / (maximum - minimum),
            cp.zeros_like(value),
        )

    # ==========================================================
    # Smoothstep
    # ==========================================================

    def _smoothstep(self, value):

        value = cp.clip(
            value,
            0.0,
            1.0,
        )

        return value * value * (
            3.0 - 2.0 * value
        )

    # ==========================================================
    # Suuret manteret
    # ==========================================================

    def _generoi_mantereet(self, x, y, z):

        frequency = (
            1.35
            * self.mantereen_koko
        )

        large = self._fbm(
            x * frequency,
            y * frequency,
            z * frequency,
            seed=self.seed + 100,
            octaves=16,
            base_grid=2,
        )

        large = self._normalisoi_01(
            large
        )

        if self.mantereen_muoto > 0.0:

            detail = self._fbm(
                x * frequency * 2.0,
                y * frequency * 2.0,
                z * frequency * 2.0,
                seed=self.seed + 200,
                octaves=3,
                base_grid=3,
            )

            detail = self._normalisoi_01(
                detail
            )

            large = (
                (1.0 - self.mantereen_muoto)
                * large
                +
                self.mantereen_muoto
                * (
                    0.75 * large
                    + 0.25 * detail
                )
            )

        return large

    # ==========================================================
    # Saarikenttä
    # ==========================================================

    def _generoi_saaret(self, x, y, z):

        frequency = self.saaren_koko

        islands = self._fbm(
            x * frequency,
            y * frequency,
            z * frequency,
            seed=self.seed + 300,
            octaves=15,
            base_grid=3,
        )

        return self._normalisoi_01(
            islands
        )

    # ==========================================================
    # Manner + saaret
    # ==========================================================

    def _generoi_maakentta(self, x, y, z):

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

        # Python-float -> GPU float32.
        # Ei tarvita cp.clip()ia tässä.
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

        # ------------------------------------------------------
        # Mantereiden pinta-ala kvantiililla
        # ------------------------------------------------------

        manner_threshold = cp.quantile(
            large,
            1.0 - tavoite_manner_osuus,
        )

        manner_maski = (
            large >= manner_threshold
        )

        # ------------------------------------------------------
        # Saarten määrä
        # ------------------------------------------------------

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
            - cp.float32(0.20) * maara,
            cp.float32(0.30),
            cp.float32(0.85),
        )

        saari_maski = (
            saari_arvot >= saari_threshold
        )

        # Saaret eivät saa mennä mantereiden päälle.
        saari_maski &= ~manner_maski

        # ------------------------------------------------------
        # Rajataan saarten määrä kokonaismaaosuuteen
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Jos manner täyttää jo maaosuuden
        # ------------------------------------------------------

        if float(sallittu_saarialue) <= 0.0:

            saari_maski = cp.zeros_like(
                manner_maski,
                dtype=cp.bool_,
            )

        # ------------------------------------------------------
        # Jos saaria on liikaa
        # ------------------------------------------------------

        elif float(nykyinen_saariosuus) > float(
            sallittu_saarialue
        ):

            saarien_arvot = saari_arvot[
                saari_maski
            ]

            if saarien_arvot.size > 0:

                pidettava_osuus = cp.clip(
                    sallittu_saarialue
                    / nykyinen_saariosuus,
                    cp.float32(0.0),
                    cp.float32(1.0),
                )

                saarien_kynnys = cp.quantile(
                    saarien_arvot,
                    1.0 - pidettava_osuus,
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
    # ==========================================================

    def _generoi_tasangot(self, x, y, z):

        terrain = self._fbm(
            x * self.tasangon_koko,
            y * self.tasangon_koko,
            z * self.tasangon_koko,
            seed=self.seed + 1000,
            octaves=14,
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        plateau = (
            1.0
            - cp.abs(
                terrain * 2.0 - 1.0
            )
        )

        return plateau ** cp.float32(1.5)

    # ==========================================================
    # Ylängöt
    # ==========================================================

    def _generoi_ylangot(self, x, y, z):

        terrain = self._fbm(
            x * self.ylangon_koko,
            y * self.ylangon_koko,
            z * self.ylangon_koko,
            seed=self.seed + 2000,
            octaves=4,
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        plateau = cp.clip(
            (terrain - cp.float32(0.52))
            / cp.float32(0.48),
            cp.float32(0.0),
            cp.float32(1.0),
        )

        return self._smoothstep(
            plateau
        )

    # ==========================================================
    # Vuoristot
    # ==========================================================

    def _generoi_vuoristot(self, x, y, z):

        mountains = self._ridged_noise(
            x * self.vuoriston_koko,
            y * self.vuoriston_koko,
            z * self.vuoriston_koko,
            seed=self.seed + 3000,
            octaves=15,
            base_grid=3,
        )

        return self._normalisoi_01(
            mountains
        )

    # ==========================================================
    # Pieni maasto
    # ==========================================================

    def _generoi_pieni_maasto(self, x, y, z):

        small = self._fbm(
            x * 5.0,
            y * 5.0,
            z * 5.0,
            seed=self.seed + 4000,
            octaves=16,
            base_grid=4,
        )

        return self._normalisoi_01(
            small
        )

    # ==========================================================
    # Merenpohja
    # ==========================================================

    def _generoi_merenpohja(self, x, y, z):

        ocean = self._fbm(
            x * self.merenpohjan_koko,
            y * self.merenpohjan_koko,
            z * self.merenpohjan_koko,
            seed=self.seed + 5000,
            octaves=15,
            base_grid=3,
        )

        return self._normalisoi_01(
            ocean
        )

    # ==========================================================
    # Raakakorkeus
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

        vuoristo = self._generoi_vuoristot(
            x,
            y,
            z,
        )

        pieni = self._generoi_pieni_maasto(
            x,
            y,
            z,
        )

        merenpohja = self._generoi_merenpohja(
            x,
            y,
            z,
        )

        # ------------------------------------------------------
        # Maan peruskorkeus
        # ------------------------------------------------------

        maan_korkeus = cp.full_like(
            tasanko,
            cp.float32(0.22),
            dtype=cp.float32,
        )

        maan_korkeus += (
            cp.float32(self.tasangon_voima)
            * tasanko
        )

        maan_korkeus += (
            cp.float32(self.ylangon_voima)
            * ylanko
        )

        maan_korkeus += (
            cp.float32(self.vuoriston_voima)
            * vuoristo
        )

        maan_korkeus += (
            cp.float32(self.pieni_maasto_voima)
            * pieni
        )

        # Saaret hieman matalammiksi.
        maan_korkeus -= (
            saari_maski
            * cp.float32(0.08)
        )

        # ------------------------------------------------------
        # Merenpohja
        # ------------------------------------------------------

        meren_korkeus = (
            cp.float32(-0.30)
            - cp.float32(self.merenpohjan_voima)
            * merenpohja
        )

        # ------------------------------------------------------
        # Rannikon loivuus
        # ------------------------------------------------------

        rannikko_noise = self._fbm(
            x * 1.5,
            y * 1.5,
            z * 1.5,
            seed=self.seed + 6000,
            octaves=13,
            base_grid=3,
        )

        rannikko_noise = (
            rannikko_noise
            + cp.float32(1.0)
        ) * cp.float32(0.5)

        rannikko_noise = self._smoothstep(
            rannikko_noise
        )

        maan_korkeus -= (
            rannikko_noise
            * cp.float32(self.rannikon_loivuus)
            * cp.float32(0.08)
        )

        return cp.where(
            maa_maski,
            maan_korkeus,
            meren_korkeus,
        )

    # ==========================================================
    # Korkeuksien skaalaus
    # ==========================================================

    def _skaalaa_korkeudet(
        self,
        raakakorkeuskartta,
        maa_maski,
        meri_maski,
    ):

        tulos = cp.zeros_like(
            raakakorkeuskartta,
            dtype=cp.float32,
        )

        # ------------------------------------------------------
        # Maa
        # ------------------------------------------------------

        maa_arvot = raakakorkeuskartta[
            maa_maski
        ]

        if maa_arvot.size > 0:

            maa_min = cp.min(
                maa_arvot
            )

            maa_max = cp.max(
                maa_arvot
            )

            normalized = cp.where(
                maa_max > maa_min,
                (
                    maa_arvot - maa_min
                )
                / (
                    maa_max - maa_min
                ),
                cp.zeros_like(
                    maa_arvot
                ),
            )

            # Korostetaan matalia ja keskikorkeita alueita.
            normalized = (
                normalized ** cp.float32(3.0)
            )

            tulos[maa_maski] = (
                normalized
                * cp.float32(
                    self.korkein_kohta
                )
            )

        # ------------------------------------------------------
        # Meri
        # ------------------------------------------------------

        meri_arvot = raakakorkeuskartta[
            meri_maski
        ]

        if meri_arvot.size > 0:

            meri_min = cp.min(
                meri_arvot
            )

            meri_max = cp.max(
                meri_arvot
            )

            normalized = cp.where(
                meri_max > meri_min,
                (
                    meri_arvot - meri_min
                )
                / (
                    meri_max - meri_min
                ),
                cp.zeros_like(
                    meri_arvot
                ),
            )

            normalized = (
                cp.float32(1.0)
                - normalized
            )

            tulos[meri_maski] = (
                normalized
                * cp.float32(
                    self.syvin_kohta
                )
            )

        return tulos

    # ==========================================================
    # Generointi
    # ==========================================================

    def generoi(self):

        # Pallokoordinaatit
        x, y, z = (
            self._pallokoordinaatit()
        )

        # Maa / manner / saaret
        (
            maa_maski,
            manner_maski,
            saari_maski,
        ) = self._generoi_maakentta(
            x,
            y,
            z,
        )

        meri_maski = ~maa_maski

        # Raakakorkeus
        raakakorkeuskartta = (
            self._generoi_raakakorkeus(
                x,
                y,
                z,
                maa_maski,
                manner_maski,
                saari_maski,
            )
        )

        self.raakakorkeuskartta = (
            raakakorkeuskartta
            .astype(
                cp.float32,
                copy=False,
            )
        )

        # Lopullinen korkeus/syvyys
        korkeus_syvyyskartta = (
            self._skaalaa_korkeudet(
                raakakorkeuskartta,
                maa_maski,
                meri_maski,
            )
        )

        self.korkeus_syvyyskartta = (
            korkeus_syvyyskartta
            .astype(
                cp.float32,
                copy=False,
            )
        )

        # Pelkkä maaston korkeuskartta:
        # meri = 0
        self.korkeuskartta = cp.where(
            maa_maski,
            korkeus_syvyyskartta,
            cp.float32(0.0),
        )

        # Maskit
        self.maa_maski = maa_maski
        self.meri_maski = meri_maski
        self.manner_maski = manner_maski
        self.saari_maski = saari_maski

        # Debug-tietoja
        if self.debug:

            self.debug_data = {
                "maa_osuus": float(
                    cp.mean(
                        maa_maski
                    )
                ),

                "meri_osuus": float(
                    cp.mean(
                        meri_maski
                    )
                ),

                "manner_osuus": float(
                    cp.mean(
                        manner_maski
                    )
                ),

                "saari_osuus": float(
                    cp.mean(
                        saari_maski
                    )
                ),

                "maa_min": float(
                    cp.min(
                        self.korkeuskartta[
                            maa_maski
                        ]
                    )
                )
                if bool(
                    cp.any(maa_maski)
                )
                else 0.0,

                "maa_max": float(
                    cp.max(
                        self.korkeuskartta[
                            maa_maski
                        ]
                    )
                )
                if bool(
                    cp.any(maa_maski)
                )
                else 0.0,

                "meri_min": float(
                    cp.min(
                        self.korkeus_syvyyskartta[
                            meri_maski
                        ]
                    )
                )
                if bool(
                    cp.any(meri_maski)
                )
                else 0.0,

                "meri_max": float(
                    cp.max(
                        self.korkeus_syvyyskartta[
                            meri_maski
                        ]
                    )
                )
                if bool(
                    cp.any(meri_maski)
                )
                else 0.0,
            }

        return self

    # ==========================================================
    # Tulokset
    # ==========================================================

    def tulokset(self, cpu=False):

        if self.korkeus_syvyyskartta is None:
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
                for key, value in tulos.items()
            }

        return tulos


def laske_ilmasto(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,

    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,

    global_moisture_coeff=1.0,

    # ------------------------------------------------------------
    # LÄMPÖTILA
    # ------------------------------------------------------------
    tmax_planet=50.0,
    K=0.73,

    meri_pilvi_max_viilennys=10.0,
    meri_pilvisyys=1.0,

    # ------------------------------------------------------------
    # TUULET
    # ------------------------------------------------------------
    tuuli_lat_scale=3.0,

    # ------------------------------------------------------------
    # MERIVIRRAT
    # ------------------------------------------------------------
    merivirta_voimakkuus=4.0,
    merivirta_etaisyys_km=300.0,
    merivirta_lampo_kerroin=0.5,

    # ------------------------------------------------------------
    # SADE
    # ------------------------------------------------------------
    itcz_max_sade=520.0,
    polar_front_max_sade=180.0,

    # ------------------------------------------------------------
    # MANNERMAISUUS
    # ------------------------------------------------------------
    kesä_mannermaisuus=12.0,
    talvi_mannermaisuus=-22.0,

    kuukausia=12,
):
    """
    Laskee kuukausittaiset lämpötilat ja sademäärät.

    Palauttaa
    ----------
    kuukausi_lämpötilat : ndarray
        Muoto (kuukausia, y, x), lämpötila °C.

    kuukausi_sateet : ndarray
        Muoto (kuukausia, y, x), sademäärä.

    kuukausi_tuuli_suunta : ndarray
        Tuulen suunnan suhteellinen kenttä kuukausittain.

    kuukausi_tuuli_voima : ndarray
        Tuulen voimakkuus 0...1 kuukausittain.

    kuukausi_merivirta : ndarray
        Merivirran suhteellinen vaikutus kuukausittain.
    """

    shape = korkeuskartta.shape

    # ============================================================
    # TULOSTEN ALUSTUS
    # ============================================================

    kuukausi_lämpötilat = np.zeros(
        (kuukausia, *shape),
        dtype=np.float64
    )

    kuukausi_sateet = np.zeros(
        (kuukausia, *shape),
        dtype=np.float64
    )

    kuukausi_tuuli_suunta = np.zeros(
        (kuukausia, *shape),
        dtype=np.float64
    )

    kuukausi_tuuli_voima = np.zeros(
        (kuukausia, *shape),
        dtype=np.float64
    )

    kuukausi_merivirta = np.zeros(
        (kuukausia, *shape),
        dtype=np.float64
    )

    # ============================================================
    # KUUKAUDET
    # ============================================================

    for kk in range(kuukausia):

        # ========================================================
        # 1. AURINKO JA VUODENAIKA
        # ========================================================

        kulma = (
            2.0
            * np.pi
            * (kk - 5.5)
            / 12.0
        )

        # Auringon / planeetan akselin vuodenaikainen kallistuma
        akselin_kallistuma = (
            tilt_planet_axis
            * np.cos(kulma)
        )

        # --------------------------------------------------------
        # Ilmastollinen päiväntasaaja / ITCZ
        # --------------------------------------------------------

        ilmastollinen_päiväntasaaja = (
            akselin_kallistuma * 0.65
        )

        # ========================================================
        # 2. RELATIIVINEN LEVEYSASTE
        #
        # KAIKKI TUULIVYÖHYKKEET PERUSTUVAT TÄHÄN
        # ========================================================

        rel_lat = (
            lat_grid_deg
            - ilmastollinen_päiväntasaaja
        )

        abs_rel_lat = np.abs(rel_lat)

        # ========================================================
        # 3. TUULEN SUUNTA
        #
        # Tuulivyöhykkeet liikkuvat ITCZ:n mukana.
        # ========================================================

        tuuli_suunta = -np.sin(
            np.radians(
                rel_lat * tuuli_lat_scale
            )
        )

        # --------------------------------------------------------
        # Hadley / pasaatit
        # --------------------------------------------------------

        tuuli_suunta = np.where(
            abs_rel_lat < 30.0,
            -np.abs(tuuli_suunta),
            tuuli_suunta
        )

        # --------------------------------------------------------
        # Ferrel / länsituulet
        # --------------------------------------------------------

        tuuli_suunta = np.where(
            (abs_rel_lat >= 30.0)
            & (abs_rel_lat < 62.0),
            np.abs(tuuli_suunta),
            tuuli_suunta
        )

        # --------------------------------------------------------
        # Polar / napapuhurit
        # --------------------------------------------------------

        tuuli_suunta = np.where(
            abs_rel_lat >= 62.0,
            -np.abs(tuuli_suunta),
            tuuli_suunta
        )

        # ========================================================
        # 4. TUULEN VOIMAKKUUS
        # ========================================================

        # Perusvoimakkuus
        tuuli_voima = np.zeros_like(
            lat_grid_deg,
            dtype=float
        )

        # --------------------------------------------------------
        # Pasaatit
        # --------------------------------------------------------

        pasaati_voima = (
            0.70
            + 0.30
            * np.abs(
                np.sin(
                    np.radians(rel_lat * 3.0)
                )
            )
        )

        tuuli_voima = np.where(
            abs_rel_lat < 30.0,
            pasaati_voima,
            tuuli_voima
        )

        # --------------------------------------------------------
        # Länsituulet
        # --------------------------------------------------------

        ferrel_voima = (
            0.60
            + 0.30
            * np.sin(
                np.radians(
                    np.clip(
                        abs_rel_lat - 30.0,
                        0.0,
                        32.0
                    )
                    * 2.8
                )
            )
        )

        tuuli_voima = np.where(
            (abs_rel_lat >= 30.0)
            & (abs_rel_lat < 62.0),
            ferrel_voima,
            tuuli_voima
        )

        # --------------------------------------------------------
        # Napa-alueet
        # --------------------------------------------------------

        tuuli_voima = np.where(
            abs_rel_lat >= 62.0,
            0.40,
            tuuli_voima
        )

        tuuli_voima = np.clip(
            tuuli_voima,
            0.30,
            1.00
        )

        # Tallennetaan
        kuukausi_tuuli_suunta[kk] = (
            tuuli_suunta
        )

        kuukausi_tuuli_voima[kk] = (
            tuuli_voima
        )

        # ========================================================
        # 5. OROGRAFIA
        # ========================================================

        kk_orografinen_pakote = (
            korkeus_gradientti_x
            * tuuli_suunta
            * tuuli_voima
        )

        # ========================================================
        # 6. MERIVIRRAT
        #
        # Merivirrat seuraavat tuulijärjestelmää.
        # ========================================================

        merivirta_lat = np.sin(
            np.radians(
                rel_lat * 2.0
            )
        )

        # Tuulen vaikutus merivirtaan
        merivirta_tuuliveto = (
            np.sign(tuuli_suunta)
            * tuuli_voima
        )

        virta_pakote = (
            merivirta_lat
            * merivirta_tuuliveto
        )

        # Meri vaikuttaa ensisijaisesti rannikon lähellä
        meri_etaisyys_kerroin = np.exp(
            -etaisyys_meresta_km
            / merivirta_etaisyys_km
        )

        kk_meri_anomalia = (
            virta_pakote
            * meri_etaisyys_kerroin
            * merivirta_voimakkuus
        )

        # Merivirta vain merellä
        kk_meri_anomalia *= meri_maski

        kuukausi_merivirta[kk] = (
            kk_meri_anomalia
        )

        # ========================================================
        # 7. AURINGON DEKLINATIO
        # ========================================================

        ratakulma_deg = kk * 30.0
        ratakulma_rad = np.radians(
            ratakulma_deg
        )

        deklinaatio = (
            tilt_planet_axis
            * np.sin(
                ratakulma_rad
                - np.radians(60.0)
            )
        )

        # ========================================================
        # 8. PLANETAARINEN ETÄISYYS AURINGOSTA
        # ========================================================

        true_anomaly = np.radians(
            ratakulma_deg
            - mvelp_planet
        )

        etäisyys_au = (
            (1.0 - ecc_planet ** 2)
            /
            (
                1.0
                + ecc_planet
                * np.cos(true_anomaly)
            )
        )

        säteily_kerroin = (
            1.0
            / np.sqrt(etäisyys_au)
        )

        # ========================================================
        # 9. ALUSTAVA LÄMPÖTILA
        # ========================================================

        zeniitti_etäisyys = np.abs(
            lat_grid_deg
            - deklinaatio
        )

        zeniitti_vaikutus = (
            K
            * zeniitti_etäisyys
        )

        # Meri pilvettää / jäähdyttää
        meri_pilvi_vaikutus = (
            meri_maski
            * meri_pilvisyys
            * meri_pilvi_max_viilennys
        )

        tasapaino_temp = (
            (tmax_planet + 273.15)
            * säteily_kerroin
            - 273.15
        )

        korkeus_vaikutus = (
            korkeuskartta / 1000.0
        ) * 6.5

        temp = (
            tasapaino_temp
            - zeniitti_vaikutus
            - meri_pilvi_vaikutus
            - korkeus_vaikutus
        )

        # ========================================================
        # 10. KOSTEUDEN KANTAVUUS
        # ========================================================

        pasaati_tehostus = np.where(
            abs_rel_lat < 16.0,
            1.8,
            1.0
        )

        kantavuus_matka_km = (
            1200.0
            * (1.0 + tuuli_voima)
            * pasaati_tehostus
        )

        # ========================================================
        # 11. TUULI -> ONSHORE / OFFSHORE
        # ========================================================

        tuulen_suunta_suhteessa_maahan = (
            korkeus_gradientti_x
            * tuuli_suunta
        )

        tuulen_kosteus_kerroin = np.where(
            tuulen_suunta_suhteessa_maahan > 0,
            1.3,
            0.2
        )

        # ITCZ:n lähellä kosteusvaikutus voimakkaampi
        tuulen_kosteus_kerroin = np.where(
            abs_rel_lat < 12.0,
            tuulen_kosteus_kerroin * 4.0,
            np.clip(
                tuulen_kosteus_kerroin,
                0.1,
                1.0
            )
        )

        # ========================================================
        # 12. SADEVARJO
        # ========================================================

        efektiivinen_etaisyys_km = (
            etaisyys_meresta_km.copy()
        )

        efektiivinen_etaisyys_km += np.where(
            korkeuskartta > 1200.0,
            1500.0,
            0.0
        )

        # ========================================================
        # 13. ALAVUUS
        # ========================================================

        alavuus_kerroin = np.clip(
            1.3
            - korkeuskartta / 800.0,
            0.4,
            1.4
        )

        # ========================================================
        # 14. RANNIKKOKERROIN
        # ========================================================

        rannikko_kerroin = np.exp(
            -efektiivinen_etaisyys_km
            / kantavuus_matka_km
        )

        rannikko_kerroin *= (
            tuulen_kosteus_kerroin
            * alavuus_kerroin
        )

        # ========================================================
        # 15. MANNERMAISUUS / GOBI
        # ========================================================

        keskileveys_maski = np.exp(
            -(
                np.abs(lat_grid_deg)
                - 45.0
            ) ** 2
            / 120.0
        )

        mannermaisuus_vaikutus = np.exp(
            -efektiivinen_etaisyys_km
            / 650.0
        )

        lopullinen_rannikko_kerroin = np.where(
            np.abs(lat_grid_deg) > 30.0,

            np.minimum(
                rannikko_kerroin,

                mannermaisuus_vaikutus
                * keskileveys_maski
                +
                rannikko_kerroin
                * (1.0 - keskileveys_maski)
            ),

            rannikko_kerroin
        )

        # Rajataan järkeväksi
        lopullinen_rannikko_kerroin = np.clip(
            lopullinen_rannikko_kerroin,
            0.0,
            1.0
        )

        manner_kerroin = (
            1.0
            - lopullinen_rannikko_kerroin
        )

        # ========================================================
        # 16. ITCZ-SADEVYÖHYKE
        # ========================================================

        itcz_vaikutus = (
            np.exp(
                -(
                    lat_grid_deg
                    - ilmastollinen_päiväntasaaja
                ) ** 2
                / 40.0
            )
            * itcz_max_sade
        )

        # ========================================================
        # 17. POLAARIRINTAMA
        # ========================================================

        polar_pohjoinen = (
            52.0
            + ilmastollinen_päiväntasaaja * 0.5
        )

        polar_etelä = (
            -52.0
            + ilmastollinen_päiväntasaaja * 0.5
        )

        polar_front_vaikutus = (
            np.exp(
                -(
                    lat_grid_deg
                    - polar_pohjoinen
                ) ** 2
                / 150.0
            )
            * polar_front_max_sade

            +

            np.exp(
                -(
                    lat_grid_deg
                    - polar_etelä
                ) ** 2
                / 150.0
            )
            * polar_front_max_sade
        )

        # ========================================================
        # 18. SUBTROPIIKIN KUIVUUS
        # ========================================================

        kuiva_pohjoinen = (
            ilmastollinen_päiväntasaaja
            + 28.0
        )

        kuiva_etelä = (
            ilmastollinen_päiväntasaaja
            - 28.0
        )

        subtropiikki_kuivuus = (
            1.0
            - 0.75
            * (
                np.exp(
                    -(
                        lat_grid_deg
                        - kuiva_pohjoinen
                    ) ** 2
                    / 70.0
                )

                +

                np.exp(
                    -(
                        lat_grid_deg
                        - kuiva_etelä
                    ) ** 2
                    / 70.0
                )
            )
        )

        subtropiikki_kuivuus = np.clip(
            subtropiikki_kuivuus,
            0.12,
            1.0
        )

        # ========================================================
        # 19. TROPIIKIN SADE
        # ========================================================

        tropiikki_sade_sisamaassa = (
            itcz_vaikutus
            * np.minimum(
                1.0,
                rannikko_kerroin + 0.65
            )
        )

        # ========================================================
        # 20. MUUT SADEVYÖHYKKEET
        # ========================================================

        muut_vyöhyke_sateet = (
            20.0
            + polar_front_vaikutus
        ) * lopullinen_rannikko_kerroin

        # ========================================================
        # 21. PERUSSADE
        # ========================================================

        perus_sade = (
            tropiikki_sade_sisamaassa
            + muut_vyöhyke_sateet
        )

        perus_sade *= (
            subtropiikki_kuivuus
        )

        # ========================================================
        # 22. OROGRAFINEN SADE
        # ========================================================

        korkeussade_pohja = np.clip(
            korkeuskartta / 150.0,
            0.0,
            80.0
        )

        tuuli_orografia_vaikutus = (
            kk_orografinen_pakote
            * 2.5
        )

        dynaaminen_orografia = (
            np.clip(
                korkeussade_pohja
                + tuuli_orografia_vaikutus,
                0.0,
                250.0
            )
            * lopullinen_rannikko_kerroin
            * 0.1
        )

        # ========================================================
        # 23. LOPULLINEN SADE
        # ========================================================

        sade = (
            perus_sade
            + dynaaminen_orografia
        )

        sade *= global_moisture_coeff

        sade = np.maximum(
            sade,
            0.0
        )

        kuukausi_sateet[kk] = sade

        # ========================================================
        # 24. SADE -> PILVISYYS -> LÄMPÖTILA
        # ========================================================

        pilvisyys_dt = (
            sade / 100.0
        )

        temp -= pilvisyys_dt

        # ========================================================
        # 25. MANNERMAISUUS
        # ========================================================

        kuivuus_anomalia = np.clip(
            1.0
            - sade / 150.0,
            0.0,
            1.0
        )

        leveysaste_painotus = (
            np.abs(lat_grid_deg)
            / 90.0
        )

        mannermaisuus_indeksi = (
            manner_kerroin
            * (
                0.5
                + 0.5 * kuivuus_anomalia
            )
            * leveysaste_painotus
        )

        # ========================================================
        # 26. KESÄ / TALVI
        # ========================================================

        kesä_maski = (
            np.sign(lat_grid_deg)
            * np.sign(akselin_kallistuma)
        )

        lämpötila_muutos = np.where(
            kesä_maski >= 0,
            mannermaisuus_indeksi
            * kesä_mannermaisuus,

            mannermaisuus_indeksi
            * talvi_mannermaisuus
        )

        temp += lämpötila_muutos

        # ========================================================
        # 27. MERIVIRRAN LÄMPÖVAIKUTUS
        # ========================================================

        meri_dt = (
            kk_meri_anomalia
            * merivirta_lampo_kerroin
        )

        temp += meri_dt

        # ========================================================
        # 28. TALLENNUS
        # ========================================================

        kuukausi_lämpötilat[kk] = temp

    # ============================================================
    # PALAUTUS
    # ============================================================

    return (
        kuukausi_lämpötilat,
        kuukausi_sateet,
        kuukausi_tuuli_suunta,
        kuukausi_tuuli_voima,
        kuukausi_merivirta,
    )


from numba import njit


#@njit(cache=True)
def warmup_numba(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,

    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    global_moisture_coeff=1.0,
):
    # Pieni pala oikean datan muotoisena
    lat_w = lat_grid_deg[:2, :2]
    korkeus_w = korkeuskartta[:2, :2]
    gradientti_w = korkeus_gradientti_x[:2, :2]
    maa_w = maa_maski[:2, :2]
    meri_w = meri_maski[:2, :2]
    etaisyys_w = etaisyys_meresta_km[:2, :2]

    laske_ilmasto_numba(
        lat_w,
        korkeus_w,
        gradientti_w,
        maa_w,
        meri_w,
        etaisyys_w,

        tilt_planet_axis,
        ecc_planet,
        mvelp_planet,

        global_moisture_coeff,
    )





#@njit(cache=True)

@njit(parallel=True, cache=True)
def laske_ilmasto_numba(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,

    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,

    global_moisture_coeff=1.0,

    tmax_planet=50.0,
    K=0.73,

    meri_pilvi_max_viilennys=10.0,
    meri_pilvisyys=1.0,

    tuuli_lat_scale=3.0,

    merivirta_voimakkuus=4.0,
    merivirta_etaisyys_km=300.0,
    merivirta_lampo_kerroin=0.5,

    itcz_max_sade=520.0,
    polar_front_max_sade=180.0,

    kesä_mannermaisuus=12.0,
    talvi_mannermaisuus=-22.0,

    kuukausia=12,
):

    ny, nx = korkeuskartta.shape

    # ============================================================
    # TULOSTEN ALUSTUS
    # ============================================================

    kuukausi_lämpötilat = np.zeros(
        (kuukausia, ny, nx),
        dtype=np.float64
    )

    kuukausi_sateet = np.zeros(
        (kuukausia, ny, nx),
        dtype=np.float64
    )

    kuukausi_tuuli_suunta = np.zeros(
        (kuukausia, ny, nx),
        dtype=np.float64
    )

    kuukausi_tuuli_voima = np.zeros(
        (kuukausia, ny, nx),
        dtype=np.float64
    )

    kuukausi_merivirta = np.zeros(
        (kuukausia, ny, nx),
        dtype=np.float64
    )

    # ============================================================
    # KUUKAUDET
    # ============================================================

    for kk in range(kuukausia):

        # --------------------------------------------------------
        # 1. AURINKO
        # --------------------------------------------------------

        kulma = (
            2.0
            * np.pi
            * (kk - 5.5)
            / 12.0
        )

        akselin_kallistuma = (
            tilt_planet_axis
            * np.cos(kulma)
        )

        # ITCZ:n siirtymä
        ilmastollinen_päiväntasaaja = (
            akselin_kallistuma
            * 0.65
        )

        # --------------------------------------------------------
        # 2. AURINGON DEKLINATIO
        # --------------------------------------------------------

        ratakulma_deg = kk * 30.0

        ratakulma_rad = (
            ratakulma_deg
            * np.pi
            / 180.0
        )

        deklinaatio = (
            tilt_planet_axis
            * np.sin(
                ratakulma_rad
                - np.pi / 3.0
            )
        )

        # --------------------------------------------------------
        # 3. PLANETAARINEN ETÄISYYS
        # --------------------------------------------------------

        true_anomaly = (
            (ratakulma_deg - mvelp_planet)
            * np.pi
            / 180.0
        )

        etäisyys_au = (
            (1.0 - ecc_planet * ecc_planet)
            /
            (
                1.0
                + ecc_planet
                * np.cos(true_anomaly)
            )
        )

        säteily_kerroin = (
            1.0
            / np.sqrt(etäisyys_au)
        )

        # ========================================================
        # HILAN LASKENTA
        # ========================================================

        for y in range(ny):

            for x in range(nx):

                lat = lat_grid_deg[y, x]
                korkeus = korkeuskartta[y, x]
                etaisyys = etaisyys_meresta_km[y, x]

                # =================================================
                # 4. RELATIIVINEN LEVEYSASTE
                # =================================================

                rel_lat = (
                    lat
                    - ilmastollinen_päiväntasaaja
                )

                abs_rel_lat = abs(rel_lat)

                # =================================================
                # 5. TUULEN SUUNTA
                # =================================================

                tuuli_suunta = -np.sin(
                    rel_lat
                    * tuuli_lat_scale
                    * np.pi
                    / 180.0
                )

                # Pasaatit
                if abs_rel_lat < 30.0:

                    tuuli_suunta = (
                        -abs(tuuli_suunta)
                    )

                # Länsituulet
                elif abs_rel_lat < 62.0:

                    tuuli_suunta = (
                        abs(tuuli_suunta)
                    )

                # Napapuhurit
                else:

                    tuuli_suunta = (
                        -abs(tuuli_suunta)
                    )

                # =================================================
                # 6. TUULEN VOIMAKKUUS
                # =================================================

                if abs_rel_lat < 30.0:

                    tuuli_voima = (
                        0.70
                        + 0.30
                        * abs(
                            np.sin(
                                rel_lat
                                * 3.0
                                * np.pi
                                / 180.0
                            )
                        )
                    )

                elif abs_rel_lat < 62.0:

                    kulma_ferrel = (
                        (abs_rel_lat - 30.0)
                        * 2.8
                        * np.pi
                        / 180.0
                    )

                    tuuli_voima = (
                        0.60
                        + 0.30
                        * np.sin(kulma_ferrel)
                    )

                else:

                    tuuli_voima = 0.40

                if tuuli_voima < 0.30:
                    tuuli_voima = 0.30

                if tuuli_voima > 1.0:
                    tuuli_voima = 1.0

                kuukausi_tuuli_suunta[
                    kk, y, x
                ] = tuuli_suunta

                kuukausi_tuuli_voima[
                    kk, y, x
                ] = tuuli_voima

                # =================================================
                # 7. OROGRAFIA
                # =================================================

                kk_orografinen_pakote = (
                    korkeus_gradientti_x[y, x]
                    * tuuli_suunta
                    * tuuli_voima
                )

                # =================================================
                # 8. MERIVIRTA
                # =================================================

                merivirta_lat = np.sin(
                    rel_lat
                    * 2.0
                    * np.pi
                    / 180.0
                )

                merivirta_tuuliveto = (
                    np.sign(tuuli_suunta)
                    * tuuli_voima
                )

                virta_pakote = (
                    merivirta_lat
                    * merivirta_tuuliveto
                )

                meri_etaisyys_kerroin = np.exp(
                    -etaisyys
                    / merivirta_etaisyys_km
                )

                meri_anomalia = (
                    virta_pakote
                    * meri_etaisyys_kerroin
                    * merivirta_voimakkuus
                )

                if not meri_maski[y, x]:
                    meri_anomalia = 0.0

                kuukausi_merivirta[
                    kk, y, x
                ] = meri_anomalia

                # =================================================
                # 9. LÄMPÖTILAN PERUSOSA
                # =================================================

                zeniitti_etäisyys = abs(
                    lat
                    - deklinaatio
                )

                zeniitti_vaikutus = (
                    K
                    * zeniitti_etäisyys
                )

                if meri_maski[y, x]:

                    meri_pilvi_vaikutus = (
                        meri_pilvisyys
                        * meri_pilvi_max_viilennys
                    )

                else:

                    meri_pilvi_vaikutus = 0.0

                tasapaino_temp = (
                    (tmax_planet + 273.15)
                    * säteily_kerroin
                    - 273.15
                )

                korkeus_vaikutus = (
                    korkeus
                    / 1000.0
                    * 6.5
                )

                temp = (
                    tasapaino_temp
                    - zeniitti_vaikutus
                    - meri_pilvi_vaikutus
                    - korkeus_vaikutus
                )

                # =================================================
                # 10. KOSTEUDEN KANTAVUUS
                # =================================================

                if abs_rel_lat < 16.0:

                    pasaati_tehostus = 1.8

                else:

                    pasaati_tehostus = 1.0

                kantavuus_matka_km = (
                    1200.0
                    * (1.0 + tuuli_voima)
                    * pasaati_tehostus
                )

                # =================================================
                # 11. ONSHORE / OFFSHORE
                # =================================================

                tuulen_suunta_suhteessa_maahan = (
                    korkeus_gradientti_x[y, x]
                    * tuuli_suunta
                )

                if (
                    tuulen_suunta_suhteessa_maahan
                    > 0.0
                ):

                    tuulen_kosteus_kerroin = 1.3

                else:

                    tuulen_kosteus_kerroin = 0.2

                if abs_rel_lat < 12.0:

                    tuulen_kosteus_kerroin *= 4.0

                else:

                    if tuulen_kosteus_kerroin < 0.1:
                        tuulen_kosteus_kerroin = 0.1

                    if tuulen_kosteus_kerroin > 1.0:
                        tuulen_kosteus_kerroin = 1.0

                # =================================================
                # 12. SADEVARJO
                # =================================================

                efektiivinen_etaisyys = etaisyys

                if korkeus > 1200.0:

                    efektiivinen_etaisyys += 1500.0

                # =================================================
                # 13. ALAVUUS
                # =================================================

                alavuus_kerroin = (
                    1.3
                    - korkeus / 800.0
                )

                if alavuus_kerroin < 0.4:
                    alavuus_kerroin = 0.4

                if alavuus_kerroin > 1.4:
                    alavuus_kerroin = 1.4

                # =================================================
                # 14. RANNIKKOKERROIN
                # =================================================

                rannikko_kerroin = np.exp(
                    -efektiivinen_etaisyys
                    / kantavuus_matka_km
                )

                rannikko_kerroin *= (
                    tuulen_kosteus_kerroin
                    * alavuus_kerroin
                )

                # =================================================
                # 15. MANNERMAISUUS
                # =================================================

                keskileveys_maski = np.exp(
                    -(
                        abs(lat) - 45.0
                    ) ** 2
                    / 120.0
                )

                mannermaisuus_vaikutus = np.exp(
                    -efektiivinen_etaisyys
                    / 650.0
                )

                if abs(lat) > 30.0:

                    arvo = (
                        mannermaisuus_vaikutus
                        * keskileveys_maski
                        +
                        rannikko_kerroin
                        * (
                            1.0
                            - keskileveys_maski
                        )
                    )

                    lopullinen_rannikko = min(
                        rannikko_kerroin,
                        arvo
                    )

                else:

                    lopullinen_rannikko = (
                        rannikko_kerroin
                    )

                if lopullinen_rannikko < 0.0:
                    lopullinen_rannikko = 0.0

                if lopullinen_rannikko > 1.0:
                    lopullinen_rannikko = 1.0

                manner_kerroin = (
                    1.0
                    - lopullinen_rannikko
                )

                # =================================================
                # 16. ITCZ
                # =================================================

                itcz_vaikutus = (
                    np.exp(
                        -(
                            lat
                            - ilmastollinen_päiväntasaaja
                        ) ** 2
                        / 40.0
                    )
                    * itcz_max_sade
                )

                # =================================================
                # 17. POLAARIRINTAMA
                # =================================================

                polar_pohjoinen = (
                    52.0
                    + ilmastollinen_päiväntasaaja
                    * 0.5
                )

                polar_etelä = (
                    -52.0
                    + ilmastollinen_päiväntasaaja
                    * 0.5
                )

                polar_front_vaikutus = (
                    np.exp(
                        -(
                            lat
                            - polar_pohjoinen
                        ) ** 2
                        / 150.0
                    )
                    * polar_front_max_sade
                    +

                    np.exp(
                        -(
                            lat
                            - polar_etelä
                        ) ** 2
                        / 150.0
                    )
                    * polar_front_max_sade
                )

                # =================================================
                # 18. SUBTROPIIKIN KUIVUUS
                # =================================================

                kuiva_pohjoinen = (
                    ilmastollinen_päiväntasaaja
                    + 28.0
                )

                kuiva_etelä = (
                    ilmastollinen_päiväntasaaja
                    - 28.0
                )

                subtropiikki_kuivuus = (
                    1.0
                    - 0.75
                    * (
                        np.exp(
                            -(
                                lat
                                - kuiva_pohjoinen
                            ) ** 2
                            / 70.0
                        )
                        +
                        np.exp(
                            -(
                                lat
                                - kuiva_etelä
                            ) ** 2
                            / 70.0
                        )
                    )
                )

                if subtropiikki_kuivuus < 0.12:
                    subtropiikki_kuivuus = 0.12

                if subtropiikki_kuivuus > 1.0:
                    subtropiikki_kuivuus = 1.0

                # =================================================
                # 19. TROPIIKIN SADE
                # =================================================

                tropiikki_sade = (
                    itcz_vaikutus
                    * min(
                        1.0,
                        rannikko_kerroin + 0.65
                    )
                )

                # =================================================
                # 20. MUUT SADEALUEET
                # =================================================

                muut_sateet = (
                    20.0
                    + polar_front_vaikutus
                ) * lopullinen_rannikko

                perus_sade = (
                    tropiikki_sade
                    + muut_sateet
                ) * subtropiikki_kuivuus

                # =================================================
                # 21. OROGRAFINEN SADE
                # =================================================

                korkeussade_pohja = (
                    korkeus / 150.0
                )

                if korkeussade_pohja < 0.0:
                    korkeussade_pohja = 0.0

                if korkeussade_pohja > 80.0:
                    korkeussade_pohja = 80.0

                tuuli_orografia_vaikutus = (
                    kk_orografinen_pakote
                    * 2.5
                )

                dynaaminen_arvo = (
                    korkeussade_pohja
                    + tuuli_orografia_vaikutus
                )

                if dynaaminen_arvo < 0.0:
                    dynaaminen_arvo = 0.0

                if dynaaminen_arvo > 250.0:
                    dynaaminen_arvo = 250.0

                dynaaminen_orografia = (
                    dynaaminen_arvo
                    * lopullinen_rannikko
                    * 0.1
                )

                # =================================================
                # 22. LOPULLINEN SADE
                # =================================================

                sade = (
                    perus_sade
                    + dynaaminen_orografia
                )

                sade *= global_moisture_coeff

                if sade < 0.0:
                    sade = 0.0

                kuukausi_sateet[
                    kk, y, x
                ] = sade

                # =================================================
                # 23. SADE -> PILVISYYS -> LÄMPÖTILA
                # =================================================

                pilvisyys_dt = (
                    sade / 100.0
                )

                temp -= pilvisyys_dt

                # =================================================
                # 24. KUIVUUS
                # =================================================

                kuivuus_anomalia = (
                    1.0
                    - sade / 150.0
                )

                if kuivuus_anomalia < 0.0:
                    kuivuus_anomalia = 0.0

                if kuivuus_anomalia > 1.0:
                    kuivuus_anomalia = 1.0

                leveysaste_painotus = (
                    abs(lat) / 90.0
                )

                mannermaisuus_indeksi = (
                    manner_kerroin
                    * (
                        0.5
                        + 0.5
                        * kuivuus_anomalia
                    )
                    * leveysaste_painotus
                )

                # =================================================
                # 25. KESÄ / TALVI
                # =================================================

                kesä_maski = (
                    np.sign(lat)
                    * np.sign(
                        akselin_kallistuma
                    )
                )

                if kesä_maski >= 0.0:

                    temp += (
                        mannermaisuus_indeksi
                        * kesä_mannermaisuus
                    )

                else:

                    temp += (
                        mannermaisuus_indeksi
                        * talvi_mannermaisuus
                    )

                # =================================================
                # 26. MERIVIRRAN LÄMPÖVAIKUTUS
                # =================================================

                temp += (
                    meri_anomalia
                    * merivirta_lampo_kerroin
                )

                # =================================================
                # 27. TALLENNUS
                # =================================================

                kuukausi_lämpötilat[
                    kk, y, x
                ] = temp

    return (
        kuukausi_lämpötilat,
        kuukausi_sateet,
        kuukausi_tuuli_suunta,
        kuukausi_tuuli_voima,
        kuukausi_merivirta,
    )


import numpy as np
import cupy as cp


def laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,

    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,

    global_moisture_coeff=1.0,

    palauta_cpu=True,
):
    """
    Koko ilmastomalli GPU:lle CuPyllä.

    GPU:
        NVIDIA GeForce RTX 5060

    Palauttaa:
        kuukausi_lampotilat
        kuukausi_sateet
        kuukausi_tuuli_suunta
        kuukausi_tuuli_voima
        kuukausi_merivirta

    Muodot:
        (12, ny, nx)

    Jos palauta_cpu=True:
        palautetaan NumPy-taulukot.

    Jos palauta_cpu=False:
        palautetaan CuPy-taulukot GPU-muistissa.
    """

    # ============================================================
    # 1. DATA GPU:LLE
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
    # 2. KOKO / TULOSARRAYT
    # ============================================================

    ny, nx = lat.shape

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

    kuukausi_merivirta = cp.empty(
        (12, ny, nx),
        dtype=cp.float32
    )

    # ============================================================
    # 3. KIINTEÄT KARTTATERMIT
    # ============================================================

    korkeus_vaikutus = (
        korkeus / cp.float32(1000.0)
    ) * cp.float32(6.5)

    # ============================================================
    # 4. KUUKAUDET
    # ============================================================

    for kk in range(12):

        # ========================================================
        # AURINKO / ITCZ
        # ========================================================

        kulma = (
            2.0
            * np.pi
            * (kk - 5.5)
            / 12.0
        )

        akselin_kallistuma = (
            tilt_planet_axis
            * np.cos(kulma)
        )

        ilmastollinen_paivantaasaaja = (
            akselin_kallistuma
            * 0.65
        )

        # ========================================================
        # TUULI
        # ========================================================

        relatiivinen_lat = (
            lat
            - ilmastollinen_paivantaasaaja
        )

        abs_rel_lat = cp.abs(
            relatiivinen_lat
        )

        # Perustuuli
        tuuli_suunta = -cp.sin(
            cp.deg2rad(
                relatiivinen_lat * 6.0
            )
        )

        # --------------------------------------------------------
        # Hadley
        # --------------------------------------------------------

        tuuli_suunta = cp.where(
            abs_rel_lat < 30.0,
            -cp.abs(tuuli_suunta),
            tuuli_suunta
        )

        # --------------------------------------------------------
        # Ferrel
        # --------------------------------------------------------

        tuuli_suunta = cp.where(
            (
                (abs_rel_lat >= 30.0)
                &
                (abs_rel_lat < 62.0)
            ),
            cp.abs(tuuli_suunta),
            tuuli_suunta
        )

        # --------------------------------------------------------
        # Polar
        # --------------------------------------------------------

        tuuli_suunta = cp.where(
            abs_rel_lat >= 62.0,
            -cp.abs(tuuli_suunta),
            tuuli_suunta
        )

        # --------------------------------------------------------
        # Tuulen voimakkuus
        # --------------------------------------------------------

        tuuli_voima = cp.clip(
            cp.abs(tuuli_suunta),
            0.3,
            1.0
        )

        kuukausi_tuuli_suunta[
            kk
        ] = tuuli_suunta

        kuukausi_tuuli_voima[
            kk
        ] = tuuli_voima

        # ========================================================
        # OROGRAFIA
        # ========================================================

        orografinen_pakote = (
            gradientti_x
            * tuuli_suunta
            * tuuli_voima
        )

        # ========================================================
        # MERIVIRTA
        # ========================================================

        virta_pakote = (
            cp.sin(
                cp.deg2rad(
                    lat * 2.0
                )
            )
            * cp.sign(
                tuuli_suunta
            )
        )

        meri_anomalia = (
            virta_pakote
            * cp.exp(
                -etaisyys_meresta / 200.0
            )
            * 4.0
            * tuuli_voima
        )

        kuukausi_merivirta[
            kk
        ] = cp.where(
            maa,
            meri_anomalia,
            0.0
        )

        # ========================================================
        # AURINGON DEKLINAATIO
        # ========================================================

        ratakulma_deg = kk * 30.0

        ratakulma_rad = np.radians(
            ratakulma_deg
        )

        deklinaatio = (
            tilt_planet_axis
            * np.sin(
                ratakulma_rad
                - np.radians(60.0)
            )
        )

        # ========================================================
        # PLANETAARINEN ETÄISYYS
        # ========================================================

        true_anomaly = np.radians(
            ratakulma_deg
            - mvelp_planet
        )

        etaisyys_au = (
            1.0
            - ecc_planet ** 2
        ) / (
            1.0
            + ecc_planet
            * np.cos(true_anomaly)
        )

        sateily_kerroin = (
            1.0
            / np.sqrt(
                etaisyys_au
            )
        )

        # ========================================================
        # PERUSLÄMPÖTILA
        # ========================================================

        zeniitti_etaisyys = cp.abs(
            lat - deklinaatio
        )

        K = cp.float32(0.73)

        zeniitti_vaikutus = (
            K
            * zeniitti_etaisyys
        )

        meri_pilvi_max_viilennys = 10.0
        meri_pilvisyys = 1.0

        meri_pilvi_vaikutus = (
            meri.astype(cp.float32)
            * meri_pilvisyys
            * meri_pilvi_max_viilennys
        )

        tmax_planet = 50.0

        tasapaino_temp = (
            (
                tmax_planet
                + 273.15
            )
            * sateily_kerroin
            - 273.15
        )

        perus_temp = (
            tasapaino_temp
            - zeniitti_vaikutus
            - meri_pilvi_vaikutus
        )

        # ========================================================
        # SADE
        # ========================================================

        # --------------------------------------------------------
        # ITCZ
        # --------------------------------------------------------

        itcz_vaikutus = (
            cp.exp(
                -(
                    (
                        lat
                        - ilmastollinen_paivantaasaaja
                    ) ** 2
                )
                / 40.0
            )
            * 520.0
        )

        # --------------------------------------------------------
        # Polar front
        # --------------------------------------------------------

        polar_pohjoinen = (
            52.0
            + ilmastollinen_paivantaasaaja
            * 0.5
        )

        polar_etela = (
            -52.0
            + ilmastollinen_paivantaasaaja
            * 0.5
        )

        polar_front_vaikutus = (
            cp.exp(
                -(
                    (
                        lat
                        - polar_pohjoinen
                    ) ** 2
                )
                / 150.0
            )
            * 180.0
            +
            cp.exp(
                -(
                    (
                        lat
                        - polar_etela
                    ) ** 2
                )
                / 150.0
            )
            * 180.0
        )

        # ========================================================
        # SUBTROPIIKIN KUIVUUS
        # ========================================================

        kuiva_pohjoinen = (
            ilmastollinen_paivantaasaaja
            + 28.0
        )

        kuiva_etela = (
            ilmastollinen_paivantaasaaja
            - 28.0
        )

        subtropiikki_kuivuus = (
            1.0
            - 0.75
            * (
                cp.exp(
                    -(
                        (
                            lat
                            - kuiva_pohjoinen
                        ) ** 2
                    )
                    / 70.0
                )
                +
                cp.exp(
                    -(
                        (
                            lat
                            - kuiva_etela
                        ) ** 2
                    )
                    / 70.0
                )
            )
        )

        subtropiikki_kuivuus = cp.clip(
            subtropiikki_kuivuus,
            0.12,
            1.0
        )

        # ========================================================
        # GOBI / SÄDESUOJA
        # ========================================================

        efektiivinen_etaisyys = (
            etaisyys_meresta
            + cp.where(
                korkeus > 1200.0,
                1500.0,
                0.0
            )
        )

        # ========================================================
        # KOSTEUDEN KANTAVUUS
        # ========================================================

        pasaati_tehostus = cp.where(
            abs_rel_lat < 16.0,
            1.8,
            1.0
        )

        kantavuus_matka = (
            1200.0
            * (1.0 + tuuli_voima)
            * pasaati_tehostus
        )

        # ========================================================
        # ONSHORE / OFFSHORE
        # ========================================================

        tuulen_suunta_suhteessa_maahan = (
            gradientti_x
            * tuuli_suunta
        )

        tuulen_kosteus_kerroin = cp.where(
            tuulen_suunta_suhteessa_maahan > 0.0,
            1.3,
            0.2
        )

        tuulen_kosteus_kerroin = cp.where(
            abs_rel_lat < 12.0,
            tuulen_kosteus_kerroin * 4.0,
            cp.clip(
                tuulen_kosteus_kerroin,
                0.1,
                1.0
            )
        )

        # ========================================================
        # ALAVUUS
        # ========================================================

        alavuus_kerroin = cp.clip(
            1.3
            - korkeus / 800.0,
            0.4,
            1.4
        )

        # ========================================================
        # RANNIKKOKERROIN
        # ========================================================

        rannikko_kerroin = cp.exp(
            -efektiivinen_etaisyys
            / kantavuus_matka
        )

        rannikko_kerroin *= (
            tuulen_kosteus_kerroin
            * alavuus_kerroin
        )

        # ========================================================
        # MANNERMAISUUS
        # ========================================================

        keskileveys_maski = cp.exp(
            -(
                (
                    cp.abs(lat)
                    - 45.0
                ) ** 2
            )
            / 120.0
        )

        mannermaisuus_vaikutus = cp.exp(
            -efektiivinen_etaisyys
            / 650.0
        )

        yhdistelma = (
            mannermaisuus_vaikutus
            * keskileveys_maski
            +
            rannikko_kerroin
            * (
                1.0
                - keskileveys_maski
            )
        )

        lopullinen_rannikko_kerroin = cp.where(
            cp.abs(lat) > 30.0,
            cp.minimum(
                rannikko_kerroin,
                yhdistelma
            ),
            rannikko_kerroin
        )

        # ========================================================
        # SADE
        # ========================================================

        tropiikki_sade_sisamaassa = (
            itcz_vaikutus
            * cp.minimum(
                1.0,
                rannikko_kerroin + 0.65
            )
        )

        muut_vyohyke_sateet = (
            20.0
            + polar_front_vaikutus
        ) * lopullinen_rannikko_kerroin

        perus_sade = (
            tropiikki_sade_sisamaassa
            + muut_vyohyke_sateet
        ) * subtropiikki_kuivuus

        # ========================================================
        # VUORISTOSADE
        # ========================================================

        korkeussade_pohja = cp.clip(
            korkeus / 150.0,
            0.0,
            80.0
        )

        tuuli_orografia_vaikutus = (
            orografinen_pakote
            * 2.5
        )

        dynaaminen_orografia = cp.clip(
            korkeussade_pohja
            + tuuli_orografia_vaikutus,
            0.0,
            250.0
        )

        dynaaminen_orografia *= (
            lopullinen_rannikko_kerroin
            * 0.1
        )

        sade = (
            perus_sade
            + dynaaminen_orografia
        )

        sade *= global_moisture_coeff

        kuukausi_sateet[
            kk
        ] = sade

        # ========================================================
        # MANNERMAISUUDEN LÄMPÖTILAVAIKUTUS
        #
        # Nyt sade on jo laskettu.
        # ========================================================

        manner_kerroin = (
            1.0
            - lopullinen_rannikko_kerroin
        )

        kuivuus_anomalia = cp.clip(
            1.0
            - sade / 150.0,
            0.0,
            1.0
        )

        leveysaste_painotus = (
            cp.abs(lat)
            / 90.0
        )

        mannermaisuus_indeksi = (
            manner_kerroin
            * (
                0.5
                + 0.5
                * kuivuus_anomalia
            )
            * leveysaste_painotus
        )

        kesä_maski = (
            cp.sign(lat)
            * np.sign(
                akselin_kallistuma
            )
        )

        lämpötila_muutos = cp.where(
            kesä_maski >= 0.0,
            mannermaisuus_indeksi * 12.0,
            mannermaisuus_indeksi * -22.0
        )

        # ========================================================
        # PILVISYYDEN JÄÄHDYTYS
        # ========================================================

        pilvisyys_dt = sade / 100.0

        # ========================================================
        # LOPULLINEN LÄMPÖTILA
        # ========================================================

        kuukausi_lampotilat[
            kk
        ] = (
            perus_temp
            - korkeus_vaikutus
            + lämpötila_muutos
            - pilvisyys_dt
        )

    # ============================================================
    # GPU SYNKRONOINTI
    # ============================================================

    cp.cuda.Stream.null.synchronize()

    # ============================================================
    # GPU -> CPU
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
                kuukausi_merivirta
            ),
        )

    # ============================================================
    # JÄTETÄÄN GPU-MUISTIIN
    # ============================================================

    return (
        kuukausi_lampotilat,
        kuukausi_sateet,
        kuukausi_tuuli_suunta,
        kuukausi_tuuli_voima,
        kuukausi_merivirta,
    )




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



def plot_precip_raster(data, title="Vuotuinen sademäärä"):
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

def plot_tmean_raster(data, title="Mean temp (tmean) Celsius"):
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





def mallinna_merijaa_cpu(korkeuskartta, maa_maski, kuukausi_lampotilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg):
    """
    Laskee merijään paksuuden ja peittävyyden ottaen huomioon termodynamiikan
    sekä tuulen ja merivirtojen aiheuttaman jään liikkeen (ajojää).
    """
    meri_maski = korkeuskartta <= 0
    shape_3d = kuukausi_lampotilat.shape
    jaa_paksuus_vuosi = np.zeros(shape_3d)
    jaa_peittavyys_vuosi = np.zeros(shape_3d)
    
    nykyinen_paksuus = np.zeros_like(korkeuskartta, dtype=float)
    stefan_vakio = 2.0
    paivat_kk = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    
#plt.imshow(merijaa_peittävyys_vuosi[6])
    # Määritetään tuulen suuntavektorit (U = Länsi-Itä, V = Etelä-Pohjoinen)
    # tuuli_suunta_base on Länsituuli (positiivinen = itään päin)
    tuuli_u = tuuli_suunta_base * 1.5  # Nopeuskerroin siirtoa varten
    
    # Coriolis-ilmiö kääntää tuulta ja virtoja pohjoisella/eteläisellä pallonpuoliskolla (Pohjoinen > 0)
    tuuli_v = np.sin(np.radians(lat_grid_deg)) * tuuli_u * 0.5
    
    for kk in range(12):
        T_ilma = kuukausi_lampotilat[kk]
        Sade = kuukausi_sateet[kk]
        T_meri_eff = T_ilma + meri_anomalia
        
        # --- 1. TERMODYNAMIIKKA ---
        pakkasaste_paivat = np.zeros_like(korkeuskartta)
        sulamis_paivat = np.zeros_like(korkeuskartta)
        T_jaatyminen = -1.8
        
        jaatymismaski = T_meri_eff < T_jaatyminen
        pakkasaste_paivat[jaatymismaski] = (T_jaatyminen - T_meri_eff[jaatymismaski]) * paivat_kk[kk]
        sulamis_paivat[~jaatymismaski] = (T_meri_eff[~jaatymismaski] - T_jaatyminen) * paivat_kk[kk]
        
        lumi_eriste = 1.0 + (Sade * 0.005)  
        kasvu = np.sqrt(nykyinen_paksuus**2 + (stefan_vakio * pakkasaste_paivat) / lumi_eriste) - nykyinen_paksuus
        sulaminen = sulamis_paivat * 0.015
        
        nykyinen_paksuus = np.clip(nykyinen_paksuus + kasvu - sulaminen, 0, 10.0)
        
        # --- 2. DYNAMIIKKA (Ajojää ja siirtymä) ---
        # Lasketaan jään liikenopeus (vektori), johon vaikuttaa tuuli ja merivirran suunta rannikoilla
        # Merivirran suunta myötäilee tuulta, mutta anomalia muuttaa sen voimakkuutta
        liike_u = tuuli_u + (meri_anomalia * 0.2)
        liike_v = tuuli_v
        
        # Keskimääräinen siirtymä kyseisessä kuukaudessa (pikseleinä tai indekseinä)
        # Skaalataan siirto sopivaksi matriisille (esim. max 1-2 pikseliä per kuukausi)
        shift_x = int(np.clip(np.mean(liike_u), -2, 2))
        shift_y = int(np.clip(np.mean(liike_v), -2, 2))
        
        if shift_x != 0 or shift_y != 0:
            # Siirretään jäämattoa tuulen ja virran mukana
            liikutettu_paksuus = shift(nykyinen_paksuus, shift=(shift_y, shift_x), cval=0.0)
            
            # TÖRMÄYS RANNIKKOON (Pakkautuminen):
            # Jos jää liikkuu maata (maa_maski) päin, se ei katoa vaan pakkautuu rannikon edustalle
            rannikko_tormays = liikutettu_paksuus * maa_maski
            
            # Palautetaan rannikolle törmännyt jää takaisin mereen rannikon viereiseen soluun (pakkautuminen)
            if np.any(rannikko_tormays > 0):
                # Heijastetaan/pakkautetaan rannikon edustalle korottamalla paksuutta
                nykyinen_paksuus = shift(liikutettu_paksuus * meri_maski, shift=(-shift_y, -shift_x), cval=0.0)
                nykyinen_paksuus += shift(rannikko_tormays, shift=(-shift_y, -shift_x), cval=0.0) * 1.5
            else:
                nykyinen_paksuus = liikutettu_paksuus * meri_maski
                
        # --- 3. PEITTÄVYYS ---
        # Peruspeittävyys paksuudesta
        peittavyys = np.clip(nykyinen_paksuus / 0.4, 0.0, 1.0)
        
        # Tuulen mekaaninen hajotus avomerellä (suuri tuuli avaa railoja)
        tuuli_rasitus = np.abs(tuuli_suunta_base) * 0.15
        peittavyys = np.clip(peittavyys - tuuli_rasitus, 0.0, 1.0)
        
        # Tyhjennetään maaalueet varmuuden vuoksi
        nykyinen_paksuus[~meri_maski] = 0
        peittavyys[~meri_maski] = 0
        
        jaa_paksuus_vuosi[kk] = nykyinen_paksuus
        jaa_peittavyys_vuosi[kk] = peittavyys
        
    return jaa_paksuus_vuosi, jaa_peittavyys_vuosi




def aja_jaatikko_malli_cpu(korkeus_metreina, kuukausi_lampotilat, kuukausi_sateet, vuodet=1):
    """
    Laskee mannerjäätikön, lumen ja jään paksuuden kehityksen SEKÄ jään virtauksen alaville maille.
    """
    Y, X = korkeus_metreina.shape
    
    # Alustetaan muuttujat nolliksi
    lumi_paksuus = np.zeros((Y, X))
    jaa_paksuus = np.zeros((Y, X))
    
    # Mallin parametrit
    LUMEN_TIHEYS = 300.0
    JAAN_TIHEYS = 917.0
    LUMI_KYNNYS = 2.0          
    SULAMIS_KERROIN = 0.005    
    
    # --- VIRTAUSPARAMETRIT ---
    KRIITTINEN_PAKSUUS = 40.0   # Jää alkaa virrata kun paksuus ylittää tämän (metreinä)
    VIRTAUS_NOPEUS = 0.02       # Kuvailee jään juoksevuutta/valumisnopeutta naapureihin per kuukausi
    
    for vuosi in range(vuodet):
        for kk in range(12):
            T = kuukausi_lampotilat[kk, :, :]
            Sade = kuukausi_sateet[kk, :, :]
            
            # 1. Kertymä (Akkumulaatio)
            sade_lumena_m = np.where(T < 0, Sade / LUMEN_TIHEYS, 0.0)
            lumi_paksuus += sade_lumena_m
            
            # 2. Sulaminen (Ablation)
            potentiaalinen_sulaminen = np.where(T > 0, T * SULAMIS_KERROIN * 30, 0.0)
            sulava_lumi = np.minimum(lumi_paksuus, potentiaalinen_sulaminen)
            lumi_paksuus -= sulava_lumi
            
            jaljella_sulamista = potentiaalinen_sulaminen - sulava_lumi
            sulava_jaa = np.minimum(jaa_paksuus, jaljella_sulamista)
            jaa_paksuus -= sulava_jaa
            
            # 3. Jäätymisprosessi (Firnifikaatio)
            ylimaara_lumi = np.maximum(0.0, lumi_paksuus - LUMI_KYNNYS)
            lumi_paksuus = np.minimum(lumi_paksuus, LUMI_KYNNYS)
            uusi_jaa = (ylimaara_lumi * LUMEN_TIHEYS) / JAAN_TIHEYS
            jaa_paksuus += uusi_jaa
            
            # --- 4. JÄÄN VIRTAUS JA VALUMINEN (UUSI OSIO) ---
            # Jäätikön pinnan kokonaiskorkeus määrittää mihin suuntaan jää valuu
            pinnan_korkeus = korkeus_metreina + jaa_paksuus + lumi_paksuus
            
            # Lasketaan minne jää voi valua katsomalla naapurisoluja (siirros 4 suuntaan)
            # Luodaan kopio jään paksuudesta siirtojen laskentaa varten
            uusi_jaa_paksuus = jaa_paksuus.copy()
            
            # Tarkistetaan virtaus vain niissä pisteissä, joissa jää ylittää kriittisen rajan
            virtaava_jaa_maski = jaa_paksuus > KRIITTINEN_PAKSUUS
            
            if np.any(virtaava_jaa_maski):
                # Käydään läpi neljä ilmansuuntaa (Y+1, Y-1, X+1, X-1)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    # Vieritetty korkeus- ja jääkartta naapurin suuntaan
                    naapurin_korkeus = np.roll(pinnan_korkeus, shift=(-dy, -dx), axis=(0, 1))
                    
                    # Korkeusero (positiivinen, jos nykyinen piste on korkeammalla kuin naapuri)
                    korkeusero = pinnan_korkeus - naapurin_korkeus
                    
                    # Jäätä valuu vain jos ollaan korkeammalla JA jää ylittää kriittisen paksuuden
                    valuva_massa = np.where((korkeusero > 0) & virtaava_jaa_maski, 
                                            korkeusero * VIRTAUS_NOPEUS, 0.0)
                    
                    # Rajoitetaan valuminen siten, ettei pisteestä lähde enemmän jäätä kuin mitä kriittisen rajan yli on
                    maksimi_valuma = (jaa_paksuus - KRIITTINEN_PAKSUUS) / 4.0
                    valuva_massa = np.minimum(valuva_massa, maksimi_valuma)
                    
                    # Vähennetään lähtevä jää nykyisestä solusta ja lisätään se naapuriin
                    uusi_jaa_paksuus -= valuva_massa
                    uusi_jaa_paksuus += np.roll(valuva_massa, shift=(dy, dx), axis=(0, 1))
                
                # Päivitetään jään paksuus virtauslaskennan jälkeen
                jaa_paksuus = np.maximum(0.0, uusi_jaa_paksuus)

    # 5. Lopputulokset
    jaan_huippu_metreina = korkeus_metreina + jaa_paksuus + lumi_paksuus
    return lumi_paksuus, jaa_paksuus, jaan_huippu_metreina



import cupy as cp


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



###############################
## main program #####################



kartantekija = KartanTekijaCuPy(
    leveys=leveys,
    korkeus=korkeus,
    syvin_kohta=syvin_kohta,
    korkein_kohta=korkein_kohta,
    manner_osuus=manner_osuus,
    seed=seed1,
)

kartantekija.generoi()

#korkeus_syvyyskartta = (
#    kartantekija.korkeus_syvyyskartta
#)

#korkeuskartta = (
#    kartantekija.korkeuskartta
#)

#maa_maski = (
#    kartantekija.maa_maski
#)

#meri_maski = (
#    kartantekija.meri_maski
#)

korkeus_syvyyskartta = kartantekija.korkeus_syvyyskartta.get()
korkeuskartta = kartantekija.korkeuskartta.get()
maa_maski = kartantekija.maa_maski.get()
meri_maski = kartantekija.meri_maski.get()

print(korkeus_syvyyskartta.shape)
print(korkeuskartta.shape)
print(maa_maski.shape)
print(meri_maski.shape)

#plt.imshow(korkeuskartta )
#plt.show()

#quit(-1)


# KÄYTTÖESIMERKKI:
# oletetaan että 'korkeus' on 2D-numpy-taulukko ja 'vari_kuva' on RGB-kuva (0-1 float)
varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)
# valmis_kuva = sekoita_soft_light(vari_kuva, varjo_maski)

# =====================================================================
# 3. MERIETÄISYYSRASATERIN LASKENTA
# =====================================================================
lon_grid_deg, lat_grid_deg = np.meshgrid(pituusasteet, leveysasteet)

korkeus, leveys = maa_maski.shape
lats = np.linspace(90, -90, korkeus)   # Y-akseli (pohjoisesta etelään)
lons = np.linspace(-180, 180, leveys) # X-akseli (lännestä itään)
lon_grid, lat_grid = np.meshgrid(lons, lats)


import numpy as np
from sklearn.neighbors import BallTree


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

# ------------------------------------------------------------
# Vain maapikselit
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# BallTree
# ------------------------------------------------------------

#tree = BallTree(
#    meri_coords,
#    metric="haversine",
#)

# ------------------------------------------------------------
# Lähin meri jokaiselle maapikselille
# ------------------------------------------------------------

#distances, _ = tree.query(
#    maa_coords,
#    k=1,
#)

# ------------------------------------------------------------
# Kilometreiksi
# ------------------------------------------------------------

#maapallon_sade_km = 6371.0088

#maa_etaisyydet = (
#    distances[:, 0]
#    * maapallon_sade_km
#)

# ------------------------------------------------------------
# Takaisin rasteriksi
# ------------------------------------------------------------

#etaisyys_meresta_km = np.zeros(
#    (korkeus, leveys),
#    dtype=np.float32,
#)

#etaisyys_meresta_km[
#    maa_y,
#    maa_x
#] = maa_etaisyydet.astype(
#    np.float32
#)


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






# =====================================================================
# 3.5 GLOBAALIT TUULET JA MERIVIRRAT
# =====================================================================
# Mallinnetaan vallitsevat tuulet leveysasteittain (1=Länsituuli, -1=Itätuuli/Pasaati)
# Pasaatit: 0° - 30° (itästä länteen, negatiivinen) | Länsituulet: 30° - 60° (lännestä itään, positiivinen)
#tuuli_suunta_base = -np.sin(np.radians(lat_grid_deg * 3.0))

# Lasketaan orografinen sateenvarjo (Rain Shadow) tuulen suunnasta riippuen.
# Käytetään NumPyn gradienttia (korkeuden muutos) itä-länsi-suunnassa (akseli 1).
korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)

# Tuulen ja maaston muodon yhteisvaikutus
#orografinen_pakote = korkeus_gradientti_x * tuuli_suunta_base

# Mallinnetaan merivirtojen anomaliat (Lämpimät vs kylmät virrat rannikoilla)
#meri_anomalia = np.zeros_like(korkeuskartta)
#if np.any(maa_maski):
#    virta_pakote = np.sin(np.radians(lat_grid_deg * 2.0)) * np.sign(tuuli_suunta_base)
#    # Suodatetaan vaikutus vain rannikoiden läheisyyteen (maks 400km merestä)
#    meri_anomalia = virta_pakote * np.exp(-etaisyys_meresta_km / 200.0) * 4.0





# =====================================================================
# 4. KUUKAUSITTAINEN LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS (12 KUUKAUTTA)
# =====================================================================
kuukausi_lämpötilat = np.zeros((12, korkeus, leveys))
kuukausi_sateet = np.zeros((12, korkeus, leveys))

korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)


import numpy as np

# Oletetaan, että dem-taulukkosi on nimeltään "korkeus_metrina"
# Muoto: [korkeus, leveys], esim. [korkeus=180, leveys=360] tai vastaava resoluutio
# Alue: [-180, 180, -90, 90]

# 1. Haetaan taulukon mitat
n_lats, n_lons = korkeuskartta.shape

# 2. Luodaan leveysastevektori (-90:stä 90:een asti) taulukon pystyrivien mukaan
# Huom: varmista, alkaako taulukon ensimmäinen rivi (index 0) etelä- vai pohjoisnavalta.
# Tässä oletetaan, että se alkaa etelänavalta (-90) pohjoisnavalle (90).
lats = np.linspace(-90, 90, n_lats)

# 3. Lasketaan kosinipainot kullekin leveysasteelle (muutetaan ensin asteet radiaaneiksi)
cos_weights = np.cos(np.radians(lats))

# 4. Laajennetaan painot 2D-taulukoksi, jotta se täsmää DEM-taulukon muotoon [n_lats, n_lons]
# np.newaxis tekee vektorista sarakkeen, ja kertomalla ykkösillä se monistetaan pituusasteille
weight_grid = cos_weights[:, np.newaxis] * np.ones(n_lons)

# 5. Luodaan maskit maalle (> 0 m) ja merelle (<= 0 m)
maa_maski = np.copy(korkeuskartta) > 0
meri_maski = np.copy(korkeuskartta) <= 0

#plt.imshow(meri_maski)
#plt.show()


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

#gkosteus_kerroin=meri_prosentti/70
#gkosteus_kerroin=np.pow((meri_prosentti/70),1) #3 hatusta vedetty, meri alle 70 %
#if(meri_prosentti>80):
#	gkosteus_kerroin=np.pow((meri_prosentti/70),2) ## hatusta vedetty
#global_moisture_coeff=gkosteus_kerroin
global_moisture_coeff=f_precip_relto_earthlike(meri_prosentti/71.0)
 
    
    
#kuukausi_lämpötilat, kuukausi_sateet = laske_ilmasto(
#   lat_grid_deg=lat_grid_deg,
#    korkeuskartta=korkeuskartta,
#    korkeus_gradientti_x=korkeus_gradientti_x,
#    maa_maski=maa_maski,
#    meri_maski=meri_maski,
#    etaisyys_meresta_km=etaisyys_meresta_km,
#    tilt_planet_axis=tilt_planet_axis,
#    ecc_planet=ecc_planet,
#    mvelp_planet=mvelp_planet,
#    global_moisture_coeff=global_moisture_coeff,
#    tmax_planet=50.0,
#    K=0.73,
#)

#_ = warmup_numba(
#    lat_grid_deg,
#    korkeuskartta,
#    korkeus_gradientti_x,
#    maa_maski,
#    meri_maski,
#    etaisyys_meresta_km,
#    tilt_planet_axis,
#    ecc_planet,
#    mvelp_planet,
#    global_moisture_coeff,
#)

#(
#    kuukausi_lämpötilat,
#    kuukausi_sateet,
#    kuukausi_tuuli_suunta,
#    kuukausi_tuuli_voima,
#    kuukausi_merivirta,
#) = laske_ilmasto_numba(
#    lat_grid_deg=lat_grid_deg,
#    korkeuskartta=korkeuskartta,
#    korkeus_gradientti_x=korkeus_gradientti_x,
#    maa_maski=maa_maski,
#    meri_maski=meri_maski,
#    etaisyys_meresta_km=etaisyys_meresta_km,

#    tilt_planet_axis=tilt_planet_axis,
#    ecc_planet=ecc_planet,
#    mvelp_planet=mvelp_planet,

#    global_moisture_coeff=global_moisture_coeff,
#)

import time

print ("GPU warming up ...")

# Ensimmäinen ajo = warm-up
laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    global_moisture_coeff,
)


print ("GPU calculation ...")
# Varsinainen mittaus
cp.cuda.Stream.null.synchronize()

t0 = time.perf_counter()

#tulos = laske_ilmasto_gpu(
#    lat_grid_deg,
#    korkeuskartta,
#    korkeus_gradientti_x,
#    maa_maski,
#    meri_maski,
#    etaisyys_meresta_km,
#    tilt_planet_axis,
#    ecc_planet,
#    mvelp_planet,
#    global_moisture_coeff,
#)
kuukausi_lämpötilat_gpu, kuukausi_sateet_gpu, \
kuukausi_tuuli_suunta_gpu, kuukausi_tuuli_voima_gpu, \
kuukausi_merivirta_gpu = laske_ilmasto_gpu(
    lat_grid_deg,
    korkeuskartta,
    korkeus_gradientti_x,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    global_moisture_coeff=global_moisture_coeff,
)

cp.cuda.Stream.null.synchronize()

dt = time.perf_counter() - t0

print(f"GPU-laskenta: {dt:.4f} s")

kuukausi_lämpötilat=kuukausi_lämpötilat_gpu
kuukausi_sateet=kuukausi_sateet_gpu

kuukausi_tuuli_suunta=kuukausi_tuuli_suunta_gpu
kuukausi_tuuli_voima=kuukausi_tuuli_voima_gpu
kuukausi_merivirta=kuukausi_merivirta_gpu


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




#plt.imshow(merijaa_peittavyys_vuosi[6])
#plt.imshow(merijaa_paksuus_vuosi[1])

#plt.contour(maa_maski, levels=[0,1,2], colors=["red"], lw=2)
#plt.show()

#quit(-1)



# Tiivistetään kuukausitiedot vuositason rastereiksi Köppen-laskentaa varten
lämpötila_ka = np.mean(kuukausi_lämpötilat, axis=0)
lämpötila_min = np.min(kuukausi_lämpötilat, axis=0)
lämpötila_max = np.max(kuukausi_lämpötilat, axis=0)

vuosi_sademäärä_kartta = np.sum(kuukausi_sateet, axis=0)
kuivimman_kuukauden_sade = np.min(kuukausi_sateet, axis=0)
sateisimman_kuukauden_sade = np.max(kuukausi_sateet, axis=0)


# Nollataan sateet merialueilta puhtaita maarasteritietoja varten
vuosi_sademäärä_kartta[~maa_maski] = 0

#planet_mean_temp_simu=np.mean(lämpötila_ka*weight_grid) 
planet_mean_temp_simu=np.mean(lämpötila_ka*weight_grid) 
planet_annual_precipitation=np.mean(vuosi_sademäärä_kartta*weight_grid) 

print(planet_mean_temp_simu)
print(planet_annual_precipitation)

#plt.imshow(lämpötila_ka)
#plt.imshow(kuukausi_lämpötilat[0])
#plt.imshow(kuukausi_sateet[6])
plot_tmean_raster(lämpötila_ka, title="Mean temperature (tmean)")
plot_precip_raster(vuosi_sademäärä_kartta, title="Annual precipitaion in mm")
		
#plt.show()
#quit(-1)


# =====================================================================
# 5. KÖPPENIN TARKEMMAT ALAVYÖHYKEMASKIT (Dynaaminen kuukausivaihtelu)
# =====================================================================
ilmastovyohyke_tarkka = np.zeros_like(lämpötila_ka, dtype=int)

maski_E = maa_maski & (lämpötila_max < 10.0)
maski_ET = maa_maski & (lämpötila_max < 10.0)
maski_EF = maa_maski & (lämpötila_max < 0.0)
# B - KUIVAT ILMASTOT (Dynaamiset rajat Köppenin virallisen kaavan mukaan)
suojattu_temp = np.maximum(lämpötila_ka, 0.0)
aro_raja = 20.0 * suojattu_temp
aavikko_raja = 10.0 * suojattu_temp

maski_BWh = maa_maski & (~maski_E) & (vuosi_sademäärä_kartta < aavikko_raja) & (lämpötila_ka >= 18.0)
maski_BWk = maa_maski & (~maski_E) & (vuosi_sademäärä_kartta < aavikko_raja) & (lämpötila_ka < 18.0)
maski_BSh = maa_maski & (~maski_E) & (vuosi_sademäärä_kartta >= aavikko_raja) & (vuosi_sademäärä_kartta < aro_raja) & (lämpötila_ka >= 18.0)
maski_BSk = maa_maski & (~maski_E) & (vuosi_sademäärä_kartta >= aavikko_raja) & (vuosi_sademäärä_kartta < aro_raja) & (lämpötila_ka < 18.0)

maski_B_kaikki = maski_BWh | maski_BWk | maski_BSh | maski_BSk

# A - TROPIIKKI (Aito kuivimman kuukauden erottelu)
pohja_A = maa_maski & (~maski_E) & (~maski_B_kaikki) & (lämpötila_min >= 18.0)
maski_Af = pohja_A & (kuivimman_kuukauden_sade >= 60.0)
maski_Aw = pohja_A & (kuivimman_kuukauden_sade < 60.0) & (kuivimman_kuukauden_sade < (100.0 - vuosi_sademäärä_kartta / 25.0))
maski_Am = pohja_A & (~maski_Af) & (~maski_Aw)

# C - LAUHKEA ILMASTO
pohja_C = maa_maski & (~maski_E) & (~maski_B_kaikki) & (~pohja_A) & (lämpötila_min >= -3.0) & (lämpötila_min < 18.0)
maski_Csa_Csb = pohja_C & (kuivimman_kuukauden_sade < 40.0) & (kuivimman_kuukauden_sade < sateisimman_kuukauden_sade / 3.0)
maski_Cfb = pohja_C & (~maski_Csa_Csb) & (etaisyys_meresta_km <= 400.0)
maski_Cfa_Cwa = pohja_C & (~maski_Csa_Csb) & (~maski_Cfb)

# D - MANNERILMASTOT (Jaettu lehtimetsä-mannerilmastoon ja kylmään Taigaan)
pohja_D = maa_maski & (~maski_E) & (~maski_B_kaikki) & (~pohja_A) & (~pohja_C) & (lämpötila_min < -3.0) & (lämpötila_max >= 10.0)
maski_Dfa_Dfb = pohja_D & (lämpötila_max >= 22.0)
maski_Dfc_Dfd = pohja_D & (lämpötila_max < 22.0)  & (~maski_E)

# Numerointi rasteripikseleille
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

im = plt.imshow(ilmastovyohyke_tarkka, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])

variables = {
    "keski_lampotila": lämpötila_ka,
    "keski_sademaara": vuosi_sademäärä_kartta,
    "maaston_korkeus_m": korkeuskartta,
    "lampotila_min": lämpötila_min,
    "lampotila_max": lämpötila_max,
    "kuivimman_kuukauden_sade": kuivimman_kuukauden_sade,
    "sateisimman_kuukauden_sade": sateisimman_kuukauden_sade,
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

#new_rgb = histrgb_im(
#    im,
#    variables,
#    variable_names,
#    max_iter=100,
#    learning_rate=0.05,
#    max_leaf_nodes=31,
#    min_samples_leaf=30,
#    l2_regularization=1.0,
 #   train_samples=50000,
#)

#ax.imshow(new_rgb)


im2 = plt.imshow(new_rgb, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])

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
