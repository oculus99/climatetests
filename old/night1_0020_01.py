
#############################
#
## simple planet climate 
#
## 20.09.2026 0000.00020.01
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
korkeus = 180*1
leveys = 360*1

seed1=9 ##ok
seed1 = 55

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


import numpy as np


class KartanTekija:
    """
    Pallomaisen maailman proseduraalinen korkeuskartan generaattori.
    """

    def __init__(
        self,
        leveys=360,
        korkeus=180,

        #syvin_kohta=-11000,
        #korkein_kohta=8848,
        syvin_kohta=-3000,
        korkein_kohta=1000,
        manner_osuus=0.30,

        #suurten_mantereiden_osuus=0.85,
        suurten_mantereiden_osuus=0.99,
        mantereen_koko=0.55,
        #mantereen_muoto=0.15,
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

        self.maa_maski = None
        self.meri_maski = None

        self.manner_maski = None
        self.saari_maski = None

        self.tasanko_maski = None
        self.ylanko_maski = None
        self.vuoristo_maski = None

        self.vedenpinta = None
        self.raakakorkeuskartta = None

        self.debug_data = {}

    # ==========================================================
    # Pallokoordinaatit
    # ==========================================================

    def _pallokoordinaatit(self):

        leveysasteet = np.linspace(
            -90.0,
            90.0,
            self.korkeus
        )

        pituusasteet = np.linspace(
            -180.0,
            180.0,
            self.leveys,
            endpoint=False
        )

        lat = np.radians(leveysasteet)
        lon = np.radians(pituusasteet)

        lon_grid, lat_grid = np.meshgrid(
            lon,
            lat
        )

        x = (
            np.cos(lat_grid)
            * np.cos(lon_grid)
        )

        y = (
            np.cos(lat_grid)
            * np.sin(lon_grid)
        )

        z = np.sin(lat_grid)

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

        px = x * grid_size
        py = y * grid_size
        pz = z * grid_size

        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        z0 = np.floor(pz).astype(np.int64)

        fx = px - x0
        fy = py - y0
        fz = pz - z0

        fx = fx * fx * (
            3.0 - 2.0 * fx
        )

        fy = fy * fy * (
            3.0 - 2.0 * fy
        )

        fz = fz * fz * (
            3.0 - 2.0 * fz
        )

        def random_value(
            ix,
            iy,
            iz,
        ):

            h = (
                ix * 374761393
                + iy * 668265263
                + iz * 2147483647
                + seed * 1274126177
            )

            h = h ^ (
                h >> 13
            )

            h = h * 1274126177

            h = h ^ (
                h >> 16
            )

            h = h & 0xffffffff

            return (
                h.astype(np.float64)
                / 2147483647.5
                - 1.0
            )

        c000 = random_value(
            x0,
            y0,
            z0
        )

        c100 = random_value(
            x0 + 1,
            y0,
            z0
        )

        c010 = random_value(
            x0,
            y0 + 1,
            z0
        )

        c110 = random_value(
            x0 + 1,
            y0 + 1,
            z0
        )

        c001 = random_value(
            x0,
            y0,
            z0 + 1
        )

        c101 = random_value(
            x0 + 1,
            y0,
            z0 + 1
        )

        c011 = random_value(
            x0,
            y0 + 1,
            z0 + 1
        )

        c111 = random_value(
            x0 + 1,
            y0 + 1,
            z0 + 1
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
        octaves=6,
        base_grid=4,
    ):

        value = np.zeros_like(
            x,
            dtype=np.float64
        )

        amplitude = 1.0
        frequency = 1.0
        amplitude_sum = 0.0

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

            frequency *= 2.0
            amplitude *= 0.5

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
        octaves=5,
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
            - np.abs(noise)
        )

        return ridge ** 2.0

    # ==========================================================
    # Normalisointi
    # ==========================================================

    def _normalisoi_01(
        self,
        value,
    ):

        value = np.asarray(
            value,
            dtype=np.float64
        )

        minimum = np.min(value)
        maximum = np.max(value)

        if maximum <= minimum:

            return np.zeros_like(
                value,
                dtype=np.float64
            )

        return (
            (value - minimum)
            / (maximum - minimum)
        )

    # ==========================================================
    # Smoothstep
    # ==========================================================

    def _smoothstep(
        self,
        value,
    ):

        value = np.clip(
            value,
            0.0,
            1.0
        )

        return (
            value
            * value
            * (3.0 - 2.0 * value)
        )

    # ==========================================================
    # Suuret manteret
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
            octaves=4,
            base_grid=2,
        )

        large = self._normalisoi_01(
            large
        )

        if self.mantereen_muoto > 0:

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

    def _generoi_saaret(
        self,
        x,
        y,
        z,
    ):

        frequency = (
            1.0
            * self.saaren_koko
        )

        islands = self._fbm(
            x * frequency,
            y * frequency,
            z * frequency,
            seed=self.seed + 300,
            octaves=5,
            base_grid=3,
        )

        return self._normalisoi_01(
            islands
        )

    # ==========================================================
    # Manner + saaret
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

        # ------------------------------------------------------
        # Tavoiteltu kokonaismaaosuus
        # ------------------------------------------------------

        tavoite_maaosuus = np.clip(
            self.manner_osuus,
            0.001,
            0.999
        )

        suuri_osuus = np.clip(
            self.suurten_mantereiden_osuus,
            0.0,
            1.0
        )

        tavoite_manner_osuus = (
            tavoite_maaosuus
            * suuri_osuus
        )

        # ------------------------------------------------------
        # Manner
        # ------------------------------------------------------

        manner_threshold = np.quantile(
            large,
            1.0 - tavoite_manner_osuus
        )

        manner_maski = (
            large >= manner_threshold
        )

        # ------------------------------------------------------
        # Saarien peruskynnys
        # ------------------------------------------------------

        maara = np.clip(
            self.saaren_maara,
            0.0,
            2.0
        )

        saari_threshold = (
            0.72
            - 0.20 * maara
        )

        saari_threshold = np.clip(
            saari_threshold,
            0.30,
            0.85
        )

        saari_maski = (
            saari_arvot
            >= saari_threshold
        )

        # Mantereen päälle ei tehdä saarta.

        saari_maski = (
            saari_maski
            & ~manner_maski
        )

        # ------------------------------------------------------
        # Rajoitetaan saarten kokonaismäärää
        # ------------------------------------------------------

        nykyinen_manner_osuus = np.mean(
            manner_maski
        )

        sallittu_saarialue = max(
            tavoite_maaosuus
            - nykyinen_manner_osuus,
            0.0
        )

        nykyinen_saariosuus = np.mean(
            saari_maski
        )

        if (
            sallittu_saarialue <= 0.0
        ):

            saari_maski[:] = False

        elif (
            nykyinen_saariosuus
            > sallittu_saarialue
        ):

            saarien_arvot = (
                saari_arvot[
                    saari_maski
                ]
            )

            pidettava_osuus = (
                sallittu_saarialue
                / nykyinen_saariosuus
            )

            pidettava_osuus = np.clip(
                pidettava_osuus,
                0.0,
                1.0
            )

            saarien_kynnys = np.quantile(
                saarien_arvot,
                1.0 - pidettava_osuus
            )

            saari_maski = (
                saari_maski
                & (
                    saari_arvot
                    >= saarien_kynnys
                )
            )

        # ------------------------------------------------------
        # Lopullinen maa
        # ------------------------------------------------------

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
            octaves=4,
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        plateau = (
            1.0
            - np.abs(
                terrain * 2.0
                - 1.0
            )
        )

        return plateau ** 1.5

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
            octaves=4,
            base_grid=3,
        )

        terrain = self._normalisoi_01(
            terrain
        )

        plateau = np.clip(
            (terrain - 0.52)
            / 0.48,
            0.0,
            1.0
        )

        return self._smoothstep(
            plateau
        )

    # ==========================================================
    # Vuoristot
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
            octaves=5,
            base_grid=3,
        )

        return self._normalisoi_01(
            mountains
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
            octaves=6,
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
            octaves=5,
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
        # Maa
        # ------------------------------------------------------

        maan_korkeus = np.full_like(
            tasanko,
            0.22,
            dtype=np.float64
        )

        maan_korkeus += (
            self.tasangon_voima
            * tasanko
        )

        maan_korkeus += (
            self.ylangon_voima
            * ylanko
        )

        maan_korkeus += (
            self.vuoriston_voima
            * vuoristo
        )

        maan_korkeus += (
            self.pieni_maasto_voima
            * pieni
        )

        maan_korkeus -= (
            saari_maski * 0.08
        )

        # ------------------------------------------------------
        # Merenpohja
        # ------------------------------------------------------

        meren_korkeus = (
            -0.30
            - (
                self.merenpohjan_voima
                * merenpohja
            )
        )

        # ------------------------------------------------------
        # Rannikko
        # ------------------------------------------------------

        rannikko_noise = self._fbm(
            x * 1.5,
            y * 1.5,
            z * 1.5,
            seed=self.seed + 6000,
            octaves=3,
            base_grid=3,
        )

        rannikko_noise = (
            rannikko_noise + 1.0
        ) * 0.5

        rannikko_noise = self._smoothstep(
            rannikko_noise
        )

        maan_korkeus -= (
            rannikko_noise
            * self.rannikon_loivuus
            * 0.08
        )

        terrain = np.where(
            maa_maski,
            maan_korkeus,
            meren_korkeus,
        )

        if self.debug:

            self.debug_data["tasanko"] = (
                tasanko
            )

            self.debug_data["ylanko"] = (
                ylanko
            )

            self.debug_data["vuoristo"] = (
                vuoristo
            )

            self.debug_data["pieni_maasto"] = (
                pieni
            )

            self.debug_data["merenpohja"] = (
                merenpohja
            )

            self.debug_data["maan_korkeus"] = (
                maan_korkeus
            )

            self.debug_data["meren_korkeus"] = (
                meren_korkeus
            )

        return terrain

    # ==========================================================
    # Korkeuksien skaalaus
    # ==========================================================

    def _skaalaa_korkeudet(
        self,
        raakakorkeuskartta,
        maa_maski,
        meri_maski,
    ):

        tulos = np.zeros_like(
            raakakorkeuskartta,
            dtype=np.float64
        )

        # ------------------------------------------------------
        # Maa
        # ------------------------------------------------------

        maa_arvot = raakakorkeuskartta[
            maa_maski
        ]

        if maa_arvot.size > 0:

            maa_min = np.min(
                maa_arvot
            )

            maa_max = np.max(
                maa_arvot
            )

            if maa_max > maa_min:

                normalized = (
                    maa_arvot - maa_min
                ) / (
                    maa_max - maa_min
                )

                normalized = (
                    #normalized ** 0.85
                    normalized ** 3
                )

                tulos[
                    maa_maski
                ] = (
                    normalized
                    * self.korkein_kohta
                )

            else:

                tulos[
                    maa_maski
                ] = 0.0

        # ------------------------------------------------------
        # Meri
        # ------------------------------------------------------

        meri_arvot = raakakorkeuskartta[
            meri_maski
        ]

        if meri_arvot.size > 0:

            meri_min = np.min(
                meri_arvot
            )

            meri_max = np.max(
                meri_arvot
            )

            if meri_max > meri_min:

                normalized = (
                    meri_arvot - meri_min
                ) / (
                    meri_max - meri_min
                )

                normalized = (
                    1.0 - normalized
                )

                tulos[
                    meri_maski
                ] = (
                    normalized
                    * self.syvin_kohta
                )

            else:

                tulos[
                    meri_maski
                ] = 0.0

        return tulos

    # ==========================================================
    # Generointi
    # ==========================================================

    def generoi(self):

        x, y, z = (
            self._pallokoordinaatit()
        )

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
            .astype(np.float32)
        )

        self.vedenpinta = 0.0

        korkeus_syvyyskartta = (
            self._skaalaa_korkeudet(
                raakakorkeuskartta,
                maa_maski,
                meri_maski,
            )
        )

        korkeuskartta = np.where(
            maa_maski,
            korkeus_syvyyskartta,
            0.0,
        )

        self.korkeus_syvyyskartta = (
            korkeus_syvyyskartta
            .astype(np.float32)
        )

        self.korkeuskartta = (
            korkeuskartta
            .astype(np.float32)
        )

        self.maa_maski = maa_maski
        self.meri_maski = meri_maski

        self.manner_maski = (
            manner_maski
        )

        self.saari_maski = (
            saari_maski
        )

        if self.debug:

            self.tasanko_maski = (
                self.debug_data["tasanko"]
                > 0.65
            )

            self.ylanko_maski = (
                self.debug_data["ylanko"]
                > 0.45
            )

            self.vuoristo_maski = (
                self.debug_data["vuoristo"]
                > 0.55
            )

            self.debug_data["maa_maski"] = (
                maa_maski
            )

            self.debug_data["meri_maski"] = (
                meri_maski
            )

            self.debug_data["manner_maski"] = (
                manner_maski
            )

            self.debug_data["saari_maski"] = (
                saari_maski
            )

            self.debug_data["tasanko_maski"] = (
                self.tasanko_maski
            )

            self.debug_data["ylanko_maski"] = (
                self.ylanko_maski
            )

            self.debug_data["vuoristo_maski"] = (
                self.vuoristo_maski
            )

        return self

    # ==========================================================
    # Tulokset
    # ==========================================================

    def tulokset(self):

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
        }

        if self.debug:
            tulos["debug"] = self.debug_data

        return tulos


# ==============================================================
# Esimerkki
# ==============================================================

if __name__ == "__main__":

    kartantekija = KartanTekija(
        leveys=360,
        korkeus=180,

        manner_osuus=0.30,

        suurten_mantereiden_osuus=0.85,

        mantereen_koko=0.55,
        mantereen_muoto=0.15,

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
        debug=True,
    )

    kartantekija.generoi()

    tulokset = kartantekija.tulokset()

    print(
        "Kartta generoitu."
    )

    print(
        "Koko:",
        tulokset[
            "korkeuskartta"
        ].shape
    )

    print(
        "Maaosuus:",
        np.mean(
            tulokset[
                "maa_maski"
            ]
        )
    )

    print(
        "Mannerosuus:",
        np.mean(
            tulokset[
                "manner_maski"
            ]
        )
    )

    print(
        "Saarialue:",
        np.mean(
            tulokset[
                "saari_maski"
            ]
        )
    )

    print(
        "Maan korkein:",
        np.max(
            tulokset[
                "korkeuskartta"
            ]
        ),
        "m"
    )

    print(
        "Meren syvin:",
        np.min(
            tulokset[
                "korkeus_syvyyskartta"
            ]
        ),
        "m"
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


def mallinna_merijaa(korkeuskartta, maa_maski, kuukausi_lampotilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg):
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




def aja_jaatikko_malli(korkeus_metreina, kuukausi_lampotilat, kuukausi_sateet, vuodet=1):
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



kartantekija = KartanTekija(
    leveys=leveys,
    korkeus=korkeus,
    syvin_kohta=syvin_kohta,
    korkein_kohta=korkein_kohta,
    manner_osuus=manner_osuus,
    seed=seed1,
)

kartantekija.generoi()

korkeus_syvyyskartta = (
    kartantekija.korkeus_syvyyskartta
)

korkeuskartta = (
    kartantekija.korkeuskartta
)

maa_maski = (
    kartantekija.maa_maski
)

meri_maski = (
    kartantekija.meri_maski
)


#plt.imshow(korkeuskartta )
#plt.show()

#quit(-1)

korkeuskartta=korkeuskartta

# KÄYTTÖESIMERKKI:
# oletetaan että 'korkeus' on 2D-numpy-taulukko ja 'vari_kuva' on RGB-kuva (0-1 float)
# varjo_maski = laske_hillshade(korkeus, azimuth=315, altitude=45)
# valmis_kuva = sekoita_soft_light(vari_kuva, varjo_maski)

# =====================================================================
# 3. MERIETÄISYYSRASATERIN LASKENTA
# =====================================================================
lon_grid_deg, lat_grid_deg = np.meshgrid(pituusasteet, leveysasteet)


import numpy as np
from sklearn.neighbors import BallTree

# 1. Luodaan koordinaattiverkot (lat/lon) rasterisi koon mukaan
# Oletetaan, että rasterin koko on (korkeus, leveys)
korkeus, leveys = maa_maski.shape
lats = np.linspace(90, -90, korkeus)   # Y-akseli (pohjoisesta etelään)
lons = np.linspace(-180, 180, leveys) # X-akseli (lännestä itään)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 2. Erotetaan meri- ja maapikselit koordinaatteina (radiaaneina Haversinea varten)
# Huom: etäisyys lasketaan MERESTÄ maalle, joten etsitään meripikselit
meri_y, meri_x = np.where(maa_maski == 0) # 0 = meri, 1 = maa
meri_coords = np.radians(np.vstack((lats[meri_y], lons[meri_x])).T)

tai_coords = np.radians(np.vstack((lat_grid.ravel(), lon_grid.ravel())).T)

# 3. Rakennetaan BallTree merikoordinaateista
# Haversine-metriikka vaatii syötteeksi (lat, lon) radiaaneina
tree = BallTree(meri_coords, metric='haversine')

# 4. Haetaan jokaiselle maailman pikselille lähin meripikseli
distances, _ = tree.query(tai_coords, k=1)

# 5. Muutetaan etäisyys radiaaneista kilometreiksi (Maapallon säde ~6371 km)
# Ja muotoillaan takaisin alkuperäisen rasterin muotoon

etaisyys_meresta_km = distances.reshape(korkeus, leveys) * maapallon_sade_km

# Maskataan merialueet nollaksi, jos halutaan vain maanpäällinen etäisyys
etaisyys_meresta_km[maa_maski == 0] = 0


# =====================================================================
# 3.5 GLOBAALIT TUULET JA MERIVIRRAT
# =====================================================================
# Mallinnetaan vallitsevat tuulet leveysasteittain (1=Länsituuli, -1=Itätuuli/Pasaati)
# Pasaatit: 0° - 30° (itästä länteen, negatiivinen) | Länsituulet: 30° - 60° (lännestä itään, positiivinen)
tuuli_suunta_base = -np.sin(np.radians(lat_grid_deg * 3.0))

# Lasketaan orografinen sateenvarjo (Rain Shadow) tuulen suunnasta riippuen.
# Käytetään NumPyn gradienttia (korkeuden muutos) itä-länsi-suunnassa (akseli 1).
korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)

# Tuulen ja maaston muodon yhteisvaikutus
orografinen_pakote = korkeus_gradientti_x * tuuli_suunta_base

# Mallinnetaan merivirtojen anomaliat (Lämpimät vs kylmät virrat rannikoilla)
meri_anomalia = np.zeros_like(korkeuskartta)
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
 
    
    
    # =================================================================
    # 3. LÄMPÖTILA
    # =================================================================



# Oletetaan, että nämä on määritelty aiemmin koodissasi:
# kuukausia = 12
# lat_grid_deg = leveysastehila (esim. -90 ... 90)
# korkeuskartta = topografiahila
# kk_meri_anomalia = merellisyyden vaikutus kullekin kuukaudelle

for kk in range(12):
    # =================================================================
    # 1. AURINKO JA ILMASTOLLINEN PÄIVÄNTASAAJA (ITCZ)
    # =================================================================
    # Auringon deklinaatio (akselin kallistuma)
    kulma = 2.0 * np.pi * (kk - 5.5) / 12.0
    akselin_kallistuma = tilt_planet_axis * np.cos(kulma)
    
    # Kaikki vyöhykkeet vaeltavat yhtenäisesti tämän pisteen ympärillä!
    ilmastollinen_päiväntasaaja = akselin_kallistuma * 0.65  # Liikkuu maks ~15 astetta
    
    # =================================================================
    # 2. DYNAAMISET TUULET
    # =================================================================
    # Suhteutetaan leveysaste ilmastolliseen päiväntasaajaan
    relatiivinen_lat = lat_grid_deg - ilmastollinen_päiväntasaaja
    abs_rel_lat = np.abs(relatiivinen_lat)
    
    # Perusaalto tuulen suunnalle
    kk_tuuli_suunta = -np.sin(np.radians(relatiivinen_lat * 6.0))
    
    # Pakotetaan ilmakehän solujen (Hadley, Ferrel, Polar) oikeat suunnat:
    # Pasaatit (0-30): itään/negatiivinen
    kk_tuuli_suunta = np.where(abs_rel_lat < 30.0, -np.abs(kk_tuuli_suunta), kk_tuuli_suunta)
    # Länsituulet (30-62): länteen/positiivinen
    kk_tuuli_suunta = np.where((abs_rel_lat >= 30.0) & (abs_rel_lat < 62.0), np.abs(kk_tuuli_suunta), kk_tuuli_suunta)
    # Napapuhurit (>62): itään/negatiivinen
    kk_tuuli_suunta = np.where(abs_rel_lat >= 62.0, -np.abs(kk_tuuli_suunta), kk_tuuli_suunta)
    
    # Korjaus eteläiselle pallonpuoliskolle suunnan pitämiseksi loogisena
    kk_tuuli_suunta = np.where(relatiivinen_lat < 0, kk_tuuli_suunta * 1.0, kk_tuuli_suunta)

    # Tuulen voimakkuus (varmistetaan rannikkotuuli aina)
    kk_tuuli_voimakkuus = np.clip(np.abs(kk_tuuli_suunta), 0.3, 1.0)

    # Orografinen pakote (paikallinen nousu/lasku tuulen suunnassa)
    kk_orografinen_pakote = korkeus_gradientti_x * kk_tuuli_suunta * kk_tuuli_voimakkuus

    # Merivirta-anomalia rannikoilla
    kk_meri_anomalia = np.zeros_like(korkeuskartta)
    if np.any(maa_maski):
        virta_pakote = np.sin(np.radians(lat_grid_deg * 2.0)) * np.sign(kk_tuuli_suunta)
        kk_meri_anomalia = virta_pakote * np.exp(-etaisyys_meresta_km / 200.0) * 4.0 * kk_tuuli_voimakkuus	
	
	
    # 1. Lasketaan kuukauden keskimääräinen kulma radalla (keskianomalia / pituusaste)
    # Jaetaan rata 12 osaan (30 astetta per kuukausi). Aloitetaan esim. tammikuusta (kulma 0)
    ratakulma_deg = kk * 30
    ratakulma_rad = np.radians(ratakulma_deg)
    
    # 2. Auringon deklinaatio (kallistuskulma kohti leveysasteita)
    # Kevätpäiväntasauksessa (esim. maaliskuu, kk=2 -> ratakulma ~60 tai siirrettynä kohdalleen)
    # Yksinkertaistettu deklinaatio kausi-ilmiölle:
    # Siirretään kulmaa niin, että kesäpäivänseisaus (maksimikallistus) osuu oikeaan kohtaan (esim. kesäkuu, kk=5)
    deklinaatio = tilt_planet_axis * np.sin(ratakulma_rad - np.radians(60)) # Kevätpäiväntasaus maaliskuussa
    
    # 3. Planeetan etäisyys Auringosta (Keplerin radan yhtälöstä Keplerin 1. lain mukaan)
    # true_anomaly (todellinen anomalia) riippuu ratakulmasta ja perihelin sijainnista (mvelp)
    true_anomaly = np.radians(ratakulma_deg - mvelp_planet)
    etäisyys_au = (1 - ecc_planet**2) / (1 + ecc_planet * np.cos(true_anomaly))
    
    # 4. Säteilyn suhteellinen voimakkuuskerroin (Käänteisen neliön laki)
    # Mitä lähempänä aurinkoa (pieni etäisyys), sitä enemmän energiaa -> kerroin kasvaa
    säteily_kerroin = 1.0 / (etäisyys_au ** 0.5)
    
    # 5. Tehollinen etäisyys auringon zeniitistä kullakin leveysasteella
    # np.abs(lat_grid_deg - deklinaatio) kertoo kuinka kaukana aurinko on zeniitistä (suoraan yläpuolelta)
    zeniitti_etäisyys = np.abs(lat_grid_deg - deklinaatio)
 
 
    #K=0.73
    K_land = 0.85
    K_ocean = 0.55
    #K_land = 0.73-0.05
    #K_ocean = 0.73+0.05   
    #K_land = 1
    #K_ocean = 1
    # 6. Lämpötilan perusyhtälö
    # 1. Maa/meri vaikuttaa zeniittireaktioon
    #K = maa_maski * K_land + meri_maski * K_ocean
    K=0.73
    #zeniitti_vaikutus = K * zeniitti_etäisyys
    #zeniitti_vaikutus = K * pow(np.sin(np.deg2rad(np.abs(zeniitti_etäisyys))),0.5)*100
    zeniitti_vaikutus = K * np.pow(zeniitti_etäisyys,1)
    #meri_pilvi_max_viilennys=30*np.pow(zeniitti_etäisyys/180,1)
    meri_pilvi_max_viilennys=10
    meri_pilvisyys=1
    # 2. Meren pilvisyys jäähdyttää merta
    meri_pilvi_vaikutus = (
    meri_maski *
    meri_pilvisyys *
    meri_pilvi_max_viilennys
    )
    tmax_planet=50
    tasapaino_temp=((tmax_planet + 273.15) * säteily_kerroin) - 273.15
    # 3. Peruslämpötila
    perus_temp = (
    tasapaino_temp
     - zeniitti_vaikutus
     - meri_pilvi_vaikutus
    )

    # 4. Korkeus
    korkeus_vaikutus = (korkeuskartta / 1000.0) * 6.5
    
    # 5. lopullinen lämpötila
    kk_temp_ka = perus_temp - korkeus_vaikutus
    kuukausi_lämpötilat[kk, :, :]=kk_temp_ka
    
    #plt.imshow(kk_temp_ka)
    #plt.show()
    #quit(-1)
    
    # =================================================================
    # 4. SADEMÄÄRÄ JA PAINEVYÖHYKKEET
    # =================================================================
    # ITCZ (Päiväntasaajan sateet)
    ###itcz_vaikutus = np.exp(-((lat_grid_deg - ilmastollinen_päiväntasaaja)**2) / 50.0) * 420.0
    # =================================================================
    # 4. SADEMÄÄRÄ JA PAINEVYÖHYKKEET (Korjattu zeniittisade)
    # =================================================================
    # Lasketaan auringon zeniittietäisyys asteina kullekin leveysasteelle.
    # Mitä lähempänä nollaa tämä on, sitä suorimmin aurinko paistaa pään päältä.
    zeniitti_etaisyys = np.abs(lat_grid_deg - akselin_kallistuma)
    
    # ITCZ (Päiväntasaajan sateet): Tehdään sateesta erittäin piikikäs ja rankka
    # juuri siellä, missä aurinko ylittää maanpinnan (zeniitti_etaisyys lähellä 0).
    # Nostetaan maksimisademäärää (esim. 580.0) ja pienennetään jakajaa (35.0),
    # jotta sade keskittyy kapeammalle, rajummalle vyöhykkeelle.
    ##itcz_vaikutus = np.exp(-(zeniitti_etaisyys**2) / 20.0) * 580.0
    itcz_vaikutus = np.exp(-((lat_grid_deg - ilmastollinen_päiväntasaaja)**2) / 40.0) * 520.0       
    # Naparintama (Lauhkea matalapainevyöhyke vaeltaa mukana)
    polar_pohjoinen = 52.0 + ilmastollinen_päiväntasaaja * 0.5
    polar_etelä = -52.0 + ilmastollinen_päiväntasaaja * 0.5
    polar_front_vaikutus = (
        np.exp(-((lat_grid_deg - polar_pohjoinen)**2) / 150.0) * 180.0 +
        np.exp(-((lat_grid_deg - polar_etelä)**2) / 150.0) * 180.0
    )
    
    # Subtrooppinen korkeapaine (Aavikkokuivuus 25-30 asteen päässä ITCZ:stä)
    kuiva_pohjoinen = ilmastollinen_päiväntasaaja + 28.0
    kuiva_etelä = ilmastollinen_päiväntasaaja - 28.0
    subtropiikki_kuivuus = 1.0 - 0.75 * (
        np.exp(-((lat_grid_deg - kuiva_pohjoinen)**2) / 70.0) +
        np.exp(-((lat_grid_deg - kuiva_etelä)**2) / 70.0)
    )
    subtropiikki_kuivuus = np.clip(subtropiikki_kuivuus, 0.12, 1.0)
    
    # =================================================================
    # 5. GOBI-KORJAUS: MANTEREIKKUUS JA SADESUOJA
    # =================================================================
    # 5.1 Sadesuoja (Vuoret syövät pilvien kantavuuden)
    # Jos maasto on korkeaa vuoristoa (>1200m), kasvatetaan "efektiivistä etäisyyttä merestä" takana olevilla alueilla.
    efektiivinen_etaisyys_km = etaisyys_meresta_km.copy()
    efektiivinen_etaisyys_km += np.where(korkeuskartta > 1200.0, 1500.0, 0.0)
    
    # =================================================================
    # 5. DYNAMINEN KORJAUS: TUULEN SUUNTA JA MAASTON ALAVUUS
    # =================================================================
    
    # Kosteuden kantavuuden perusmatka
    pasaati_tehostus = np.where(abs_rel_lat < 16.0, 1.8*1, 1.0)
    #ekvaattori_tehostus = np.where(abs_rel_lat < 5.0, 100, 1.0)
    kantavuus_matka_km = 1200.0 * (1.0 + kk_tuuli_voimakkuus) * pasaati_tehostus

    # --- UUSI MERITUULEN LASKENTA (Onshore / Offshore Wind) ---
    # Tarkistetaan tuulen suunta suhteessa rannikkoon hyödyntämällä korkeusgradienttia (x-akselilla).
    # Jos korkeus_gradientti_x > 0, maa nousee itään (meri on lännessä).
    # Jos kk_tuuli_suunta > 0, tuuli puhaltaa lännestä itään (mereltä maalle = onshore).
    # Jos molemmat ovat samanmerkkisiä, tuuli puhaltaa mereltä maalle ja tuo kosteutta!
    tuulen_suunta_suhteessa_maahan = korkeus_gradientti_x * kk_tuuli_suunta
    
    # Luodaan kerroin: 1.0 = tuuli puhaltaa mereltä maalle, 0.2 = tuuli puhaltaa mantereelta (kuiva)
    tuulen_kosteus_kerroin = np.where(tuulen_suunta_suhteessa_maahan > 0, 1.3, 0.2)
    # Päiväntasaajalla (pasaatit) tuuli on erittäin dominoiva, joten lukitaan maatuulen kuivatus sinne tiukasti
    #tuulen_kosteus_kerroin = np.where(abs_rel_lat < 15.0, tuulen_kosteus_kerroin, np.clip(tuulen_kosteus_kerroin, 0.5, 1.0))
    tuulen_kosteus_kerroin = np.where(abs_rel_lat < 12.0, tuulen_kosteus_kerroin*4, np.clip(tuulen_kosteus_kerroin, 0.1, 1.0))

    # --- ALAVAN MAAN SADEPAINOTUS ---
    # Mitä matalampi maa (lähellä merenpintaa, esim. alle 400m), sitä helpommin sademetsä viihtyy.
    # Korkea maasto (>1000m) muuttuu vuoristoksi, jolloin sademetsä väistyy vuoristosademetsän tai sumumetsän tieltä.
    alavuus_kerroin = np.clip(1.3 - (korkeuskartta / 800.0), 0.4, 1.4)

    # --- EFREKTIIVINEN ETÄISYYS JA RANNIKKOKERROIN ---
    efektiivinen_etaisyys_km = etaisyys_meresta_km.copy()
    efektiivinen_etaisyys_km += np.where(korkeuskartta > 1200.0, 1500.0, 0.0)
    
    # Lopullinen rannikkokerroin ottaa nyt huomioon tuulen suunnan ja maaston alavuuden!
    rannikko_kerroin = np.exp(-efektiivinen_etaisyys_km / kantavuus_matka_km)
    rannikko_kerroin = rannikko_kerroin * tuulen_kosteus_kerroin * alavuus_kerroin
    
    # 5.3 Äärimmäinen mannermaisuus keskileveysasteille (Säilytetään Gobi ennallaan)
    keskileveys_maski = np.exp(-((np.abs(lat_grid_deg) - 45.0)**2) / 120.0)
    mannermaisuus_vaikutus = np.exp(-efektiivinen_etaisyys_km / 650.0)
    
    lopullinen_rannikko_kerroin = np.where(
        np.abs(lat_grid_deg) > 30.0,
        np.minimum(rannikko_kerroin, mannermaisuus_vaikutus * keskileveys_maski + rannikko_kerroin * (1 - keskileveys_maski)),
        rannikko_kerroin
    )


    # =================================================================
    # 7. MANTEREINEN LÄMPÖTILAVAIHTELU (Gobi & Siperia -efekti)
    # =================================================================
    # Mitä kauempana merestä ollaan, sitä suurempi on lämpötilan amplitudi.
    # rannikko_kerroin (tai lopullinen_rannikko_kerroin) on 1.0 rannikolla ja lähestyy nollaa sisämaassa.
    manner_kerroin = 1.0 - lopullinen_rannikko_kerroin
    
    # Kuivuus voimistaa vaihtelua (pilvettömyys): jos kuukauden sademäärä on pieni, maaperä kuumenee/jäähtyy nopeammin
    kuivuus_anomalia = np.clip(1.0 - (kuukausi_sateet[kk, :, :] / 150.0), 0.0, 1.0)
    
    # Yhdistetty mannermaisuusvoimakkuus (skaalataan leveysasteen mukaan, sillä ilmiö on voimakkain keskileveysasteilla ja navoilla)
    leveysaste_painotus = np.abs(lat_grid_deg) / 90.0
    mannermaisuus_indeksi = manner_kerroin * (0.5 + 0.5 * kuivuus_anomalia) * leveysaste_painotus
    
    # Lasketaan, onko kyseessä kesä- vai talvikuukausi kyseisellä pallonpuoliskolla.
    # akselin_kallistuma on positiivinen pohjoisen kesällä (kesä-heinäkuu) ja negatiivinen etelän kesällä (joulu-tammikuu).
    # np.sign(lat_grid_deg) kertoo pallonpuoliskon (1 = Pohjoinen, -1 = Etelä).
    kesä_maski = np.sign(lat_grid_deg) * np.sign(akselin_kallistuma)
    
    # Jos kesä_maski > 0, on kesä -> mannermaisuus nostaa lämpötilaa (max +12 astetta syvällä sisämaassa)
    # Jos kesä_maski < 0, on talvi -> mannermaisuus pudottaa lämpötilaa rajummin (max -22 astetta syvällä sisämaassa)
    lämpötila_muutos = np.where(kesä_maski >= 0, 
                                mannermaisuus_indeksi * 12.0,   # Kesäkuumennus
                                mannermaisuus_indeksi * -22.0)  # Talvijäähdytys
    
    # Päivitetään kuukauden lopulliset lämpötilat
    kuukausi_lämpötilat[kk, :, :] += lämpötila_muutos

    
    # =================================================================
    # 6. SADEMÄÄRÄN LOPULLINEN YHDISTÄMINEN (Korjattu tropiikin sademetsä)
    # =================================================================
    # ITCZ-sateet (konvektiosateet) syntyvät suoraan auringon lämmöstä, 
    # joten ne eivät tarvitse rannikkokerrointa samalla tavalla kuin mereltä tuleva kosteus.
    # Polar front (lauhkea vyöhyke) sen sijaan tarvitsee rannikkokertoimen.
    
    # Jaetaan sateet kahteen osaan:
    # 1. Konvektiivinen sademetsäsade (ITCZ), joka ulottuu syvälle sisämaahan (rannikkokerroin korotettu/ohitettu)
    tropiikki_sade_sisamaassa = itcz_vaikutus * np.minimum(1.0, rannikko_kerroin + 0.65)
    
    # 2. Muut vyöhykesateet, jotka tottelevat tiukasti rannikon etäisyyttä
    muut_vyöhyke_sateet = (20.0 + polar_front_vaikutus) * lopullinen_rannikko_kerroin
    
    # Yhdistetään ja sovelletaan subtrooppista kuivuutta (joka puree vasta n. 25-30 asteen päässä ITCZ:stä)
    perus_sade = (tropiikki_sade_sisamaassa + muut_vyöhyke_sateet) * subtropiikki_kuivuus
    
    # Dynaaminen vuoristosade (pysyy samana kuin ennen)
    korkeussade_pohja = np.clip(korkeuskartta / 150.0, 0, 80.0)
    tuuli_orografia_vaikutus = kk_orografinen_pakote * 2.5 
    dynaaminen_orografia = np.clip(korkeussade_pohja + tuuli_orografia_vaikutus, 0.0, 250.0) * lopullinen_rannikko_kerroin * 0.1   
    
    # Tallennetaan kuukauden sademäärä
    kuukausi_sateet[kk, :, :] = perus_sade + dynaaminen_orografia
    kuukausi_sateet[kk, :, :] = kuukausi_sateet[kk, :, :]*global_moisture_coeff ## default 1.0

    pilvisyys_dt= kuukausi_sateet[kk, :, :]/100
    #plt.imshow(pilvisyys_dt)
    #plt.show()
    kuukausi_lämpötilat[kk, :, :] -= pilvisyys_dt
    ## meri-dt tropiikissa muuten meri liian kuuma!
    #kuukausi_lämpötilat[kk, :, :] -= np.where(kuukausi_lämpötilat[kk, :, :] > (tmax_planet), tlimit-10, kuukausi_lämpötilat[kk, :, :] )
    #ehto = (kuukausi_lämpötilat > (tmax_planet - 10)) & (meri_maski == 1)
    # Vähennetään näistä kohdista lämpötilaa 8 asteella
    #kuukausi_lämpötilat[ehto] -= 10  


meri_anomalia=np.mean(kk_meri_anomalia, axis=0)


lumi, jaa, kokonais_korkeus = aja_jaatikko_malli(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuodet=200)

#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia*0+1,lat_grid_deg)

merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, maa_maski, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg)

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
