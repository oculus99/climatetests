
#################################################
#
## Simple planet climate Köppen classes
#  Wordbuilding and game purposes
#
# Python3+CuPy - requires CUDA
#
## 23.09.2026 0000.0025.08
#
#################################################

import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import shift
from scipy.interpolate import griddata

from pygam import LinearGAM, s

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor ## good
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

import cupy as cp
import cupyx.scipy.ndimage as ndimage

from noise import pnoise3



# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus =180*1
leveys = 360*1

#seed1=9 ##ok
#seed1 = 53 ## hyva
#seed1=3333
#seed1=44
#seed1=33
#seed1 = 233 ## hyva
#seed1=61 ## hyva monsuuniin

seed1=15

#maapallon_sade_km = 6371.0
#syvin_kohta=-11000
#korkein_kohta=8848
#manner_osuus=0.30

maapallon_sade_km = 6371.0
syvin_kohta=-2000
korkein_kohta=3000
#manner_osuus=0.30
manner_osuus=0.3 ## continnet fraction preset

oceanfrac=1-manner_osuus


#tilt_planet_axis=23.44
#ecc_planet=0.013
#mvelp_planet=102.0
# Parametrit (esimerkkinä Maan arvot)

planet_mass_me=1
planet_radius_re=math.pow(planet_mass_me,0.27) ## 
planet_atmosphere_pressure=math.pow(planet_radius_re, 1.6) ## hrom hat
planet_axis_tilt_degrees = 23.44*1
planet_ecc= 0.013*1
planet_mvelp_degrees = 102.0  # Perihelin pituus asteina (Maa saavuttaa perihelin tammikuun alussa)
planet_orbital_period=1
planet_rotation_period_hours=24

## guess planet tmean

planet_tmean=14.8
planet_global_moisture_coeff=1



###########################################
#######################################
### kartta


import heapq
import numpy as np
from noise import pnoise3
import numpy as np
from noise import pnoise3
from dataclasses import dataclass


# ============================================================
# PLANEETAN REFERENSSI
# ============================================================




# ============================================================
# PLANET REFERENCE
# ============================================================

class PlanetReference:

    def __init__(
        self,
        seed,
        noise_scale,
        octaves,
        persistence,
        lacunarity,
        distortion,
        distortion_scale,
        syvin_kohta,
        korkein_kohta,

        # ----------------------------------------------------
        # PLANEETTA
        # ----------------------------------------------------

        sade=6371.0,

        # ----------------------------------------------------
        # MERI
        #
        # meri_pinta_ala:
        #     haluttu meren pinta-ala km²
        #
        # valtameret:
        #     Maan valtamerten vesimäärän kerroin
        #
        # Esim.
        #
        #     valtameret=1.0
        #
        # tarkoittaa yhtä Maan valtamerten
        # vesimäärää.
        # ----------------------------------------------------

        meri_pinta_ala=None,
        valtameret=None,

        # ----------------------------------------------------
        # Vanha oceanfrac säilytetään yhteensopivuutta varten
        # ----------------------------------------------------

        oceanfrac=None,

        merenpinta=None,
    ):

        self.seed = seed

        self.noise_scale = noise_scale

        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

        self.distortion = distortion
        self.distortion_scale = distortion_scale

        self.syvin_kohta = syvin_kohta
        self.korkein_kohta = korkein_kohta

        # ====================================================
        # PLANEETAN SÄDE
        # ====================================================

        self.sade = float(
            sade
        )

        # ====================================================
        # MEREN MÄÄRITYS
        # ====================================================

        self.meri_pinta_ala = (
            None
            if meri_pinta_ala is None
            else float(meri_pinta_ala)
        )

        self.valtameret = (
            None
            if valtameret is None
            else float(valtameret)
        )

        # ----------------------------------------------------
        # Vanha arvo säilytetään.
        #
        # Jos vanhaa koodia käytetään oceanfrac-arvolla,
        # se toimii edelleen.
        # ----------------------------------------------------

        self.oceanfrac = oceanfrac

        # ====================================================
        # RATKAISTU MERENPINTA
        # ====================================================

        self.merenpinta = (
            None
            if merenpinta is None
            else float(merenpinta)
        )


# ============================================================
# PLANET MAP
# ============================================================

class PlanetMap:

    # ========================================================
    # MAAN VAKIOT
    # ========================================================

    # Maan keskimääräinen säde kilometreinä.
    MAAN_SADE_KM = 6371.0

    # Maan valtamerten pinta-ala.
    #
    # Käytetään oletuksena, jos käyttäjä ei anna
    # meri_pinta_ala- eikä valtameret-arvoa.
    #
    MAAN_VALTAMERIEN_PINTA_ALA_KM2 = (
        361_900_000.0
    )

    # Maan valtamerten arvioitu vesitilavuus km³.
    #
    # valtameret=1.0 tarkoittaa yhtä tämän suuruista
    # vesimäärää.
    #
    MAAN_VALTAMERIEN_TILAVUUS_KM3 = (
        1_332_000_000.0
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        leveys=360,
        korkeus=180,
        syvin_kohta=-4000,
        korkein_kohta=3000,
        noise_scale=1.0,
        seed=42,

        # ----------------------------------------------------
        # Planeetan säde kilometreinä.
        #
        # Maa:
        #     6371 km
        # ----------------------------------------------------

        sade=6371.0,
    ):

        self.leveys = int(
            leveys
        )

        self.korkeus = int(
            korkeus
        )

        self.syvin_kohta = float(
            syvin_kohta
        )

        self.korkein_kohta = float(
            korkein_kohta
        )

        self.noise_scale = float(
            noise_scale
        )

        self.seed = float(
            seed
        )

        # ====================================================
        # PLANEETAN SÄDE
        # ====================================================

        self.sade = float(
            sade
        )

        if self.sade <= 0.0:
            raise ValueError(
                "sade pitää olla > 0"
            )

        # ====================================================
        # KARTTA
        # ====================================================

        self.kartta = np.empty(
            (
                self.korkeus,
                self.leveys
            ),
            dtype=np.float32
        )

    # ========================================================
    # REFERENSSIN LUONTI
    # ========================================================

    def _luo_referenssi(
        self,
        octaves,
        persistence,
        lacunarity,
        distortion,
        distortion_scale,
        meri_pinta_ala=None,
        valtameret=None,
        oceanfrac=None,
    ):

        return PlanetReference(

            # ------------------------------------------------
            # Noise
            # ------------------------------------------------

            seed=self.seed,

            noise_scale=self.noise_scale,

            octaves=int(
                octaves
            ),

            persistence=float(
                persistence
            ),

            lacunarity=float(
                lacunarity
            ),

            distortion=float(
                distortion
            ),

            distortion_scale=float(
                distortion_scale
            ),

            # ------------------------------------------------
            # Korkeudet
            # ------------------------------------------------

            syvin_kohta=self.syvin_kohta,

            korkein_kohta=self.korkein_kohta,

            # ------------------------------------------------
            # Planeetta
            # ------------------------------------------------

            sade=self.sade,

            # ------------------------------------------------
            # Meri
            # ------------------------------------------------

            meri_pinta_ala=(
                meri_pinta_ala
            ),

            valtameret=(
                valtameret
            ),

            # ------------------------------------------------
            # Vanha oceanfrac
            # ------------------------------------------------

            oceanfrac=(
                oceanfrac
            ),

            # ------------------------------------------------
            # Aluksi None.
            #
            # Ensimmäinen kartta ratkaisee tämän.
            # ------------------------------------------------

            merenpinta=None,
        )

    # ========================================================
    # MERENPINTA-ALAN LASKEMINEN
    # ========================================================

    def _laske_alueen_pinta_ala(
        self,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):

        """
        Laskee annetun maantieteellisen alueen
        pallopinta-alan km².

        Kaava:

            A = R² * Δlon *
                (sin(lat_max) - sin(lat_min))

        Säde on kilometreinä, joten tulos on km².
        """

        lat_min_rad = np.deg2rad(
            float(lat_min)
        )

        lat_max_rad = np.deg2rad(
            float(lat_max)
        )

        lon_min_rad = np.deg2rad(
            float(lon_min)
        )

        lon_max_rad = np.deg2rad(
            float(lon_max)
        )

        delta_lon = (
            lon_max_rad
            - lon_min_rad
        )

        pinta_ala = (
            self.sade
            * self.sade
            * delta_lon
            * (
                np.sin(
                    lat_max_rad
                )
                - np.sin(
                    lat_min_rad
                )
            )
        )

        return float(
            abs(pinta_ala)
        )

    # ========================================================
    # MERENPINTA-ALAN RATKAISU
    # ========================================================

    def _laske_meripinta_ala_tavoite(
        self,
        meri_pinta_ala,
        valtameret,
        oceanfrac,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):

        """
        Määrittää tavoitellun meren pinta-alan.

        Prioriteetti:

        1. meri_pinta_ala
        2. valtameret
        3. oceanfrac
        4. Maan valtamerten pinta-ala

        Näin oletuksena käytetään Maan vastaavaa
        valtamerten pinta-alaa.
        """

        # ====================================================
        # 1. ABSOLUUTTINEN MERIPINTA-ALA
        # ====================================================

        if meri_pinta_ala is not None:

            tavoite = float(
                meri_pinta_ala
            )

            if tavoite <= 0.0:
                raise ValueError(
                    "meri_pinta_ala pitää olla > 0"
                )

            return tavoite

        # ====================================================
        # 2. VALTAMERET
        # ====================================================

        if valtameret is not None:

            valtameret = float(
                valtameret
            )

            if valtameret <= 0.0:
                raise ValueError(
                    "valtameret pitää olla > 0"
                )

            # ------------------------------------------------
            # Tässä vaiheessa valtameret tarkoittaa
            # Maan valtamerten pinta-alan kerrointa.
            #
            # Tilavuuspohjainen ratkaisu tulee erilliseen
            # tulvitukseen, jossa veden määrä ratkaisee
            # merenpinnan.
            # ------------------------------------------------

            return (
                self.MAAN_VALTAMERIEN_PINTA_ALA_KM2
                * valtameret
            )

        # ====================================================
        # 3. VANHA OCEANFRAC
        # ====================================================

        if oceanfrac is not None:

            oceanfrac = float(
                oceanfrac
            )

            if not 0.0 < oceanfrac < 1.0:
                raise ValueError(
                    "oceanfrac pitää olla välillä 0.0 ... 1.0"
                )

            alueen_pinta_ala = (
                self._laske_alueen_pinta_ala(
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                )
            )

            return (
                alueen_pinta_ala
                * oceanfrac
            )

        # ====================================================
        # 4. OLETUS:
        #
        # MAAN VALTAMERIEN PINTA-ALA
        # ====================================================

        return float(
            self.MAAN_VALTAMERIEN_PINTA_ALA_KM2
        )

    # ========================================================
    # MERENPINTA-ALAN TULVITUS
    # ========================================================

    def _etsi_merenpinta_tulvituksella(
        self,
        kartta,
        tavoite_pinta_ala,
        lat_min,
        lat_max,
    ):

        """
        Tulvittaa karttaa alimmasta kohdasta ylöspäin.

        Meren pinta-ala lasketaan todellisen pallopinta-alan
        mukaan.

        Longitude wrap:
            x=0 <-> x=W-1

        Latitude ei wrapata.
        """

        H, W = kartta.shape

        tavoite_pinta_ala = float(
            tavoite_pinta_ala
        )

        if tavoite_pinta_ala <= 0.0:
            raise ValueError(
                "tavoite_pinta_ala pitää olla > 0"
            )

        # ====================================================
        # LATITUDIRAJAT
        # ====================================================

        lat_edges = np.linspace(
            float(lat_min),
            float(lat_max),
            H + 1,
            dtype=np.float64,
        )

        lat_edges_rad = np.deg2rad(
            lat_edges
        )

        # ====================================================
        # RIVIN PINTA-ALAPAINO
        #
        # Delta sin(lat)
        #
        # Longitude jätetään tässä pois, koska kaikkien
        # saman rivin solujen longitude-leveys on sama.
        # ====================================================

        rivien_paino = (
            np.sin(
                lat_edges_rad[1:]
            )
            - np.sin(
                lat_edges_rad[:-1]
            )
        )

        # ====================================================
        # LONGITUDE-LEVEYS
        # ====================================================

        # Tämä saadaan myöhemmin todellisesta alueesta.
        #
        # Tässä metodi tietää vain latitudin, joten
        # suhteellinen paino riittää, kun tavoitepinta-ala
        # muunnetaan samaan suhteelliseen yksikköön.
        #
        # Käytetään siksi normalisoitua pinta-alaa.
        # ====================================================

        koko_paino = (
            float(
                np.sum(
                    rivien_paino
                )
            )
            * W
        )

        # ====================================================
        # Tavoite suhteellisena pinta-alana
        # ====================================================

        # Huom:
        # Tämä metodi tarvitsee kutsujalta todellisen
        # tavoitealan suhteuttamisen koko alueen alaan.
        #
        # Tätä käytetään osassa 2.
        # ====================================================

        tavoite_suhteellinen = (
            tavoite_pinta_ala
        )

        # ====================================================
        # KÄYDYT SOLUT
        # ====================================================

        kayty = np.zeros(
            (H, W),
            dtype=np.bool_
        )

        # ====================================================
        # MIN-HEAP
        # ====================================================

        heap = []

        indeksi = int(
            np.argmin(kartta)
        )

        sy = (
            indeksi // W
        )

        sx = (
            indeksi % W
        )

        heapq.heappush(
            heap,
            (
                float(
                    kartta[sy, sx]
                ),
                sy,
                sx,
            )
        )

        kertynyt_pinta_ala = 0.0

        merenpinta = float(
            kartta[sy, sx]
        )

        # ====================================================
        # TULVITUS
        # ====================================================

        while (
            heap
            and kertynyt_pinta_ala
            < tavoite_suhteellinen
        ):

            korkeus, y, x = (
                heapq.heappop(
                    heap
                )
            )

            if kayty[y, x]:
                continue

            kayty[y, x] = True

            # ------------------------------------------------
            # Solun pinta-alapaino
            # ------------------------------------------------

            solun_paino = float(
                rivien_paino[y]
            )

            kertynyt_pinta_ala += (
                solun_paino
            )

            merenpinta = float(
                korkeus
            )

            # =================================================
            # POHJOINEN
            # =================================================

            if y > 0:

                if not kayty[
                    y - 1,
                    x
                ]:

                    heapq.heappush(
                        heap,
                        (
                            float(
                                kartta[
                                    y - 1,
                                    x
                                ]
                            ),
                            y - 1,
                            x,
                        )
                    )

            # =================================================
            # ETELÄ
            # =================================================

            if y < H - 1:

                if not kayty[
                    y + 1,
                    x
                ]:

                    heapq.heappush(
                        heap,
                        (
                            float(
                                kartta[
                                    y + 1,
                                    x
                                ]
                            ),
                            y + 1,
                            x,
                        )
                    )

            # =================================================
            # LÄNSI
            # =================================================

            vasen = (
                x - 1
            ) % W

            if not kayty[
                y,
                vasen
            ]:

                heapq.heappush(
                    heap,
                    (
                        float(
                            kartta[
                                y,
                                vasen
                            ]
                        ),
                        y,
                        vasen,
                    )
                )

            # =================================================
            # ITÄ
            # =================================================

            oikea = (
                x + 1
            ) % W

            if not kayty[
                y,
                oikea
            ]:

                heapq.heappush(
                    heap,
                    (
                        float(
                            kartta[
                                y,
                                oikea
                            ]
                        ),
                        y,
                        oikea,
                    )
                )

        return merenpinta

    # ========================================================
    # MEREN TILAVUUDEN LASKEMINEN
    # ========================================================

    def _laske_veden_tilavuus(
        self,
        kartta,
        merenpinta,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):

        """
        Laskee meren alla olevan veden tilavuuden km³.

        kartta sisältää korkeudet suhteessa merenpintaan.

        Jos kartan arvo on esimerkiksi:

            -500

        ja merenpinta on:

            0

        vettä on kyseisessä solussa 500 metriä.

        Pinta-ala lasketaan pallon pinnalta ja muutetaan
        tilavuudeksi kertomalla veden syvyydellä.
        """

        H, W = kartta.shape

        lat_edges = np.linspace(
            float(lat_min),
            float(lat_max),
            H + 1,
            dtype=np.float64,
        )

        lat_edges_rad = np.deg2rad(
            lat_edges
        )

        lon_min_rad = np.deg2rad(
            float(lon_min)
        )

        lon_max_rad = np.deg2rad(
            float(lon_max)
        )

        delta_lon = (
            lon_max_rad
            - lon_min_rad
        )

        # ----------------------------------------------------
        # Rivien pallopinta-alat.
        #
        # km²
        # ----------------------------------------------------

        rivien_pinta_ala = (
            self.sade ** 2
            * delta_lon
            * (
                np.sin(
                    lat_edges_rad[1:]
                )
                - np.sin(
                    lat_edges_rad[:-1]
                )
            )
        )

        # ----------------------------------------------------
        # Veden syvyys metreinä.
        #
        # kartta on metreinä.
        # ----------------------------------------------------

        syvyys = np.maximum(
            np.float64(merenpinta)
            - np.asarray(
                kartta,
                dtype=np.float64
            ),
            0.0,
        )

        # ----------------------------------------------------
        # Solun pinta-ala km².
        #
        # Jokaisella rivillä sama pinta-ala.
        # ----------------------------------------------------

        solun_pinta_ala = (
            rivien_pinta_ala[:, None]
            / float(W)
        )

        # ----------------------------------------------------
        # Tilavuus:
        #
        # km² * m
        #
        # -> km³
        #
        # koska 1 km² * 1 m = 0.001 km³
        # ----------------------------------------------------

        tilavuus_km3 = (
            syvyys
            * solun_pinta_ala
            * 0.001
        )

        return float(
            np.sum(
                tilavuus_km3
            )
        )

    # ========================================================
    # MERENPINTA TILAVUUDEN PERUSTEELLA
    # ========================================================

    def _etsi_merenpinta_tilavuudella(
        self,
        kartta,
        valtameret,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):

        """
        Etsii merenpinnan korkeuden siten, että veden
        kokonaismäärä vastaa haluttua määrää.

        valtameret=1.0

            = Maan valtamerten vesimäärä

        valtameret=0.5

            = puolet Maan valtamerten vesimäärästä

        valtameret=2.0

            = kaksi kertaa Maan valtamerten vesimäärä.

        Merenpinta ratkaistaan binäärihaulla.
        """

        valtameret = float(
            valtameret
        )

        if valtameret <= 0.0:
            raise ValueError(
                "valtameret pitää olla > 0"
            )

        tavoite_tilavuus = (
            self.MAAN_VALTAMERIEN_TILAVUUS_KM3
            * valtameret
        )

        kartta_min = float(
            np.min(kartta)
        )

        kartta_max = float(
            np.max(kartta)
        )

        # ====================================================
        # Jos kaikki haluttu vesi ei mahdu kartalle
        # edes korkeimman kohdan saavuttamiseen, nostetaan
        # ylärajaa.
        # ====================================================

        ala = kartta_min

        yla = kartta_max

        tilavuus_yla = (
            self._laske_veden_tilavuus(
                kartta=kartta,
                merenpinta=yla,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )
        )

        # ----------------------------------------------------
        # Laajennetaan ylärajaa tarvittaessa.
        # ----------------------------------------------------

        while (
            tilavuus_yla
            < tavoite_tilavuus
        ):

            yla += (
                max(
                    1000.0,
                    yla - ala
                )
            )

            tilavuus_yla = (
                self._laske_veden_tilavuus(
                    kartta=kartta,
                    merenpinta=yla,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                )
            )

        # ====================================================
        # BINÄÄRIHAKU
        # ====================================================

        for _ in range(64):

            keski = (
                ala + yla
            ) * 0.5

            tilavuus = (
                self._laske_veden_tilavuus(
                    kartta=kartta,
                    merenpinta=keski,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                )
            )

            if (
                tilavuus
                < tavoite_tilavuus
            ):

                ala = keski

            else:

                yla = keski

        return float(
            (ala + yla) * 0.5
        )

    # ========================================================
    # KARTAN GENEROINTI
    # ========================================================

    def generoi_kartta(
        self,

        # ----------------------------------------------------
        # Maantieteellinen alue
        # ----------------------------------------------------

        alue=(
            -180.0,
            180.0,
            -90.0,
            90.0
        ),

        # ----------------------------------------------------
        # Resoluutio
        # ----------------------------------------------------

        leveys=None,
        korkeus=None,

        # ----------------------------------------------------
        # Noise
        # ----------------------------------------------------

        octaves=16,
        persistence=0.5,
        lacunarity=2.0,

        # ----------------------------------------------------
        # Domain warp
        # ----------------------------------------------------

        distortion=0.5,
        distortion_scale=1.5,

        # ----------------------------------------------------
        # RAM / suorituskyky
        # ----------------------------------------------------

        chunk_rows=16,

        # ----------------------------------------------------
        # VANHA MERI
        #
        # Säilytetään yhteensopivuus.
        # ----------------------------------------------------

        oceanfrac=None,

        # ----------------------------------------------------
        # UUSI MERI
        #
        # meri_pinta_ala:
        #
        #     haluttu merialue km²
        #
        # valtameret:
        #
        #     Maan valtamerten vesimäärän kerroin
        #
        # Jos kumpaakaan ei anneta:
        #
        #     käytetään Maan valtamerten pinta-alaa.
        # ----------------------------------------------------

        meri_pinta_ala=None,
        valtameret=None,

        # ----------------------------------------------------
        # REFERENSSI
        # ----------------------------------------------------

        referenssi=None,
    ):

        # ====================================================
        # RESOLUUTIO
        # ====================================================

        if leveys is None:

            leveys = self.leveys

        if korkeus is None:

            korkeus = self.korkeus

        W = int(
            leveys
        )

        H = int(
            korkeus
        )

        if W < 1 or H < 1:

            raise ValueError(
                "leveys ja korkeus pitää olla > 0"
            )

        # ====================================================
        # REFERENSSI
        # ====================================================

        if referenssi is None:

            referenssi = (
                self._luo_referenssi(
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    distortion=distortion,
                    distortion_scale=distortion_scale,
                    meri_pinta_ala=meri_pinta_ala,
                    valtameret=valtameret,
                    oceanfrac=oceanfrac,
                )
            )

        else:

            # ------------------------------------------------
            # Käytetään AINA alkuperäisen ajon parametreja.
            # ------------------------------------------------

            self.seed = (
                referenssi.seed
            )

            self.noise_scale = (
                referenssi.noise_scale
            )

            octaves = (
                referenssi.octaves
            )

            persistence = (
                referenssi.persistence
            )

            lacunarity = (
                referenssi.lacunarity
            )

            distortion = (
                referenssi.distortion
            )

            distortion_scale = (
                referenssi.distortion_scale
            )

            self.syvin_kohta = (
                referenssi.syvin_kohta
            )

            self.korkein_kohta = (
                referenssi.korkein_kohta
            )

            # ------------------------------------------------
            # Planeetan säde kuuluu referenssiin.
            # ------------------------------------------------

            self.sade = float(
                referenssi.sade
            )

            # ------------------------------------------------
            # Meriparametrit kuuluvat referenssiin.
            # ------------------------------------------------

            meri_pinta_ala = (
                referenssi.meri_pinta_ala
            )

            valtameret = (
                referenssi.valtameret
            )

            oceanfrac = (
                referenssi.oceanfrac
            )

        # ====================================================
        # ALUE
        # ====================================================

        lon_min, lon_max, lat_min, lat_max = map(
            float,
            alue
        )

        if lon_min < -180.0 or lon_min > 180.0:

            raise ValueError(
                "lon_min pitää olla välillä -180 ... 180"
            )

        if lon_max < -180.0 or lon_max > 180.0:

            raise ValueError(
                "lon_max pitää olla välillä -180 ... 180"
            )

        if lat_min < -90.0 or lat_min > 90.0:

            raise ValueError(
                "lat_min pitää olla välillä -90 ... 90"
            )

        if lat_max < -90.0 or lat_max > 90.0:

            raise ValueError(
                "lat_max pitää olla välillä -90 ... 90"
            )

        if lon_max <= lon_min:

            raise ValueError(
                "lon_max pitää olla suurempi kuin lon_min"
            )

        if lat_max <= lat_min:

            raise ValueError(
                "lat_max pitää olla suurempi kuin lat_min"
            )

        # ====================================================
        # NOISE-VAKIOT
        # ====================================================

        seed_offset_x = (
            self.seed * 100.0
        )

        seed_offset_y = (
            self.seed * 200.0
        )

        seed_offset_z = (
            self.seed * 300.0
        )

        base_seed = int(
            abs(self.seed)
        )

        height_range = (
            self.korkein_kohta
            - self.syvin_kohta
        )

        # ====================================================
        # LONGITUDE
        # ====================================================

        lon = (
            lon_min
            + (
                np.arange(
                    W,
                    dtype=np.float32
                )
                + 0.5
            )
            * np.float32(
                lon_max - lon_min
            )
            / np.float32(W)
        )

        lon_rad = np.deg2rad(
            lon
        ).astype(
            np.float32
        )

        cos_lon = np.cos(
            lon_rad
        ).astype(
            np.float32
        )

        sin_lon = np.sin(
            lon_rad
        ).astype(
            np.float32
        )

        # ====================================================
        # KARTTA
        # ====================================================

        kartta = np.empty(
            (
                H,
                W
            ),
            dtype=np.float32
        )

        # ====================================================
        # CHUNKIT
        # ====================================================

        for y0 in range(
            0,
            H,
            chunk_rows
        ):

            y1 = min(
                y0 + chunk_rows,
                H
            )

            rows = (
                y1 - y0
            )

            # ------------------------------------------------
            # Latitude
            # ------------------------------------------------

            lat = (
                lat_min
                + (
                    np.arange(
                        y0,
                        y1,
                        dtype=np.float32
                    )
                    + 0.5
                )
                * np.float32(
                    lat_max - lat_min
                )
                / np.float32(H)
            )

            lat_rad = np.deg2rad(
                lat
            ).astype(
                np.float32
            )

            cos_lat = np.cos(
                lat_rad
            ).astype(
                np.float32
            )

            sin_lat = np.sin(
                lat_rad
            ).astype(
                np.float32
            )

            # ------------------------------------------------
            # Pallon XYZ
            # ------------------------------------------------

            nx = (
                cos_lat[:, None]
                * cos_lon[None, :]
            )

            ny = (
                cos_lat[:, None]
                * sin_lon[None, :]
            )

            # =================================================
            # DOMAIN WARP
            # =================================================

            ox = np.empty(
                (
                    rows,
                    W
                ),
                dtype=np.float32
            )

            oy = np.empty(
                (
                    rows,
                    W
                ),
                dtype=np.float32
            )

            oz = np.empty(
                (
                    rows,
                    W
                ),
                dtype=np.float32
            )

            for iy in range(
                rows
            ):

                for ix in range(
                    W
                ):

                    xx = float(
                        nx[iy, ix]
                    )

                    yy = float(
                        ny[iy, ix]
                    )

                    zz = float(
                        sin_lat[iy]
                    )

                    ox[
                        iy,
                        ix
                    ] = pnoise3(
                        xx
                        * distortion_scale
                        + seed_offset_x,

                        yy
                        * distortion_scale,

                        zz
                        * distortion_scale,

                        octaves=3,
                    )

                    oy[
                        iy,
                        ix
                    ] = pnoise3(
                        xx
                        * distortion_scale,

                        yy
                        * distortion_scale
                        + seed_offset_y,

                        zz
                        * distortion_scale,

                        octaves=3,
                    )

                    oz[
                        iy,
                        ix
                    ] = pnoise3(
                        xx
                        * distortion_scale,

                        yy
                        * distortion_scale,

                        zz
                        * distortion_scale
                        + seed_offset_z,

                        octaves=3,
                    )

            # =================================================
            # LOPULLINEN NOISE
            # =================================================

            for iy in range(
                rows
            ):

                for ix in range(
                    W
                ):

                    xx = (
                        float(
                            nx[
                                iy,
                                ix
                            ]
                        )
                        + float(
                            ox[
                                iy,
                                ix
                            ]
                        )
                        * distortion
                    ) * self.noise_scale

                    yy = (
                        float(
                            ny[
                                iy,
                                ix
                            ]
                        )
                        + float(
                            oy[
                                iy,
                                ix
                            ]
                        )
                        * distortion
                    ) * self.noise_scale

                    zz = (
                        float(
                            sin_lat[iy]
                        )
                        + float(
                            oz[
                                iy,
                                ix
                            ]
                        )
                        * distortion
                    ) * self.noise_scale

                    n = pnoise3(
                        xx,
                        yy,
                        zz,

                        octaves=octaves,

                        persistence=persistence,

                        lacunarity=lacunarity,

                        base=base_seed,
                    )

                    # -----------------------------------------
                    # [-1, 1] -> [0, 1]
                    # -----------------------------------------

                    n = (
                        n + 1.0
                    ) * 0.5

                    if n < 0.0:

                        n = 0.0

                    elif n > 1.0:

                        n = 1.0

                    # -----------------------------------------
                    # ABSOLUUTTINEN KORKEUS
                    # -----------------------------------------

                    kartta[
                        y0 + iy,
                        ix
                    ] = (
                        self.syvin_kohta
                        + np.float32(n)
                        * np.float32(
                            height_range
                        )
                    )

            # ------------------------------------------------
            # Vapautetaan väliaikaiset taulukot
            # ------------------------------------------------

            del nx
            del ny
            del ox
            del oy
            del oz

        # ====================================================
        # MERENPINTA
        # ====================================================

        if referenssi.merenpinta is None:

            # =================================================
            # VALTAMERET:
            #
            # Tilavuus määrää merenpinnan.
            # =================================================

            if valtameret is not None:

                merenpinta = (
                    self._etsi_merenpinta_tilavuudella(
                        kartta=kartta,

                        valtameret=valtameret,

                        lat_min=lat_min,
                        lat_max=lat_max,

                        lon_min=lon_min,
                        lon_max=lon_max,
                    )
                )

            else:

                # =============================================
                # MERI PINTA-ALAN PERUSTEELLA
                # =============================================

                tavoite_pinta_ala = (
                    self._laske_meripinta_ala_tavoite(
                        meri_pinta_ala=meri_pinta_ala,

                        valtameret=None,

                        oceanfrac=oceanfrac,

                        lat_min=lat_min,
                        lat_max=lat_max,

                        lon_min=lon_min,
                        lon_max=lon_max,
                    )
                )

                # ---------------------------------------------
                # Muutetaan absoluuttinen km² pinta-ala
                # tulvituksen käyttämään suhteelliseen
                # pinta-alapainoon.
                # ---------------------------------------------

                koko_alueen_pinta_ala = (
                    self._laske_alueen_pinta_ala(
                        lat_min=lat_min,
                        lat_max=lat_max,
                        lon_min=lon_min,
                        lon_max=lon_max,
                    )
                )

                if (
                    tavoite_pinta_ala
                    > koko_alueen_pinta_ala
                ):

                    raise ValueError(
                        "haluttu meripinta-ala on "
                        "suurempi kuin generoitu alue"
                    )

                # ---------------------------------------------
                # Tulvituksen käyttämä normalisoitu pinta-ala.
                # ---------------------------------------------

                tavoite_suhteellinen = (
                    (
                        tavoite_pinta_ala
                        / koko_alueen_pinta_ala
                    )
                    * (
                        np.sum(
                            np.sin(
                                np.deg2rad(
                                    np.linspace(
                                        lat_min,
                                        lat_max,
                                        H + 1
                                    )[1:]
                                )
                            )
                            - np.sin(
                                np.deg2rad(
                                    np.linspace(
                                        lat_min,
                                        lat_max,
                                        H + 1
                                    )[:-1]
                                )
                            )
                        )
                        * W
                    )
                )

                merenpinta = (
                    self._etsi_merenpinta_tulvituksella(
                        kartta=kartta,

                        tavoite_pinta_ala=(
                            tavoite_suhteellinen
                        ),

                        lat_min=lat_min,
                        lat_max=lat_max,
                    )
                )

            # =================================================
            # TALLENNETAAN REFERENSSIIN
            # =================================================

            referenssi.merenpinta = (
                float(
                    merenpinta
                )
            )

        else:

            # =================================================
            # ZOOMAUKSESSA KÄYTETÄÄN AINA
            # ALKUPERÄISTÄ MERENPINTAA
            # =================================================

            merenpinta = float(
                referenssi.merenpinta
            )

        # ====================================================
        # SIIRRETÄÄN MERENPINTA NOLLAAN
        # ====================================================

        kartta = (
            kartta
            - np.float32(
                merenpinta
            )
        ).astype(
            np.float32
        )

        # ====================================================
        # MASKIT
        # ====================================================

        maa_maski = (
            kartta
            > np.float32(0.0)
        )

        korkeuskartta_maa = np.maximum(
            kartta,
            np.float32(0.0)
        )

        # ====================================================
        # TULOS
        # ====================================================

        return (
            kartta,
            korkeuskartta_maa,
            maa_maski,
            referenssi,
            merenpinta,
        )







#######################################################
##############################



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


import matplotlib.pyplot as plt
import numpy as np


def koordinaatit_indkseiksi(lon, lat, leveys, korkeus):
    # Muunnetaan lon [-180, 180] -> sarake [0, leveys-1]
    col = int(((lon + 180) / 360) * leveys)
    # Muunnetaan lat [-90, 90] -> rivi [0, korkeus-1]
    # Huom: Matriiseissa ylin rivi (0) on yleensä pohjoisin (+90 lat)
    row = int(((90 - lat) / 180) * korkeus)

    # Varmistetaan, etteivät indeksit mene rajojen yli
    row = max(0, min(korkeus - 1, row))
    col = max(0, min(leveys - 1, col))
    return row, col


def maarita_koppen_paavyohyke(kuukausi_lampotilat, kuukausi_sateet):
    """Yksinkertaistettu Köppenin päävyöhykkeen (A, B, C, D, E) määritys

    kuukausiarvojen perusteella.
    """
    t_mean = np.mean(kuukausi_lampotilat)
    t_min = np.min(kuukausi_lampotilat)
    t_max = np.max(kuukausi_lampotilat)
    p_sum = np.sum(kuukausi_sateet)

    # 1. E - Jäätikkö / Tundra (Kylmin vyöhyke)
    if t_max < 10:
        return "E"

    # 2. B - Kuiva ilmasto (Yksinkertaistettu sademääräraja)
    # Oikeassa Köppenissä raja riippuu lämpötilan ja sateen kausittaisuudesta,
    # mutta karkea nyrkkisääntö globaalisti on p_sum < 200-400mm maastosta
    # riippuen.
    # Tässä käytetään karkeaa kynnystä kuivuudelle:
    if p_sum < 250:  # Tai tarkempi kaava: 20 * t_mean + jatkot
        return "B"

    # 3. A - Trooppinen (Kaikki kuukaudet > 18 °C)
    if t_min >= 18:
        return "A"

    # 4. C - Lauhkea (Kylmin kuukausi -3 °C ja 18 °C välillä)
    if -3 <= t_min < 18:
        return "C"

    # 5. D - Mannermainen / Luminen (Kylmin kuukausi < -3 °C, lämpimin > 10 °C)
    if t_min < -3 and t_max >= 10:
        return "D"

    return "Tuntematon"

import numpy as np

def laske_ilmastotilastot(
    lampotilat,
    sateet,
    alue=[-180, 180, -90, 90]
):
    """
    Laskee cos(lat)-painotetun keskilämpötilan, sadesumman
    sekä min/max-lämpötilat.

    Parametrit:
    - lampotilat: numpy-taulukko muodossa (12, korkeus, leveys)
    - sateet: numpy-taulukko muodossa (12, korkeus, leveys)
    - alue: [lon_min, lon_max, lat_min, lat_max]

    Leveysasteet generoidaan automaattisesti rasterin korkeuden
    ja annetun alueen perusteella.
    """

    # Tarkistetaan syötteiden muodot
    if lampotilat.ndim != 3:
        raise ValueError("lampotilat pitää olla 3-ulotteinen taulukko (12, korkeus, leveys)")

    if sateet.shape != lampotilat.shape:
        raise ValueError("lampotilat ja sateet pitää olla samanmuotoisia")

    if lampotilat.shape[0] != 12:
        raise ValueError("Ensimmäisen dimension pitää sisältää 12 kuukautta")

    lon_min, lon_max, lat_min, lat_max = alue

    korkeus = lampotilat.shape[1]
    leveys = lampotilat.shape[2]

    # Generoidaan automaattisesti rasterin keskusten leveysasteet.
    # Esim. maailmanlaajuiselle rasterille [-90, 90].
    lat_rajat = np.linspace(lat_min, lat_max, korkeus + 1)
    leveysasteet = (lat_rajat[:-1] + lat_rajat[1:]) / 2

    # Muutetaan radiaaneiksi
    lat_rad = np.deg2rad(leveysasteet)

    # Cos(lat)-painot
    cos_lat = np.cos(lat_rad)

    # Muutetaan muotoon (korkeus, 1), jolloin NumPy
    # laajentaa sen automaattisesti leveys-suunnassa.
    painot = cos_lat[:, np.newaxis]

    # --------------------------------------------------
    # Lämpötila
    # --------------------------------------------------

    validoi_lampo = ~np.isnan(lampotilat)

    painot_lampo = np.where(
        validoi_lampo,
        painot,
        0
    )

    painotettu_keskilampo = (
        np.nansum(lampotilat * painot_lampo)
        / np.sum(painot_lampo)
    )

    min_lampo = np.nanmin(lampotilat)
    max_lampo = np.nanmax(lampotilat)

    # --------------------------------------------------
    # Sade
    # --------------------------------------------------

    # 12 kuukauden sade yhteen jokaiselle hilapisteelle
    vuosi_sateet_hila = np.nansum(sateet, axis=0)

    # Huomioidaan myös sateen NaN-arvot painotuksessa
    validoi_sade = ~np.isnan(vuosi_sateet_hila)

    sade_painot = np.where(
        validoi_sade,
        painot,
        0
    )

    keskiarvoinen_vuosisade = (
        np.nansum(vuosi_sateet_hila * sade_painot)
        / np.sum(sade_painot)
    )

    return {
        "painotettu_keskilampo": painotettu_keskilampo,
        "min_lampo": min_lampo,
        "max_lampo": max_lampo,
        "keskiarvoinen_vuosisade": keskiarvoinen_vuosisade,
        "vuosisateet_hila": vuosi_sateet_hila,
        "leveysasteet": leveysasteet
    }

def piirra_ilmastodiagrammi(
    lon, lat, korkeuskartta, kuukausi_lampotilat, kuukausi_sateet
):
    """lon, lat: halutun paikan koordinaatit korkeuskartta: 2D array [korkeus,

    leveys] kuukausi_lampotilat: 3D array [12, korkeus, leveys]
    kuukausi_sateet: 3D array [12, korkeus, leveys]
    """
    korkeus, leveys = korkeuskartta.shape
    row, col = koordinaatit_indkseiksi(lon, lat, leveys, korkeus)

    # Poimitaan data valitusta pisteestä
    piste_korkeus = int(korkeuskartta[row, col])
    piste_temps = kuukausi_lampotilat[:, row, col]  # 12 arvoa
    piste_precip = kuukausi_sateet[:, row, col]  # 12 arvoa

    # Lasketaan vaaditut yhteenvedot
    tmean = np.mean(piste_temps)
    tmin = np.min(piste_temps)
    tmax = np.max(piste_temps)

    precipsum = np.sum(piste_precip)
    precipmin = np.min(piste_precip)
    precipmax = np.max(piste_precip)

    koppen = maarita_koppen_paavyohyke(piste_temps, piste_precip)

    # --- PIIRTÄMINEN (Matplotlib) ---
    kuukaudet = [
        "Tam",
        "Hel",
        "Maa",
        "Huht",
        "Tou",
        "Kes",
        "Hei",
        "Elo",
        "Syy",
        "Lok",
        "Mar",
        "Jou",
    ]
    x = np.arange(12)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Ensimmäinen y-akseli: Sademäärä (Pylväät)
    color = "tab:blue"
    ax1.set_xlabel("Kuukausi")
    ax1.set_ylabel("Sademäärä (mm)", color=color)
    bars = ax1.bar(
        x, piste_precip, color=color, alpha=0.6, label="Sademäärä", width=0.6
    )
    ax1.tick_params(axis="y", labelcolor=color)
    # Asetetaan sademäärän maksimi sopivaksi, jotta kuva on siisti
    ax1.set_ylim(0, max(precipmax * 1.2, 50))

    # Toinen y-akseli lämpötilalle (Viiva)
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Lämpötila (°C)", color=color)
    (line,) = ax2.plot(
        x,
        piste_temps,
        color=color,
        marker="o",
        linewidth=2,
        label="Lämpötila",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    # Kuukaudet x-akselille
    plt.xticks(x, kuukaudet)

    # Yhteenvetoteksti otsikkoon tai laatikkoon
    otsikko = (
        f"Ilmastodiagrammi: Lon {lon}°, Lat {lat}°\n"
        f"Korkeus: {piste_korkeus} m | Köppen: Päävyöhyke {koppen}\n"
        f"Tmean: {tmean:.1f}°C (Min: {tmin:.1f}°C, Max: {tmax:.1f}°C)\n"
        f"Sadesumma: {precipsum:.1f} mm (Min: {precipmin:.1f} mm, Max: {precipmax:.1f} mm)"
    )
    plt.title(otsikko, fontsize=11, loc="left", pad=15)

    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()





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
    contours = ax.contour(
        dem, levels=[0,0.5,1], extent=extent, colors="black", linewidths=2
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

import numpy as np
import cupy as cp


# ================================================================
# 1. DATAN VALMISTELU GPU:LLE
# ================================================================

def valmistele_ilmastodata(
    lat_grid_deg,
    korkeuskartta,
    korkeus_syvyyskartta,
    korkeus_gradientti_x,
    korkeus_gradientti_y,
    maa_maski,
    meri_maski,
    etaisyys_meresta_km,
):
    """
    Muuttaa kaikki ilmastomallin kartat CuPy-taulukoiksi.

    korkeus_syvyyskartta:
        Meren syvyys metreinä.
        Maa-alueilla arvo voidaan olla 0.
    """

    lat = cp.asarray(
        lat_grid_deg,
        dtype=cp.float32
    )

    korkeus = cp.asarray(
        korkeuskartta,
        dtype=cp.float32
    )

    syvyys = cp.asarray(
        korkeus_syvyyskartta,
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

    # ------------------------------------------------------------
    # Tarkistukset
    # ------------------------------------------------------------

    if lat.ndim != 2:
        raise ValueError(
            "lat_grid_deg pitää olla 2D-taulukko."
        )

    shape = lat.shape

    tarkistettavat = {
        "korkeuskartta": korkeus,
        "korkeus_syvyyskartta": syvyys,
        "korkeus_gradientti_x": gradientti_x,
        "korkeus_gradientti_y": gradientti_y,
        "maa_maski": maa,
        "meri_maski": meri,
        "etaisyys_meresta_km": etaisyys_meresta,
    }

    for nimi, data in tarkistettavat.items():

        if data.shape != shape:
            raise ValueError(
                f"{nimi} väärän kokoinen: "
                f"{data.shape}, odotettu {shape}."
            )

    # ------------------------------------------------------------
    # Binaariset maskit
    # ------------------------------------------------------------

    maa_bin = (
        maa > 0
    ).astype(cp.float32)

    meri_bin = (
        meri > 0
    ).astype(cp.float32)

    # ------------------------------------------------------------
    # Syvyys ei saa olla negatiivinen
    # ------------------------------------------------------------

    syvyys = cp.maximum(
        syvyys,
        cp.float32(0.0)
    )

    # ------------------------------------------------------------
    # Korkeus ei saa olla negatiivinen maan korkeutena
    # ------------------------------------------------------------

    korkeus = cp.maximum(
        korkeus,
        cp.float32(0.0)
    )

    return {
        "lat": lat,
        "korkeus": korkeus,
        "syvyys": syvyys,

        "gradientti_x": gradientti_x,
        "gradientti_y": gradientti_y,

        "maa": maa_bin,
        "meri": meri_bin,

        "etaisyys_meresta": etaisyys_meresta,

        "ny": shape[0],
        "nx": shape[1],
    }





def warmup_gpu(seconds=0):
    """
    Käynnistää CUDA-contextin ja tekee GPU:lla
    muutaman laskun ennen varsinaista benchmarkia.
    """
    x = cp.ones((4096, 4096), dtype=cp.float32)

    for _ in range(5):
        x = x * 1.000001
        x = cp.sqrt(x)

    cp.cuda.Stream.null.synchronize()

    del x



def benchmark_gpu():
    """
    Mittaa puhdasta GPU-laskentaa.
    """
    x = cp.random.random(
        (4096, 4096),
        dtype=cp.float32
    )

    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()

    for _ in range(20):
        x = cp.sqrt(x)
        x = x * 1.000001

    cp.cuda.Stream.null.synchronize()

    dt = time.perf_counter() - t0

    del x

    return dt


def benchmark_ilmasto(korkeus_syvyyskartta):
    """
    Varsinainen ilmastomallin benchmark.
    """

    # --------------------------------------------------
    # CPU warm-up
    # --------------------------------------------------

    print("CPU warm-up...")
    warmup_cpu(2.0)

    # --------------------------------------------------
    # GPU warm-up
    # --------------------------------------------------

    print("GPU warm-up...")
    warmup_gpu()

    # --------------------------------------------------
    # GPU benchmark
    # --------------------------------------------------

    print("GPU benchmark...")
    gpu_time = benchmark_gpu()

    print(f"GPU benchmark: {gpu_time:.3f} s")

    # --------------------------------------------------
    # Ilmastomalli
    # --------------------------------------------------

    print("Ilmastolasku...")

    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()

    tulos = laske_ilmasto_gpu(
        korkeus_syvyyskartta,
        planet_mass_me=1,
        planet_radius_re=1.0,
        planet_tilt=23.44,
        planet_ecc=0.013,
        planet_mvelp_deg=101.2,
        tmax_default=57.0,
        global_moisture_coeff=1.0,
    )

    cp.cuda.Stream.null.synchronize()

    ilmasto_time = time.perf_counter() - t0

    print(f"Ilmastolasku: {ilmasto_time:.3f} s")

    return tulos
    
def benchmark_multiple(korkeus_syvyyskartta, n=5):

    ajat = []

    print("GPU warm-up...")
    warmup_gpu()

    for i in range(n):

        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()

        tulos = laske_ilmasto_gpu(
            korkeus_syvyyskartta,
            planet_mass_me=1,
            planet_radius_re=1.0,
            planet_tilt=23.44,
            planet_ecc=0.013,
            planet_mvelp_deg=101.2,
            tmax_default=57.0,
            global_moisture_coeff=1.0,
        )

        cp.cuda.Stream.null.synchronize()

        dt = time.perf_counter() - t0
        ajat.append(dt)

        print(f"Ajo {i+1}: {dt:.3f} s")

    print()
    print(f"Ensimmäinen: {ajat[0]:.3f} s")
    print(f"Keskiarvo:   {np.mean(ajat):.3f} s")
    print(f"Minimi:      {np.min(ajat):.3f} s")

    if n > 1:
        print(f"Muut ajot:   {np.mean(ajat[1:]):.3f} s")

    return ajat


# ================================================================
# YHTEISET APUTOIMINNOT
# ================================================================

def _tarkista_2d_samat_muodot(
    lat,
    korkeus,
    syvyys,
    gradientti_x,
    gradientti_y,
    maa,
    meri,
    etaisyys_meresta,
):
    """
    Tarkistaa että kaikki kartat ovat 2D ja saman kokoisia.
    """

    if lat.ndim != 2:
        raise ValueError(
            "lat_grid_deg pitää olla 2D-taulukko."
        )

    odotettu = lat.shape

    kartat = {
        "korkeuskartta": korkeus,
        "korkeus_syvyyskartta": syvyys,
        "korkeus_gradientti_x": gradientti_x,
        "korkeus_gradientti_y": gradientti_y,
        "maa_maski": maa,
        "meri_maski": meri,
        "etaisyys_meresta_km": etaisyys_meresta,
    }

    for nimi, data in kartat.items():

        if data.shape != odotettu:
            raise ValueError(
                f"{nimi} väärän kokoinen: "
                f"{data.shape}, odotettu {odotettu}."
            )


def _tarkista_parametrit(
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    global_moisture_coeff,
):
    """
    Planeetan parametrien tarkistus.
    """

    tilt_planet_axis = float(
        tilt_planet_axis
    )

    ecc_planet = float(
        ecc_planet
    )

    mvelp_planet = float(
        mvelp_planet
    )

    global_moisture_coeff = float(
        global_moisture_coeff
    )

    if not np.isfinite(tilt_planet_axis):
        raise ValueError(
            "tilt_planet_axis ei ole kelvollinen."
        )

    if not np.isfinite(ecc_planet):
        raise ValueError(
            "ecc_planet ei ole kelvollinen."
        )

    if ecc_planet < 0.0 or ecc_planet >= 1.0:
        raise ValueError(
            "ecc_planet pitää olla välillä 0 <= e < 1."
        )

    if not np.isfinite(mvelp_planet):
        raise ValueError(
            "mvelp_planet ei ole kelvollinen."
        )

    if global_moisture_coeff < 0.0:
        raise ValueError(
            "global_moisture_coeff ei voi olla negatiivinen."
        )

    return (
        tilt_planet_axis,
        ecc_planet,
        mvelp_planet,
        global_moisture_coeff,
    )


# ================================================================
# AURINGON PERUSLÄMPÖ
# ================================================================

def laske_auringon_peruslampotila(
    maa_maski2,
    lat,
    time_years,
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    tmax_planet_param,
    orbital_period_years,
    planet_mass_me=1.0,
    planet_radius_re=1.0,
    planet_atmosphere_pressure=101325.0,
    planet_rotation_period=24.0,
    leveysaste_viilennys_kerroin=0.13,
):
    """
    Laskee planeetan aurinkoperuslämpötilan.

    Parametrit
    ----------
    lat :
        Leveysaste asteina. Voi olla numpy- tai CuPy-taulukko.

    time_years :
        Simulaatioaika Maan vuosina.

    orbital_period_years :
        Planeetan kiertoaika Maan vuosina.

    planet_mass_me :
        Planeetan massa Maan massoina.

    planet_radius_re :
        Planeetan säde Maan säteinä.

    planet_atmosphere_pressure :
        Keskimääräinen pintapaine pascaleina.
        Maa = 101325 Pa.

    planet_rotation_period :
        Pyörähdysaika Maan vuorokausina.
        Maa = 1.0.

    Palauttaa
    ----------
    perus_lampotila
    deklinaatio
    sateily_kerroin
    ratakulma_rad
    """

    if orbital_period_years <= 0.0:
        raise ValueError(
            "orbital_period_years pitää olla suurempi kuin nolla."
        )

    if planet_mass_me <= 0.0:
        raise ValueError(
            "planet_mass_me pitää olla suurempi kuin nolla."
        )

    if planet_radius_re <= 0.0:
        raise ValueError(
            "planet_radius_re pitää olla suurempi kuin nolla."
        )

    if planet_atmosphere_pressure < 0.0:
        raise ValueError(
            "planet_atmosphere_pressure ei voi olla negatiivinen."
        )

    if planet_rotation_period <= 0.0:
        raise ValueError(
            "planet_rotation_period pitää olla suurempi kuin nolla."
        )
    planet_rotation_period=planet_rotation_period/24
    # ============================================================
    # PLANEETAN RATA-ASEMA
    # ============================================================

    rata_vaihe = (
        time_years
        /
        orbital_period_years
    )

    ratakulma_rad = (
        2.0
        *
        np.pi
        *
        rata_vaihe
    )

    ratakulma_rad = (
        ratakulma_rad
        %
        (2.0 * np.pi)
    )

    # ============================================================
    # TRUE ANOMALY
    # ============================================================

    true_anomaly = (
        ratakulma_rad
        -
        np.radians(mvelp_planet)
    )

    # ============================================================
    # ELLIPTISEN RADAN ETÄISYYS
    # ============================================================

    nimittaja = (
        1.0
        +
        ecc_planet
        *
        np.cos(true_anomaly)
    )

    etaisyys_au = (
        1.0
        -
        ecc_planet ** 2
    ) / nimittaja

    if (
        not np.isfinite(etaisyys_au)
        or
        etaisyys_au <= 0.0
    ):
        raise ValueError(
            "Planeetan rataetäisyys ei ole kelvollinen."
        )

    # ============================================================
    # SÄTEILYN SUHTEELLINEN VOIMAKKUUS
    # ============================================================

    sateily_kerroin = (
        1.0
        /
        np.sqrt(etaisyys_au)
    )

    # ============================================================
    # AURINGON DEKLIINAATIO
    # ============================================================

    deklinaatio = (
        tilt_planet_axis
        *
        np.sin(
            ratakulma_rad
            -
            np.radians(60.0)
        )
    )

    # ============================================================
    # REFERENSSILÄMPÖTILA
    # ============================================================

    tmax = cp.float32(
        tmax_planet_param
    )

    sateily_poikkeama = (
        cp.float32(
            sateily_kerroin
        )
        -
        cp.float32(1.0)
    )

    sateily_lampotila_muutos = (
        cp.float32(0.50)
        *
        tmax
        *
        sateily_poikkeama
    )

    tasapaino_temp = (
        tmax
        +
        sateily_lampotila_muutos
    )

    # ============================================================
    # PLANEETAN PAINOVOIMA
    # ============================================================

    gravity_me = (
        planet_mass_me
        /
        (
            planet_radius_re
            *
            planet_radius_re
        )
    )

    # ============================================================
    # ILMAKEHÄN SUHTEELLINEN MASSA / LÄMPÖINERTIA
    #
    # Pintapaine kertoo ilmakehän massasta pinta-alaa kohti.
    # Painovoima vaikuttaa siihen, kuinka paljon kaasumassaa
    # tietty paine vastaa.
    #
    # Tämä on tässä normalisoitu suhteellinen suure, ei
    # absoluuttinen lämpökapasiteetti.
    # ============================================================

    paine_suhde = (
        planet_atmosphere_pressure
        /
        101325.0
    )

    ilmakeha_inertia = (
        paine_suhde
        /
        gravity_me
    )

    # Rajataan täysin äärimmäiset arvot.
    ilmakeha_inertia = np.clip(
        ilmakeha_inertia,
        0.01,
        100.0,
    )

    # ============================================================
    # ORBITAALISEN VUODEN PITUUDEN VAIKUTUS
    #
    # Mitä pidempi vuosi, sitä hitaammin aurinkopakote muuttuu
    # ja sitä paremmin lämpöjärjestelmä ehtii seurata sitä.
    #
    # Maa = 1.0
    # ============================================================

    orbital_period_factor = np.sqrt(
        orbital_period_years
    )

    # Ilmakehän suuri lämpöinertia vaimentaa vuodenaikavaihtelua.
    #
    # Tämä on tarkoituksella maltillinen empiirinen kerroin,
    # jotta nykyinen mallisi ei muutu rajusti.
    inertia_factor = (
        np.sqrt(ilmakeha_inertia)
    )
    ## JN WARNING KALIBROI TÄMÄ!!!!
    seasonal_response = (
        orbital_period_factor
        /
        (
            orbital_period_factor
            +
            0.50
            *
            inertia_factor
        )
    )

    seasonal_response = np.clip(
        seasonal_response,
        0.05,
        1.0,
    )

    # ============================================================
    # SÄTEILYN VUODENAIKAISVAIHTELUN VAIMENNUS
    # ============================================================

    sateily_lampotila_muutos = (
        sateily_lampotila_muutos
        *
        cp.float32(seasonal_response)
    )

    tasapaino_temp = (
        tmax
        +
        sateily_lampotila_muutos
    )

    # ============================================================
    # LEVEYSASTEEN VAIKUTUS
    # ============================================================

    lat_cp = cp.asarray(
        lat,
        dtype=cp.float32
    )

    deklinaatio_cp = cp.float32(
        deklinaatio
    )

    zeniitti_etaisyys = cp.abs(
        lat_cp
        -
        deklinaatio_cp
    )

    zeniitti_etaisyys = cp.clip(
        zeniitti_etaisyys,
        cp.float32(0.0),
        cp.float32(90.0),
    )

    # ============================================================
    # PYÖRIMISEN VAIKUTUS
    #
    # Nopea pyöriminen -> tehokkaampi leveysasteiden välinen
    # ilmakehän dynamiikka.
    #
    # Hidas pyöriminen -> suurempi päiväntasaaja/napaero.
    #
    # Maa = 1.0
    # ============================================================

    rotation_factor = np.sqrt(
        planet_rotation_period
    )

    rotation_factor = np.clip(
        rotation_factor,
        0.25,
        4.0,
    )

    # Maa-arvolla = 1.
    #
    # Hitaasti pyörivällä planeetalla kerroin kasvaa.
    # Nopeasti pyörivällä pienenee.
    rotation_latitude_factor = (
        rotation_factor
    )
    kerroin_taulukko = cp.array([1*leveysaste_viilennys_kerroin, 1.5*leveysaste_viilennys_kerroin], dtype=cp.float32)
    leveysaste_viilennys = (
    #leveysaste_viilennys_kerroin
    kerroin_taulukko[maa_maski2.astype(cp.int32)]
    * zeniitti_etaisyys
    * cp.float32(rotation_latitude_factor)
    )
    #leveysaste_viilennys = (
    #    cp.float32(
    #        leveysaste_viilennys_kerroins[maa_maski2]
    #    )
    #    *
    #    zeniitti_etaisyys
    #    *
    #    cp.float32(
    #        rotation_latitude_factor
    #    )
    #)

    # ============================================================
    # LOPULLINEN PERUSLÄMPÖTILA
    # ============================================================

    perus_lampotila = (
        tasapaino_temp
        -
        leveysaste_viilennys
    )

    return (
        perus_lampotila,
        cp.float32(deklinaatio),
        cp.float32(sateily_kerroin),
        cp.float32(ratakulma_rad),
    )



## kokeilua vain


def koe_laske_vuosikierron_tavoite(
    perus_lampotila,
    ratakulma_rad,
    vuodenaika_amplitudi=8.0,
):
    """
    Muodostaa aurinkoperuslämpötilasta
    vuodenaikaisen tavoitelämpötilan.

    vuodenaika_amplitudi:
        Vuodenaikaisen maa-merijärjestelmän
        lisävaihtelun amplitudi Celsius-asteina.
    """

    vuodenaika = cp.sin(
        ratakulma_rad
    )

    tavoite_lampotila = (
        perus_lampotila
        +
        cp.float32(
            vuodenaika_amplitudi
        )
        *
        vuodenaika
    )

    return tavoite_lampotila

def koe_paivita_maa_meri_lampotila(
    edellinen_lampotila,
    tavoite_lampotila,
    landmask,
    dt_years,
    maa_lampoaika_years=0.10,
    meri_lampoaika_years=1.50,
):
    """
    Päivittää maa-merisysteemin lämpötilan.

    landmask:
        0.0 = meri
        1.0 = maa
        0..1 = maa/meri-sekoitus

    dt_years:
        Simulaation timestep Maan vuosina.

    maa_lampoaika_years:
        Maan lämpötilan reagointiaika.

    meri_lampoaika_years:
        Meren lämpötilan reagointiaika.

    Palauttaa:
        uuden lämpötilakentän.
    """

    landmask = cp.asarray(
        landmask,
        dtype=cp.float32,
    )

    landmask = cp.clip(
        landmask,
        cp.float32(0.0),
        cp.float32(1.0),
    )

    if dt_years <= 0.0:
        raise ValueError(
            "dt_years pitää olla suurempi kuin nolla."
        )

    if maa_lampoaika_years <= 0.0:
        raise ValueError(
            "maa_lampoaika_years pitää olla suurempi kuin nolla."
        )

    if meri_lampoaika_years <= 0.0:
        raise ValueError(
            "meri_lampoaika_years pitää olla suurempi kuin nolla."
        )

    # ============================================================
    # MAA / MERI LÄMPÖAIKA
    # ============================================================

    lampoaika = (
        landmask
        *
        cp.float32(
            maa_lampoaika_years
        )
        +
        (cp.float32(1.0) - landmask)
        *
        cp.float32(
            meri_lampoaika_years
        )
    )

    # ============================================================
    # REAKTION NOPEUS
    #
    # Eksponentiaalinen ratkaisu:
    #
    # T_new =
    # T_old + alpha * (T_target - T_old)
    #
    # alpha = 1 - exp(-dt/tau)
    # ============================================================

    alfa = (
        cp.float32(1.0)
        -
        cp.exp(
            -
            cp.float32(dt_years)
            /
            lampoaika
        )
    )

    # ============================================================
    # UUSI LÄMPÖTILA
    # ============================================================

    uusi_lampotila = (
        edellinen_lampotila
        +
        alfa
        *
        (
            tavoite_lampotila
            -
            edellinen_lampotila
        )
    )

    return uusi_lampotila


# ================================================================
# TUULIMALLI
# ================================================================

def laske_tuulet(
    lat,
    kk,
    tilt_planet_axis, itcz_kerroin
):
    """
    Laskee planetaarisen tuulikentän.

    Palauttaa:

        tuuli_x
        tuuli_y
        tuuli_x_yksikko
        tuuli_y_yksikko
        tuuli_suunta
        tuuli_voima
        ilmastollinen_paivantaasaaja
    """

    ratakulma_rad = np.radians(
        kk * 30.0
    )

    # ------------------------------------------------------------
    # Ilmastollinen päiväntasaaja
    # ------------------------------------------------------------

    akselin_kallistuma = (
        tilt_planet_axis
        *
        np.cos(ratakulma_rad)
    )

    ilmastollinen_paivantaasaaja = (
        akselin_kallistuma
        *
        itcz_kerroin #0.65
    )

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

    # ------------------------------------------------------------
    # Perustuuli
    # ------------------------------------------------------------

    perus_tuuli = cp.sin(
        cp.deg2rad(
            relatiivinen_lat * 2.0
        )
    )

    # ------------------------------------------------------------
    # Ilmastovyöhykkeet
    # ------------------------------------------------------------

    hadley_maski = (
        abs_rel_lat < 30.0
    )

    ferrel_maski = (
        (abs_rel_lat >= 30.0)
        &
        (abs_rel_lat < 62.0)
    )

    polar_maski = (
        abs_rel_lat >= 62.0
    )

    # ------------------------------------------------------------
    # Itä-länsisuuntainen komponentti
    # ------------------------------------------------------------

    tuuli_x = cp.zeros_like(
        lat,
        dtype=cp.float32
    )

    tuuli_x = cp.where(
        hadley_maski,
        -cp.abs(perus_tuuli),
        tuuli_x
    )

    tuuli_x = cp.where(
        ferrel_maski,
        cp.abs(perus_tuuli),
        tuuli_x
    )

    tuuli_x = cp.where(
        polar_maski,
        -cp.abs(perus_tuuli),
        tuuli_x
    )

    # ------------------------------------------------------------
    # Meridionaalinen komponentti
    # ------------------------------------------------------------

    vuodenaika_sign = np.sign(
        akselin_kallistuma
    )

    tuuli_y = (
        cp.float32(
            0.12 * vuodenaika_sign
        )
        *
        cp.tanh(
            relatiivinen_lat
            /
            cp.float32(25.0)
        )
    )

    tuuli_y = cp.asarray(
        tuuli_y,
        dtype=cp.float32
    )

    # ------------------------------------------------------------
    # Tuulen nopeus
    # ------------------------------------------------------------

    tuuli_voima_raaka = cp.sqrt(
        tuuli_x * tuuli_x
        +
        tuuli_y * tuuli_y
    )

    tuuli_voima = cp.clip(
        tuuli_voima_raaka
        *
        cp.float32(3.0),
        cp.float32(0.3),
        cp.float32(3.0)
    )

    # ------------------------------------------------------------
    # Yksikkövektori
    # ------------------------------------------------------------

    tuuli_normi = cp.sqrt(
        tuuli_x * tuuli_x
        +
        tuuli_y * tuuli_y
        +
        cp.float32(1.0e-8)
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

    # ------------------------------------------------------------
    # Tuulen kulma
    # ------------------------------------------------------------

    tuuli_suunta = (
        cp.degrees(
            cp.arctan2(
                tuuli_y_yksikko,
                tuuli_x_yksikko
            )
        )
        %
        cp.float32(360.0)
    )

    return (
        tuuli_x,
        tuuli_y,
        tuuli_x_yksikko,
        tuuli_y_yksikko,
        tuuli_suunta,
        tuuli_voima,
        cp.float32(
            ilmastollinen_paivantaasaaja
        ),
    )


def laske_monsuunituuli(tuuli_x, tuuli_y, lampotila_kk, maa_maski, dx=1.0, dy=1.0, monsuuni_voimakkuus=0.5):
    """
    Manipuloi tuulikenttää siten, että lämpimät maamassat imevät tuulta mereltä.
    
    lampotila_kk: Nykyisen kuukauden lämpötilaruudukko
    maa_maski: 1.0 = maa, 0.0 = meri
    monsuuni_voimakkuus: Kerroin sille, kuinka voimakkaasti lämpötilaerot vaikuttavat tuuleen
    """
    # Lasketaan lämpötilan muutossuunta (gradientti)
    # grad_y on pohjois-eteläsuuntainen, grad_x on itä-länsisuuntainen muutoksen suunta
    grad_y, grad_x = cp.gradient(lampotila_kk, dy, dx)
    
    # Tuuli pyrkii puhaltamaan kohti kuumempaa aluetta (positiivinen gradientti).
    # Monsuuniefekti on voimakkaimmillaan maalla ja rannikoilla, joten kerrotaan maa_maskilla.
    # (Jos halutaan imu myös rannikon läheiseltä mereltä, maskia voi hieman "pehmentää" sumennuksella)
    monsuuni_paine_imu_x = grad_x * maa_maski * monsuuni_voimakkuus
    monsuuni_paine_imu_y = grad_y * maa_maski * monsuuni_voimakkuus
    
    # Lisätään monsuuni-imu olemassa olevaan taustatuuleen (esim. pasaatituuliin)
    uusi_tuuli_x = tuuli_x + monsuuni_paine_imu_x
    uusi_tuuli_y = tuuli_y + monsuuni_paine_imu_y
    
    return uusi_tuuli_x, uusi_tuuli_y


def laske_monsuunin_sade_muutos(tuuli_x, tuuli_y, maa_maski, lampotila_kk, 
                                monsuuni_sade_kerroin=50.0):
    """
    Laskee PELKÄN monsuunin aiheuttaman sademäärän muutoksen (positiivinen tai negatiivinen).
    Ei vaadi latitudi- tai ITCZ-tietoja, vaan reagoi pelkästään tuuleen ja lämpötilaan.
    
    tuuli_x, tuuli_y: Manipuloidut tuulikentät (jotka sisältävät jo monsuuni-imun)
    maa_maski: 1.0 = maa, 0.0 = meri
    lampotila_kk: Kuluvan kuukauden lämpötilaruudukko
    monsuuni_sade_kerroin: Säätöruuvi monsuunisateen voimakkuudelle
    """
    # 1. Lasketaan maan ja meren raja (gradientti maa_maskista)
    # maa_grad on positiivinen, kun siirrytään mereltä maalle
    maa_grad_y, maa_grad_x = cp.gradient(maa_maski)
    
    # 2. Kosteuden advektio (virtaus rannikon yli)
    # Positiivinen arvo = tuuli puhaltaa mereltä maalle (tuo kosteutta)
    # Negatiivinen arvo = tuuli puhaltaa maalta merelle (kuiva maatuuli)
    merelta_maalle_virtaus = (tuuli_x * maa_grad_x) + (tuuli_y * maa_grad_y)
    import cupyx.scipy.ndimage as ndimage
    merelta_maalle_virtaus = ndimage.gaussian_filter(merelta_maalle_virtaus, sigma=5.0)    
    # 3. Lämpötilatekijä
    # Monsuunikonvektio vaatii lämpöä. Otetaan kynnykseksi esim. 18°C.
    # Jos on kylmempää, konvektio nollataan cp.clipillä.
    lampo_kynnys = cp.clip(lampotila_kk - 18.0, 0.0, None)
    
    # --- KESÄMONSUUNI (Sateen lisäys) ---
    # Kun virtaus on mereltä maalle (> 0) ja maa on lämmin
    kesamonsuuni_sade = cp.clip(merelta_maalle_virtaus, 0.0, None) * lampo_kynnys * maa_maski
    
    # --- TALVIMONSUUNI (Sateen vähennys / kuiva kausi) ---
    # Jos haluat, että talvella maalta merelle puhaltava tuuli kuivattaa maata entisestään,
    # poimitaan negatiivinen virtaus.
    talvimonsuuni_kuivuus = cp.clip(-merelta_maalle_virtaus, 0.0, None) * maa_maski
    
    # 4. Yhdistetään vaikutukset
    # Kesällä monsuuni LISÄÄ sadetta, talvella se VÄHENTÄÄ sitä (pienentää perussadetta)
    # Voit säätää talvikuiva-kertoimen (esim. 5.0) haluamaksesi tai jättää sen nollaksi,
    # jos haluat vain sateen lisäystä.
    monsuuni_muutos = (kesamonsuuni_sade * monsuuni_sade_kerroin) - (talvimonsuuni_kuivuus * 5.0)
    
    return monsuuni_muutos

def laske_monsuunin_sade_muutos_laaja_koe_yksi(tuuli_x, tuuli_y, maa_maski, lampotila_kk, 
                                      monsuuni_sade_kerroin=50.0):
    """
    Laskee monsuunin laajempana vyöhykkeenä koko maamassan päälle ilman kapeaa rannikkogradienttia.
    """
    maa_maski = maa_maski.astype(cp.float32)
    
    # Katsotaan, mihin suuntaan maapallon suuret ilmamassat liikkuvat.
    # Jos tuuli_y on positiivinen (pohjoiseen) ja ollaan pohjoisella pallonpuoliskolla, 
    # se tuo usein kosteutta päiväntasaajalta. 
    # Yleispäiväisempi tapa on katsoa, onko tuulella voimakas suunta lämpimällä maalla:
    
    # Konvergenssi (tuulen koonpression laskenta) on loistava indikaattori laajoille sateille:
    # Kun tuulet kohtaavat maalla ja hidastuvat, ilma nousee ylös ja sataa laajasti.
    div_y, _ = cp.gradient(tuuli_y)
    _, div_x = cp.gradient(tuuli_x)
    konvergenssi = -(div_x + div_y) # Positiivinen arvo tarkoittaa, että tuulet pakkautuvat kasaan
    
    # Lämpötilatekijä (kuten ennenkin)
    lampo_kynnys = cp.clip(lampotila_kk - 18.0, 0.0, None)
    
    # Kesämonsuuni aktivoituu maalla, kun tuulet kohtaavat (konvergenssi) ja on kuuma
    kesamonsuuni_sade = cp.clip(konvergenssi, 0.0, None) * lampo_kynnys * maa_maski
    
    # Säädetään kerrointa isommaksi, koska konvergenssiarvot ovat usein pieniä
    monsuuni_muutos = kesamonsuuni_sade * (monsuuni_sade_kerroin * 10.0)
    
    return monsuuni_muutos


def laske_monsuunin_sade_muutos_laaja_kaksi(tuuli_x, tuuli_y, maa_maski, lampotila_kk, 
                                      rannikko_etäisyys_km,
                                      monsuuni_sade_kerroin=50.0,
                                      kosteuden_kantama_km=800.0):
    """
    Laskee laajan monsuunin maamassoille ja vaimentaa sitä sisämaassa
    valmiin rannikkoetäisyystaulukon avulla.
    
    kosteuden_kantama_km: Kuinka monta kilometriä meren kosteus jaksaa taivaltaa 
                          sisämaahan ennen kuin monsuuni puolittuu/hiipuu.
    """
    maa_maski = maa_maski.astype(cp.float32)
    
    # --- 1. KOSTEUDEN EHTYMINEN (Etäisyys mereen) ---
    # Eksponentiaalinen vaimeneminen: rannikolla (0 km) kerroin on 1.0.
    # Kun etäisyys kasvaa, kerroin lähestyy nollaa.
    kosteus_saatavuus = cp.exp(-rannikko_etäisyys_km / kosteuden_kantama_km) * maa_maski

    # --- 2. TUULEN KONVERGENSSI (Saderintama) ---
    div_y, _ = cp.gradient(tuuli_y)
    _, div_x = cp.gradient(tuuli_x)
    konvergenssi = -(div_x + div_y)
    
    # --- 3. LÄMPÖTILA ---
    lampo_kynnys = cp.clip(lampotila_kk - 18.0, 0.0, None)
    
    # --- 4. YHDISTETÄÄN JA RAJOITETAAN SISÄMAASSA ---
    # Konvergenssi * kosteuden saatavuus estää sateen kuivilla aavikoilla
    kesamonsuuni_sade = cp.clip(konvergenssi, 0.0, None) * kosteus_saatavuus * lampo_kynnys
    
    monsuuni_muutos = kesamonsuuni_sade * (monsuuni_sade_kerroin * 10.0)
    
    return monsuuni_muutos

def laske_monsuunin_sade_muutos_laaja(tuuli_x, tuuli_y, maa_maski, lampotila_kk, 
                                      rannikko_etäisyys_km,
                                      monsuuni_sade_kerroin=50.0,
                                      kosteuden_kantama_km=200.0,
                                      talvikuivuus_kerroin=15.0):
    """
    Laskee laajan monsuunin maamassoille. 
    - Kesällä lisää sadetta rannikon läheisyydessä (konvergenssi + lämpö).
    - Talvella vähentää sadetta (divergenssi / maalta merelle puhaltava kuiva tuuli).
    """
    maa_maski = maa_maski.astype(cp.float32)
    
    # --- 1. KOSTEUDEN EHTYMINEN (Etäisyys mereen) ---
    kosteus_saatavuus = cp.exp(-rannikko_etäisyys_km / kosteuden_kantama_km) * maa_maski

    # --- 2. TUULEN LIIKE (Konvergenssi vs Divergenssi) ---
    div_y, _ = cp.gradient(tuuli_y)
    _, div_x = cp.gradient(tuuli_x)
    
    # net_divergenssi > 0 tarkoittaa, että tuulet hajoavat (talvimonsuunin kuiva laskuvirtaus)
    # net_divergenssi < 0 tarkoittaa, että tuulet pakkautuvat kasaan (kesämonsuunin saderintama)
    net_divergenssi = div_x + div_y 
    
    # --- 3. KESÄMONSUUNI (Sateen lisäys) ---
    # Konvergenssi on -net_divergenssi
    konvergenssi = cp.clip(-net_divergenssi, 0.0, None)
    lampo_kynnys = cp.clip(lampotila_kk - 18.0, 0.0, None)
    
    kesamonsuuni_sade = konvergenssi * kosteus_saatavuus * lampo_kynnys
    
    # --- 4. TALVIMONSUUNI (Sateen vähennys / kuiva kausi) ---
    # Talvella tuuli puhaltaa maalta pois, eli divergenssi on positiivinen.
    # Talvikuivuus puree parhaiten silloin, kun maa on kylmä (esim. alle 15 astetta).
    divergenssi = cp.clip(net_divergenssi, 0.0, None)
    kylmyys_kynnys = cp.clip(15.0 - lampotila_kk, 0.0, None)
    
    # Talvikuivuus leviää rannikolta sisämaahan (käytetään samaa tai hieman eri etäisyyskerrointa)
    talvimonsuuni_kuivuus = divergenssi * kosteus_saatavuus * kylmyys_kynnys

    # --- 5. YHDISTETÄÄN VAIKUTUKSET ---
    # Kesä lisää sadetta, talvi vähentää sitä teoreettisesta perussateesta
    monsuuni_muutos = (kesamonsuuni_sade * (monsuuni_sade_kerroin * 10.0)) - (talvimonsuuni_kuivuus * talvikuivuus_kerroin)
    
    return monsuuni_muutos

def laske_mantereiset_painesolut_ja_tuuli_koe_yksi(tuuli_x, tuuli_y, lampotila_kk, maa_maski, 
                                         rannikko_etäisyys_km, dy=1.0, dx=1.0, 
                                         paine_efekti_voimakkuus=0.8):
    """
    Simuloi mantereen kokoon kytkeytyviä korkeapainesoluja (talvella) ja 
    matalapainesoluja (kesällä), ja päivittää tuulikentän niiden mukaan.
    """
    maa_maski = maa_maski.astype(cp.float32)
    
    # 1. LASKETAAN LEVEYSASTEEN KESKILÄMPÖTILA (Vyöhykkeellinen keskiarvo)
    # Lasketaan jokaisen leveysasterivin keskilämpötila maapallolla
    leveysaste_keskiarvot = cp.mean(lampotila_kk, axis=1, keepdims=True)
    
    # Lämpötila-anomalia: Kuinka paljon ruutu poikkeaa oman leveysasteensa keskiarvosta
    # Talvella sisämaa on paljon KYLMEMPI (anomalia < 0) -> korkeapaine
    # Kesällä sisämaa on paljon KUUMEMPI (anomalia > 0) -> matalapaine
    lampo_anomalia = lampotila_kk - leveysaste_keskiarvot
    
    # 2. LUODAAN PAINESOLU (Paineanomalia)
    # Mitä syvemmällä sisämaassa ollaan (suuri rannikkoetäisyys) ja mitä suurempi 
    # lämpötilaerotus on, sitä voimakkaampi painesolu syntyy.
    # Eksponentiaalinen kasvu rannikolta sisämaahan (esim. 500 km skaalalla voimistuen)
    manner_kerroin = (1.0 - cp.exp(-rannikko_etäisyys_km / 500.0)) * maa_maski
    
    # Kylmyys luo korkeapainetta (+), kuumuus matalapainetta (-)
    painesolu = -lampo_anomalia * manner_kerroin
    
    # 3. PAINESOLUN VAIKUTUS TUULEEN (Gradientti)
    # Tuuli virtaa korkeapaineesta matalapaineeseen (eli painegradienttia *vastaan*)
    # grad_y on pohjois-etelä, grad_x on itä-länsi
    paine_grad_y, paine_grad_x = cp.gradient(painesolu, dy, dx)
    
    # Puhallus suoraan korkeapaineesta ulos (ja matalapaineeseen sisään)
    paine_tuuli_x = -paine_grad_x * paine_efekti_voimakkuus
    paine_tuuli_y = -paine_grad_y * paine_efekti_voimakkuus
    
    # 4. [BONUS: CORIOLIS-EFEKTI / PYÖRIMISLIIKE]
    # Oikeassa maailmassa korkeapainesolu alkaa pyöriä (pohjoisella pallonpuoliskolla myötäpäivään).
    # Jos haluat tuulista spiraalimaisia, voit kääntää gradienttia hieman ristiriitaan:
    # tuuli_x_pyörintä = -paine_grad_y, tuuli_y_pyörintä = paine_grad_x
    
    # 5. YHDISTETÄÄN TAUSTATUULEEN
    uusi_tuuli_x = tuuli_x + paine_tuuli_x
    uusi_tuuli_y = tuuli_y + paine_tuuli_y
    
    return uusi_tuuli_x, uusi_tuuli_y

def laske_painesolujen_pyorteet(tuuli_x, tuuli_y, lat_grid_deg, lampotila_kk, maa_maski, 
                               rannikko_etäisyys_km, dy=1.0, dx=1.0, 
                               paine_suora_voimakkuus=0.3,   # Suoraan paineesta ulos/sisään puhaltava tuuli
                               paine_pyörre_voimakkuus=0.7): # Coriolis-efektin luoma pyörivä tuuli
    """
    Laskee mantereille korkea- ja matalapainesolut ja luo niiden ympärille 
    leveysasteesta riippuvat Coriolis-pyörteet.
    """
    maa_maski = maa_maski.astype(cp.float32)
    
    # 1. Lasketaan painesolu lämpötila-anomalian avulla (kuten aiemmin)
    leveysaste_keskiarvot = cp.mean(lampotila_kk, axis=1, keepdims=True)
    lampo_anomalia = lampotila_kk - leveysaste_keskiarvot
    manner_kerroin = (1.0 - cp.exp(-rannikko_etäisyys_km / 500.0)) * maa_maski
    
    # Kylmä manner = korkeapaine (+), kuuma manner = matalapaine (-)
    painesolu = -lampo_anomalia * manner_kerroin
    
    # 2. Lasketaan paineen gradientti (muutossuunta)
    paine_grad_y, paine_grad_x = cp.gradient(painesolu, dy, dx)
    
    # 3. LASKETAAN CORIOLIS-KERROIN (Sini leveysasteesta)
    # Coriolis on nolla päiväntasaajalla, positiivinen pohjoisessa, negatiivinen etelässä
    lat_rad = cp.deg2rad(lat_grid_deg)
    coriolis_kerroin = cp.sin(lat_rad)
    
    # 4. SUORA TUULI (Korkeasta matalaan)
    # Puhaltaa suoraan painetta vastaan
    suora_x = -paine_grad_x * paine_suora_voimakkuus
    suora_y = -paine_grad_y * paine_suora_voimakkuus
    
    # 5. PYÖRIVÄ TUULI (Coriolis-käännös 90 astetta)
    # Käännetään gradienttivektoria 90 astetta:
    # Pohjoisella pallonpuoliskolla korkeapaineen ympärille syntyy myötäpäivään kiertävä tuuli.
    # Kerrotaan coriolis_kertoimella, jotta suunta vaihtuu etelässä ja vaimenee päiväntasaajalla.
    pyorre_x = paine_grad_y * coriolis_kerroin * paine_pyörre_voimakkuus
    pyorre_y = -paine_grad_x * coriolis_kerroin * paine_pyörre_voimakkuus
    
    # 6. YHDISTETÄÄN UUDET TUULET ALKUPERÄISEEN TAUSTATUULEEN
    uusi_tuuli_x = tuuli_x + suora_x + pyorre_x
    uusi_tuuli_y = tuuli_y + suora_y + pyorre_y
    
    return uusi_tuuli_x, uusi_tuuli_y

# ================================================================
# OROGRAFIA
# ================================================================

def laske_orografia(
    korkeus,
    gradientti_x,
    gradientti_y,
    tuuli_x_yksikko,
    tuuli_y_yksikko,
    tuuli_voima,
):
    """
    Laskee maaston vaikutuksen tuuleen.

    Positiivinen gradientti:
        ilma nousee maastoa vasten.

    Negatiivinen:
        ilma laskee maastosta poispäin.
    """

    maasto_gradientti_tuulen_suunnassa = (
        gradientti_x
        *
        tuuli_x_yksikko
        +
        gradientti_y
        *
        tuuli_y_yksikko
    )

    orografinen_pakote = (
        maasto_gradientti_tuulen_suunnassa
        *
        tuuli_voima
    )

    nousu = cp.maximum(
        orografinen_pakote,
        cp.float32(0.0)
    )

    lasku = cp.maximum(
        -orografinen_pakote,
        cp.float32(0.0)
    )

    noususade = cp.clip(
        nousu
        *
        cp.float32(1.2),
        cp.float32(0.0),
        cp.float32(180.0)
    )

    sadevarjo = cp.clip(
        lasku
        *
        cp.float32(0.45),
        cp.float32(0.0),
        cp.float32(80.0)
    )

    return (
        maasto_gradientti_tuulen_suunnassa,
        orografinen_pakote,
        noususade,
        sadevarjo,
    )



# ================================================================
# MERIVIRRAT
# ================================================================

def laske_merivirrat(
    lat,
    syvyys,
    meri_bin,
    etaisyys_meresta,
    tuuli_x_yksikko,
    tuuli_y_yksikko,
    tuuli_voima,
    kk,
    ilmastollinen_paivantaasaaja,
):
    """
    Laskee yksinkertaistetun planetaarisen merivirran.

    Mallissa on:

        1. pintavirta
        2. termohaliininen / syvempi virtaus
        3. Coriolis-vaikutus
        4. syvyyden vaikutus
        5. rannikon vaikutus
        6. merivirran lämpötila-anomalia

    Tärkeä:

        syvyys = meren syvyys metreinä.

    Oletus:

        meri:
            syvyys >= 0

        maa:
            meri_bin == 0
    """

    # ============================================================
    # 1. TURVALLINEN SYVYYS
    # ============================================================

    syvyys = cp.nan_to_num(
        syvyys,
        nan=0.0,
        posinf=12000.0,
        neginf=0.0,
    )

    syvyys = cp.maximum(
        syvyys,
        cp.float32(0.0)
    )

    # ------------------------------------------------------------
    # Jos syvyys olisi tarkoituksella negatiivinen kartassasi,
    # muuta yllä oleva tähän:
    #
    # syvyys = cp.abs(syvyys)
    #
    # ------------------------------------------------------------

    # ============================================================
    # 2. SYVYYDEN NORMALISOINTI
    # ============================================================

    syvyys_paino = cp.clip(
        syvyys
        /
        cp.float32(4000.0),
        cp.float32(0.0),
        cp.float32(1.0)
    )

    # ============================================================
    # 3. LEVEYSASTEEN CORIOLIS
    # ============================================================

    coriolis = cp.sin(
        cp.deg2rad(lat)
    )

    coriolis_abs = cp.abs(
        coriolis
    )

    # ============================================================
    # 4. PINTAVIRRAN PERUSVEKTORI
    # ============================================================

    pintavirta_x = (
        tuuli_x_yksikko
        *
        tuuli_voima
    )

    pintavirta_y = (
        tuuli_y_yksikko
        *
        tuuli_voima
    )

    # ------------------------------------------------------------
    # Coriolis kääntää virtausta.
    #
    # Pohjoisella pallonpuoliskolla:
    #     oikealle
    #
    # Eteläisellä:
    #     vasemmalle
    # ------------------------------------------------------------

    coriolis_x = (
        -coriolis
        *
        pintavirta_y
    )

    coriolis_y = (
        coriolis
        *
        pintavirta_x
    )

    pintavirta_x = (
        pintavirta_x
        +
        coriolis_x
        *
        cp.float32(0.45)
    )

    pintavirta_y = (
        pintavirta_y
        +
        coriolis_y
        *
        cp.float32(0.45)
    )

    # ============================================================
    # 5. SUBTROPIIKIN GYRE-VAIKUTUS
    # ============================================================

    abs_lat = cp.abs(
        lat
    )

    subtrooppinen_paino = cp.exp(
        -(
            (
                abs_lat
                -
                cp.float32(25.0)
            )
            ** 2
        )
        /
        cp.float32(500.0)
    )

    gyre_x = (
        cp.sign(
            lat
        )
        *
        subtrooppinen_paino
        *
        cp.float32(0.55)
    )

    gyre_y = (
        -subtrooppinen_paino
        *
        cp.float32(0.20)
    )

    pintavirta_x = (
        pintavirta_x
        +
        gyre_x
    )

    pintavirta_y = (
        pintavirta_y
        +
        gyre_y
    )

    # ============================================================
    # 6. SYVÄVIRTA
    # ============================================================
    #
    # Syvävirta ei seuraa suoraan pintatuulta.
    #
    # Se on hitaampi ja paljon tasaisempi.
    #
    # ============================================================

    syvavirta_x = (
        -cp.sign(lat)
        *
        cp.float32(0.18)
        *
        cp.exp(
            -abs_lat
            /
            cp.float32(45.0)
        )
    )

    syvavirta_y = (
        cp.float32(0.10)
        *
        cp.sin(
            cp.deg2rad(lat)
        )
    )

    # Syvyys mahdollistaa syvän veden kierron.
    syvavirta_x *= (
        cp.float32(0.35)
        +
        syvyys_paino
        *
        cp.float32(0.65)
    )

    syvavirta_y *= (
        cp.float32(0.35)
        +
        syvyys_paino
        *
        cp.float32(0.65)
    )

    # ============================================================
    # 7. TERMALINEN VESIMASSA-VAIKUTUS
    # ============================================================
    #
    # Korkeilla leveysasteilla kylmä, tiheä vesi painuu.
    #
    # Tämä antaa syvävirralle napojen ja tropiikin välisen
    # kiertokomponentin.
    # ============================================================

    pohjoinen_upotus = cp.exp(
        -(
            (
                lat
                -
                cp.float32(65.0)
            )
            ** 2
        )
        /
        cp.float32(250.0)
    )

    etelainen_upotus = cp.exp(
        -(
            (
                lat
                +
                cp.float32(65.0)
            )
            ** 2
        )
        /
        cp.float32(250.0)
    )

    termohaliini_x = (
        pohjoinen_upotus
        -
        etelainen_upotus
    )

    termohaliini_y = (
        cp.sign(lat)
        *
        cp.float32(0.12)
        *
        (
            pohjoinen_upotus
            +
            etelainen_upotus
        )
    )

    syvavirta_x += (
        termohaliini_x
        *
        syvyys_paino
    )

    syvavirta_y += (
        termohaliini_y
        *
        syvyys_paino
    )

    # ============================================================
    # 8. PINTA + SYVÄVIRTA
    # ============================================================

    syvyyden_vaimennus = (
        cp.float32(0.15)
        +
        syvyys_paino
        *
        cp.float32(0.85)
    )

    virta_x = (
        pintavirta_x
        *
        cp.float32(0.72)
        +
        syvavirta_x
        *
        cp.float32(0.28)
        *
        syvyyden_vaimennus
    )

    virta_y = (
        pintavirta_y
        *
        cp.float32(0.72)
        +
        syvavirta_y
        *
        cp.float32(0.28)
        *
        syvyyden_vaimennus
    )

    # ============================================================
    # 9. VAIN MERELLÄ
    # ============================================================

    virta_x = cp.where(
        meri_bin > 0.0,
        virta_x,
        cp.float32(0.0)
    )

    virta_y = cp.where(
        meri_bin > 0.0,
        virta_y,
        cp.float32(0.0)
    )

    # ============================================================
    # 10. RANNIKON LÄMPÖKAPASITEETTI / VIRTA
    # ============================================================

    rannikko_paino = cp.exp(
        -etaisyys_meresta
        /
        cp.float32(500.0)
    )

    # ============================================================
    # 11. VIRRAN NOPEUS
    # ============================================================

    virta_voima = cp.hypot(
        virta_x,
        virta_y
    )

    virta_voima = cp.clip(
        virta_voima,
        cp.float32(0.0),
        cp.float32(4.0)
    )

    # ============================================================
    # 12. VIRRAN SUUNTA
    # ============================================================

    virta_suunta = (
        cp.degrees(
            cp.arctan2(
                virta_y,
                virta_x
            )
        )
        %
        cp.float32(360.0)
    )

    # ============================================================
    # 13. MERIVEDEN LÄMPÖANOMALIA
    # ============================================================
    #
    # Positiivinen:
    #     lämmin merivirta
    #
    # Negatiivinen:
    #     kylmä merivirta
    #
    # ============================================================

    trooppinen_lahde = cp.exp(
        -(
            lat
            -
            cp.float32(
                ilmastollinen_paivantaasaaja
            )
        )
        ** 2
        /
        cp.float32(1400.0)
    )

    polaarinen_lahde = cp.exp(
        -(
            cp.abs(lat)
            -
            cp.float32(65.0)
        )
        ** 2
        /
        cp.float32(500.0)
    )

    merivesi_anomalia = (
        trooppinen_lahde
        *
        cp.float32(8.0)
        -
        polaarinen_lahde
        *
        cp.float32(8.0)
    )

    # Virran kuljettama lämpö
    merivesi_anomalia *= (
        cp.float32(0.35)
        +
        cp.clip(
            virta_voima,
            0.0,
            2.0
        )
        *
        cp.float32(0.35)
    )

    # Syvä vesi on lähempänä neutraalia lämpötilaa
    merivesi_anomalia *= (
        cp.float32(1.0)
        -
        syvyys_paino
        *
        cp.float32(0.35)
    )

    # ============================================================
    # 14. RANNIKON MERIVIRTA-ANOMALIA
    # ============================================================

    rannikko_merivirta_anomalia = (
        merivesi_anomalia
        *
        rannikko_paino
        *
        cp.float32(0.65)
    )

    return (
        virta_x,
        virta_y,
        virta_voima,
        virta_suunta,
        merivesi_anomalia,
        rannikko_merivirta_anomalia,
    )





# ================================================================
# SADEMALLI
# ================================================================


def laske_perus_sade(lat_grid_deg,
                     ilmastollinen_paivantaasaaja,
                     perussade=20.0,
                     tropiikin_max_sade=1800.0,
                     lauhkean_max_sade=100.0,
                     tropiikin_leveys=8.0,
                     lauhkean_leveys=15.0):

    # Perussade: pieni taustasade kaikkialla
    #perussade = cp.clip(perussade, 0.0, 50.0)

    # Ilmastollisen päiväntasaajan sijainti
    itcz_keskus = ilmastollinen_paivantaasaaja * 0.25

    # Etäisyys ITCZ:stä
    etaisyys_itcz = cp.abs(lat_grid_deg - itcz_keskus)

    # -------------------------
    # TROPIIKIN VYÖHYKE
    # -------------------------

    tropiikki = cp.exp(
        -(etaisyys_itcz / tropiikin_leveys) ** 2
    )

    tropiikin_sade = (
        tropiikin_max_sade * tropiikki
    )

    # -------------------------
    # LAUHKEAN VYÖHYKKEET
    # -------------------------

    # Etäisyys päiväntasaajasta
    abs_lat = cp.abs(lat_grid_deg)

    # Lauhkea vyöhyke alkaa tropiikin ulkopuolelta.
    # Maksimi noin 45 asteen kohdalla.
    lauhkea_keskus = 45.0

    etaisyys_lauhkea = cp.abs(abs_lat - lauhkea_keskus)

    lauhkea = cp.exp(
        -(etaisyys_lauhkea / lauhkean_leveys) ** 2
    )

    lauhkean_sade = (
        lauhkean_max_sade * lauhkea
    )

    # -------------------------
    # NAPA-ALUEEN VÄHENTYMINEN
    # -------------------------

    napakerroin = (
        cp.cos(cp.deg2rad(abs_lat)) ** 0.5
    )

    # -------------------------
    # YHDISTETÄÄN VYÖHYKKEET
    # -------------------------

    sade = (
        perussade
        + tropiikin_sade
        + lauhkean_sade
    )

    sade = sade * napakerroin

    return sade



def tarkenna_sadevyohykkeet(
    perussade,
    lat_grid_deg,
    ilmastollinen_paivantaasaaja,
):

    d = cp.abs(
        lat_grid_deg - ilmastollinen_paivantaasaaja
    )

    # ITCZ:n vaikutus
    #itcz = cp.exp(-(d / 10.0) ** 2)
    itcz = cp.exp(-(d / 8.0) ** 2)
    # Subtrooppisen korkeapainevyöhykkeen kuivuus
    subtrooppinen = cp.exp(
        -((d - 27.0) / 15.0) ** 2
    )

    # Korjauskertoimet
    itcz_korjaus = 1.0 + 1.5 * itcz
    #subtrooppinen_korjaus = 1.0 - 0.65 * subtrooppinen
    subtrooppinen_korjaus = 1.0 - 0.8 * subtrooppinen
    sade = (
        perussade
        * itcz_korjaus
        * subtrooppinen_korjaus
    )

    return cp.maximum(sade, 0.0)

def laske_rannikkokerroin(
    etaisyys_meresta_km,
    kosteus_max=1.5,
    vaikutusmatka_km=500.0,
):
    rannikkokerroin = 1.0 + (
        kosteus_max - 1.0
    ) * cp.exp(
        -etaisyys_meresta_km / vaikutusmatka_km
    )

    return rannikkokerroin



def laske_tuulietaisyys_meresta_gpu(
    meri_maski,
    tuuli_x_yksikko,
    tuuli_y_yksikko,
    lat_grid_deg,
    planet_radius_re=1.0,
    max_pixel_etaisyys=1000,
):
    """
    Suora vastatuulihaku.

    Jokainen solu käyttää lähtösolunsa tuulivektoria koko matkan.
    Rannikkoa etsitään ensin harppomalla pikselietäisyyksissä ja
    sen jälkeen binäärihaulla.

    max_pixel_etaisyys ei ole kilometriraja.
    Se on maksimi etäisyys rasterikoordinaateissa.
    """

    korkeus, leveys = meri_maski.shape

    # ============================================================
    # GEOMETRIA
    # ============================================================

    R = cp.float32(
        6371.0088 * planet_radius_re
    )

    rad = cp.float32(np.pi / 180.0)

    dlat = cp.float32(
        180.0 / (korkeus - 1)
    )

    dlon = cp.float32(
        360.0 / (leveys - 1)
    )

    km_y = R * dlat * rad

    km_x_grid = (
        R
        * dlon
        * rad
        * cp.cos(lat_grid_deg * rad)
    )

    # ============================================================
    # ALKUPISTEET
    # ============================================================

    y0, x0 = cp.meshgrid(
        cp.arange(korkeus, dtype=cp.float32),
        cp.arange(leveys, dtype=cp.float32),
        indexing="ij",
    )

    # ============================================================
    # LÄHTÖPISTEEN TUULIVEKTORI
    #
    # Tätä EI päivitetä matkan aikana.
    # ============================================================

    dx = -tuuli_x_yksikko
    dy = -tuuli_y_yksikko

    # ============================================================
    # TULOS
    # ============================================================

    tulos = cp.full(
        (korkeus, leveys),
        cp.inf,
        dtype=cp.float32,
    )

    tulos[meri_maski] = 0.0

    # Maa = vielä etsitään
    aktiivinen = ~meri_maski

    # ============================================================
    # APUFUNKTIO:
    # onko piste meressä?
    # ============================================================

    def piste_meressa(etaisyys_px):

        px = x0 + dx * etaisyys_px
        py = y0 + dy * etaisyys_px

        # Maapallo jatkuu vaakasuunnassa
        px = cp.mod(px, leveys)

        # Y-akselilla ei wrapata
        py_int = cp.rint(py).astype(cp.int32)
        px_int = cp.rint(px).astype(cp.int32)

        sisalla = (
            (py_int >= 0)
            &
            (py_int < korkeus)
        )

        py_safe = cp.clip(
            py_int,
            0,
            korkeus - 1
        )

        meri = meri_maski[
            py_safe,
            px_int
        ]

        return meri & sisalla

    # ============================================================
    # 1. HARPPOMINEN
    #
    # 1, 2, 4, 8, 16...
    # ============================================================

    edellinen = cp.zeros(
        (korkeus, leveys),
        dtype=cp.float32,
    )

    nykyinen = cp.ones(
        (korkeus, leveys),
        dtype=cp.float32,
    )

    loydetty = cp.zeros(
        (korkeus, leveys),
        dtype=cp.bool_,
    )

    # Merellä olevat ovat valmiita
    loydetty |= meri_maski

    harppaus = 1.0

    while harppaus <= max_pixel_etaisyys:

        meri = piste_meressa(
            cp.float32(harppaus)
        )

        # Ensimmäinen meri
        uusi = (
            aktiivinen
            &
            (~loydetty)
            &
            meri
        )

        # Näillä kohdilla rannikko on:
        #
        # edellinen < rannikko <= nykyinen

        edellinen = cp.where(
            uusi,
            harppaus / 2.0,
            edellinen
        )

        nykyinen = cp.where(
            uusi,
            harppaus,
            nykyinen
        )

        loydetty |= uusi

        # Seuraava harppaus
        harppaus *= 2.0

        # Jos kaikki löytyivät, loppu
        if bool(cp.all(loydetty)):
            break

    # ============================================================
    # 2. NIILLE JOILLE MERI LÖYTYI:
    # BINÄÄRIHAKU
    # ============================================================

    aktiivinen_tarkennus = (
        aktiivinen &
        loydetty
    )

    vasen = edellinen.copy()
    oikea = nykyinen.copy()

    # noin 10 kierrosta antaa melko tarkan pikselisijainnin
    for _ in range(10):

        keski = (
            vasen + oikea
        ) * cp.float32(0.5)

        meri = piste_meressa(keski)

        # Jos keskellä on meri,
        # rannikko on vasemman ja keskikohdan välillä.
        oikea = cp.where(
            aktiivinen_tarkennus & meri,
            keski,
            oikea
        )

        # Jos keskellä on maa,
        # rannikko on keskikohdan ja oikean välillä.
        vasen = cp.where(
            aktiivinen_tarkennus & (~meri),
            keski,
            vasen
        )

    # ============================================================
    # 3. ARVIOITU PIKSELIETÄISYYS
    # ============================================================

    etaisyys_px = (
        vasen + oikea
    ) * cp.float32(0.5)

    # ============================================================
    # 4. MUUNNOS KILOMETREIKSI
    #
    # Koska x- ja y-suunnilla on eri fyysinen mittakaava,
    # lasketaan suoran reitin komponentit.
    # ============================================================

    matka_x_px = (
        dx * etaisyys_px
    )

    matka_y_px = (
        dy * etaisyys_px
    )

    # x-matkan leveysastekohtainen km
    km_x = (
        km_x_grid
        * cp.abs(matka_x_px)
    )

    km_y_total = (
        cp.abs(matka_y_px)
        * km_y
    )

    etaisyys_km = cp.sqrt(
        km_x * km_x
        +
        km_y_total * km_y_total
    )

    # ============================================================
    # 5. TULOS
    # ============================================================

    tulos = cp.where(
        aktiivinen_tarkennus,
        etaisyys_km,
        tulos
    )

    return tulos




def laske_merelta_kulkeva_kosteus(
    tuulietaisyys_meresta_km,
    kosteus_vaikutusmatka_km=700.0,
    kosteus_max=1.5,
):
    """
    Laskee mereltä tuulen mukana kulkevan kosteuden kertoimen.
    """

    kosteuskerroin = (
        1.0
        +
        (kosteus_max - 1.0)
        *
        cp.exp(
            -tuulietaisyys_meresta_km
            /
            kosteus_vaikutusmatka_km
        )
    )

    # Jos merta ei löytynyt tuulireitiltä,
    # ei anneta ylimääräistä merellistä kosteutta.
    kosteuskerroin = cp.where(
        cp.isinf(tuulietaisyys_meresta_km),
        cp.float32(1.0),
        kosteuskerroin
    )

    return kosteuskerroin

def laske_orografiakerroin(
    korkeus,
    orografinen_pakote,
    korkeus_kerroin=0.00005,
    nousu_kerroin=0.15,
    lasku_kerroin=0.08,
):
    """
    Laskee orografisen sademäärän kertoimen.
    """

    # ------------------------------------------------------------
    # Korkeuden vaikutus
    # ------------------------------------------------------------

    korkeusvaikutus = (
        1.0
        + korkeus_kerroin
        * cp.maximum(korkeus, 0.0)
    )

    # ------------------------------------------------------------
    # Tuulenpuoleinen rinne
    # ------------------------------------------------------------

    nousu = cp.maximum(
        orografinen_pakote,
        0.0
    )

    nousukerroin = (
        1.0
        + nousu_kerroin
        * cp.tanh(nousu)
    )

    # ------------------------------------------------------------
    # Suojanpuoleinen rinne
    # ------------------------------------------------------------

    lasku = cp.maximum(
        -orografinen_pakote,
        0.0
    )

    varjokerroin = (
        1.0
        -
        lasku_kerroin
        * cp.tanh(lasku)
    )

    # ------------------------------------------------------------
    # Yhdistetään
    # ------------------------------------------------------------

    orografiakerroin = (
        korkeusvaikutus
        * nousukerroin
        * varjokerroin
    )

    return orografiakerroin



def laske_jaat(korkeuskartta, kuukausi_lämpotilat, kuukausi_sateet,vuosia=200):
    meri_anomalia=np.mean(kuukausi_sateet, axis=0)/1000
    meri_anomalia=np.where(meri_anomalia==np.nan,0, meri_anomalia)
    tuuli_suunta_base=np.mean(kuukausi_tuuli_suunta, axis=0)/1000
    korkeuskartta_gpu = cp.asarray(korkeuskartta)
    maa_maski_gpu = cp.asarray(maa_maski)
    lampotilat_gpu = cp.asarray(kuukausi_lämpötilat)
    sateet_gpu = cp.asarray(kuukausi_sateet)

    tuuli_gpu = cp.asarray(tuuli_suunta_base)
    anomalia_gpu = cp.asarray(meri_anomalia)
    lat_gpu = cp.asarray(lat_grid_deg)
	
    #lumi, jaa, kokonais_korkeus = aja_jaatikko_malli(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuodet=200)
    gpu_lumi, gpu_jaa, gpu_kokonais_korkeus = aja_jaatikko_malli(korkeuskartta_gpu, lampotilat_gpu, sateet_gpu, vuodet=vuosia)
    lumi = cp.asnumpy(gpu_lumi)
    jaatikko = cp.asnumpy(gpu_jaa)
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
    merijaa_peittavyys = cp.asnumpy(merijaa_peittavyys_gpu)
    merijaa_paksuus = cp.asnumpy(merijaa_paksuus_gpu)
    merijaa_laajin=np.sum(merijaa_peittavyys_vuosi, axis=1)
    merijaa_aina = np.all(merijaa_peittavyys_vuosi > 0, axis=0)
    merijaa_joskus_tai_aina = np.any(merijaa_peittavyys_vuosi > 0, axis=0)
    # Vähennetään tästä ne alueet, joissa jäätä on aina
    merijaa_vain_joskus = merijaa_joskus_tai_aina & ~merijaa_aina
    return(jaatikko, merijaa_aina)



import math

def lapse_rate_helppo(T, p, gee, kosteus):
    """
    Laskee pystygradientin (°C/km).
    T: Lämpötila (°C)
    p: Ilmanpaine (hPa)
    gee: Painovoima g-yksiköissä (Maa = 1)
    kosteus: Suhteellinen kosteus välillä 0.0 (kuiva) - 1.0 (täysin kostea)
    """
    # 1. Kuiva gradientti pohjaksi g-voiman mukaan
    dalr = gee * 9.8
    
    # 2. Kyllästysvesihöyrynpaine (Tetens-kaava)
    e_s = 6.11 * math.exp((17.27 * T) / (T + 237.3))
    
    # 3. Kostutuskerroin: jos kosteus=0, kerroin=1.0 (kuiva gradientti).
    # Jos kosteus=1, vähennetään maksimimäärä piilevää lämpöä.
    kerroin = 1.0 - (kosteus * (4.5 * e_s / p))
    
    # Suojaraja, ettei gradientti romahda epäfysikaaliseksi extreme-oloissa
    if kerroin < 0.3:
        kerroin = 0.3
        
    return (dalr * kerroin/1000)



import cupy as cp

def laske_lapse_rate_gpu(T_celsius, sade, korkeus, P0, g=9.81, sademuutos_raja=0.0):
    """Laskee dynaamisen pystygradientin (lapse rate) rasteri- tai cupy-datasta
    arvioimalla suhteellisen Fluxin/kosteuden (RH) kuukausittaisen sademäärän perusteella GPU:lla.

    Parametrit:
    -----------
    T_celsius : cupy.ndarray tai float
        Kuukauden keskilämpötila celsiusasteina (°C)
    sade : cupy.ndarray tai float
        Sademäärä (mm/kk)
    korkeus : cupy.ndarray tai float
        Maaston korkeus metreinä (m)
    P0 : cupy.ndarray tai float
        Ilmanpaine 0-tasolla (pohjalla) pascaleina (Pa)
    g : float, vapaaehtoinen
        Planeetan painovoimakiihtyvyys (m/s^2), oletus 9.81 (Maa)
    sademuutos_raja : float, vapaaehtoinen
        Sademäärän kynnysarvo (mm/kk), jonka alapuolella olevat solut
        käsitellään täysin sateettomina (RH = 40 %).

    Palauttaa:
    ----------
    cupy.ndarray tai float
        Lapse rate yksikössä °C / m maaston pinnalla.
    """
    # --- Fysikaaliset vakiot ---
    cp_air = 1005  # Kuivan ilman ominaislämpökapasiteetti (J/kg*K) - nimetty uudelleen, ettei sekoitu CuPyyn
    Rd = 287.05    # Kuivan ilman kaasuvakio (J/kg*K)
    Rv = 461.5     # Vesihöyryn kaasuvakio (J/kg*K)
    Lv = 2.501e6   # Veden piilevä höyrystymislämpö (J/kg)

    # 1. Yksikkömuunnos Kelvineiksi ja ilmanpaineen korjaus korkeudelle z
    T_kelvin = T_celsius + 273.15
    P_korkeudella = P0 * cp.exp(-g * korkeus / (Rd * T_kelvin))

    # 2. Teoreettinen kuiva adiabaattinen pystygradientti (K/m)
    gamma_dry = g / cp_air

    # 3. Teoreettinen tyydyttynyt gradientti (Clausius-Clapeyron)
    es = 611.2 * cp.exp((17.67 * T_celsius) / (T_celsius + 243.5))  # (Pa)
    rs = 0.622 * es / (P_korkeudella - es)  # Kyllästyssekoitussuhde

    osoittaja = 1 + (Lv * rs) / (Rd * T_kelvin)
    nimittaja = 1 + (Lv**2 * rs) / (cp_air * Rv * T_kelvin**2)
    gamma_moist = gamma_dry * (osoittaja / nimittaja)

    # 4. MUUNNETAAN SADEMÄÄRÄ SUHTEELLISEKSI KOSTEUDEKSI (RH)
    # Käytetään cp.where ja cp.log1p suoraan GPU-taulukoille
    rh = cp.where(
        sade > sademuutos_raja,
        0.40 + 0.12 * cp.log1p(sade - sademuutos_raja),
        0.40
    )

    # Rajataan suhteellinen kosteus välille (40 % - 95 %) CuPyn clip-funktiolla
    rh = cp.clip(rh, 0.40, 0.95)

    # 5. LASKETAAN DYNAAMINEN GRADIENTTI PAINOTETTUNA KESKIARVONA
    lapse_rate_m = (1.0 - rh) * gamma_dry + rh * gamma_moist

    return lapse_rate_m




################################
## ilmasto


import time
from collections import defaultdict


def laske_ilmasto_gpu(
    korkeus_syvyyskartta,
    planet_mass_me=1,
    planet_radius_re=1.0,
    planet_tilt=23.44,
    planet_ecc=0.013,
    planet_mvelp_deg=101.2,
    planet_orbital_period=1,
    planet_rotation_hours=24,
    planet_atmosphere_pressure=1,
    landfrac=0.3,
    planet_tmean=13.8, 
    global_moisture_coeff=1.0,
):

    oceanfrac=1-landfrac
    tmax_default=planet_tmean+32 ##
    
    gee=planet_mass_me/(planet_radius_re*planet_radius_re)

    #leveysaste_viilennys_kerroin=0.73
    landfrac_vaikutus_leveysaste_viilennys=(landfrac/0.3)
    leveysaste_viilennys_kerroin=0.49 *math.sqrt(planet_radius_re)*math.sqrt(24/planet_rotation_hours)*(1/planet_atmosphere_pressure)*landfrac_vaikutus_leveysaste_viilennys
    #leveysaste_viilennys_kerroin=0.65
    #lapse_rate_coeff=gee*np.pow(planet_atmosphere_pressure, 0.18)
    #lapse_rate = (-6.5 / 1000)*lapse_rate_coeff
    lapse_rate=lapse_rate_helppo(planet_tmean, planet_atmosphere_pressure*1013.25, gee, global_moisture_coeff)
    lapse_rate=lapse_rate*oceanfrac*1.126
    
    #itcz_coeff=0.5+(planet_tmean-14)*0.1
    itcz_coeff=0.5
    #print(lapse_rate)    
    #quit(-1)
    korkeus, leveys = korkeus_syvyyskartta.shape



    # ============================================================
    # PROFILOINTI
    # ============================================================

    # Kokonaisajat
    profiili = defaultdict(float)

    # Kuukausikohtaiset ajat
    kuukausi_profiili = {
        nimi: [0.0] * 12
        for nimi in [
            "aurinko",
            "tuulet",
            "orografia",
            "merivirrat",
            "perus_sade",
            "tarkenna_sadevyohykkeet",
            "tuulietaisyys_meresta",
            "merelta_tuleva_kosteus",
            "orografiakerroin",
        ]
    }

    # Staattisen vaiheen ajat
    staattinen_profiili = defaultdict(float)

    def mittaa_gpu(nimi, funktio, *args, kuukausi=None, **kwargs):
        """
        Mittaa GPU-funktion CUDA Event -ajoilla.

        Tulos palautetaan normaalisti kuten alkuperäinen funktio.

        nimi:
            Profilointinimi.

        kuukausi:
            Jos annettu 0-11, tallennetaan myös kuukausikohtainen aika.
        """

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()

        tulos = funktio(*args, **kwargs)

        end.record()
        end.synchronize()

        aika_ms = cp.cuda.get_elapsed_time(start, end)
        aika_s = aika_ms / 1000.0

        profiili[nimi] += aika_s

        if kuukausi is not None:
            if nimi not in kuukausi_profiili:
                kuukausi_profiili[nimi] = [0.0] * 12

            kuukausi_profiili[nimi][kuukausi] += aika_s

        return tulos

    def mittaa_gpu_staattinen(nimi, funktio, *args, **kwargs):
        """
        Staattisen GPU-vaiheen mittaus.
        """

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        start.record()

        tulos = funktio(*args, **kwargs)

        end.record()
        end.synchronize()

        aika_ms = cp.cuda.get_elapsed_time(start, end)
        aika_s = aika_ms / 1000.0

        staattinen_profiili[nimi] += aika_s

        return tulos

    # ============================================================
    # KOKO FUNKTION CPU-AJAN MITTAUS
    # ============================================================

    kokonais_start = time.perf_counter()

    print("GPU laskenta ...")

    # ============================================================
    # 1. KAIKKI STAATTINEN DATA GPU:LLE
    # ============================================================

    staattinen_start = time.perf_counter()

    leveysasteet = np.linspace(
        90,
        -90,
        korkeus,
        dtype=np.float32
    )

    pituusasteet = np.linspace(
        -180,
        180,
        leveys,
        dtype=np.float32
    )

    dlat = 180.0 / (korkeus - 1)
    dlon = 360.0 / (leveys - 1)

    lon_grid_deg_cpu, lat_grid_deg_cpu = np.meshgrid(
        pituusasteet,
        leveysasteet
    )

    # ------------------------------------------------------------
    # CPU -> GPU
    # ------------------------------------------------------------

    gpu_transfer_start = time.perf_counter()

    lat_grid_deg = cp.asarray(lat_grid_deg_cpu)
    lon_grid_deg = cp.asarray(lon_grid_deg_cpu)

    korkeuskartta = cp.asarray(
        np.maximum(korkeus_syvyyskartta, 0),
        dtype=cp.float32
    )

    syvyyskartta = cp.asarray(
        np.minimum(korkeus_syvyyskartta, 0),
        dtype=cp.float32
    )

    cp.cuda.Stream.null.synchronize()

    gpu_transfer_aika = (
        time.perf_counter() - gpu_transfer_start
    )

    staattinen_profiili["CPU -> GPU siirrot"] += gpu_transfer_aika

    # ------------------------------------------------------------
    # Lämpötilan korkeusvaikutus
    # ------------------------------------------------------------

    lampotila_start = cp.cuda.Event()
    lampotila_end = cp.cuda.Event()

    lampotila_start.record()

    lämpötila_vähete = korkeuskartta * lapse_rate

    lampotila_end.record()
    lampotila_end.synchronize()

    staattinen_profiili["lämpötila_vähete"] += (
        cp.cuda.get_elapsed_time(
            lampotila_start,
            lampotila_end
        ) / 1000.0
    )

    staattinen_cpu_aika = time.perf_counter() - staattinen_start

    # ============================================================
    # 2. GEOMETRIA
    # ============================================================

    geometria_start = cp.cuda.Event()
    geometria_end = cp.cuda.Event()

    geometria_start.record()

    R = 6371.0088 * planet_radius_re

    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    lat_rad = cp.deg2rad(lat_grid_deg)

    solukoko_y_km = R * dlat_rad

    solukoko_x_km = (
        R *
        dlon_rad *
        cp.cos(lat_rad)
    )

    geometria_end.record()
    geometria_end.synchronize()

    staattinen_profiili["geometria"] += (
        cp.cuda.get_elapsed_time(
            geometria_start,
            geometria_end
        ) / 1000.0
    )

    # ============================================================
    # 3. GRADIENTIT GPU:LLA
    # ============================================================

    gradient_start = cp.cuda.Event()
    gradient_end = cp.cuda.Event()

    gradient_start.record()

    korkeus_gradientti_x = cp.gradient(
        korkeuskartta,
        axis=1
    )

    korkeus_gradientti_y = cp.gradient(
        korkeuskartta,
        axis=0
    )

    syvyys_gradientti_x = cp.gradient(
        syvyyskartta,
        axis=1
    )

    syvyys_gradientti_y = cp.gradient(
        syvyyskartta,
        axis=0
    )

    gradient_end.record()
    gradient_end.synchronize()

    staattinen_profiili["gradientit"] += (
        cp.cuda.get_elapsed_time(
            gradient_start,
            gradient_end
        ) / 1000.0
    )

    # ============================================================
    # 4. MAA / MERI
    # ============================================================

    maa_start = cp.cuda.Event()
    maa_end = cp.cuda.Event()

    maa_start.record()

    maa_maski = korkeuskartta > 0
    meri_bin = ~maa_maski
    maa_maski2 = np.where(korkeuskartta>0,1,0).astype(int)
 
    maa_end.record()
    maa_end.synchronize()

    staattinen_profiili["maa_meri_maskit"] += (
        cp.cuda.get_elapsed_time(
            maa_start,
            maa_end
        ) / 1000.0
    )

    # ============================================================
    # 5. ETÄISYYS RANNIKOSTA
    # ============================================================

    etaisyys_meresta_km = mittaa_gpu_staattinen(
        "etaisyys_rannikosta",
        laske_etaisyys_rannikosta_gpu,
        maa_maski,
        maapallon_sade_km=R,
    )

    # ============================================================
    # 6. OUTPUT-ARRAYT
    # ============================================================

    output_start = cp.cuda.Event()
    output_end = cp.cuda.Event()

    output_start.record()

    kuukausi_lämpötilat_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_sateet_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_tuuli_suunta_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_tuuli_voima_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_merivirta_x_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_merivirta_y_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    output_end.record()
    output_end.synchronize()

    staattinen_profiili["output-arrayt"] += (
        cp.cuda.get_elapsed_time(
            output_start,
            output_end
        ) / 1000.0
    )

    # ============================================================
    # 7. KUUKAUSISILMUKKA
    # ============================================================


    for kk in range(12):

        print(f"Kuukausi {kk + 1}/12")

        # ========================================================
        # AURINKO
        # ========================================================
        ## vanha koodi: vain ilmastolline päiväntassaja
        #peruslampotulos = mittaa_gpu(
        #    "aurinko",
        #    laske_auringon_peruslampotila,
        #    lat_grid_deg,
        #    kk,
        #    planet_tilt,
        #    planet_ecc,
        #    planet_mvelp_deg,
        #    tmax_default,
        #    kuukausi=kk,leveysaste_viilennys_kerroin=leveysaste_viilennys_kerroin
        #)

        #peruslampotila = peruslampotulos[0]

        #ilmastollinen_paivantaasaaja = (
        #    peruslampotulos[1]
        #)

        #kuukausi_lämpötilat_gpu[kk] = peruslampotila
        dt_years = planet_orbital_period/ 12.0
        time_years = kk * dt_years
        
        #leveysaste_viilennys_kerroin=3

        (
        perus_lampotila,
        deklinaatio,
        sateily_kerroin,
        ratakulma_rad,
        ) = laske_auringon_peruslampotila(maa_maski2=maa_maski2,
        lat=lat_grid_deg,
        time_years=time_years,
        tilt_planet_axis=planet_tilt,
        ecc_planet=planet_ecc,
        mvelp_planet=planet_mvelp_deg,
        tmax_planet_param=tmax_default,
        orbital_period_years=planet_orbital_period,
        planet_mass_me=planet_mass_me,
        planet_radius_re=planet_radius_re,
        planet_atmosphere_pressure=planet_atmosphere_pressure*1013000,
        planet_rotation_period=planet_rotation_period_hours,
        leveysaste_viilennys_kerroin=leveysaste_viilennys_kerroin,
        )
        

        ilmastollinen_paivantaasaaja=deklinaatio*itcz_coeff
        
        #tavoite_lampotila = laske_vuosikierron_tavoite(
        #perus_lampotila=perus_lampotila,
        #ratakulma_rad=ratakulma,
        #vuodenaika_amplitudi=8.0,
        #)

        #lampotila = paivita_maa_meri_lampotila(
        #edellinen_lampotila=tavoite_lampotila,
        #tavoite_lampotila=tavoite_lampotila,
        #landmask=maa_maski,
        #dt_years=dt_years,
        #)
        #kuukausi_lämpötilat_gpu[kk] = lampotila
        kuukausi_lämpötilat_gpu[kk] = perus_lampotila

        # ========================================================
        # TUULI
        # ========================================================

        perustuulitulos = mittaa_gpu(
            "tuulet",
            laske_tuulet,
            lat_grid_deg,
            kk,
            planet_tilt,
            kuukausi=kk, itcz_kerroin=itcz_coeff
        )

        tuuli_x = perustuulitulos[0]*1
        tuuli_y = perustuulitulos[1]*1
        tuuli_x_yksikko = perustuulitulos[2]
        tuuli_y_yksikko = perustuulitulos[3]
        tuuli_suunta = perustuulitulos[4]
        tuuli_voima = perustuulitulos[5]

        kuukausi_tuuli_suunta_gpu[kk] = tuuli_suunta
        kuukausi_tuuli_voima_gpu[kk] = tuuli_voima

 			
        # ========================================================
        # OROGRAFIA
        # ========================================================

        orografiatulos = mittaa_gpu(
            "orografia",
            laske_orografia,
            korkeuskartta,
            korkeus_gradientti_x,
            korkeus_gradientti_y,
            tuuli_x_yksikko,
            tuuli_y_yksikko,
            tuuli_voima,
            kuukausi=kk,
        )

        maasto_gradientti_tuulen_suunnassa = (
            orografiatulos[0]
        )

        orografinen_pakote = (
            orografiatulos[1]
        )

        noususade = orografiatulos[2]
        sadevarjo = orografiatulos[3]

        # ========================================================
        # MERIVIRRAT
        # ========================================================

        merivirrat_tulos = mittaa_gpu(
            "merivirrat",
            laske_merivirrat,
            lat_grid_deg,
            syvyyskartta,
            meri_bin,
            etaisyys_meresta_km,
            tuuli_x_yksikko,
            tuuli_y_yksikko,
            tuuli_voima,
            kk,
            ilmastollinen_paivantaasaaja,
            kuukausi=kk,
        )

        virta_x = merivirrat_tulos[0]
        virta_y = merivirrat_tulos[1]
        virta_voima = merivirrat_tulos[2]
        virta_suunta = merivirrat_tulos[3]
        merivesi_anomalia = merivirrat_tulos[4]

        kuukausi_merivirta_x_gpu[kk] = virta_x
        kuukausi_merivirta_y_gpu[kk] = virta_y

        # ========================================================
        # PERUSSATEET
        # ========================================================

        perussade = mittaa_gpu(
            "perus_sade",
            laske_perus_sade,
            lat_grid_deg,
            ilmastollinen_paivantaasaaja,
            tropiikin_max_sade=180.0,
            lauhkean_max_sade=60.0,
            lauhkean_leveys=15.0,
            tropiikin_leveys=8.0,
            kuukausi=kk,
        )
        perussade=perussade*global_moisture_coeff
        sade=perussade
        # ========================================================
        # SADEVYÖHYKKEIDEN TARKENNUS
        # ========================================================

        #perussade = mittaa_gpu(
        #    "tarkenna_sadevyohykkeet",
        perussade=tarkenna_sadevyohykkeet(
            perussade,
            lat_grid_deg,
            ilmastollinen_paivantaasaaja,
            #kuukausi=kk,
        )
        #kuukausi_sateet_gpu[kk]=perussade
        #sade=perussade
        #sade_end.record()
        #sade_end.synchronize()

        #profiili["sateen_tallennus"] += (
        #    cp.cuda.get_elapsed_time(
        #        sade_start,
        #        sade_end
        #    ) / 1000.0
        #)

        # ========================================================
        # SATEEN VAIKUTUS LÄMPÖTILAAN
        # ========================================================

        vaikutus_start = cp.cuda.Event()
        vaikutus_end = cp.cuda.Event()

        vaikutus_start.record()

        kuukausi_lämpötilat_gpu[kk] = (
            kuukausi_lämpötilat_gpu[kk]
            - sade / 100.0
        )

        meri_pilvisyys_vaikutus = -7+meri_bin * -5
        #mantereisuus_anomalia=(rannikko_etaisyys_gpu/1000)*kuukausi_lämpötilat_gpu[kk]/cp.abs(kuukausi_lämpötilat_gpu[kk])
        #mantereisuus_anomalia = (rannikko_etaisyys_gpu / 100) * cp.sign(kuukausi_lämpötilat_gpu[kk])
        # Mitä enemmän sadetta (mm), sitä pienempi kerroin (asteikko esim. 100 mm)
        #sade_vaimennus_lampotilaan = cp.exp(-kuukausi_sateet_gpu[kk] / 100.0)

        # Lämpötilan pehmennys: lähellä nollaa vaikutus on pieni, kaukana nollasta se vakiintuu
        #lampotila_pehmeä = cp.tanh(kuukausi_lämpötilat_gpu[kk] / 5.0)
        #mantereisuus_anomalia = (rannikko_etaisyys_gpu / 500) * lampotila_pehmeä * sade_vaimennus_lampotilaan
        # Pehmennetään etäisyyden kasvu logaritmillä (lisätään 1, jotta log(1) = 0 rannikolla)
        #etaisyys_pehmeä = cp.log1p(rannikko_etaisyys_gpu / 20)+1

        # Sade vähentää mantereisuutta (mitä kuivempaa, sen mantereisempaa)
        # cp.maximum estää nollalla jakamisen, jos sadetta on 0 mm
        #sade_kerroin_ma = 100.0 / cp.maximum(kuukausi_sateet_gpu[kk], 10.0)
        #sade_kerroin_ma=1

        #mantereisuus_anomalia = etaisyys_pehmeä * cp.tanh(kuukausi_lämpötilat_gpu[kk] / 100.0) * sade_vaimennus_lampotilaan
        #sade_efekti_lampotila = 1.0 / (1.0 + (kuukausi_sateet_gpu[kk] / 50.0))

        #lampotila_vaikutus = kuukausi_lämpötilat_gpu[kk] / (cp.abs(kuukausi_lämpötilat_gpu[kk]) + 2.0)

        #mantereisuus_anomalia = (rannikko_etaisyys_gpu / 100) * lampotila_vaikutus * sade_efekti_lampotila
        # Määritetään herkkyys: esim. 75 mm kuukausisadetta puolittaa ääri-ilmiöt
        sade_asteikko = 75.0 

        # Mitä enemmän sadetta, sitä lähemmäs nollaa tämä kerroin painuu
        sade_puskuri = cp.exp(-kuukausi_sateet_gpu[kk] / sade_asteikko)

        # Pehmeä lämpötilaefektin kasvu tanh-funktiolla (esim. 5 asteen skaalauksella)
        lampotila_pehmeä = cp.tanh(kuukausi_lämpötilat_gpu[kk] / 5.0)

        # Lopullinen anomalia
        mantereisuus_anomalia = (rannikko_etaisyys_gpu / 300) * lampotila_pehmeä * sade_puskuri

 
        #plt.imshow(mantereisuus_anomalia.get())
        #plt.show()
        kuukausi_lämpötilat_gpu[kk]=kuukausi_lämpötilat_gpu[kk]+meri_pilvisyys_vaikutus+merivesi_anomalia
        kuukausi_lämpötilat_gpu[kk]=kuukausi_lämpötilat_gpu[kk]+mantereisuus_anomalia      
    
        lapse_ratex=laske_lapse_rate_gpu(kuukausi_lämpötilat_gpu[kk], sade, korkeuskartta, planet_atmosphere_pressure*101325, g=9.81*gee, sademuutos_raja=0.0)
        lämpötila_vähetex=lapse_ratex*korkeuskartta
        kuukausi_lämpötilat_gpu[kk]=kuukausi_lämpötilat_gpu[kk]+lämpötila_vähetex
        #kuukausi_lämpötilat_gpu[kk]=kuukausi_lämpötilat_gpu[kk]+lämpötila_vähete
        #kuukausi_lämpötilat_gpu[kk] = (
        #    kuukausi_lämpötilat_gpu[kk]
        #    + meri_pilvisyys_vaikutus
        #    + lämpötila_vähete+merivesi_anomalia
        #)
        #plt.imshow(tuuli_x.get())
        #plt.show()
        #quit(-1)
        #tuuli_x, tuuli_y=laske_mantereiset_painesolut_ja_tuuli(tuuli_x, tuuli_y, kuukausi_lämpötilat_gpu[kk], maa_maski, 
        #                                 rannikko_etaisyys_gpu, dy=1.0, dx=1.0, 
        #                                 paine_efekti_voimakkuus=0.8)


        tuuli_x, tuuli_y=laske_painesolujen_pyorteet(tuuli_x, tuuli_y, lat_grid_deg, kuukausi_lämpötilat_gpu[kk], maa_maski, 
                               rannikko_etaisyys_gpu, dy=1.0, dx=1.0, 
                               paine_suora_voimakkuus=0.3,   # Suoraan paineesta ulos/sisään puhaltava tuuli
                               paine_pyörre_voimakkuus=0.7*1) # Coriolis-efektin
        tuuli_x, tuuli_y=laske_monsuunituuli(tuuli_x, tuuli_y, kuukausi_lämpötilat_gpu[kk], maa_maski, dx=1.0, dy=1.0, monsuuni_voimakkuus=0.5)
        #plot_grid_streamplot(tuuli_x.get(), tuuli_y.get())
  
        #monsuuni_sade_muutos=laske_monsuunin_sade_muutos(tuuli_x, tuuli_y, cp.array(maa_maski, dtype=cp.float32), kuukausi_lämpötilat_gpu[kk], 
        #                        monsuuni_sade_kerroin=50.0)
        #monsuuni_sade_muutos=laske_monsuunin_sade_muutos_laaja(tuuli_x, tuuli_y, maa_maski, kuukausi_lämpötilat_gpu[kk], 
        #                              monsuuni_sade_kerroin=50.0)
        #monsuuni_sade_muutos=laske_monsuunin_sade_muutos_laaja(tuuli_x, tuuli_y, maa_maski, 
        #                              kuukausi_lämpötilat_gpu[kk],rannikko_etaisyys_gpu, 
        #                              monsuuni_sade_kerroin=50.0,
        #                              kosteuden_kantama_km=200.0)
        monsuuni_sade_muutos=laske_monsuunin_sade_muutos_laaja(tuuli_x, tuuli_y, maa_maski,kuukausi_lämpötilat_gpu[kk], 
                                      rannikko_etaisyys_gpu,
                                      monsuuni_sade_kerroin=15.0,
                                      kosteuden_kantama_km=400.0,
                                      talvikuivuus_kerroin=25.0)

        #plt.imshow(monsuuni_sade_muutos.get())
        #plt.show()
        #quit(-1)
        kuukausi_sateet_gpu[kk]=kuukausi_sateet_gpu[kk]+monsuuni_sade_muutos
        # ========================================================
        # TUULEN ETÄISYYS MERESTÄ
        # ========================================================
        sade=kuukausi_sateet_gpu[kk]
        tuulietaisyys_tulos = mittaa_gpu(
            "tuulietaisyys_meresta",
            laske_tuulietaisyys_meresta_gpu,
            meri_bin,
            tuuli_x,
            tuuli_y,
            lat_grid_deg,
            planet_radius_re=planet_radius_re,
            max_pixel_etaisyys=500,
            kuukausi=kk,
        )

        tuulietaisyys_meresta_km = (
            tuulietaisyys_tulos
        )

        # ========================================================
        # MERELTÄ TULEVA KOSTEUS
        # ========================================================

        merelta_tuleva_kosteus = mittaa_gpu(
            "merelta_tuleva_kosteus",
            laske_merelta_kulkeva_kosteus,
            tuulietaisyys_meresta_km,
            kosteus_vaikutusmatka_km=700.0,
            kosteus_max=1.5,
            kuukausi=kk,
        )

        sade = (
            perussade *
            merelta_tuleva_kosteus
        )

        # ========================================================
        # OROGRAFINEN VAIKUTUS
        # ========================================================

        sade_orografia_vaikutuskerroin = mittaa_gpu(
            "orografiakerroin",
            laske_orografiakerroin,
            korkeuskartta,
            orografinen_pakote,
            korkeus_kerroin=0.00005,
            nousu_kerroin=0.15,
            lasku_kerroin=0.08,
            kuukausi=kk,
        )

        sade = (
            sade *
            sade_orografia_vaikutuskerroin
        )

        # ========================================================
        # LOPULLINEN SADE
        # ========================================================

        sade_start = cp.cuda.Event()
        sade_end = cp.cuda.Event()

        sade_start.record()

        kuukausi_sateet_gpu[kk] = sade



        kuukausi_lämpötilat_gpu[kk]=kuukausi_lämpötilat_gpu[kk]+(monsuuni_sade_muutos/100)
        vaikutus_end.record()
        vaikutus_end.synchronize()

        profiili["lämpötilan_loppukäsittely"] += (
            cp.cuda.get_elapsed_time(
                vaikutus_start,
                vaikutus_end
            ) / 1000.0
        )

    # ============================================================
    # KOKO GPU:N SYNKRONOINTI
    # ============================================================

    cp.cuda.Stream.null.synchronize()

    kokonais_aika = (
        time.perf_counter() - kokonais_start
    )

    # ============================================================
    # PROFILOINTIRAPORTTI
    # ============================================================

    print()
    print("=" * 90)
    print("GPU-LASKENNAN PROFILOINTI")
    print("=" * 90)

    print()
    print("KUUKAUSISILMUKAN ALIOHJELMAT")
    print("-" * 90)

    # Kaikki kuukausifunktiot
    kuukausifunktiot = [
        nimi
        for nimi in profiili
        if nimi in kuukausi_profiili
    ]

    kuukausifunktiot.sort(
        key=lambda nimi: profiili[nimi],
        reverse=True
    )

    kuukausi_kokonais = sum(
        profiili[nimi]
        for nimi in kuukausifunktiot
    )

    print(
        f"{'Funktio':35s}"
        f"{'Yhteensä':>12s}"
        f"{'ka/kk':>12s}"
        f"{'%':>9s}"
    )

    print("-" * 90)

    for nimi in kuukausifunktiot:

        aika = profiili[nimi]

        prosentti = (
            100.0 * aika / kuukausi_kokonais
            if kuukausi_kokonais > 0
            else 0.0
        )

        keskiarvo = aika / 12.0

        print(
            f"{nimi:35s}"
            f"{aika:12.4f} s"
            f"{keskiarvo:12.4f} s"
            f"{prosentti:8.2f} %"
        )

    # ============================================================
    # KUUKAUSITTAINEN PROFILOINTI
    # ============================================================

    print()
    print("=" * 90)
    print("AIKA KUUKAUSITTAIN")
    print("=" * 90)

    kuukaudet = [
        "Tammi",
        "Helmi",
        "Maalis",
        "Huhti",
        "Touko",
        "Kesä",
        "Heinä",
        "Elo",
        "Syys",
        "Loka",
        "Marras",
        "Joulu",
    ]

    print()
    print(
        f"{'Funktio':30s}",
        end=""
    )

    for nimi in kuukaudet:
        print(f"{nimi:>9s}", end="")

    print()

    print("-" * 140)

    for nimi in kuukausifunktiot:

        print(f"{nimi:30s}", end="")

        ajat = kuukausi_profiili[nimi]

        for aika in ajat:
            print(
                f"{aika:9.3f}",
                end=""
            )

        print()

    # ============================================================
    # HITAIN KUUKAUSI / FUNKTIO
    # ============================================================

    print()
    print("=" * 90)
    print("HITAIN KUUKAUSI KULLEKIN FUNKTIOLLE")
    print("=" * 90)

    for nimi in kuukausifunktiot:

        ajat = kuukausi_profiili[nimi]

        hitain_kk = max(
            range(12),
            key=lambda i: ajat[i]
        )

        hitain_aika = ajat[hitain_kk]

        print(
            f"{nimi:35s}: "
            f"{kuukaudet[hitain_kk]:>6s} "
            f"{hitain_aika:.4f} s"
        )

    # ============================================================
    # STAATTINEN VAIHE
    # ============================================================

    print()
    print("=" * 90)
    print("STAATTINEN ALUSTUS")
    print("=" * 90)

    print(
        f"{'Toiminto':35s}"
        f"{'Aika':>15s}"
    )

    print("-" * 90)

    staattinen_yhteensa = sum(
        staattinen_profiili.values()
    )

    for nimi, aika in sorted(
        staattinen_profiili.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{nimi:35s}"
            f"{aika:15.4f} s"
        )

    print("-" * 90)

    print(
        f"{'Staattinen yhteensä':35s}"
        f"{staattinen_yhteensa:15.4f} s"
    )

    # ============================================================
    # KOKO PROFIILI
    # ============================================================

    print()
    print("=" * 90)
    print("KOKO PROFILOINTI")
    print("=" * 90)

    print(
        f"{'Toiminto':35s}"
        f"{'Aika':>15s}"
        f"{'Osuus':>10s}"
    )

    print("-" * 90)

    kaikki_profiilit = {}

    for nimi, aika in staattinen_profiili.items():
        kaikki_profiilit[
            "STAATTINEN: " + nimi
        ] = aika

    for nimi, aika in profiili.items():
        kaikki_profiilit[
            nimi
        ] = aika

    mitattu_yhteensa = sum(
        kaikki_profiilit.values()
    )

    for nimi, aika in sorted(
        kaikki_profiilit.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        prosentti = (
            100.0 * aika / mitattu_yhteensa
            if mitattu_yhteensa > 0
            else 0.0
        )

        print(
            f"{nimi:35s}"
            f"{aika:15.4f} s"
            f"{prosentti:9.2f} %"
        )

    # ============================================================
    # YHTEENVETO
    # ============================================================

    print()
    print("=" * 90)
    print("YHTEENVETO")
    print("=" * 90)

    print(
        f"Funktio kokonaisuudessaan: "
        f"{kokonais_aika:.4f} s"
    )

    print(
        f"Mitattu GPU-laskenta: "
        f"{mitattu_yhteensa:.4f} s"
    )

    print(
        f"Staattinen vaihe: "
        f"{staattinen_yhteensa:.4f} s"
    )

    print()

    if kuukausifunktiot:

        hitain_funktio = max(
            kuukausifunktiot,
            key=lambda nimi: profiili[nimi]
        )

        print(
            f"HITAIN kuukausifunktio: "
            f"{hitain_funktio} "
            f"({profiili[hitain_funktio]:.4f} s)"
        )

    if staattinen_profiili:

        hitain_staattinen = max(
            staattinen_profiili,
            key=staattinen_profiili.get
        )

        print(
            f"HITAIN staattinen vaihe: "
            f"{hitain_staattinen} "
            f"({staattinen_profiili[hitain_staattinen]:.4f} s)"
        )

    print("=" * 90)

    # ============================================================
    # PALUU
    # ============================================================

    return (
        kuukausi_lämpötilat_gpu,
        kuukausi_sateet_gpu,
        kuukausi_tuuli_suunta_gpu,
        kuukausi_tuuli_voima_gpu,
        kuukausi_merivirta_x_gpu,
        kuukausi_merivirta_y_gpu,
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
        mean_liike_u = cp.nan_to_num(
        cp.mean(liike_u),
        nan=0.0,
        posinf=2.0,
        neginf=-2.0
        )

        mean_liike_v = cp.nan_to_num(
        cp.mean(liike_v),
        nan=0.0,
        posinf=2.0,
        neginf=-2.0
        )

        shift_x = int(cp.clip(mean_liike_u, -2, 2).get())
        shift_y = int(cp.clip(mean_liike_v, -2, 2).get())

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


mapper = PlanetMap(
    leveys=leveys,
    korkeus=korkeus,

    syvin_kohta=syvin_kohta,
    korkein_kohta=korkein_kohta,

    seed=seed1,
)

korkeus_syvyyskartta, korkeuskartta, maa_maski, referenssi, merenpinta = (
    mapper.generoi_kartta(
        alue=(-180, 180, -90, 90),

        leveys=leveys,
        korkeus=korkeus,

        octaves=8,

        oceanfrac=oceanfrac,
    )
)

meri_maski=np.where(np.copy(maa_maski)==1,0,1)


print("Merenpinta:", merenpinta)

print(
    "Meriosuus:",
    np.count_nonzero(
        ~maa_maski
    ) / maa_maski.size
)

print(
    "Maaosuus:",
    np.count_nonzero(
        maa_maski
    ) / maa_maski.size
)

#plt.imshow(
#    korkeuskartta,
#    cmap="terrain"
#)

#plt.show()

korkeus_syvyyskartta2, korkeuskartta2, maa_maski2, referenssi2, merenpinta2 = (
    mapper.generoi_kartta(
        alue=(0, 90, 0, 90),

        leveys=100,
        korkeus=100,

        referenssi=referenssi,
    )
)

print("Alkuperäinen merenpinta:", merenpinta)
print("Zoomin merenpinta:", merenpinta2)
print("Sama:", merenpinta == merenpinta2)

#plt.imshow(
#    korkeuskartta2,
#    cmap="terrain"
#)

#plt.show()



#quit(-1)



#plt.imshow(korkeuskartta )
#plt.show()
#plt.imshow(zoomattu_korkeuskartta )
#plt.show()
#quit(-1)



korkeus, leveys = maa_maski.shape
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


maa_maski_gpu = cp.asarray(maa_maski)


n_lats, n_lons = korkeuskartta.shape
lats = np.linspace(-90, 90, n_lats)
cos_weights = np.cos(np.radians(lats))
weight_grid = cos_weights[:, np.newaxis] * np.ones(n_lons)
maa_painotettu = np.sum(weight_grid[maa_maski])
meri_painotettu = np.sum(weight_grid[meri_maski])
kokonais_paino = np.sum(weight_grid)
maa_prosentti = (maa_painotettu / kokonais_paino) * 100
meri_prosentti = (meri_painotettu / kokonais_paino) * 10
maan_keskikorkeus = np.sum(korkeuskartta[maa_maski] * weight_grid[maa_maski]) /maa_painotettu

print(f"Land : {maa_prosentti:.2f} %")
print(f"Ocean: {meri_prosentti:.2f} %")
print(f"Mean land height: {maan_keskikorkeus:.1f} metriä")

landfrac=maa_prosentti/100
oceanfrac=meri_prosentti/100

print("landfrac, oceanfrac in effect", landfrac, oceanfrac)





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



#plt.imshow(etaisyys_meresta_km)

#plt.show()
#quit(-1)


#meri_y, meri_x = np.where(
#    ~maa_maski
#)


#meri_coords = np.radians(
#    np.column_stack(
#        (
#            lats[meri_y],
#            lons[meri_x],
#        )
#    )
#)


#maa_y, maa_x = np.where(
#    maa_maski
#)

#maa_coords = np.radians(
#    np.column_stack(
#        (
#            lats[maa_y],
#            lons[maa_x],
#        )
#    )
#)

print("Distance to coast ..")

varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)

#korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)
#korkeus_gradientti_y = np.gradient(korkeuskartta, axis=0)




# =====================================================================
# 4. KUUKAUSITTAINEN LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS (12 KUUKAUTTA)
# =====================================================================
kuukausi_lämpötilat = np.zeros((12, korkeus, leveys))
kuukausi_sateet = np.zeros((12, korkeus, leveys))

planet_global_moisture_coeff=planet_global_moisture_coeff*f_precip_relto_earthlike(oceanfrac/0.71)
#planet_global_moisture_coeff=oceanfrac 
 
 
print("Global moisture coeff ",planet_global_moisture_coeff ) 
 
#warmup_gpu(seconds=2.0)
    
#benchmark_ilmasto(korkeus_syvyyskartta)
#benchmark_multiple(korkeus_syvyyskartta)
#quit(-1)



print ("Calculate climate ...")

# Ensimmäinen ajo = warm-up


## 0) Laske ilmasto

kuukausi_lämpötilat_gpu, kuukausi_sateet_gpu, \
kuukausi_tuuli_suunta_gpu, kuukausi_tuuli_voima_gpu, \
kuukausi_merivirta_x_gpu, kuukausi_merivirta_y_gpu=laske_ilmasto_gpu(
    korkeus_syvyyskartta,
    planet_mass_me=planet_mass_me,
    planet_radius_re=planet_radius_re,
    planet_tilt=planet_axis_tilt_degrees,
    planet_ecc=planet_ecc,
    planet_mvelp_deg=planet_mvelp_degrees,
    planet_orbital_period=planet_orbital_period,
    planet_rotation_hours=planet_rotation_period_hours,
    planet_atmosphere_pressure=planet_atmosphere_pressure,
    landfrac=landfrac,
    planet_tmean=planet_tmean, ## 14.8
    global_moisture_coeff=planet_global_moisture_coeff,)


#korkeus_syvyyskartta, planet_mass_me=1, planet_radius_re=1.0, planet_tilt=23.44, planet_ecc=0.013, planet_mvelp_deg=101.2, tmax_default=tmax_planet, global_moisture_coeff=global_moisture_coeff)

#print("Ilmasto laskettu, debug BRK")

#quit(-1)

#warmup_gpu()




print("...")
kuukausi_lämpötilat=kuukausi_lämpötilat_gpu.get()
kuukausi_sateet=kuukausi_sateet_gpu.get()

kuukausi_tuuli_suunta=kuukausi_tuuli_suunta_gpu.get()
kuukausi_tuuli_voima=kuukausi_tuuli_voima_gpu.get()
kuukausi_merivirta_x=kuukausi_merivirta_x_gpu.get()
kuukausi_merivirta_y=kuukausi_merivirta_y_gpu.get()
kuukausi_merivirta=np.sqrt(kuukausi_merivirta_x*kuukausi_merivirta_x+kuukausi_merivirta_y*kuukausi_merivirta_y)

   

jaa, merijaa_aina=laske_jaat(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuosia=200)  
    
    

#plot_grid_streamplot(kuukausi_merivirta_x[0], kuukausi_merivirta_y[0], alue=[-180, 180, -90, 90])


#plt.imshow(maa_maski)

maa_maski_gpu=maa_maski.astype(bool)

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
    maa_maski_gpu,
    etaisyys_meresta_km
)


tilasto_tulokset = laske_ilmastotilastot(
    kuukausi_lämpötilat,
    kuukausi_sateet
)

print(" Mean annual temp  Ta_mean :", round(tilasto_tulokset[ "painotettu_keskilampo"],2))
print(" Mean sum  precip  Pr_ann  :", round(tilasto_tulokset[ "keskiarvoinen_vuosisade"]))
print(" Min annual temp   T_min   :", round(tilasto_tulokset["min_lampo" ],2))
print(" Max annual temp   T_max   :", round(tilasto_tulokset[ "max_lampo"],2))



plot_tmean_raster(korkeuskartta, tmean_annual, title="Annual mean temperature degC")

plot_precip_raster(korkeuskartta, precip_annual, title="Annual precipitation mm")

# Ajetaan funktio esimerkiksi koordinaateilla Lon=25.0, Lat=60.0 (Helsingin suunnilla)
piirra_ilmastodiagrammi(
    lon=25.0,
    lat=60.0,
    korkeuskartta=korkeuskartta,
    kuukausi_lampotilat=kuukausi_lämpötilat,
    kuukausi_sateet=kuukausi_sateet,
)

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

plt.imshow(np.exp(varjo_maski), cmap="gray", alpha=0.1,extent=[-180, 180, -90, 90])

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
