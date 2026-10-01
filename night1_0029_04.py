
#################################################
#
## Simple planet climate Köppen classes
#  Wordbuilding and game purposes
#
# Python3+CuPy - requires CUDA
#
## 1.10.2026 0000.0029.04
#
#################################################


import sys, os

import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import LightSource  
from matplotlib.colors import Normalize

import math
import numpy as np

import heapq
from dataclasses import dataclass

import scipy
import scipy.ndimage as ndimage

from scipy.ndimage import distance_transform_edt
from scipy.ndimage import shift

from scipy.interpolate import griddata

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor ## good
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

import cupy as cp
import cupyx.scipy.ndimage as ndimage2

from noise import pnoise3
from pygam import LinearGAM, s

import cartopy.crs as ccrs




# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus =180*5
leveys = 360*5

#korkeus =64

#leveys = 128



#seed1=9 ##ok
#seed1 = 53 ## hyva
#seed1=3333
#seed1=44
#seed1=33
#seed1 = 233 ## hyva
#seed1=61 ## hyva monsuuniin

seed1=77


#maapallon_sade_km = 6371.0
#syvin_kohta=-11000
#korkein_kohta=8848
#manner_osuus=0.30

maapallon_sade_km = 6371.0
syvin_kohta=-10000
korkein_kohta=8000
#manner_osuus=0.30
manner_osuus=0.3 ## continent fraction preset

oceanfrac=1-manner_osuus


#tilt_planet_axis=23.44
#ecc_planet=0.013
#mvelp_planet=102.0
# Parametrit (esimerkkinä Maan arvot)



planet_mass_me=1
planet_radius_re=math.pow(planet_mass_me,0.27) ## 
planet_atmosphere_pressure=math.pow(planet_radius_re, 1.6) ## hrom hat
planet_co2_ppm=280

planet_axis_tilt_degrees = 23.44*1
planet_ecc= 0.013*1
planet_mvelp_degrees = 102.0  # Perihelin pituus asteina (Maa saavuttaa perihelin tammikuun alussa)
planet_orbital_period=1
planet_rotation_period_hours=24

## guess planet tmean

planet_tmean=14.8
planet_global_moisture_coeff=1


star_teff_k=5778
star_mass_msun=1
star_luminosity_lsun=1



h = 6.62607015e-34
c = 299792458.0
k = 1.380649e-23
NA = 6.02214076e23

P_EARTH = 101325.0

###########################################
#######################################
### kartta





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
        syvin_kohta=-11000,
        korkein_kohta=8000,
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




def coord_to_pixel_global(lons, lats, kuvan_korkeus, kuvan_leveys):
    """
    Muuntaa globaalin lon/lat-koordinaatin rasteripikseliksi.

    Kartan alue:
        lon: -180 ... 180
        lat:   90 ... -90

    Palauttaa:
        y, x
    """

    lons = np.asarray(lons)
    lats = np.asarray(lats)

    x = (lons + 180) / 360 * (kuvan_leveys - 1)
    y = (90 - lats) / 180 * (kuvan_korkeus - 1)

    x = np.rint(x).astype(int)
    y = np.rint(y).astype(int)

    # Varmistetaan, etteivät koordinaatit mene rasterin ulkopuolelle
    x = np.clip(x, 0, kuvan_leveys - 1)
    y = np.clip(y, 0, kuvan_korkeus - 1)

    return y, x
def coord_to_pixel(
    lon,
    lat,
    kuvan_korkeus,
    kuvan_leveys
):
    """ ## uusi
    WGS84 -> rasterin (y, x).

    Longitude on periodinen:
        -180 == +180
    """

    # Longitude wrap
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0

    x = (
        (lon + 180.0)
        / 360.0
        * kuvan_leveys
    ).astype(int)

    y = (
        (90.0 - np.asarray(lat))
        / 180.0
        * kuvan_korkeus
    ).astype(int)

    x = np.clip(x, 0, kuvan_leveys - 1)
    y = np.clip(y, 0, kuvan_korkeus - 1)

    return y, x


def laske_pikselien_pinta_alat_re2(
    leveys, korkeus, alue=[-180, 180, -90, 90], planeetan_sade=1
):
    """
    Laskee jokaisen pikselin pinta-alan (km2) koko 2D-rasterille [korkeus, leveys].
    
    Parametrit:
    - leveys: Rasterin sarake-määrä (X-akseli)
    - korkeus: Rasterin rivi-määrä (Y-akseli)
    - alue: [lon_min, lon_max, lat_min, lat_max] asteina
    - planeetan_sade: Planeetan säde kilometreinä (oletuksena Maa = 6371.0 km)
    """
    lon_min, lon_max, lat_min, lat_max = alue

    planeetan_sade=planeetan_sade*1
    # Pituuspiirien väli radiaaneina (vakio koko rasterissa)
    dlon = np.radians((lon_max - lon_min) / leveys)
    
    # Leveyspiirien väli asteina kullekin pikselille
    dlat = (lat_max - lat_min) / korkeus

    # Lasketaan jokaisen rivin ylä- ja alareunan leveysasteet radiaaneina
    # np.arange(korkeus) luo indeksit ylhäältä alaspäin
    lat1 = np.radians(lat_max - np.arange(korkeus) * dlat)
    lat2 = np.radians(lat_max - (np.arange(korkeus) + 1) * dlat)

    # Lasketaan 1D-taulukko rivien (leveyspiirivyöhykkeiden) pinta-aloista
    rivien_pinta_alat = (
        (planeetan_sade ** 2)
        * dlon
        * (np.sin(lat1) - np.sin(lat2))
    )

    # Laajennetaan (levitetään) 1D-taulukko 2D-rasteriksi muotoon [korkeus, leveys]
    pinta_alat_2d = np.repeat(rivien_pinta_alat[:, np.newaxis], leveys, axis=1)

    return pinta_alat_2d



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

## piirtoja

## fantasy 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. HILLSHADE
# ============================================================

def make_hillshade(elevation, azimuth=315, altitude=45):
    """
    Luo DEM:stä hillshaden.
    elevation: 2D numpy array
    """

    dy, dx = np.gradient(elevation.astype(float))

    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))

    aspect = np.arctan2(-dx, dy)

    az = np.radians(azimuth)
    alt = np.radians(altitude)

    hillshade = (
        np.sin(alt) * np.sin(slope)
        + np.cos(alt)
        * np.cos(slope)
        * np.cos(az - aspect)
    )

    hillshade = (hillshade - hillshade.min()) / (
        hillshade.max() - hillshade.min() + 1e-12
    )

    return hillshade


# ============================================================
# 2. BIOME / CLIMATE MASKS
# ============================================================

def make_biome_masks(elevation, tmean, precip):
    """
    Yksinkertaistettu fantasiamaailman biomejako.

    tmean = vuosikeskilämpötila °C
    precip = vuosisade mm
    """

    masks = {}

    # -------------------------
    # DESERT
    # -------------------------

    masks["desert"] = (
        (precip < 250) &
        (tmean > 5)
    )

    # -------------------------
    # STEPPE
    # -------------------------

    masks["steppe"] = (
        (precip >= 250) &
        (precip < 500) &
        (tmean > 2)
    )

    # -------------------------
    # TEMPERATE FOREST
    # -------------------------

    masks["forest"] = (
        (precip >= 500) &
        (precip < 1800) &
        (tmean > 5) &
        (tmean < 18)
    )

    # -------------------------
    # TROPICAL FOREST
    # -------------------------

    masks["tropical_forest"] = (
        (precip >= 1800) &
        (tmean > 18)
    )

    # -------------------------
    # COLD / BOREAL
    # -------------------------

    masks["boreal"] = (
        (tmean >= -5) &
        (tmean <= 5) &
        (precip > 300)
    )

    # -------------------------
    # TUNDRA
    # -------------------------

    masks["tundra"] = (
        (tmean < -5) |
        ((tmean < 2) & (elevation > 1500))
    )

    # -------------------------
    # HIGH MOUNTAINS
    # -------------------------

    masks["high_mountain"] = elevation > 2500

    return masks


# ============================================================
# 3. RANDOM BUT STABLE POINTS
# ============================================================

def random_points_in_mask(mask, density, seed=1234):
    """
    Palauttaa pisteitä maskin sisältä.

    density = pisteiden määrä suhteessa maskin pinta-alaan.
    """

    rng = np.random.default_rng(seed)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.empty((0, 2))

    n = int(len(xs) * density)

    if n <= 0:
        return np.empty((0, 2))

    n = min(n, len(xs))

    indices = rng.choice(len(xs), n, replace=False)

    return np.column_stack([
        xs[indices],
        ys[indices]
    ])


# ============================================================
# 4. DESERT DOTS
# ============================================================

def draw_desert(ax, mask, seed=1234):
    """
    Aavikko pistekuviona.
    """

    points = random_points_in_mask(
        mask,
        density=0.003,
        seed=seed
    )

    if len(points) == 0:
        return

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=1.0,
        color="#9c7650",
        alpha=0.55,
        linewidths=0
    )


# ============================================================
# 5. FOREST SYMBOLS
# ============================================================

def draw_tree(ax, x, y, size=5, color="#334b32"):
    """
    Yksinkertainen fantasiakartan puusymboli.
    """

    # runko
    ax.plot(
        [x, x],
        [y, y + size * 0.55],
        color=color,
        linewidth=0.5,
        alpha=0.8
    )

    # latvus
    ax.plot(
        [x, x - size * 0.45],
        [y + size * 0.25, y + size * 0.25],
        color=color,
        linewidth=0.7
    )

    ax.plot(
        [x, x + size * 0.45],
        [y + size * 0.25, y + size * 0.25],
        color=color,
        linewidth=0.7
    )

    ax.plot(
        [x, x],
        [y, y + size],
        color=color,
        linewidth=0.8
    )


def draw_forest(ax, mask, seed=123):
    """
    Piirtää satunnaisen määrän pieniä puita metsämaskiin.
    """

    points = random_points_in_mask(
        mask,
        density=0.0015,
        seed=seed
    )

    for x, y in points:

        draw_tree(
            ax,
            x,
            y,
            size=4.0
        )


# ============================================================
# 6. MOUNTAINS
# ============================================================

def draw_mountain(ax, x, y, size=10, color="#403d38"):
    """
    Fantasiakartan pieni vuorisymboli.

          /\
         /  \
        / /\ \
       /_/  \_\
    """

    # päävuori
    left = x - size
    right = x + size
    top = y - size * 1.5

    ax.plot(
        [left, x, right],
        [y, top, y],
        color=color,
        linewidth=0.8,
        alpha=0.9
    )

    # toinen huippu / sisäinen V
    ax.plot(
        [x, x + size * 0.35, x + size * 0.7],
        [top, y + size * 0.65, y],
        color=color,
        linewidth=0.55,
        alpha=0.75
    )

    # vasemman rinteen varjo
    ax.plot(
        [x, x - size * 0.35],
        [top, y + size * 0.55],
        color=color,
        linewidth=0.45,
        alpha=0.7
    )


def draw_mountains(ax, elevation, seed=42):
    """
    Sijoittaa vuorisymboleita korkeille alueille.

    Tärkeää:
    symbolit eivät täytä koko vuorta, vaan
    muodostavat hajanaisia vuorijonoja.
    """

    # Korkeuden perusteella maski
    threshold = np.percentile(elevation, 82)

    mountain_mask = elevation > threshold

    # hieman tasoitettu maski
    smooth = gaussian_filter(
        mountain_mask.astype(float),
        sigma=8
    )

    mountain_mask = smooth > 0.45

    points = random_points_in_mask(
        mountain_mask,
        density=0.00035,
        seed=seed
    )

    rng = np.random.default_rng(seed)

    for x, y in points:

        local_elevation = elevation[
            int(np.clip(y, 0, elevation.shape[0] - 1)),
            int(np.clip(x, 0, elevation.shape[1] - 1))
        ]

        size = np.interp(
            local_elevation,
            [threshold, elevation.max()],
            [5, 14]
        )

        # pieni satunnaisuus
        size *= rng.uniform(0.75, 1.25)

        draw_mountain(
            ax,
            x,
            y,
            size=size
        )


# ============================================================
# 7. RIVERS
# ============================================================

def draw_rivers(ax, rivers):
    """
    Piirtää joet.

    Tämä olettaa, että rivers on joko:
        - binary rasteri
        - tai jo valmiiksi sopiva maski

    Jos sinulla on jokigeometrioita, tehdään myöhemmin
    oma versio niille.
    """

    river_mask = rivers > 0

    ys, xs = np.where(river_mask)

    if len(xs) == 0:
        return

    ax.scatter(
        xs,
        ys,
        s=3,
        color="#526f7f",
        alpha=0.8,
        linewidths=0
    )
    #plot_wrapped_line(ax, xs, ys, lw=1,color="#526f7f", alpha=0.8 )


# ============================================================
# 8. LAKES
# ============================================================

def draw_lakes(ax, lakes):

    lake_mask = lakes > 0

    ax.imshow(
        np.ma.masked_where(
            ~lake_mask,
            lake_mask
        ),
        cmap="Blues",
        alpha=0.45,
        interpolation="nearest"
    )

def draw_water(ax, rivers, lakes, water_color="#6f8fa3"):
    """
    Piirtää järvet ja joet fantasiakartan vesityylillä.

    rivers ja lakes ovat rastereita samassa koordinaatistossa.
    """

    # -------------------------
    # LAKES
    # -------------------------

    lake_mask = lakes > 0

    lake_layer = np.zeros(
        (*lakes.shape, 4),
        dtype=float
    )

    rgb = plt.matplotlib.colors.to_rgb(water_color)

    lake_layer[..., 0] = rgb[0]
    lake_layer[..., 1] = rgb[1]
    lake_layer[..., 2] = rgb[2]
    lake_layer[..., 3] = lake_mask * 0.75

    ax.imshow(
        lake_layer,
        interpolation="nearest"
    )

    # -------------------------
    # RIVERS
    # -------------------------

    river_mask = rivers > 0

    ys, xs = np.where(river_mask)

    if len(xs):
        ax.scatter(
            xs,
            ys,
            s=0.7,
            color=water_color,
            alpha=0.9,
            linewidths=0
        )
        

def draw_ocean_00(ax, dem, ocean, water_color="#0000ff"):
    """
    ocean == 1 -> sininen
    ocean == 0 -> NaN -> läpinäkyvä
    """

    ocean_data = np.copy(dem)
    

    # Maa pois piirroksesta
    ocean_data[ocean_data > 0] = np.nan
    ocean_data[ocean_data == 0] = 1
    # Sininen colormap
    cmap = plt.matplotlib.colors.ListedColormap([water_color])
    cmap.set_bad(alpha=0)

    ax.imshow(
        ocean_data,
        cmap=cmap,
        interpolation="nearest",
        origin="upper"
    )


def draw_ocean(ax, dem, ocean, water_color="#0000ff",
               coast_color="#000000"):
    """
    ocean == 1 -> sininen
    ocean == 0 -> läpinäkyvä

    Piirtää lisäksi rannikon meren ja maan rajalle.
    """

    # -------------------------
    # OCEAN
    # -------------------------
    ocean_data = np.copy(dem).astype(float)

    ocean_data[ocean_data > 0] = np.nan
    ocean_data[ocean_data == 0] = 1

    cmap = plt.matplotlib.colors.ListedColormap([water_color])
    cmap.set_bad(alpha=0)

    ax.imshow(
        ocean_data,
        cmap=cmap,
        interpolation="nearest",
        origin="upper"
    )

    # -------------------------
    # COASTLINE
    # -------------------------

    # Meri = True
    sea = dem == 0

    # Etsi kohdat, joissa naapuri on maata
    coast = (
        sea &
        (
            np.roll(dem > 0,  1, axis=0) |
            np.roll(dem > 0, -1, axis=0) |
            np.roll(dem > 0,  1, axis=1) |
            np.roll(dem > 0, -1, axis=1)
        )
    )

    # Piirrä rannikko
    y, x = np.where(coast)

    ax.scatter(
        x,
        y,
        s=1,
        c=coast_color,
        marker="s"
    )


# ============================================================
# 9. MAIN RENDERER
# ============================================================

def piirra_fantasy_map(
    elevation,
    tmean_annual,
    precip_annual,
    rivers,
    lakes,
    figsize=(12, 6),
    dpi=200
):
    """
    Pääfunktio.

    Kaikkien rasterien pitää olla saman kokoisia.
    """

    ocean=np.copy(elevation)
    ocean=np.where(elevation<0,1,0)
    # -------------------------
    # CHECK
    # -------------------------

    shape = elevation.shape

    for name, raster in [
        ("tmean_annual", tmean_annual),
        ("precip_annual", precip_annual),
        ("rivers", rivers),
        ("lakes", lakes),
    ]:
        if raster.shape != shape:
            raise ValueError(
                f"{name} shape {raster.shape} != "
                f"elevation shape {shape}"
            )

    # -------------------------
    # HILLSHADE
    # -------------------------

    hillshade = make_hillshade(elevation)

    # -------------------------
    # BIOMES
    # -------------------------

    biomes = make_biome_masks(
        elevation,
        tmean_annual,
        precip_annual
    )

    # -------------------------
    # FIGURE
    # -------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    ax.set_facecolor("#e8dfc8")

    # -------------------------
    # TERRAIN
    # -------------------------

    ax.imshow(
        hillshade,
        cmap="Greys",
        alpha=0.25,
        interpolation="bilinear"
    )

    # -------------------------
    # BIOME COLORS
    # -------------------------

    colors = {
        "desert": "#d9bd82",
        "steppe": "#c5b878",
        "forest": "#78916a",
        "tropical_forest": "#557c58",
        "boreal": "#70856b",
        "tundra": "#b8b9a6",
        "high_mountain": "#aaa79b",
    }

    for name, mask in biomes.items():

        layer = np.zeros(
            (*shape, 4),
            dtype=float
        )

        color = colors[name]

        # matplotlib osaa muuntaa hexin RGB:ksi
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(color)

        layer[..., 0] = rgb[0]
        layer[..., 1] = rgb[1]
        layer[..., 2] = rgb[2]
        layer[..., 3] = mask * 0.35

        ax.imshow(
            layer,
            interpolation="nearest"
        )

    # -------------------------
    # DESERT TEXTURE
    # -------------------------

    draw_desert(
        ax,
        biomes["desert"]
    )

    # -------------------------
    # FORESTS
    # -------------------------

    draw_forest(
        ax,
        biomes["forest"]
    )

    draw_forest(
        ax,
        biomes["tropical_forest"],
        seed=456
    )

    draw_forest(
        ax,
        biomes["boreal"],
        seed=789
    )

    # -------------------------
    # MOUNTAINS
    # -------------------------

    draw_mountains(
        ax,
        elevation
    )

    # -------------------------
    # LAKES
    # -------------------------

    #draw_lakes(
    #    ax,
    #    lakes
    #)

    # -------------------------
    # RIVERS
    # -------------------------

    draw_rivers(
        ax,
        rivers
    )
    draw_water(ax, lakes, lakes, water_color="#6f8fa3")
    draw_ocean(ax, elevation, ocean, water_color="#7f7fff")
    # -------------------------
    # MAP STYLE
    # -------------------------

    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0)

    ax.set_aspect("equal")

    ax.axis("off")

    plt.tight_layout(pad=0)

    return fig, ax




## .. fantasy











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




def piirra_korkeuskartta(
    data, bounds=(-180.0, 180.0, -90.0, 90.0), otsikko="Planet Topography"
):
    """Visualisoi NumPy-taulukon pelkkänä korkeusdatana (maa-alueet).

    Käyttää mukautettua maastopalettia ja määriteltyjä kontuuritasoja:
    [0, 250, 500, 1000, 2000, 4000, 6000, 8000]
    """
    min_lon, max_lon, min_lat, max_lat = bounds

    # 1. MÄÄRITETÄÄN KONTUURITASOT (Määritelty lista)
    kontuuri_tasot = [0,1, 250, 500, 1000, 2000, 4000, 6000, 8000]

    # 2. LUODAAN KORKEUSKARTALLE SOPIVA PALETTI
    # Alanko (vihreä) -> Keskikorkeus (keltainen/ruskea) -> Vuoristo (tummansuklaa) -> Huiput (valkoinen)
    maan_varit = [
        "#a0a0ff",  # 0m (sininen)
        "#1b7837",  # 1m (Tummanvihreä)
        "#5aae61",  # 250m (Vaaleanvihreä)
        "#a6dba0",  # 500m (Kellertävä vihreä)
        "#f4a582",  # 1000m (Hiekanruskea)
        "#d6604d",  # 2000m (Oranssinruskea)
        "#b2182b",  # 4000m (Punaruskea)
        "#67001f",  # 6000m (Tumma vuoristonruskea)
        "#ffffff",  # 8000m+ (Ikilumi/Valkoinen)
    ]
    oma_cmap = mcolors.LinearSegmentedColormap.from_list(
        "korkeus_teema", maan_varit
    )

    # Koska kontuurivälit eivät ole tasaisia (0->250 on pieni askel, 2000->4000 suuri),
    # käytetään BoundaryNormia. Se varmistaa, että värit vaihtuvat täsmälleen kontuurirajojen kohdalla.
    normi = mcolors.BoundaryNorm(
        boundaries=kontuuri_tasot, ncolors=oma_cmap.N, extend="max"
    )

    # 3. ALUSTETAAN KARTTA
    plt.figure(figsize=(14, 8))

    if min_lon == -180.0 and max_lon == 180.0:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    else:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    #ax.coastlines(resolution="110m", color="#333333", linewidth=0.8)
    ls = LightSource(azdeg=315, altdeg=45)

    # Muunnetaan värit ja varjostus RGB-kuvaksi. fraction säätää varjon voimakkuutta (0.0 - 1.0)
    rgb_varjostettu = ls.shade(
        data, cmap=oma_cmap, norm=normi, blend_mode="overlay", fraction=0.4
    )
    # 4. PIIRRETÄÄN POHJAKARTTA
    im = ax.imshow(
        rgb_varjostettu,
        extent=[min_lon, max_lon, min_lat, max_lat],
        transform=ccrs.PlateCarree(),
        cmap=oma_cmap,
        norm=normi,
        origin="lower",
    )

    # 5. LISÄTÄÄN MÄÄRITYKSEN MUKAISET KONTUURIVIIVAT
    lons = np.linspace(min_lon, max_lon, data.shape[1])
    lats = np.linspace(min_lat, max_lat, data.shape[0])

    kontuurit = ax.contour(
        lons,
        lats,
        data,
        levels=kontuuri_tasot,
        colors="#111111",
        linewidths=0.4,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
    )

    # Lisätään tekstilaput kontuuriviivoihin (esim. "500", "1000") selkeyden vuoksi
    plt.clabel(
        kontuurit, fmt="%d", fontsize=8, colors="#111111", inline=True, inline_spacing=3
    )

    # 6. VÄRISELITE JA OTSIKKO
    # BoundaryNormia käytettäessä colorbar näyttää automaattisesti oikeat epätasaiset välit
    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.7, ticks=kontuuri_tasot
    )
    cbar.set_label("Korkeus merenpinnasta (metriä)")

    plt.title(otsikko, fontsize=14, pad=10)
    plt.show()


def piirra_korkeus_ja_syvyyskartta(
    data, bounds=(-180.0, 180.0, -90.0, 90.0), otsikko="Planet dem"
):
    """Visualisoi NumPy-taulukon korkeus- ja syvyysdatana karttaprojektiossa.

    Lukitsee nollapisteen merenpinnaksi ja lisää kontuuriviivat 1000m välein.
    """
    min_lon, max_lon, min_lat, max_lat = bounds
    korkein = data.max()
    syvin = data.min()
    
    data2=np.copy(data)
    data2=np.where(data2>0, data2,0)

    # 1. LUODAAN MUKAUTETTU VÄRIKAAVIO (Maa ja meri)
    # Tasaiset värisiirtymät: syvänsininen -> vaaleansininen (meri) | vihreä -> ruskea -> valkoinen (maa)
    meren_varit = ["#000044", "#1144aa", "#4499ff", "#b3d1ff"]
    maan_varit = ["#22aa22", "#77cc44", "#aa8844", "#885522", "#ffffff"]
    kaikki_varit = meren_varit + maan_varit

    oma_cmap = mcolors.LinearSegmentedColormap.from_list(
        "planeetta_teema", kaikki_varit
    )

    # 2. LUKITAAN NOLLAPISTE MERENPINNAKSI
    # TwoSlopeNorm pakottaa arvon 0.0 tismalleen väriskaalan keskikohtaan
    if syvin < 0 and korkein > 0:
        normi = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=syvin, vmax=korkein)
    else:
        normi = mcolors.Normalize(vmin=syvin, vmax=korkein)

    # 3. ALUSTETAAN KARTTA
    plt.figure(figsize=(14, 8))

    if min_lon == -180.0 and max_lon == 180.0:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    else:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    norm = Normalize(vmin=data2.min(), vmax=data2.max())
    rgb_unshaded = oma_cmap(norm(data2))
    ls = LightSource(azdeg=315, altdeg=45)
    rgb_shaded = ls.shade(data, cmap=oma_cmap, norm=normi, blend_mode='overlay')
    final_rgb = rgb_shaded.copy()

    # Luodaan Boolean-maski niistä kohdista, joissa data2 on pienempi kuin 0
    negative_mask = data2 < 0

    # Korvataan nämä kohdat (kaikki 4 RGBA-kanavaa) alkuperäisillä väreillä
    final_rgb[negative_mask] = rgb_unshaded[negative_mask]
    # 4. PIIRRETÄÄN POHJAKARTTA
    im = ax.imshow(
        final_rgb,
        extent=[min_lon, max_lon, min_lat, max_lat],
        transform=ccrs.PlateCarree(),
        cmap=oma_cmap,
        norm=normi,
        origin="lower",
    )

    # 5. LISÄTÄÄN KONTUURIVIIVAT (1000 metrin välein)
    # Luodaan tasaväliset tasot nollasta ylös- ja alaspäin
    positiiviset_tasot = np.arange(1000, korkein, 500)
    negatiiviset_tasot = np.arange(-1000, syvin, -1000)[::-1]  # Järjestys pienimmästä suurimpaan
    kontuuri_tasot = np.concatenate([negatiiviset_tasot, [0], positiiviset_tasot])

    # Luodaan X- ja Y-koordinaatit imshow-dataa vastaaviksi contour-funktiota varten
    lons = np.linspace(min_lon, max_lon, data.shape[1])
    lats = np.linspace(min_lat, max_lat, data.shape[0])

    # Piirretään ohuet kontuuriviivat
    kontuurit = ax.contour(
        lons,
        lats,
        data,
        levels=kontuuri_tasot,
        colors="black",
        linewidths=0.3,
        alpha=0.6,  # Tekee viivoista hieman läpinäkyviä, jotta ne eivät hypi silmille
        transform=ccrs.PlateCarree(),
    )
    kontuurit2 = ax.contour(
        lons,
        lats,
        data,
        levels=[-1,0,1],
        colors="black",
        linewidths=1,
        alpha=0.6,  # Tekee viivoista hieman läpinäkyviä, jotta ne eivät hypi silmille
        transform=ccrs.PlateCarree(),
    )

    # Vaihtoehtoinen: Lisää tekstit kontuuriviivoihin (esim. "1000") ottamalla alta kommentti pois
    # plt.clabel(kontuurit, fmt='%d', fontsize=8, colors='black')

    # 6. VÄRISELITE JA OTSIKKO
    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.7
    )
    cbar.set_label("Korkeus / Syvyys merenpinnasta (metriä)")

    plt.title(otsikko, fontsize=14, pad=10)
    plt.show()

def plot_wrapped_line(ax, lon, lat, **kwargs):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # Katkaise viiva antimeridiaanin kohdalla
    jumps = np.abs(np.diff(lon)) > 180

    start = 0

    for i in np.where(jumps)[0]:
        plt.plot(
            lon[start:i + 1],
            lat[start:i + 1],
            **kwargs
        )
        start = i + 1

    # viimeinen segmentti
    plt.plot(
        lon[start:],
        lat[start:],
        **kwargs
    )

def piirra_ilmasto_kartta(
    koppen,
    tmean_annual=None,
    precip_annual=None,
    korkeuskartta_m=None,
    tmin_annual=None,
    tmax_annual=None,
    precip_dry=None,
    precip_wet=None,
    korkeuskartta=None,
    merijaa_aina=None,
    rivers=None,
    lakes=None,
    jaa=None,
    paletti=None,
    figsize=(14, 7),
    dpi=100,
    otsikko="Planet map",
):
    """
    Piirtää Köppen-ilmastokartan.

    Parametrit
    ----------
    koppen : rasteri
        Köppen-ilmastoluokat 0–14.

    tmean_annual : rasteri, optional
        Vuotuinen keskilämpötila.

    precip_annual : rasteri, optional
        Vuotuinen sademäärä.

    korkeuskartta_m : rasteri, optional
        Korkeus metreinä.

    tmin_annual : rasteri, optional
        Vuotuinen minimilämpötila.

    tmax_annual : rasteri, optional
        Vuotuinen maksimilämpötila.

    precip_dry : rasteri, optional
        Kuivimman kuukauden sademäärä.

    precip_wet : rasteri, optional
        Sateisimman kuukauden sademäärä.

    korkeuskartta : rasteri, optional
        Hillshadea varten käytettävä korkeusrasteri.

    merijaa_aina : rasteri, optional
        Pysyvän merijään rasteri.

    rivers : rasteri, optional
        Jokirasteri.

    lakes : rasteri, optional
        Järvirasteri.

    jaa : rasteri, optional
        Lumirasteri.

    paletti : str/list/Colormap, optional
        "paletti1", "paletti2" tai "paletti3".
        Oletus: "paletti3".

    Returns
    -------
    fig, ax
        Matplotlibin figure ja axes.
    """

    # ================================================================
    # PALETIT
    # ================================================================

    paletti1 = [
        "#1a233a",  # 0: Meri
        "#0000fe",  # 1: Af
        "#0077fe",  # 2: Am
        "#41c2ff",  # 3: Aw
        "#fe0000",  # 4: BWh
        "#fe9999",  # 5: BWk
        "#fec200",  # 6: BSh
        "#fffe00",  # 7: BSk
        "#c6ff4e",  # 8: Csa/Csb
        "#007e00",  # 9: Cfb
        "#96ff96",  # 10: Cfa/Cwa
        "#327e65",  # 11: Dfa/Dfb
        "#00465f",  # 12: Dfc/Dfd
        "#7f7f7f",  # 13: ET
        "#ffffff",  # 14: EF
    ]

    paletti2 = [
        "#1a233a",  # 0: Meri
        "#0b5345",  # 1: Af
        "#117a65",  # 2: Am
        "#7dcea0",  # 3: Aw
        "#edbb99",  # 4: BWh
        "#e59866",  # 5: BWk
        "#f8c471",  # 6: BSh
        "#f7dc6f",  # 7: BSk
        "#a2d9ce",  # 8: Csa/Csb
        "#1e8449",  # 9: Cfb
        "#2ecc71",  # 10: Cfa/Cwa
        "#273746",  # 11: Dfa/Dfb
        "#1c2833",  # 12: Dfc/Dfd
        "#a6acaf",  # 13: ET
        "#ffffff",  # 14: EF
    ]

    # Köppenpasta
    paletti3 = [
        "#0f1e38",  # 0: Meri
        "#2a410f",  # 1: Af
        "#385714",  # 2: Am
        "#98a94f",  # 3: Aw
        "#d5b785",  # 4: BWh
        "#b89c6c",  # 5: BWk
        "#cbb27a",  # 6: BSh
        "#ddcb94",  # 7: BSk
        "#717b43",  # 8: Csa/Csb
        "#4c6a21",  # 9: Cfb
        "#557525",  # 10: Cfa/Cwa
        "#2f421f",  # 11: Dfa/Dfb
        "#1d2a13",  # 12: Dfc/Dfd
        "#828675",  # 13: ET
        "#cfcfcf",  # 14: EF
    ]

    # ================================================================
    # VALITSE PALETTI
    # ================================================================

    if paletti is None:
        paletti = paletti3

    elif isinstance(paletti, str):
        paletit = {
            "paletti1": paletti1,
            "paletti2": paletti2,
            "paletti3": paletti3,
        }

        if paletti not in paletit:
            raise ValueError(
                "Tuntematon paletti. Käytä 'paletti1', 'paletti2' tai 'paletti3'."
            )

        paletti = paletit[paletti]

    cmap_tarkka = ListedColormap(paletti)

    rajat_tarkka = np.arange(-0.5, 15.5, 1.0)
    norm_tarkka = BoundaryNorm(
        rajat_tarkka,
        cmap_tarkka.N
    )

    # ================================================================
    # ILMASTOLUOKKIEN NIMET
    # ================================================================

    tarkat_nimet = [
        "Sea",
        "Rain forest (Af)",
        "Monsoon (Am)",
        "Savanna (Aw)",
        "Hot desert (BWh)",
        "Cold desert (BWk)",
        "Hot steppe (BSh)",
        "Cold steppe (BSk)",
        "Mediterranean climate (Csa/Csb)",
        "Oceanic climate (Cfb)",
        "Subtropical moist (Cfa)",
        "Continental (Dfa/Dfb)",
        "Taiga/Boreal (Dfc/Dfd)",
        "Tundra (ET)",
        "Polar desert / ice (EF)",
    ]

    # ================================================================
    # KUVA
    # ================================================================

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi
    )

    # ================================================================
    # ALKUPERÄINEN KÖPPEN-RASTERI
    # ================================================================

    im = ax.imshow(
        koppen,
        cmap=cmap_tarkka,
        norm=norm_tarkka, origin="upper",
        extent=[-180, 180, -90, 90],
    )

    # ================================================================
    # ET / GAM / SVM / MUU KORJAUS
    # ================================================================

    variables = {
        "keski_lampotila": tmean_annual,
        "keski_sademaara": precip_annual,
        "maaston_korkeus_m": korkeuskartta_m,
        "lampotila_min": tmin_annual,
        "lampotila_max": tmax_annual,
        "kuivimman_kuukauden_sade": precip_dry,
        "sateisimman_kuukauden_sade": precip_wet,
    }

    # Poistetaan puuttuvat muuttujat
    variables = {
        nimi: rasteri
        for nimi, rasteri in variables.items()
        if rasteri is not None
    }

    variable_names = list(variables.keys())

    if variable_names:
        new_rgb = et_im(
            im,
            variables,
            variable_names,
        )

        ax.imshow(
            new_rgb,
            cmap=cmap_tarkka,
            norm=norm_tarkka,origin="upper",
            extent=[-180, 180, -90, 90],
        )

    # ================================================================
    # HILLSHADE
    # ================================================================

    if korkeuskartta is not None:

        varjo_maski = laske_hillshade(
            korkeuskartta,
            azimuth=315,
            altitude=45,
        )

        ax.imshow(
            np.exp(varjo_maski),
            cmap="gray",
            alpha=0.1,origin="upper",
            extent=[-180, 180, -90, 90],
        )

    # ================================================================
    # MERIJÄÄ
    # ================================================================

    if merijaa_aina is not None:

        ax.contourf(
            np.flipud(merijaa_aina),
            levels=[0.25, 1],
            extent=[-180, 180, -90, 90],
            alpha=0.7,origin="upper",
            colors=["lightblue"],
        )

    # ================================================================
    # JOET
    # ================================================================

    if rivers is not None:

        ax.imshow(
            rivers,
            cmap="Blues_r",
            extent=[-180, 180, -90, 90],
            origin="upper",
            alpha=0.7,
        )

    # ================================================================
    # JÄRVET
    # ================================================================

    if lakes is not None:

        ax.imshow(
            lakes,
            cmap="Blues_r",
            extent=[-180, 180, -90, 90],
            origin="upper",
            alpha=1.0,
        )

    # ================================================================
    # LUMI
    # ================================================================

    if jaa is not None:

        ax.contourf(
            np.flipud(jaa),
            levels=[20, 50, 100, 200],
            extent=[-180, 180, -90, 90],
            colors=["white"],
        )

    # ================================================================
    # VÄRIPALKIT
    # ================================================================

    cbar = fig.colorbar(
        im,
        ax=ax,
        ticks=list(range(15)),
        orientation="horizontal",
        pad=0.15,
        shrink=0.9,
    )

    cbar.ax.set_xticklabels(
        tarkat_nimet,
        rotation=25,
        ha="right",
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    # ================================================================
    # VIIMEISTELY
    # ================================================================

    ax.set_title(
        otsikko,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")

    ax.grid(
        color="black",
        linestyle="--",
        alpha=0.1,
    )

    plt.tight_layout()
    plt.show()

    return fig, ax


def laske_makkink_evaporation(
    local_radiation_Wm2,
    temperature_C
):
    """
    Makkink potential evapotranspiration.

    Input:
        local_radiation_Wm2 : päivittäinen keskimääräinen säteily [W/m²]
        temperature_C       : lämpötila [°C]

    Output:
        ET0_mm_day          : potentiaalinen evapotranspiraatio [mm/day]
    """

    # W/m² -> MJ/m²/day
    Rs = local_radiation_Wm2 * 0.0864
    Rs=cp.where(Rs==0, 0.001, Rs) ## Warning jn kludge
    #plt.imshow(Rs.get())
    #plt.show()
    #quit(-1)

    
    # Saturaatiohöyrynpaine [kPa]
    es = (
        0.6108
        * cp.exp(
            17.27 * temperature_C
            / (temperature_C + 237.3)
        )
    )

    # Delta [kPa/°C]
    Delta = (
        4098.0 * es
        / (temperature_C + 237.3)**2
    )

    # Psykrometrinen vakio
    gamma = 0.066

    # Veden höyrystymislämpö [MJ/kg]
    lambda_v = (
        2.501
        - 0.002361 * temperature_C
    )

    # Makkink
    ET0_mm_day = (
        0.61
        * Delta
        / (Delta + gamma)
        * Rs
        / lambda_v
    )

    # Ei negatiivista haihtumista
    ET0_mm_day = cp.maximum(
        ET0_mm_day,
        0.0
    )

    return ET0_mm_day





def laske_etaisyys_rannikosta_gpu(
    maa_maski0,
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
        maa_maski0,
        dtype=cp.bool_,
    )

    #rasteri = maa_maski.astype(
    #    cp.float32
    #)
    rasteri = cp.asarray(
        maa_maski0,
        dtype=cp.float32,
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



    etaisyys_pixeleina = (
        ndimage2.distance_transform_edt(
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

def calculate_climate_rivers_and_lakes_original_sure(dem_cpu, rain_cpu, temp_cpu, iterations=500, river_threshold=5000, lake_threshold=20000):
    """
    Laskee jokiverkoston ja järvet ottaen huomioon sademäärän ja haihtumisen.
    
    dem_cpu: Korkeuskartta (2D numpy array)
    rain_cpu: Vuosisademäärä mm/vuosi (luku tai 2D numpy array)
    temp_cpu: Vuosikeskilämpötila °C (luku tai 2D numpy array)
    iterations: Virtausaskeleet
    river_threshold: Jokikynnys (vesivolyymi joen näkymiselle)
    lake_threshold: Järvikynnys (vesivolyymi kuopassa, jotta se lasketaan järveksi)
    """
    # Siirretään datat GPU:lle
    dem = cp.array(dem_cpu, dtype=cp.float32)
    
    if isinstance(rain_cpu, (int, float)):
        rain = cp.full(dem.shape, rain_cpu, dtype=cp.float32)
    else:
        rain = cp.array(rain_cpu, dtype=cp.float32)
        
    if isinstance(temp_cpu, (int, float)):
        temp = cp.full(dem.shape, temp_cpu, dtype=cp.float32)
    else:
        temp = cp.array(temp_cpu, dtype=cp.float32)
    
    # 1. LASKETAAN D8-VIRTAUSSUUNNAT
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    max_drop = cp.zeros_like(dem)
    flow_dir = cp.full(dem.shape, -1, dtype=cp.int8)
    
    for idx, (dy, dx) in enumerate(directions):
        neighbor = cp.roll(cp.roll(dem, -dy, axis=0), -dx, axis=1)
        drop = dem - neighbor
        mask = (drop > max_drop) & (dem > 0)
        max_drop = cp.where(mask, drop, max_drop)
        flow_dir = cp.where(mask, idx, flow_dir)
        
    # Merkitään kohdat, joista vesi ei pääse virtaamaan eteenpäin (kuopat / sinks)
    # Jos max_drop on 0 ja ollaan kuivalla maalla (dem > 0), kyseessä on kuoppa
    is_sink = (max_drop <= 0) & (dem > 0)
    
    # Meri on alue, jossa korkeus on nolla tai alle
    is_ocean = (dem <= 0)
    flow_dir = cp.where(is_ocean, -1, flow_dir)
    
    # 2. HAIHTUMINEN AND ALUSTUS
    evaporation_potential = cp.clip(temp * 5.0, 0, None)
    water_volume = cp.copy(rain)
    
    # 3. ITERATIIVINEN VIRTAUS JA HAIHTUMINEN
    for _ in range(iterations):
        next_water = cp.copy(rain)
        evaporated_water = cp.clip(water_volume - evaporation_potential, 0, None)
        
        # Luodaan rasteri vedelle, joka liikkuu tässä askeleessa
        moving_water = cp.zeros_like(water_volume)
        
        for idx, (dy, dx) in enumerate(directions):
            flow_mask = (flow_dir == idx)
            # Vesi liikkuu vain, jos solu virtaa kyseiseen suuntaan
            moving_water += cp.where(flow_mask, evaporated_water, 0)
            
            # Valutetaan vesi naapuriin
            next_water += cp.roll(cp.roll(cp.where(flow_mask, evaporated_water, 0), dy, axis=0), dx, axis=1)
            
        # KUOPPA-LOGIIKKA: Vesi, joka on kuopassa, ei virtaa minnekään.
        # Se jää paikoilleen ja siihen kertyy uutta vettä yläjuoksulta.
        staying_water = evaporated_water - moving_water
        next_water += cp.where(is_sink, staying_water, 0)
        
        # Päivitetään vesimäärä
        water_volume = next_water

    # 4. EROTELLAAN JOKI- JA JÄRVERASTERIT
    # Järvi: Vesi on kerääntynyt kuoppaan (tai sen välittömään läheisyyteen) ja ylittää järvikynnyksen
    # Ei lasketa merta järveksi
    lake_mask = (water_volume >= lake_threshold) & is_sink & (~is_ocean)
    
    # Laajennetaan järveä tulvimalla (valinnainen, mutta tekee järvistä suurempia kuin 1 solu)
    # Tässä yksinkertainen maski: jos solu on kuoppa ja vettä on paljon -> Järvi
    lake_raster = cp.where(lake_mask, water_volume, 0)
    
    # Joki: Virtaavaa vettä, joka ylittää jokikynnyksen, mutta ei ole järveä tai merta
    river_raster = cp.where((water_volume >= river_threshold) & (~lake_mask) & (~is_ocean), water_volume, 0)
    
    return river_raster.get(), lake_raster.get()



def calculate_climate_rivers_original_sure(dem_cpu, rain_cpu, temp_cpu, iterations=500, threshold=5000):
    """
    Laskee jokiverkoston ottaen huomioon alueellisen sademäärän ja lämpötilasta johtuvan haihtumisen.
    
    dem_cpu: Korkeuskartta (2D numpy array)
    rain_cpu: Vuosisademäärä mm/vuosi (luku tai 2D numpy array)
    temp_cpu: Vuosikeskilämpötila °C (luku tai 2D numpy array)
    iterations: Virtausaskeleet (riippuu kartan koosta)
    threshold: Jokikynnys (kuinka paljon vesivolyymia vaaditaan joen näkymiseen)
    """
    # Siirretään datat GPU:lle
    dem = cp.array(dem_cpu, dtype=cp.float32)
    rain = cp.array(rain_cpu, dtype=cp.float32)
    temp = cp.array(temp_cpu, dtype=cp.float32)
    
    # 1. LASKETAAN D8-VIRTAUSSUUNNAT (kuten aiemmin)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    max_drop = cp.zeros_like(dem)
    flow_dir = cp.full(dem.shape, -1, dtype=cp.int8)
    
    for idx, (dy, dx) in enumerate(directions):
        neighbor = cp.roll(cp.roll(dem, -dy, axis=0), -dx, axis=1)
        drop = dem - neighbor
        mask = (drop > max_drop) & (dem > 0)
        max_drop = cp.where(mask, drop, max_drop)
        flow_dir = cp.where(mask, idx, flow_dir)
        
    flow_dir = cp.where(dem <= 0, -1, flow_dir)
    
    # 2. MÄÄRITETÄÄN PAIKALLINEN HAIHTUMINEN (Evaporation)
    # Yksinkertaistettu potentiaalisen haihtuvuuden malli (esim. korreloi lämpötilan kanssa).
    # Jos temp < 0, haihtumista ei käytännössä tapahdu nestemäisenä.
    # Clip-funktiolla varmistetaan, ettei haihtuminen ole negatiivista.
    evaporation_potential = cp.clip(temp * 5.0, 0, None)  # Kerroin 40 on säädettävissä
    
    # Alustetaan vesimäärä solun omalla sademäärällä
    water_volume = cp.copy(rain)
    
    # 3. ITERATIIVINEN VIRTAUS JA HAIHTUMINEN
    for _ in range(iterations):
        next_water = cp.copy(rain)  # Jokainen askel tuo uutta sadetta yläjuoksulta
        
        # Kerätään vesi kaikista tähän soluun virtaavista naapureista
        incoming_water = cp.zeros_like(dem)
        for idx, (dy, dx) in enumerate(directions):
            from_neighbor = cp.roll(cp.roll(water_volume, dy, axis=0), dx, axis=1)
            neighbor_dir = cp.roll(cp.roll(flow_dir, dy, axis=0), dx, axis=1)
            incoming_water += cp.where(neighbor_dir == idx, from_neighbor, 0.0)
            
        # Solun kokonaisvesimäärä = oma sade + yläjuoksun tuoma vesi
        total_water = next_water + incoming_water
        
        # Vähennetään haihtuminen tästä vesimäärästä
        # Vesi ei voi mennä negatiiviseksi (jos kuivuu, se on 0)
        water_volume = cp.clip(total_water - evaporation_potential, 0, None)
        
        # Meri imee kaiken veden, eli nollataan merisolut
        water_volume = cp.where(dem > 0, water_volume, 0.0)
        
    # 4. TUNNISTETAAN JOET
    # Joki syntyy vain, jos virtaava vesimäärä (haihtumisen jälkeen) ylittää kynnyksen
    river_mask = (water_volume > threshold) & (dem > 0)
    
    return water_volume.get(), river_mask.get()




def calculate_climate_rivers_and_lakes(
    dem_cpu,
    rain_cpu,
    temp_cpu,
    iterations=500,
    river_threshold=1.0e10,
    lake_threshold=1.0e8,
    minimum_catchment_km2=100.0,
    minimum_lake_area_km2=10.0,
    planet_radius_re=1.0,
    atmosphere_pressure_bar=1.0,
    planet_gravity_earths=1.0,
):
    """
    Globaali D8-hydrologia equirectangular-rasterille.

    Palauttaa:
        river_raster : vuotuinen jokivirta m³/v
        lake_raster  : vuotuinen järveen saapuva valunta m³/v

    Järvet muodostetaan topografisiin painanteisiin.
    Vedenpinta kasvatetaan spill pointiin asti eikä käytetä
    kiinteää säde-/korkeustoleranssia.

    DEM:
        metriä

    rain:
        mm/v

    temp:
        °C
    """



    # ============================================================
    # 1. DATA GPU:LLE
    # ============================================================

    dem = cp.asarray(
        dem_cpu,
        dtype=cp.float32
    )

    if dem.ndim != 2:
        raise ValueError("dem_cpu pitää olla 2D-rasteri.")

    rows, cols = dem.shape
    n = rows * cols

    # ------------------------------------------------------------
    # Rain
    # ------------------------------------------------------------

    if np.isscalar(rain_cpu):

        rain = cp.full(
            dem.shape,
            float(rain_cpu),
            dtype=cp.float32
        )

    else:

        rain = cp.asarray(
            rain_cpu,
            dtype=cp.float32
        )

        if rain.shape != dem.shape:
            raise ValueError(
                "rain_cpu:n muodon pitää olla sama kuin dem_cpu:n."
            )

    # ------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------

    if np.isscalar(temp_cpu):

        temp = cp.full(
            dem.shape,
            float(temp_cpu),
            dtype=cp.float32
        )

    else:

        temp = cp.asarray(
            temp_cpu,
            dtype=cp.float32
        )

        if temp.shape != dem.shape:
            raise ValueError(
                "temp_cpu:n muodon pitää olla sama kuin dem_cpu:n."
            )

    # ============================================================
    # 2. PLANETA
    # ============================================================

    radius_re = max(
        float(planet_radius_re),
        0.01
    )

    pressure = max(
        float(atmosphere_pressure_bar),
        0.01
    )

    gravity = max(
        float(planet_gravity_earths),
        0.01
    )

    EARTH_RADIUS_M = 6_371_000.0

    planet_radius_m = (
        EARTH_RADIUS_M
        * radius_re
    )

    # ============================================================
    # 3. EQUIRECTANGULAR-GEOMETRIA
    # ============================================================

    lat = (
        90.0
        -
        (
            cp.arange(
                rows,
                dtype=cp.float32
            )
            + 0.5
        )
        *
        (
            180.0 / rows
        )
    )

    lat_rad = cp.deg2rad(lat)

    cos_lat = cp.cos(lat_rad)

    cos_lat = cp.maximum(
        cos_lat,
        1.0e-6
    )

    cos_lat = cos_lat[:, None]

    dlat = np.pi / rows
    dlon = 2.0 * np.pi / cols

    cell_dy = (
        planet_radius_m
        * dlat
    )

    cell_dx = (
        planet_radius_m
        * dlon
        * cos_lat
    )

    cell_area = (
        cell_dx
        * cell_dy
    )

    cell_area_full = cp.broadcast_to(
        cell_area,
        dem.shape
    )

    # ============================================================
    # 4. SADE -> VUOSITTAINEN VESIMÄÄRÄ
    # ============================================================

    rain_m = rain / 1000.0

    annual_rain_volume = (
        rain_m
        * cell_area_full
    )

    # ============================================================
    # 5. MERI
    # ============================================================

    is_ocean = (
        dem <= 0.0
    )

    # ============================================================
    # 6. CLIMATE RUNOFF
    # ============================================================

    pet = (
        500.0
        *
        cp.exp(
            0.055
            *
            (
                temp - 10.0
            )
        )
    )

    pet *= (
        1.0
        /
        cp.sqrt(pressure)
    )

    pet *= (
        1.0
        /
        cp.sqrt(gravity)
    )

    pet = cp.clip(
        pet,
        50.0,
        10000.0
    )

    runoff_fraction = (
        rain
        /
        (
            rain
            + pet
            + 1.0e-6
        )
    )

    runoff_fraction = (
        0.01
        +
        0.89
        *
        runoff_fraction
    )

    runoff_fraction = cp.clip(
        runoff_fraction,
        0.0,
        0.90
    )

    runoff = (
        annual_rain_volume
        *
        runoff_fraction
    )

    runoff = cp.where(
        is_ocean,
        0.0,
        runoff
    )

    # ============================================================
    # 7. D8
    # ============================================================

    directions = [
        (-1, -1),   # NW
        (-1,  0),   # N
        (-1,  1),   # NE
        ( 0, -1),   # W
        ( 0,  1),   # E
        ( 1, -1),   # SW
        ( 1,  0),   # S
        ( 1,  1),   # SE
    ]

    flow_dir = cp.full(
        dem.shape,
        -1,
        dtype=cp.int8
    )

    max_slope = cp.full(
        dem.shape,
        -cp.inf,
        dtype=cp.float32
    )

    row_grid = cp.arange(
        rows,
        dtype=cp.int32
    )[:, None]

    col_grid = cp.arange(
        cols,
        dtype=cp.int32
    )[None, :]

    # ============================================================
    # 8. LASKE D8-SUUNTA
    # ============================================================

    for idx, (dy_idx, dx_idx) in enumerate(directions):

        neighbor = cp.empty_like(dem)

        # --------------------------------------------------------
        # Y
        # --------------------------------------------------------

        if dy_idx == -1:

            neighbor[:-1, :] = dem[1:, :]
            neighbor[-1, :] = dem[-1, :]

            valid_y = cp.ones(
                dem.shape,
                dtype=cp.bool_
            )

            valid_y[-1, :] = False

        elif dy_idx == 1:

            neighbor[1:, :] = dem[:-1, :]
            neighbor[0, :] = dem[0, :]

            valid_y = cp.ones(
                dem.shape,
                dtype=cp.bool_
            )

            valid_y[0, :] = False

        else:

            neighbor[:, :] = dem

            valid_y = cp.ones(
                dem.shape,
                dtype=cp.bool_
            )

        # --------------------------------------------------------
        # Longitude wrap
        # --------------------------------------------------------

        if dx_idx == -1:

            neighbor = cp.roll(
                neighbor,
                1,
                axis=1
            )

        elif dx_idx == 1:

            neighbor = cp.roll(
                neighbor,
                -1,
                axis=1
            )

        # --------------------------------------------------------
        # Fyysinen etäisyys
        # --------------------------------------------------------

        if dy_idx == 0:

            distance = cell_dx

        elif dx_idx == 0:

            distance = cell_dy

        else:

            distance = cp.sqrt(
                cell_dy ** 2
                +
                cell_dx ** 2
            )

        # --------------------------------------------------------
        # Korkeusero
        # --------------------------------------------------------

        drop = (
            dem
            - neighbor
        )

        slope = (
            drop
            /
            (
                distance
                + 1.0e-12
            )
        )

        valid = (
            (slope > 0.0)
            &
            (slope > max_slope)
            &
            valid_y
            &
            (~is_ocean)
        )

        max_slope = cp.where(
            valid,
            slope,
            max_slope
        )

        flow_dir = cp.where(
            valid,
            idx,
            flow_dir
        )

    # ============================================================
    # 9. DOWNSTREAM
    # ============================================================

    downstream = cp.full(
        dem.shape,
        -1,
        dtype=cp.int32
    )

    for idx, (dy_idx, dx_idx) in enumerate(directions):

        mask = (
            flow_dir == idx
        )

        target_row = (
            row_grid
            +
            dy_idx
        )

        target_col = (
            col_grid
            +
            dx_idx
        )

        target_col = (
            target_col % cols
        )

        valid_y = (
            (target_row >= 0)
            &
            (target_row < rows)
        )

        target_row_safe = cp.clip(
            target_row,
            0,
            rows - 1
        )

        target_index = (
            target_row_safe
            *
            cols
            +
            target_col
        )

        target_index = target_index.astype(
            cp.int32
        )

        downstream = cp.where(
            mask & valid_y,
            target_index,
            downstream
        )

    downstream_flat = downstream.ravel()

    # ============================================================
    # 10. FLOW ACCUMULATION
    # ============================================================

    incoming_count = cp.zeros(
        n,
        dtype=cp.int32
    )

    valid_sources = (
        (downstream_flat >= 0)
        &
        (~is_ocean.ravel())
    )

    cp.add.at(
        incoming_count,
        downstream_flat[valid_sources],
        1
    )

    accumulated_runoff = runoff.ravel().copy()

    accumulated_area = (
        cp.where(
            is_ocean,
            0.0,
            cell_area_full
        )
        .ravel()
        .copy()
    )

    processed = cp.zeros(
        n,
        dtype=cp.bool_
    )

    active = (
        (incoming_count == 0)
        &
        (~is_ocean.ravel())
    )

    # ============================================================
    # 11. TOPOLOGINEN FLOW ACCUMULATION
    # ============================================================

    for _ in range(iterations):

        active_idx = cp.where(
            active
        )[0]

        if active_idx.size == 0:
            break

        processed[
            active_idx
        ] = True

        dest = downstream_flat[
            active_idx
        ]

        valid = (
            dest >= 0
        )

        src_idx = active_idx[valid]
        dst_idx = dest[valid]

        cp.add.at(
            accumulated_runoff,
            dst_idx,
            accumulated_runoff[src_idx]
        )

        cp.add.at(
            accumulated_area,
            dst_idx,
            accumulated_area[src_idx]
        )

        cp.add.at(
            incoming_count,
            dst_idx,
            -1
        )

        active[
            active_idx
        ] = False

        active = (
            (incoming_count == 0)
            &
            (~processed)
            &
            (~is_ocean.ravel())
        )

    # ============================================================
    # 12. RESOLVOIMATTOMAT TASANKOALUEET
    # ============================================================

    unresolved = (
        (~processed)
        &
        (~is_ocean.ravel())
    )

    if bool(
        cp.any(unresolved).get()
    ):

        unresolved_idx = cp.where(
            unresolved
        )[0]

        flow_dir.ravel()[
            unresolved_idx
        ] = -1

        downstream_flat[
            unresolved_idx
        ] = -1

    # ============================================================
    # 13. 2D
    # ============================================================

    accumulated_runoff = accumulated_runoff.reshape(
        rows,
        cols
    )

    accumulated_area = accumulated_area.reshape(
        rows,
        cols
    )

    accumulated_area_km2 = (
        accumulated_area
        /
        1_000_000.0
    )

    # ============================================================
    # 14. SINKIT
    # ============================================================

    sink_mask = (
        (~is_ocean)
        &
        (flow_dir < 0)
    )

    # ============================================================
    # 15. JÄRVEN SEEDIT
    # ============================================================
    #
    # Ei vaadita valtavaa valuma-aluetta.
    # Järven syntyminen määräytyy sekä veden että painanteen
    # koon perusteella.
    # ============================================================

    lake_seed = (
        sink_mask
        &
        (
            accumulated_area_km2
            >=
            minimum_catchment_km2
        )
        &
        (
            accumulated_runoff
            >=
            lake_threshold
        )
    )

    seed_indices = cp.where(
        lake_seed.ravel()
    )[0]

    seed_indices_cpu = cp.asnumpy(
        seed_indices
    )

    # ============================================================
    # 16. JÄRVIEN MASKI
    # ============================================================

    lake_mask = cp.zeros(
        dem.shape,
        dtype=cp.bool_
    )

    # ------------------------------------------------------------
    # Käsitellään suurimmat järviseedit ensin.
    # ------------------------------------------------------------

    if len(seed_indices_cpu) > 10000:

        seed_values = (
            accumulated_runoff
            *
            lake_seed
        ).ravel()

        top_idx = cp.argpartition(
            seed_values,
            -10000
        )[-10000:]

        seed_indices_cpu = cp.asnumpy(
            top_idx
        )

    # ============================================================
    # 17. JÄRVEN KASVATUS
    # ============================================================

    for seed_flat in seed_indices_cpu:

        seed_flat = int(seed_flat)

        seed_row = (
            seed_flat // cols
        )

        seed_col = (
            seed_flat % cols
        )

        seed_height = float(
            dem[
                seed_row,
                seed_col
            ].get()
        )

        # --------------------------------------------------------
        # Paikallinen maski
        # --------------------------------------------------------

        local_mask = cp.zeros(
            dem.shape,
            dtype=cp.bool_
        )

        local_mask[
            seed_row,
            seed_col
        ] = True

        # --------------------------------------------------------
        # Etsi lähin korkeampi spill point.
        #
        # Kasvatetaan painannetta askel askeleelta.
        # --------------------------------------------------------

        current_level = seed_height

        max_level_steps = 256

        for _ in range(max_level_steps):

            # --------------------------------------------
            # Naapurit
            # --------------------------------------------

            expanded = local_mask.copy()

            for dy_idx, dx_idx in directions:

                shifted = cp.zeros_like(
                    local_mask
                )

                if dy_idx == -1:

                    shifted[:-1, :] = (
                        local_mask[1:, :]
                    )

                elif dy_idx == 1:

                    shifted[1:, :] = (
                        local_mask[:-1, :]
                    )

                else:

                    shifted[:, :] = local_mask

                if dx_idx == -1:

                    shifted = cp.roll(
                        shifted,
                        1,
                        axis=1
                    )

                elif dx_idx == 1:

                    shifted = cp.roll(
                        shifted,
                        -1,
                        axis=1
                    )

                expanded |= shifted

            # --------------------------------------------
            # Etsi ulkoreunan korkeudet.
            # --------------------------------------------

            boundary = (
                expanded
                &
                (~local_mask)
                &
                (~is_ocean)
            )

            boundary_heights = cp.where(
                boundary,
                dem,
                cp.inf
            )

            next_level = float(
                cp.min(
                    boundary_heights
                ).get()
            )

            if not np.isfinite(next_level):
                break

            # --------------------------------------------
            # Jos spill point on liian korkea,
            # painanne ei enää ole järkevä järvi.
            # --------------------------------------------

            if next_level <= current_level:

                current_level += 0.01

            else:

                current_level = next_level

            # --------------------------------------------
            # Täytä kaikki tämän vedenpinnan alle jäävät
            # solut, jotka ovat yhteydessä siemeneseen.
            # --------------------------------------------

            candidate = (
                expanded
                &
                (~is_ocean)
                &
                (
                    dem
                    <=
                    current_level
                )
            )

            # --------------------------------------------
            # Vain samaan painanteeseen liittyvät solut.
            # --------------------------------------------

            new_mask = (
                local_mask
                |
                candidate
            )

            # --------------------------------------------
            # Tarkista muutos.
            # --------------------------------------------

            changed = bool(
                cp.any(
                    new_mask != local_mask
                ).get()
            )

            local_mask = new_mask

            if not changed:
                break

            # --------------------------------------------
            # Jos järvi on jo tarpeeksi suuri,
            # lopeta kasvatus.
            # --------------------------------------------

            area_now = float(
                cp.sum(
                    cp.where(
                        local_mask,
                        cell_area_full,
                        0.0
                    )
                ).get()
                / 1_000_000.0
            )

            if (
                area_now
                >=
                minimum_lake_area_km2
            ):
                break

        # ====================================================
        # 18. JÄRVEN PINTA-ALA
        # ====================================================

        local_area_km2 = float(
            cp.sum(
                cp.where(
                    local_mask,
                    cell_area_full / 1_000_000.0,
                    0.0
                )
            ).get()
        )

        if (
            local_area_km2
            <
            minimum_lake_area_km2
        ):
            continue

        # ====================================================
        # 19. JÄRVEN VALUMA-ALUE
        # ====================================================

        local_catchment = float(
            cp.sum(
                cp.where(
                    local_mask,
                    accumulated_area_km2,
                    0.0
                )
            ).get()
        )

        if (
            local_catchment
            <
            minimum_catchment_km2
        ):
            continue

        # ====================================================
        # 20. JÄRVEN VESIMÄÄRÄ
        # ====================================================

        local_runoff = float(
            cp.max(
                cp.where(
                    local_mask,
                    accumulated_runoff,
                    0.0
                )
            ).get()
        )

        if (
            local_runoff
            <
            lake_threshold
        ):
            continue

        # ====================================================
        # 21. HYVÄKSY JÄRVI
        # ====================================================

        lake_mask |= local_mask

    # ============================================================
    # 22. JÄRVIEN RUNOFF
    # ============================================================

    lake_raster = cp.where(
        lake_mask,
        accumulated_runoff,
        0.0
    )
    # ============================================================
    # 22. SUURTEN JÄRVIEN LASKUJOET
    # ============================================================
    #
    # Etsitään jokaiselle riittävän suurelle järvelle
    # luonnollinen outlet eli alin järven ulkopuolinen solu,
    # johon järven vesi voi purkautua.
    #
    # Laskujoki saa järven koko valuma-alueen runoffin.
    # ============================================================

    lake_outlet_mask = cp.zeros(
        dem.shape,
        dtype=cp.bool_
    )

    lake_outlet_flow = cp.zeros(
        dem.shape,
        dtype=cp.float32
    )

    # ------------------------------------------------------------
    # Komponentit järville.
    #
    # Jokainen järvi tunnistetaan flood fill -tyyppisesti.
    # ------------------------------------------------------------

    remaining_lakes = lake_mask.copy()

    # Vain suuret järvet saavat erityisen laskujoen.
    large_lake_area_km2 = max(
        minimum_lake_area_km2 * 5.0,
        50.0
    )

    while bool(
        cp.any(remaining_lakes).get()
    ):

        # --------------------------------------------------------
        # Ota yksi järvisolu
        # --------------------------------------------------------

        seed = cp.where(
            remaining_lakes.ravel()
        )[0]

        if seed.size == 0:
            break

        seed_flat = int(
            seed[0].get()
        )

        seed_row = (
            seed_flat // cols
        )

        seed_col = (
            seed_flat % cols
        )

        # --------------------------------------------------------
        # Etsi koko järvikomponentti
        # --------------------------------------------------------

        component = cp.zeros(
            dem.shape,
            dtype=cp.bool_
        )

        component[
            seed_row,
            seed_col
        ] = True

        for _ in range(256):

            expanded = component.copy()

            for dy_idx, dx_idx in directions:

                shifted = cp.zeros_like(
                    component
                )

                if dy_idx == -1:

                    shifted[:-1, :] = (
                        component[1:, :]
                    )

                elif dy_idx == 1:

                    shifted[1:, :] = (
                        component[:-1, :]
                    )

                else:

                    shifted[:, :] = component

                if dx_idx == -1:

                    shifted = cp.roll(
                        shifted,
                        1,
                        axis=1
                    )

                elif dx_idx == 1:

                    shifted = cp.roll(
                        shifted,
                        -1,
                        axis=1
                    )

                expanded |= shifted

            new_component = (
                expanded
                &
                lake_mask
            )

            changed = bool(
                cp.any(
                    new_component != component
                ).get()
            )

            component = new_component

            if not changed:
                break

        # --------------------------------------------------------
        # Poista komponentti käsiteltävistä
        # --------------------------------------------------------

        remaining_lakes &= ~component

        # --------------------------------------------------------
        # Järven pinta-ala
        # --------------------------------------------------------

        lake_area_km2 = float(
            cp.sum(
                cp.where(
                    component,
                    cell_area_full / 1_000_000.0,
                    0.0
                )
            ).get()
        )

        if (
            lake_area_km2
            <
            large_lake_area_km2
        ):
            continue

        # --------------------------------------------------------
        # Etsi järven ulkoreuna
        # --------------------------------------------------------

        boundary = cp.zeros(
            dem.shape,
            dtype=cp.bool_
        )

        for dy_idx, dx_idx in directions:

            shifted = cp.zeros_like(
                component
            )

            if dy_idx == -1:

                shifted[:-1, :] = (
                    component[1:, :]
                )

            elif dy_idx == 1:

                shifted[1:, :] = (
                    component[:-1, :]
                )

            else:

                shifted[:, :] = component

            if dx_idx == -1:

                shifted = cp.roll(
                    shifted,
                    1,
                    axis=1
                )

            elif dx_idx == 1:

                shifted = cp.roll(
                    shifted,
                    -1,
                    axis=1
                )

            boundary |= (
                component
                &
                (~shifted)
            )

        # --------------------------------------------------------
        # Etsi matalin mahdollinen poistumissolu järven ulkopuolelta
        # --------------------------------------------------------

        external = cp.zeros_like(
            component
        )

        for dy_idx, dx_idx in directions:

            shifted = cp.zeros_like(
                component
            )

            if dy_idx == -1:

                shifted[:-1, :] = (
                    component[1:, :]
                )

            elif dy_idx == 1:

                shifted[1:, :] = (
                    component[:-1, :]
                )

            else:

                shifted[:, :] = component

            if dx_idx == -1:

                shifted = cp.roll(
                    shifted,
                    1,
                    axis=1
                )

            elif dx_idx == 1:

                shifted = cp.roll(
                    shifted,
                    -1,
                    axis=1
                )

            external |= (
                shifted
                &
                (~component)
                &
                (~is_ocean)
            )

        outlet_height = cp.where(
            external,
            dem,
            cp.inf
        )

        outlet_flat = int(
            cp.argmin(
                outlet_height.ravel()
            ).get()
        )

        outlet_height_value = float(
            dem.ravel()[
                outlet_flat
            ].get()
        )

        if not np.isfinite(
            outlet_height_value
        ):
            continue

        outlet_row = (
            outlet_flat // cols
        )

        outlet_col = (
            outlet_flat % cols
        )

        # --------------------------------------------------------
        # Varmista että outlet on oikeasti järven vieressä
        # --------------------------------------------------------

        is_adjacent = False

        for dy_idx, dx_idx in directions:

            rr = outlet_row + dy_idx
            cc = (
                outlet_col + dx_idx
            ) % cols

            if (
                0 <= rr < rows
                and
                bool(
                    component[
                        rr,
                        cc
                    ].get()
                )
            ):
                is_adjacent = True
                break

        if not is_adjacent:
            continue

        # --------------------------------------------------------
        # Outletin pitää olla järven pintaa alempana
        # tai käytännössä ensimmäinen spill point.
        # --------------------------------------------------------

        lake_surface = float(
            cp.max(
                cp.where(
                    component,
                    dem,
                    -cp.inf
                )
            ).get()
        )

        if (
            outlet_height_value
            >
            lake_surface + 1000.0
        ):
            continue

        # --------------------------------------------------------
        # Merkitse outlet
        # --------------------------------------------------------

        lake_outlet_mask[
            outlet_row,
            outlet_col
        ] = True

        # --------------------------------------------------------
        # Järven kokonaisvalunta
        # --------------------------------------------------------

        total_lake_runoff = float(
            cp.max(
                cp.where(
                    component,
                    accumulated_runoff,
                    0.0
                )
            ).get()
        )

        lake_outlet_flow[
            outlet_row,
            outlet_col
        ] = total_lake_runoff

        # ========================================================
        # 22B. SEURAA OUTLETISTA ALASPÄIN
        # ========================================================

        current = outlet_flat

        visited = cp.zeros(
            n,
            dtype=cp.bool_
        )

        for _ in range(n):

            if current < 0:
                break

            if bool(
                visited[current].get()
            ):
                break

            visited[current] = True

            rr = (
                current // cols
            )

            cc = (
                current % cols
            )

            # ----------------------------------------------------
            # Älä tee merestä laskujokea
            # ----------------------------------------------------

            if bool(
                is_ocean[
                    rr,
                    cc
                ].get()
            ):
                break

            # ----------------------------------------------------
            # Merkitse laskujoen virtaus
            # ----------------------------------------------------

            river_value = (
                total_lake_runoff
            )

            if (
                river_value
                >=
                river_threshold
            ):

                lake_outlet_flow[
                    rr,
                    cc
                ] = max(
                    float(
                        lake_outlet_flow[
                            rr,
                            cc
                        ].get()
                    ),
                    river_value
                )

            next_cell = int(
                downstream_flat[
                    current
                ].get()
            )

            if next_cell < 0:
                break

            # ----------------------------------------------------
            # Jos saavutetaan toinen järvi,
            # yhdistetään järvien vedet.
            # ----------------------------------------------------

            nr = (
                next_cell // cols
            )

            nc = (
                next_cell % cols
            )

            if bool(
                lake_mask[
                    nr,
                    nc
                ].get()
            ):
                break

            current = next_cell

    # ============================================================
    # 23. JOET
    # ============================================================

    river_mask = (
        (~is_ocean)
        &
        (~lake_mask)
        &
        (flow_dir >= 0)
        &
        (
            accumulated_area_km2
            >=
            minimum_catchment_km2
        )
        &
        (
            accumulated_runoff
            >=
            river_threshold
        )
    )

    # ------------------------------------------------------------
    # Lisää suurten järvien laskujoet
    # ------------------------------------------------------------

    river_mask |= (
        lake_outlet_flow
        >=
        river_threshold
    )

    river_raster = cp.where(
        river_mask,
        cp.maximum(
            accumulated_runoff,
            lake_outlet_flow
        ),
        0.0
    )

    # ============================================================
    # 24. PALAUTUS CPU:LLE
    # ============================================================

    return (
        river_raster.get(),
        lake_raster.get()
    )

    # ============================================================
    # 23. JOET
    # ============================================================

    river_mask = (
        (~is_ocean)
        &
        (~lake_mask)
        &
        (flow_dir >= 0)
        &
        (
            accumulated_area_km2
            >=
            minimum_catchment_km2
        )
        &
        (
            accumulated_runoff
            >=
            river_threshold
        )
    )

    river_raster = cp.where(
        river_mask,
        accumulated_runoff,
        0.0
    )

    # ============================================================
    # 24. PALAUTUS CPU:LLE
    # ============================================================

    return (
        river_raster.get(),
        lake_raster.get()
    )







def apply_dem_erosion(dem, cell_size=30.0, iterations=20, erosion_rate=0.02, 
                      peneplain_threshold=100.0, protected_threshold=-200.0, smoothing_sigma=2.0):
    """
    Suorittaa korkeusriippuvaisen eroosioalgoritmisen metriselle DEM-datalle.
    
    Parametrit:
    - dem: 2D numpy-array (korkeudet metreinä)
    - cell_size: Pikselin koko metreinä (esim. 30m / Landsat/SRTM)
    - iterations: Algoritmin ajokerrat
    - erosion_rate: Muutosnopeus per iteraatio
    - peneplain_threshold: Korkeus (m), jonka alapuolella maasto tasoitetaan peneplaaniksi
    - protected_threshold: Korkeus (m), jonka alapuoliseen maastoon (esim. mannerjalusta) EI kosketa
    - smoothing_sigma: Alavien maiden tasoituksen voimakkuus (pikseleinä)
    """
    h = dem.copy().astype(float)
    
    for _ in range(iterations):
        # 1. Lasketaan gradientit ja rinteet (metreinä ottaen huomioon solukoko)
        grad_y, grad_x = np.gradient(h, cell_size)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Simuloidaan veden valumista ja uomanmuodostusta (Fluvial erosion)
        # Mitä suurempi sigma, sitä laajemmalle uomat ja niiden vaikutusalue leviävät
        flow_accumulation = ndimage.gaussian_filter(slope, sigma=1.5)
        # Normalisoidaan virtaustekijä paikallisesti vakauden takaamiseksi
        if flow_accumulation.max() > flow_accumulation.min():
            flow_factor = (flow_accumulation - flow_accumulation.min()) / (flow_accumulation.max() - flow_accumulation.min() + 1e-6)
        else:
            flow_factor = np.zeros_like(flow_accumulation)

        # 3. Määritetään alueelliset maskit korkeuden perusteella
        # Koskematon alue (esim. kaikki alle -200m)
        protected_mask = h <= protected_threshold
        
        # Alava maa, joka kulutetaan tasangoksi (esim. välillä -200m ja +100m)
        lowland_mask = (h > protected_threshold) & (h <= peneplain_threshold)
        
        # Vuoristo/Ylänkö, joka terävöitetään (esim. kaikki yli +100m)
        highland_mask = h > peneplain_threshold
        
        # 4. YLÄNGÖT: Virtauseroosio (Stream power law -henkinen uurtaminen)
        # Kuluttaa laaksoja siellä missä vesi virtaa ja rinne on jyrkkä, jolloin huiput jäävät teräviksi.
        # Muutetaan eroosion määrä suhteessa metriasteikkoon
        erosion_amount = flow_factor * slope * cell_size * erosion_rate
        h[highland_mask] -= erosion_amount[highland_mask]
        
        # 5. ALANGOT: Peneplanaatio (Morfologinen diffuusio)
        # Tasoitetaan alavat maat Gauss-smennuksella matalaksi puolitasangoksi.
        blurred = ndimage.gaussian_filter(h, sigma=smoothing_sigma)
        h[lowland_mask] = h[lowland_mask] * (1 - erosion_rate) + blurred[lowland_mask] * erosion_rate
        
        # 6. PALAUTUS: Varmistetaan, että suojattu alue pysyy täysin koskemattomana
        h[protected_mask] = dem[protected_mask]
        
    return h





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

    merelta_maalle_virtaus = ndimage2.gaussian_filter(merelta_maalle_virtaus, sigma=5.0)    
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


# 25.9.2026

def laske_lampotila_vaihtelu_gpu(
    lat_hila: cp.ndarray,
    etaisyys_mereen_km: cp.ndarray,
    sademara_mm: cp.ndarray,
    auringon_deklinaatio_rad: float,
    orbital_period_days: float,
    rotation_period_h: float
) -> cp.ndarray:

    # Latitudi radiaaneiksi
    lat_rad = cp.deg2rad(lat_hila)

    # Etäisyys auringon deklinaatiosta
    kulma = cp.abs(lat_rad - auringon_deklinaatio_rad)

    # Aurinkovaikutus:
    # 0° -> +1
    # 60° -> 0
    # 90° -> -1
    aurinko_faktori = 2.0 * cp.cos(kulma) - 1.0

    # Mannermaisuus
    manner_faktori = cp.log1p(etaisyys_mereen_km) * 2.5

    # Kuivuus
    kuivuus_faktori = cp.exp(-sademara_mm / 100.0) * 8.0

    # Vuorokauden pituus
    rotaatio_faktori = cp.sqrt(rotation_period_h / 24.0)

    # Kiertoaika
    kierto_faktori = 1.0 + (365.25 / orbital_period_days) * 0.1

    vaihtelu_rasteri = (
        (5.0 + manner_faktori + kuivuus_faktori)
        * aurinko_faktori
        * rotaatio_faktori
        * kierto_faktori
    )

    return vaihtelu_rasteri






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
                     perussade=2.0,
                     tropiikin_max_sade=180.0,
                     lauhkean_max_sade=100.0,
                     tropiikin_leveys=12.0,
                     lauhkean_leveys=15.0):

    # Perussade: pieni taustasade kaikkialla
    #perussade = cp.clip(perussade, 0.0, 50.0)

    # Ilmastollisen päiväntasaajan sijainti
    #itcz_keskus = ilmastollinen_paivantaasaaja * 0.25
    itcz_keskus = ilmastollinen_paivantaasaaja

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
    lauhkea_keskus = 55

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












def laske_photosynthetic_radiation(
    local_radiation_Wm2,
    star_teff_K,
    atmosphere_pressure_Pa,
    n_lambda=5000
):

    # ----------------------------------------
    # 1. Aallonpituus
    # ----------------------------------------

    wavelength = cp.linspace(
        200e-9,
        2000e-9,
        n_lambda,
        dtype=cp.float64
    )

    nm = wavelength * 1e9

    # ----------------------------------------
    # 2. Planck-spektri
    # ----------------------------------------

    B = (
        2 * h * c**2
        / wavelength**5
        / (
            cp.exp(
                h * c
                / (wavelength * k * star_teff_K)
            ) - 1
        )
    )

    # KOKONAISENERGIA
    total_energy = cp.trapezoid(
        B,
        wavelength
    )

    spectral_fraction = B / total_energy

    # ----------------------------------------
    # 3. Earth-like transmission
    # ----------------------------------------

    transmission = cp.ones_like(wavelength)

    transmission[nm < 300] = 0.02

    uv = (
        (nm >= 300) &
        (nm < 400)
    )

    transmission[uv] = (
        0.20
        + 0.80 * (nm[uv] - 300) / 100
    )

    visible = (
        (nm >= 400) &
        (nm <= 700)
    )

    transmission[visible] = 0.90

    nir = (
        (nm > 700) &
        (nm <= 1000)
    )

    transmission[nir] = 0.80

    ir = nm > 1000
    transmission[ir] = 0.55

    # ----------------------------------------
    # 4. Earth PAR
    # ----------------------------------------

    earth_mask = (
        (nm >= 400) &
        (nm <= 700)
    )

    # PAR-fraktio ennen painekorjausta
    par_fraction_earth = cp.trapezoid(
        spectral_fraction[earth_mask]
        * transmission[earth_mask],
        wavelength[earth_mask]
    )

    # ----------------------------------------
    # 5. Earth PPFD
    # ----------------------------------------

    photon_spectrum = (
        spectral_fraction
        * transmission
        * wavelength
        / (h * c)
    )

    ppfd_fraction_earth = cp.trapezoid(
        photon_spectrum[earth_mask],
        wavelength[earth_mask]
    )

    ppfd_fraction_earth = (
        ppfd_fraction_earth
        / NA
        * 1e6
    )

    # ----------------------------------------
    # 6. Optimimaalinen aallonpituus
    # ----------------------------------------
    optimal_wavelength_nm = (
    680.0
    + 0.12 * (5800.0 - star_teff_K)
    )

    optimal_wavelength_nm = cp.maximum(
    optimal_wavelength_nm,
    680.0
    )

    optimal_wavelength_nm = cp.minimum(
    optimal_wavelength_nm,
    1000.0
    )

    sigma_nm = 100.0

    alien_response = cp.exp(
        -0.5
        * (
            (nm - optimal_wavelength_nm)
            / sigma_nm
        )**2
    )

    alien_response[nm < 400] = 0
    alien_response[nm > 1100] = 0

    alien_response /= alien_response.max()

    # ----------------------------------------
    # 7. Optimized PAR / PPFD
    # ----------------------------------------

    optimized_par_fraction = cp.trapezoid(
        spectral_fraction
        * transmission
        * alien_response,
        wavelength
    )

    optimized_ppfd_fraction = cp.trapezoid(
        spectral_fraction
        * transmission
        * wavelength
        / (h * c)
        * alien_response,
        wavelength
    )

    optimized_ppfd_fraction = (
        optimized_ppfd_fraction
        / NA
        * 1e6
    )

    # ----------------------------------------
    # 8. Paine raster
    # ----------------------------------------

    pressure_ratio = (
        atmosphere_pressure_Pa
        / P_EARTH
    )

    # ----------------------------------------
    # 9. Painekorjattu läpäisy
    #
    # Käytetään yksinkertaista efektiivistä
    # PAR-läpäisyä.
    # ----------------------------------------

    earth_transmission_pressure = (
        par_fraction_earth
        ** pressure_ratio
    )

    optimized_transmission_pressure = (
        optimized_par_fraction
        ** pressure_ratio
    )

    # ----------------------------------------
    # 10. Rasteritulokset
    # ----------------------------------------

    par_earth_Wm2 = (
        local_radiation_Wm2
        * earth_transmission_pressure
    )

    ppfd_earth_umol_m2_s = (
        local_radiation_Wm2
        * ppfd_fraction_earth
        * (
            earth_transmission_pressure
            / par_fraction_earth
        )
    )

    optimized_par_Wm2 = (
        local_radiation_Wm2
        * optimized_transmission_pressure
    )

    optimized_ppfd_umol_m2_s = (
        local_radiation_Wm2
        * optimized_ppfd_fraction
        * (
            optimized_transmission_pressure
            / optimized_par_fraction
        )
    )

    return {
        "par_earth_Wm2": par_earth_Wm2,
        "ppfd_earth_umol_m2_s":
            ppfd_earth_umol_m2_s,

        "optimized_par_Wm2":
            optimized_par_Wm2,

        "optimized_ppfd_umol_m2_s":
            optimized_ppfd_umol_m2_s,

        "optimal_wavelength_nm":
            float(optimal_wavelength_nm.get()),

        "stellar_peak_nm":
            float(
                (
                    2.897771955e-3
                    / star_teff_K
                    * 1e9
                )
            )
    }






def laske_lamposummat(
    vuoden_pituus_a,
    pyorahdysaika_h,
    lampotilat,
    raja_lampotila=5.0,
    ylaraja_lampotila=30.0,
):
    """
    Laskee 12 kuukausittaisen lämpötilarasterin perusteella
    GDD-tyyppisen lämpösumman.

    Parametrit
    ----------
    vuoden_pituus_a : float
        Planeetan vuoden pituus Maan vuosina.

    pyorahdysaika_h : float
        Planeetan pyörähdysaika tunteina.
        Ei vaikuta nykyiseen GDD-laskuun, mutta mukana rajapinnassa
        myöhempää vuorokausivaihtelun huomiointia varten.

    lampotilat : cupy.ndarray
        Muoto (12, y, x).
        12 kuukausittaista keskilämpötilarasteria °C.

    raja_lampotila : float
        GDD:n alaraja. Oletus 5 °C.

    ylaraja_lampotila : float
        GDD:n yläraja. Oletus 30 °C.

    Palauttaa
    ----------
    ylittava_lamposumma : cupy.ndarray
        GDD-lämpösumma °C·päivää.

    alittava_lamposumma : cupy.ndarray
        Lämpösumma alle 5 °C:n rajan °C·päivää.
    """

    lampotilat = cp.asarray(lampotilat, dtype=cp.float32)

    if lampotilat.shape[0] != 12:
        raise ValueError(
            "lampotilat pitää sisältää 12 kuukausirasteria "
            "muodossa (12, y, x)"
        )

    # Planeetan vuoden pituus Maan päivinä.
    vuoden_pituus_paivina = 365.25 * vuoden_pituus_a

    # Oletetaan 12 yhtä pitkää vuodenaikajaksoa.
    kuukauden_pituus_paivina = vuoden_pituus_paivina / 12.0

    # -----------------------------------------
    # GDD
    # -----------------------------------------

    # Rajoitetaan lämpötila GDD:n ala- ja ylärajaan.
    t_gdd = cp.clip(
        lampotilat,
        raja_lampotila,
        ylaraja_lampotila
    )

    # GDD:n päivittäinen arvo kuukausikeskiarvosta.
    gdd_paiva = t_gdd - raja_lampotila

    # Muutetaan kuukausi °C·päiviksi.
    gdd_kuukausi = (
        gdd_paiva * kuukauden_pituus_paivina
    )

    # Summa 12 kuukaudelta.
    ylittava_lamposumma = cp.sum(
        gdd_kuukausi,
        axis=0
    )

    # -----------------------------------------
    # Alle 5 °C oleva lämpösumma
    # -----------------------------------------

    alitus = cp.maximum(
        raja_lampotila - lampotilat,
        0.0
    )

    alittava_lamposumma = cp.sum(
        alitus * kuukauden_pituus_paivina,
        axis=0
    )

    return ylittava_lamposumma, alittava_lamposumma




def laske_npp_miami_kuukausittain(
    lampotilat,
    sateet,
    vuoden_pituus_a,
):
    lampotilat = cp.asarray(lampotilat, dtype=cp.float32)
    sateet = cp.asarray(sateet, dtype=cp.float32)

    vuoden_pituus_paivina = 365.25 * vuoden_pituus_a
    #kuukauden_pituus_paivina = vuoden_pituus_paivina / 12.0

    # Miami lämpötilarajoite
    npp_t = 3000.0 / (
        1.0 + cp.exp(
            1.315 - 0.119 * lampotilat
        )
    )

    # Miami sadannan rajoite
    npp_p = 3000.0 * (
        1.0 - cp.exp(
            -0.000664 * sateet
        )
    )

    # Se kumpi rajoittaa kasvua
    npp_kuukausi = cp.minimum(npp_t, npp_p)
    # Kuukausiosuus vuosituotannosta
    npp_kuukausi *= (
        vuoden_pituus_a
    )

    return npp_kuukausi



def korjaa_npp(
    perus_npp,
    lampotilat,
    star_teff_k,
    ilmakehan_paine_maa=1.0,
    p_co2_ppm=280.0,
    kasvillisuuden_laatu="for_sunlike",
    planeetan_npp_kerroin=1.0,
    optimi_lampotila=20.0,
    lampostressi_alku=25.0,
    lampostressi_max=45.0,
):
    """
    Korjaa Miami-mallin perus-NPP:tä planeetan olosuhteisiin.

    perus_npp:
        Kuukausittainen Miami-NPP, muoto (12, y, x).

    lampotilat:
        Kuukausittaiset keskilämpötilat °C, muoto (12, y, x).

    star_teff_k:
        Tähden efektiivinen lämpötila K.

    ilmakehan_paine_maa:
        Pintapaine suhteessa Maahan.
        1.0 = Maan nykyinen paine.

    p_co2_ppm:
        CO2-pitoisuus ppm.

    kasvillisuuden_laatu:
        "for_sunlike"
        "for_k_dwarf"
        "for_red_dwarf"

    Palauttaa:
        Korjatun kuukausittaisen NPP:n muodossa (12, y, x).
    """

    perus_npp = cp.asarray(perus_npp, dtype=cp.float32)
    lampotilat = cp.asarray(lampotilat, dtype=cp.float32)

    if perus_npp.shape != lampotilat.shape:
        raise ValueError(
            "perus_npp ja lampotilat pitää olla samanmuotoisia"
        )

    # =========================================================
    # 1. KORKEA LÄMPÖTILA
    # =========================================================
    #
    # Miami ei vähennä NPP:tä riittävästi korkeissa lämpötiloissa.
    #
    # Alle lampostressi_alku:
    #     ei lisärangaistusta
    #
    # 25–45 °C:
    #     NPP pienenee asteittain
    #
    # >= 45 °C:
    #     lämpötilakerroin = 0
    #
    # Käytetään pehmeää kvadraattista pudotusta.
    #

    stressi = (
        lampotilat - lampostressi_alku
    ) / (
        lampostressi_max - lampostressi_alku
    )

    stressi = cp.clip(stressi, 0.0, 1.0)

    lampostressi_kerroin = 1.0 - stressi ** 2

    # =========================================================
    # 2. ILMAKEHÄN PAINE
    # =========================================================
    #
    # Paineen vaikutus ei ole lineaarinen.
    # Maa = 1.
    #
    # Käytetään sqrt-riippuvuutta ja rajataan vaikutus.
    #

    paine = max(float(ilmakehan_paine_maa), 0.01)

    paine_kerroin = 0.85 + 0.15 * (paine ** 0.5)

    paine_kerroin = max(
        0.5,
        min(1.3, paine_kerroin)
    )

    # =========================================================
    # 3. CO2
    # =========================================================
    #
    # Fotosynteettinen hyöty riippuu CO2:n osapaineesta.
    #
    # Vertailu:
    #     Maa = 1 atm, 280 ppm
    #
    # =========================================================

    co2_osapaine_suhde = (
        paine
        * p_co2_ppm
        / 280.0
    )

    # Michaelis-Menten-tyyppinen kyllästyminen.
    #
    # 280 ppm + 1 atm -> 1.0
    #
    # K = 280 tarkoittaa, että CO2-vaikutus
    # alkaa kyllästyä vähitellen.
    #

    co2_raakatekija = (
        co2_osapaine_suhde
        / (
            co2_osapaine_suhde
            + 1.0
        )
    )

    co2_maa = 1.0 / 2.0

    co2_kerroin = (
        co2_raakatekija / co2_maa
    )

    # Rajataan ääripäät.
    co2_kerroin = max(
        0.4,
        min(2.0, co2_kerroin)
    )

    # =========================================================
    # 4. TÄHDEN SPEKTRI
    # =========================================================
    #
    # Tämä on tarkoituksella alustava empiirinen kerroin.
    #
    # Teff ei kerro tähden kokonaiskirkkautta, vaan ensisijaisesti
    # spektrin lämpötilaa.
    #
    # Kasvillisuuden oletetaan olevan eri tavalla sopeutunutta
    # eri tähtityypeille.
    #

    if kasvillisuuden_laatu == "for_sunlike":

        # Aurinkotyyppinen kasvillisuus.
        #
        # Maa / Aurinko = 1.
        #
        spektri_kerroin = (
            star_teff_k / 5778.0
        ) ** 0.15

    elif kasvillisuuden_laatu == "for_k_dwarf":

        # K-kääpiön punaisempi spektri.
        #
        # Oletetaan sopeutunut kasvillisuus.
        #
        spektri_kerroin = (
            0.97
            * (star_teff_k / 5000.0) ** 0.10
        )

    elif kasvillisuuden_laatu == "for_red_dwarf":

        # M-kääpiön spektri.
        #
        # Kasvillisuuden oletetaan käyttävän punaista /
        # lähi-infrapunaa tehokkaammin.
        #
        spektri_kerroin = (
            0.90
            * (star_teff_k / 3500.0) ** 0.05
        )

    else:
        raise ValueError(
            f"Tuntematon kasvillisuuden_laatu: "
            f"{kasvillisuuden_laatu}"
        )

    # =========================================================
    # 5. YHDISTETTY KORJAUS
    # =========================================================

    kokonaiskerroin = (
        lampostressi_kerroin
        * paine_kerroin
        * co2_kerroin
        * spektri_kerroin
        * planeetan_npp_kerroin
    )

    # =========================================================
    # 6. KORJATTU NPP
    # =========================================================

    korjattu_npp = (
        perus_npp
        * kokonaiskerroin
    )

    return korjattu_npp


## laske_säätä



def laske_planeetan_dtr_rh_pilvet_pressure(t_mean, sade_mm_kk, paine_bar=1.0, g=1.0, rotaatio_h=24):
    """
    Kaikki syötteet ja tulokset ovat CuPy-rastereita (tai skalaareja, jotka CuPy broadcasting hoitaa).
    """
    # Suojataan g ja paine negatiivisilta tai nolla-arvoilta logaritmeja varten
    g_safe = cp.maximum(0.001, g)
    paine_bar_safe = cp.maximum(0.001, paine_bar)
    
    # Rotaatiopäivät (suojattu nollalta)
    rot_days = cp.maximum(0.01, rotaatio_h) / 24.0
    
    # Perus-DTR (Lämpötilan vuorokausivaihtelu)
    dtr_base = 8 * cp.sqrt(rot_days) / paine_bar_safe * cp.power(g_safe, -0.2)

    # Paine-indeksi pilvisyyttä varten
    paine_ind = paine_bar_safe / (paine_bar_safe + 1.0)

    # Kostutusindeksi: suojataan t_mean nollalta/negatiivisuudelta cp.maximum-funktiolla
    kostutus = sade_mm_kk / (2 * cp.maximum(1.0, t_mean))
    
    # Suhteellinen kosteus (RH) kostutusindeksin pohjalta
    rh = kostutus / 2.0

    # Paine ja g muuttavat hieman vesihöyryn käyttäytymistä
    rh += 0.04 * cp.log1p(paine_bar_safe)
    rh += 0.02 * cp.log(g_safe)
    rh = cp.clip(rh, 0.1, 1.0) # Vastaa max(0.1, min(1.0, rh)) rasterille
    
    # Pilvisyyden laskenta (sigmoid-funktio + kosteus + paine)
    pilvi = 1.0 / (1.0 + cp.exp(-(rh - 0.68) / 9.0))
    pilvi = 0.65 * pilvi + 0.35 * cp.clip(kostutus, 0.0, 1.0)
    pilvi += 0.08 * paine_ind
    pilvi = cp.clip(pilvi, 0.0, 1.0)    

    # DTR-kerroin ilman if/else-haaroitusta käyttämällä cp.where-funktiota
    # Jos kostutus > 0.2, käytetään käänteislukua. Muuten käytetään kuivan maan kaavaa.
    dtr_coeff_kostea = 1.0 / kostutus
    dtr_coeff_kuiva = 5.0 - (kostutus * 20.0)
    dtr_coeff_kuiva = cp.maximum(1.0, dtr_coeff_kuiva)
    
    dtr_coeff = cp.where(kostutus > 0.2, dtr_coeff_kostea, dtr_coeff_kuiva)
 
    dtr = dtr_base * dtr_coeff
  
    # Sademäärä pudottaa paikallista ilmanpainetta hieman (matalapaine)
    tulospaine = paine_bar - (sade_mm_kk - 50) / 5000.0
    tulospaine = cp.maximum(0.0, tulospaine)
    
    # Minimi- ja maksimilämpötilat vuorokauden aikana rastereina
    t_min = t_mean - (dtr / 2.0)
    t_max = t_mean + (dtr / 2.0)
    
    # Pyöristykset tehdään elementtikohtaisesti rastereille
    return {
        "paine_bar": cp.round(tulospaine, 3),
        "RH_prosentti": cp.round(rh * 100, 1),
        "pilvisyys_prosentti": cp.round(pilvi * 100, 1),
        "DTR_C": cp.round(dtr, 1),
        "Tmin_C": cp.round(t_min, 1),
        "Tmax_C": cp.round(t_max, 1)
    }






#### peruslämpötila


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
    rot_period_h=24.0,
    leveysaste_viilennys_kerroin=0.13,
    star_luminosity_lsun=1.0, star_teff_k=5777
):
    """
    Laskee planeetan aurinkoperuslämpötilan sekä
    leveysastekohtaisen tähden säteilyn.

    Parametrit
    ----------
    maa_maski2 :
        0 = meri
        1 = maa

    lat :
        Leveysaste asteina. Voi olla NumPy- tai CuPy-taulukko.

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

    rot_period_h :
        Planeetan pyörähdysaika tunteina.
        Maa = 24 h.

    leveysaste_viilennys_kerroin :
        Nykyisen yksinkertaisen leveysastejäähdytyksen
        peruskerroin.

    star_luminosity_lsun :
        Tähden luminositeetti Auringon luminositeettina.
        Aurinko = 1.0.

    Palauttaa
    ----------
    perus_lampotila :
        CuPy-taulukko, sama muoto kuin lat.

    deklinaatio :
        Tähden/planeetan deklinaatio asteina.

    sateily_kerroin :
        Planeetan etäisyyden ja tähden luminositeetin
        aiheuttama lämpötilakerroin.

    ratakulma_rad :
        Planeetan ratakulma radiaaneina.

    paikallinen_sateily_wm2 :
        Leveysastekohtainen päivittäinen keskimääräinen
        tähtisäteily W/m².
        Sama taulukkomuoto kuin lat.
    """

    # ============================================================
    # PARAMETRIEN TARKISTUS
    # ============================================================

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

    if rot_period_h <= 0.0:
        raise ValueError(
            "rot_period_h pitää olla suurempi kuin nolla."
        )

    if star_luminosity_lsun <= 0.0:
        raise ValueError(
            "star_luminosity_lsun pitää olla suurempi kuin nolla."
        )

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
    # TÄHDEN SÄTEILY
    #
    # Säteilyvuo:
    #
    # F ~ L / r²
    #
    # Lämpötila:
    #
    # T ~ F^(1/4)
    # ============================================================

    stellar_flux_factor = (
        star_luminosity_lsun
        /
        (
            etaisyys_au
            *
            etaisyys_au
        )
    )

    sateily_kerroin = (
        stellar_flux_factor ** 0.25
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

    sateily_lampotila_muutos = (
        tmax
        *
        cp.float32(
            sateily_kerroin - 1.0
        )
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
    # ILMAKEHÄN SUHTEELLINEN LÄMPÖINERTIA
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

    ilmakeha_inertia = np.clip(
        ilmakeha_inertia,
        0.01,
        100.0,
    )

    # ============================================================
    # ORBITAALISEN VUODEN PITUUDEN VAIKUTUS
    # ============================================================

    orbital_period_factor = np.sqrt(
        orbital_period_years
    )

    inertia_factor = np.sqrt(
        ilmakeha_inertia
    )

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
        cp.float32(
            seasonal_response
        )
    )

    tasapaino_temp = (
        tmax
        +
        sateily_lampotila_muutos
    )

    # ============================================================
    # LEVEYSASTE
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
    # rot_period_h = tunnit
    # Maa = 24 h
    # ============================================================

    rotation_period_days = (
        rot_period_h
        /
        24.0
    )

    rotation_factor = np.sqrt(
        rotation_period_days
    )

    rotation_factor = np.clip(
        rotation_factor,
        0.25,
        4.0,
    )

    # ============================================================
    # MAA / MERI
    #
    # 0 = meri
    # 1 = maa
    #
    # Aikaisemman 1.50 sijaan kokeillaan 1.10.
    # ============================================================

    kerroin_taulukko = cp.array(
        [
            1.00,
            1.10,
        ],
        dtype=cp.float32
    )

    leveysaste_viilennys = (
        kerroin_taulukko[
            maa_maski2.astype(cp.int32)
        ]
        *
        zeniitti_etaisyys
        *
        cp.float32(
            rotation_factor
        )
        *
        cp.float32(
            leveysaste_viilennys_kerroin
        )
    )

    # ============================================================
    # LOPULLINEN PERUSLÄMPÖTILA
    # ============================================================

    perus_lampotila = (
        tasapaino_temp
        -
        leveysaste_viilennys
    )

    # ============================================================
    # TÄHDEN TODELLINEN SÄTEILY W/m²
    #
    # Aurinko 1 AU:ssa:
    # ~1361 W/m²
    #
    # Tämä EI vielä sisällä albedoa.
    # ============================================================

    SOLAR_CONSTANT_W_M2 = 1361.0

    stellar_flux_wm2 = (
        SOLAR_CONSTANT_W_M2
        *
        cp.float32(
            stellar_flux_factor
        )
    )

    # ============================================================
    # LEVEYSASTEKOHTAINEN PÄIVITTÄINEN KESKIMÄÄRÄINEN SÄTEILY
    #
    # Tulos on samaa muotoa kuin perus_lampotila.
    #
    # Jokainen leveysaste saa oman W/m²-arvonsa.
    # ============================================================

    lat_rad = cp.radians(
        lat_cp
    )

    deklinaatio_rad = cp.radians(
        deklinaatio_cp
    )

    # Auringonnousun/-laskun tuntikulma.
    #
    # cos(H0) = -tan(phi) * tan(delta)
    #
    cos_h0 = (
        -cp.tan(lat_rad)
        *
        cp.tan(deklinaatio_rad)
    )

    # Napayö / yötön yö käsitellään erikseen.
    polar_night = (
        cos_h0 > 1.0
    )

    polar_day = (
        cos_h0 < -1.0
    )

    cos_h0 = cp.clip(
        cos_h0,
        cp.float32(-1.0),
        cp.float32(1.0),
    )

    h0 = cp.arccos(
        cos_h0
    )

    # Päivittäinen keskimääräinen insolaation geometrinen osa.

    daily_solar_flux = (
        stellar_flux_wm2
        /
        cp.pi
        *
        (
            h0
            *
            cp.sin(lat_rad)
            *
            cp.sin(deklinaatio_rad)
            +
            cp.cos(lat_rad)
            *
            cp.cos(deklinaatio_rad)
            *
            cp.sin(h0)
        )
    )

    # Napayö = ei auringonsäteilyä.

    daily_solar_flux = cp.where(
        polar_night,
        cp.float32(0.0),
        daily_solar_flux,
    )

    # Yötön yö:
    #
    # Aurinko on koko vuorokauden horisontin yläpuolella.
    #
    # Tällöin päivittäinen keskimääräinen säteily voidaan
    # laskea täydellä vuorokaudella.

    polar_day_flux = (
        stellar_flux_wm2
        *
        (
            cp.sin(lat_rad)
            *
            cp.sin(deklinaatio_rad)
        )
    )

    # Tämä ei kuitenkaan riitä yksinään, koska vuorokauden
    # aikana zeniittikulma muuttuu.
    #
    # Käytetään kaavaa H0 = pi yöttömän yön tapauksessa.

    polar_day_flux = (
        stellar_flux_wm2
        /
        cp.pi
        *
        (
            cp.pi
            *
            cp.sin(lat_rad)
            *
            cp.sin(deklinaatio_rad)
        )
    )

    daily_solar_flux = cp.where(
        polar_day,
        polar_day_flux,
        daily_solar_flux,
    )

    # Estetään mahdolliset pienet numeeriset negatiiviset arvot.

    paikallinen_sateily_wm2 = cp.maximum(
        daily_solar_flux,
        cp.float32(0.0)
    )

    # ============================================================
    # PALAUTUS
    # ============================================================

    return (
        perus_lampotila,
        cp.float32(deklinaatio),
        cp.float32(sateily_kerroin),
        cp.float32(ratakulma_rad),
        paikallinen_sateily_wm2,
    )





################################
## ilmasto


def laske_ilmasto_gpu(
    star_mass_msun=1.0,
    star_luminosity_lsun=1.0,
    star_teff_k=5778,
    korkeus_syvyyskartta=None,
    planet_mass_me=1,
    planet_radius_re=1.0,
    planet_tilt=23.44,
    planet_ecc=0.013,
    planet_mvelp_deg=101.2,
    planet_orbital_period=1,
    planet_rotation_hours=24,
    planet_atmosphere_pressure=1,
    planet_co2_ppm=1,
    landfrac=0.3,
    planet_tmean=13.8,
    global_moisture_coeff=1.0,
):

    orbit_days = planet_orbital_period*365.25                  # Vuoden pituus päivinä
    pyorinta = planet_rotation_period_hours                # 36 tunnin vuorokausi

    oceanfrac = 1 - landfrac
    tmax_default = planet_tmean + 25

    gee = planet_mass_me / (planet_radius_re * planet_radius_re)
    planet_gee_earths=gee

    landfrac_vaikutus_leveysaste_viilennys = (landfrac / 0.3)

    leveysaste_viilennys_kerroin = (
        0.49
        * math.sqrt(planet_radius_re)
        * math.sqrt(24 / planet_rotation_hours)
        * (1 / planet_atmosphere_pressure)
        * landfrac_vaikutus_leveysaste_viilennys
    )

    lapse_rate = lapse_rate_helppo(
        planet_tmean,
        planet_atmosphere_pressure * 1013.25,
        gee,
        global_moisture_coeff
    )

    lapse_rate = lapse_rate * oceanfrac * 1.126

    itcz_coeff = 0.4

    korkeus, leveys = korkeus_syvyyskartta.shape

    print("GPU laskenta ...")

    # ============================================================
    # 1. KAIKKI STAATTINEN DATA GPU:LLE
    # ============================================================

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

    # ------------------------------------------------------------
    # Lämpötilan korkeusvaikutus
    # ------------------------------------------------------------

    lämpötila_vähete = korkeuskartta * lapse_rate*-1

    # ============================================================
    # 2. GEOMETRIA
    # ============================================================

    R = 6371.0088 * planet_radius_re

    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    lat_rad = cp.deg2rad(lat_grid_deg)

    solukoko_y_km = R * dlat_rad

    solukoko_x_km = (
        R
        * dlon_rad
        * cp.cos(lat_rad)
    )

    # ============================================================
    # 3. GRADIENTIT GPU:LLA
    # ============================================================

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

    # ============================================================
    # 4. MAA / MERI
    # ============================================================

    maa_maski = korkeuskartta > 0
    meri_bin = ~maa_maski

    maa_maski2 = np.where(
        korkeuskartta > 0,
        1,
        0
    ).astype(int)

    # ============================================================
    # 5. ETÄISYYS RANNIKOSTA
    # ============================================================

    etaisyys_meresta_km = laske_etaisyys_rannikosta_gpu(
        maa_maski,
        maapallon_sade_km=R,
    )

    # ============================================================
    # 6. OUTPUT-ARRAYT
    # ============================================================

    kuukausi_lämpötilat_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    kuukausi_sateet_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32
    )

    #kuukausi_npp_gpu = cp.empty(
    #    (12, korkeus, leveys),
    #    dtype=cp.float32
    #)
    kuukausi_ptopet_gpu = cp.empty(
        (12, korkeus, leveys),
        dtype=cp.float32)
    
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

    # ============================================================
    # 7. KUUKAUSISILMUKKA
    # ============================================================

    for kk in range(12):

        
        #print(f"Kuukausi {kk + 1}/12")

        # ========================================================
        # AURINKO
        # ========================================================

        dt_years = planet_orbital_period / 12.0
        time_years = kk * dt_years


        (
            perus_lampotila,
            deklinaatio,
            sateily_kerroin,
            ratakulma_rad, 
            paikallinen_sateily
        ) = laske_auringon_peruslampotila(
            star_luminosity_lsun=star_luminosity_lsun,
            star_teff_k=star_teff_k,
            maa_maski2=maa_maski2,
            lat=lat_grid_deg,
            time_years=time_years,
            tilt_planet_axis=planet_tilt,
            ecc_planet=planet_ecc,
            mvelp_planet=planet_mvelp_deg,
            tmax_planet_param=tmax_default,
            orbital_period_years=planet_orbital_period,
            planet_mass_me=planet_mass_me,
            planet_radius_re=planet_radius_re,
            planet_atmosphere_pressure=planet_atmosphere_pressure *101325.0,
            rot_period_h=planet_rotation_hours,
            leveysaste_viilennys_kerroin=leveysaste_viilennys_kerroin,
            
        )
        #print(deklinaatio)
  
        zeniitti_etaisyys_deg=lat_grid_deg -deklinaatio
        ilmastollinen_paivantaasaaja = (
            deklinaatio * itcz_coeff
        )

        #plt.imshow(perus_lampotila.get())
        #plt.show()
        #quit(-1)
        kuukausi_lämpötilat_gpu[kk] = perus_lampotila

        result = laske_photosynthetic_radiation(
        local_radiation_Wm2=paikallinen_sateily,
        star_teff_K=star_teff_k,
        atmosphere_pressure_Pa=101325.0*planet_atmosphere_pressure 
        )


        # ========================================================
        # TUULI
        # ========================================================

        perustuulitulos = laske_tuulet(
            lat_grid_deg,
            kk,
            planet_tilt,
            itcz_kerroin=itcz_coeff
        )

        tuuli_x = perustuulitulos[0] * 1
        tuuli_y = perustuulitulos[1] * 1

        tuuli_x_yksikko = perustuulitulos[2]
        tuuli_y_yksikko = perustuulitulos[3]

        tuuli_suunta = perustuulitulos[4]
        tuuli_voima = perustuulitulos[5]

        kuukausi_tuuli_suunta_gpu[kk] = tuuli_suunta
        kuukausi_tuuli_voima_gpu[kk] = tuuli_voima



        # ========================================================
        # MERIVIRRAT
        # ========================================================

        merivirrat_tulos = laske_merivirrat(
            lat_grid_deg,
            syvyyskartta,
            meri_bin,
            etaisyys_meresta_km,
            tuuli_x_yksikko,
            tuuli_y_yksikko,
            tuuli_voima,
            kk,
            ilmastollinen_paivantaasaaja,
        )

        virta_x = merivirrat_tulos[0]
        virta_y = merivirrat_tulos[1]
        virta_voima = merivirrat_tulos[2]
        virta_suunta = merivirrat_tulos[3]
        merivirta_lampotila_anomalia = merivirrat_tulos[4]

        kuukausi_merivirta_x_gpu[kk] = virta_x
        kuukausi_merivirta_y_gpu[kk] = virta_y

        # ========================================================
        # PERUSSATEET
        # ========================================================

        perussade = laske_perus_sade(
            lat_grid_deg,
            ilmastollinen_paivantaasaaja,
            tropiikin_max_sade=180.0,
            lauhkean_max_sade=60.0,
            lauhkean_leveys=15.0,
            tropiikin_leveys=8.0,
        )

        perussade = perussade * global_moisture_coeff
        kuukausi_sateet_gpu[kk]=perussade
        #sade = perussade

        # ========================================================
        # SADEVYÖHYKKEIDEN TARKENNUS
        # ========================================================

        #perussade = tarkenna_sadevyohykkeet(
        #    perussade,
        #    lat_grid_deg,
        #    ilmastollinen_paivantaasaaja,
        #)

        # ========================================================
        # SATEEN VAIKUTUS LÄMPÖTILAAN
        # ========================================================

        perussade_vaikutus_lampotilaan = (
            - perussade / 100.0
        )
        #plt.imshow(perussade_vaikutus_lampotilaan.get())
        #plt.show()
        #quit(-1)
 
        meri_pilvisyys_vaikutus_lampotilaan = (
             meri_bin * -5
        )

        # --------------------------------------------------------
        # MANTEREISUUS
        # --------------------------------------------------------

        #sade_asteikko = 75.0

        #sade_puskuri = cp.exp(
        #    -kuukausi_lämpötilat_gpu[kk] * 0
        #)

        # Käytetään nykyistä sadetta, kuten alkuperäisessä
        # koodissa tässä vaiheessa.
        #sade_puskuri = cp.exp(
        #    -perussade / sade_asteikko
        #)

        #lampotila_pehmeä = cp.tanh(
        #    kuukausi_lämpötilat_gpu[kk] / 5.0
        #)

        #mantereisuus_anomalia = (
        #    (etaisyys_meresta_km / 300)
        #    * lampotila_pehmeä
        #    * sade_puskuri
        #)

        #kuukausi_lämpötilat_gpu[kk] = (
        #    kuukausi_lämpötilat_gpu[kk]
        #    + meri_pilvisyys_vaikutus
        #    + merivesi_anomalia
        #)

        #kuukausi_lämpötilat_gpu[kk] = (
        #    kuukausi_lämpötilat_gpu[kk]
        #    + mantereisuus_anomalia
        #)

        
        mantereisuus_lampotila_anomalia= laske_lampotila_vaihtelu_gpu(lat_grid_deg, etaisyys_meresta_km ,  kuukausi_sateet_gpu[kk], deklinaatio,  orbit_days, pyorinta)
        #plt.imshow(zeniitti_etaisyys_deg.get())
        #plt.imshow(lat_grid_deg.get())
        #plt.imshow(mantereisuus_lampotila_anomalia.get())
        #plt.show()
        #quit(-1)

        # Jos tulos halutaan siirtää takaisin CPU:lle (NumPy):
        # tulos_cpu = cp.asnumpy(tulos_gpu)
        # ========================================================
        # LAPSE RATE
        # ========================================================



        # ========================================================
        # PAINESOLUJEN PYÖRTEET
        # ========================================================

        tuuli_x, tuuli_y = (
            laske_painesolujen_pyorteet(
                tuuli_x,
                tuuli_y,
                lat_grid_deg,
                kuukausi_lämpötilat_gpu[kk],
                maa_maski,
                etaisyys_meresta_km,
                dy=1.0,
                dx=1.0,
                paine_suora_voimakkuus=0.3,
                paine_pyörre_voimakkuus=0.7 * 1
            )
        )

        # ========================================================
        # MONSUUNIT
        # ========================================================

        tuuli_x, tuuli_y = (
            laske_monsuunituuli(
                tuuli_x,
                tuuli_y,
                kuukausi_lämpötilat_gpu[kk],
                maa_maski,
                dx=1.0,
                dy=1.0,
                monsuuni_voimakkuus=0.5
            )
        )


        # ========================================================
        # TUULEN ETÄISYYS MERESTÄ
        # ========================================================

        #sade = kuukausi_sateet_gpu[kk]

        tuulietaisyys_meresta_km = (
            laske_tuulietaisyys_meresta_gpu(
                meri_bin,
                tuuli_x,
                tuuli_y,
                lat_grid_deg,
                planet_radius_re=planet_radius_re,
                max_pixel_etaisyys=500,
            )
        )

        # ========================================================
        # MONSUUNISADE
        # ========================================================

        monsuuni_sade_muutos = (
            laske_monsuunin_sade_muutos_laaja(
                tuuli_x,
                tuuli_y,
                maa_maski,
                kuukausi_lämpötilat_gpu[kk],
                tuulietaisyys_meresta_km,
                monsuuni_sade_kerroin=15.0,
                kosteuden_kantama_km=400.0,
                talvikuivuus_kerroin=25.0
            )
        )

        kuukausi_sateet_gpu[kk] = (
            kuukausi_sateet_gpu[kk]
            #+ monsuuni_sade_muutos
        )

        #plt.imshow(monsuuni_sade_muutos.get())
        #plt.show()
        #quit(-1)        

        # ========================================================
        # MERELTÄ TULEVA KOSTEUS
        # ========================================================

        merelta_tuleva_kosteus = (
            laske_merelta_kulkeva_kosteus(
                tuulietaisyys_meresta_km,
                kosteus_vaikutusmatka_km=700.0,
                kosteus_max=1.5,
            )
        )-1
        merelta_tuleva_sade= merelta_tuleva_kosteus*perussade*maa_maski_gpu
        
        #plt.imshow(merelta_tuleva_sade.get())
        #plt.show()
        #quit(-1)


        #sade = (
        #    perussade *
        #    merelta_tuleva_kosteus
        #)
        # ========================================================
        # OROGRAFIA
        # ========================================================

        orografiatulos = laske_orografia(
            korkeuskartta,
            korkeus_gradientti_x,
            korkeus_gradientti_y,
            tuuli_x_yksikko,
            tuuli_y_yksikko,
            tuuli_voima,
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
        # OROGRAFINEN VAIKUTUS
        # ========================================================

        sade_orografia_vaikutuskerroin = (
            laske_orografiakerroin(
                korkeuskartta,
                orografinen_pakote,
        #        korkeus_kerroin=0.00005,
                korkeus_kerroin=0.00005,
                nousu_kerroin=0.15,
                lasku_kerroin=0.08,
            )
        )

        sade = (
            perussade *
            sade_orografia_vaikutuskerroin
        ) +merelta_tuleva_sade+monsuuni_sade_muutos

        # ========================================================
        # LOPULLINEN SADE
        # ========================================================
        lapse_ratex = laske_lapse_rate_gpu(
            kuukausi_lämpötilat_gpu[kk],
            perussade,
            korkeuskartta,
            planet_atmosphere_pressure * 101325,
            g=9.81 * gee,
            sademuutos_raja=0.0
        )

        lämpötila_vähetex = (
            lapse_ratex * -korkeuskartta
        )
        #plt.imshow(lämpötila_vähetex.get())
        #plt.imshow(perus_lampotila.get())
        #plt.show()
        #quit(-1)
        #kuukausi_lämpötilat_gpu[kk] = (
        #    kuukausi_lämpötilat_gpu[kk]
        #    + lämpötila_vähetex
        #)

        sade=cp.where(sade<0,0,sade)
        sade=cp.where(sade>10000,10000,sade)
        kuukausi_sateet_gpu[kk] = sade

        #kuukausi_lämpötilat_gpu[kk] = (
        #    kuukausi_lämpötilat_gpu[kk]
        #    + (monsuuni_sade_muutos / 100)
        #)
        #meri_pilvisyys_vaikutus_lampotilaan
        #sade_vaikutus_lampotilaan?
        #mantereisuus_lampotila_anomalia ?
        #merivirta_lampotila_anomalia 
        #plt.imshow(mantereisuus_lampotila_anomalia.get())
        #plt.show()
        #quit(-1)
        
        
        totaali_lampotila=perus_lampotila+lämpötila_vähetex 
        +mantereisuus_lampotila_anomalia  -sade/100
         #+merivirta_lampotila_anomalia -sade/100
        #   meri_pilvisyys_vaikutus_lampotilaan+lämpötila_vähetex \
        #   + (monsuuni_sade_muutos / 100)
        #sade_vaikutus_lampotilaan+merivirta_lampotila_anomalia 
        #mantereisuus_lampotila_anomalia

        kuukausi_lämpötilat_gpu[kk]=totaali_lampotila

        makkink_pet=laske_makkink_evaporation(paikallinen_sateily,totaali_lampotila)
        #makkink_pet=cp.where(makkink_pet==cp.nan,0,makkink_pet)
        #makkink_pet=cp.where(makkink_pet==cp.inf,0,makkink_pet)
        kuukausi_ptopet_gpu[kk]=sade/makkink_pet
        kuukausi_ptopet_gpu[kk]=cp.where(kuukausi_ptopet_gpu[kk]>50, 50,kuukausi_ptopet_gpu[kk]) 
        
        saa_param=laske_planeetan_dtr_rh_pilvet_pressure(totaali_lampotila, sade, paine_bar=planet_atmosphere_pressure, g=planet_gee_earths, rotaatio_h=planet_rotation_hours)
        # Siirretään arvot sanakirjasta omiin GPU-muuttujiinsa
        rh_gpu = saa_param["RH_prosentti"]
        pilvi_gpu = saa_param["pilvisyys_prosentti"]
        paine_gpu = saa_param["paine_bar"]
        dtr_gpu = saa_param["DTR_C"]
        tmin_gpu = saa_param["Tmin_C"]
        tmax_gpu = saa_param["Tmax_C"]      

        #plt.imshow(p_pet.get())
        #plt.show()
        #quit(-1)
    # gdd
    gdd5, below5 =laske_lamposummat(
    planet_orbital_period,planet_rotation_period_hours,kuukausi_lämpötilat_gpu,raja_lampotila=5.0)
    # npp
    kuukausi_npp_gpu = laske_npp_miami_kuukausittain(
    kuukausi_lämpötilat_gpu,
    kuukausi_sateet_gpu,
    planet_orbital_period,
    )

    planet_npp_coeff=planet_orbital_period,planet_rotation_period_hours
    planet_npp_coeff=1.0

    kuukausi_npp_gpu = korjaa_npp(
    perus_npp=kuukausi_npp_gpu,
    lampotilat= kuukausi_lämpötilat_gpu,
    star_teff_k=star_teff_k,
    ilmakehan_paine_maa=planet_atmosphere_pressure,
    p_co2_ppm=planet_co2_ppm,
    kasvillisuuden_laatu="for_sunlike",
    planeetan_npp_kerroin=planet_npp_coeff,
    )
    #plt.imshow(kuukausi_npp_gpu.sum(axis=0).get())
    #plt.imshow(kuukausi_ptopet_gpu.sum(axis=0).get())
    #plt.show()
    #quit(-1) 
    #kuukausi_ptopet_gpu=ptopet
    #kuukausi_npp_gpu=npp

    
    #plt.imshow(npp_kk_gpu[0].get())
    #plt.imshow(gdd5.get())
    #plt.show()
    #quit(-1)


    # ============================================================
    # GPU SYNKRONOINTI
    # ============================================================

    cp.cuda.Stream.null.synchronize()

    # ============================================================
    # PALUU
    # ============================================================

    return (
        kuukausi_lämpötilat_gpu,
        kuukausi_sateet_gpu,
        kuukausi_npp_gpu,
        kuukausi_ptopet_gpu,       
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


##############################################
###############################################






# ============================================================
# APUFUNKTIOT
# ============================================================

def pixel_idx(y, x, w):
    return int(y * w + x)


def pixel_to_coord(y, x, h, w):
    """
    Rasteripikseli -> WGS84.

    Oletus:
        y=0     -> +90°
        y=h-1   -> -90°
        x=0     -> -180°
        x=w-1   -> lähes +180°
    """

    lon = -180.0 + (x / w) * 360.0
    lat = 90.0 - (y / h) * 180.0

    return lon, lat


def coord_to_pixel(lons, lats, h, w):
    """
    WGS84 -> rasteripikselit.
    """

    lons = np.asarray(lons)
    lats = np.asarray(lats)

    x = (
        (lons + 180.0)
        / 360.0
        * w
    )

    y = (
        (90.0 - lats)
        / 180.0
        * h
    )

    x = np.clip(
        np.rint(x).astype(int),
        0,
        w - 1
    )

    y = np.clip(
        np.rint(y).astype(int),
        0,
        h - 1
    )

    return y, x


# ============================================================
# KAUPUNKIPISTEET
# ============================================================

def tee_kaupunkipisteet(
    korkeuskartta,
    annual_npp,
    rivers2,
    lakes2,
    meri_maski,

    npp_kerroin=2.0,
    korkeus_kerroin=2.0,
    joki_kerroin=2.0,
    ranta_kerroin=5.0,
    jokisuu_kerroin=20.0
):
    """
    Luo rasterin, joka kertoo kuinka hyvä kukin kuivan maan
    pikseli on kaupungin paikaksi.

    Tätä samaa rasteria voidaan käyttää kaikilla
    hierarkiatasoilla.
    """

    korkeus = np.asarray(
        korkeuskartta,
        dtype=float
    )

    rivers2 = np.asarray(
        rivers2
    ).astype(bool)

    lakes2 = np.asarray(
        lakes2
    ).astype(bool)

    meri = np.asarray(
        meri_maski
    ).astype(bool)

    # --------------------------------------------------------
    # Kuiva maa
    # --------------------------------------------------------

    maa = (
        ~meri &
        ~rivers2 &
        ~lakes2
    )

    pisteet = np.full(
        korkeus.shape,
        -np.inf,
        dtype=float
    )

    if not np.any(maa):
        return pisteet

    # --------------------------------------------------------
    # Korkeus
    # --------------------------------------------------------

    korkeus_pos = np.maximum(
        korkeus,
        0
    )

    h95 = np.nanpercentile(
        korkeus_pos[maa],
        95
    )

    korkeus_norm = np.clip(
        korkeus_pos / (h95 + 1e-9),
        0,
        1
    )

    # --------------------------------------------------------
    # NPP
    # --------------------------------------------------------

    npp = np.asarray(
        annual_npp,
        dtype=float
    )

    npp95 = np.nanpercentile(
        npp[maa],
        95
    )

    npp_norm = np.clip(
        npp / (npp95 + 1e-9),
        0,
        1
    )

    # --------------------------------------------------------
    # Etäisyys jokeen
    # --------------------------------------------------------

    etaisyys_jokeen = (
        ndimage.distance_transform_edt(
            ~rivers2
        )
    )

    joki_lahistolla = (
        etaisyys_jokeen > 0
    ) & (
        etaisyys_jokeen <= 10
    )

    # --------------------------------------------------------
    # Etäisyys mereen
    # --------------------------------------------------------

    etaisyys_mereen = (
        ndimage.distance_transform_edt(
            ~meri
        )
    )

    meren_ranta = (
        etaisyys_mereen > 0
    ) & (
        etaisyys_mereen <= 10
    )

    # --------------------------------------------------------
    # Jokisuu
    # --------------------------------------------------------

    jokisuu = (
        joki_lahistolla &
        meren_ranta
    )

    # --------------------------------------------------------
    # Pisteet
    # --------------------------------------------------------

    pisteet[maa] = (
        npp_kerroin * npp_norm[maa]
        +
        korkeus_kerroin *
        (1.0 - korkeus_norm[maa])
    )

    pisteet[
        joki_lahistolla & maa
    ] += joki_kerroin

    pisteet[
        meren_ranta & maa
    ] += ranta_kerroin

    pisteet[
        jokisuu & maa
    ] += jokisuu_kerroin

    return pisteet


# ============================================================
# KUSTANNUSMATRIISI
# ============================================================

def tee_kustannusmatriisi(
    korkeuskartta,
    lakes2,
    meri,
    reittityyppi="maa",
    maa_kerroin=1.0,
    meri_kerroin=1.0
):
    """
    Luo maa-, meri- tai yhdistetyn reitityksen kustannuksen.
    """

    korkeus = np.asarray(
        korkeuskartta,
        dtype=float
    )

    lakes2 = np.asarray(
        lakes2
    ).astype(bool)

    meri = np.asarray(
        meri
    ).astype(bool)

    # ========================================================
    # MAA
    # ========================================================

    maa = (
        ~meri &
        ~lakes2
    )

    korkeus_pos = np.maximum(
        korkeus,
        0
    )

    h95 = np.nanpercentile(
        korkeus_pos[maa],
        95
    )

    korkeus_norm = np.clip(
        korkeus_pos / (h95 + 1e-9),
        0,
        1
    )

    gy, gx = np.gradient(
        korkeus
    )

    jyrkkyys = np.sqrt(
        gx**2 + gy**2
    )

    slope95 = np.nanpercentile(
        jyrkkyys[maa],
        95
    )

    jyrkkyys_norm = np.clip(
        jyrkkyys / (slope95 + 1e-9),
        0,
        1
    )

    maa_kustannus = (
        maa_kerroin *
        (
            1.0
            +
            3.0 * korkeus_norm
            +
            20.0 * jyrkkyys_norm**2
        )
    )

    # ========================================================
    # MERI
    # ========================================================

    rannikko_etaisyys = (
        ndimage.distance_transform_edt(
            ~maa
        )
    )

    rannikko_sade = 2000.0
    rannikko_rangaistus = 5.0

    meri_kustannus = (
        meri_kerroin *
        (
            1.0
            +
            (
                rannikko_etaisyys /
                rannikko_sade
            )**2
            * rannikko_rangaistus
        )
    )

    # ========================================================
    # VALINTA
    # ========================================================

    if reittityyppi == "maa":

        kustannus = maa_kustannus.copy()

        # Myös meri ja järvet ovat esteitä
        kustannus[~maa] = np.inf

    elif reittityyppi == "meri":

        kustannus = meri_kustannus.copy()

        # Vain meri
        kustannus[maa] = np.inf

    elif reittityyppi == "yhdistetty":

        kustannus = np.where(
            maa,
            maa_kustannus,
            meri_kustannus
        )

    else:

        raise ValueError(
            f"Tuntematon reittityyppi: "
            f"{reittityyppi}"
        )

    return kustannus


# ============================================================
# PALLOMAINEN SEAMLESS-VERKKO
# ============================================================

def rakenna_kustannusverkko(
    kustannus_matriisi,
    korkeuskartta
):
    """
    Rakentaa pallomaisen rasteriverkon.

    Longitude wrapataan.
    Latitude ei wrapata.

    TÄRKEÄ:
    korkeuskartta annetaan parametrina eikä lueta
    globaalista muuttujasta.
    """

    h, w = kustannus_matriisi.shape
    n = h * w

    lat = np.linspace(
        90.0,
        -90.0,
        h
    )

    lat_rad = np.deg2rad(
        lat
    )

    dlat = np.pi / max(
        h - 1,
        1
    )

    dlon = (
        2.0 * np.pi / w
    )

    y, x = np.indices(
        (h, w)
    )

    rows = []
    cols = []
    data = []

    suunnat = (
        (-1,  0),
        ( 1,  0),
        ( 0, -1),
        ( 0,  1),
        (-1, -1),
        (-1,  1),
        ( 1, -1),
        ( 1,  1)
    )

    for dy, dx in suunnat:

        ny = y + dy
        nx = (x + dx) % w

        valid = (
            (ny >= 0) &
            (ny < h)
        )

        if not np.any(valid):
            continue

        sy = y[valid]
        sx = x[valid]

        ty = ny[valid]
        tx = nx[valid]

        source_cost = (
            kustannus_matriisi[
                sy,
                sx
            ]
        )

        target_cost = (
            kustannus_matriisi[
                ty,
                tx
            ]
        )

        ok = (
            np.isfinite(
                source_cost
            )
            &
            np.isfinite(
                target_cost
            )
        )

        if not np.any(ok):
            continue

        sy = sy[ok]
        sx = sx[ok]

        ty = ty[ok]
        tx = tx[ok]

        source_cost = source_cost[ok]
        target_cost = target_cost[ok]

        # ----------------------------------------------------
        # Pallopinnan etäisyys
        # ----------------------------------------------------

        lat1 = lat_rad[sy]
        lat2 = lat_rad[ty]

        dlat_move = (
            lat2 - lat1
        )

        if dx != 0:

            lat_center = (
                lat1 + lat2
            ) / 2.0

            dlon_move = dlon

        else:

            lat_center = (
                lat1 + lat2
            ) / 2.0

            dlon_move = 0.0

        distance = np.sqrt(
            dlat_move**2
            +
            (
                np.cos(lat_center)
                *
                dlon_move
            )**2
        )

        # ----------------------------------------------------
        # Nousu
        # ----------------------------------------------------

        h1 = korkeuskartta[
            sy,
            sx
        ]

        h2 = korkeuskartta[
            ty,
            tx
        ]

        nousu = np.maximum(
            h2 - h1,
            0.0
        )

        nousu_norm = np.clip(
            nousu / 1000.0,
            0,
            1
        )

        base_cost = (
            source_cost +
            target_cost
        ) / 2.0

        edge_cost = (
            distance
            *
            base_cost
            *
            (
                1.0
                +
                20.0 * nousu_norm
            )
        )

        source = (
            sy * w + sx
        ).astype(np.int64)

        target = (
            ty * w + tx
        ).astype(np.int64)

        rows.extend(
            source.tolist()
        )

        cols.extend(
            target.tolist()
        )

        data.extend(
            edge_cost.tolist()
        )

    return csr_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(n, n)
    )


# ============================================================
# YHDEN KAUPUNGIN DIJKSTRA
# ============================================================

def laske_etaisyydet(
    graph,
    alku_yx
):
    """
    Dijkstra yhdestä pikselistä kaikkiin muihin.
    """

    h = None

    w = None

    # graphin koko on N x N.
    # Rasterin leveys annetaan kutsujalta tarvittaessa.
    return dijkstra(
        graph,
        directed=True,
        indices=alku_yx
    )


# ============================================================
# KAUPUNKIEN ALUEET
# ============================================================

def laske_kustannusalueet(
    graph,
    keskukset,
    shape
):
    """
    Kustannuspohjainen Voronoi-jako.

    Jokainen pikseli saa lähimmän keskuksen indeksin.
    """

    h, w = shape

    if len(keskukset) == 0:

        return (
            np.full(
                (h, w),
                np.nan
            ),
            np.full(
                (h, w),
                np.inf
            )
        )

    indeksit = np.array(
        [
            pixel_idx(y, x, w)
            for y, x in keskukset
        ],
        dtype=np.int64
    )

    distances = dijkstra(
        csgraph=graph,
        directed=True,
        indices=indeksit
    )

    distances = np.atleast_2d(
        distances
    )

    alueet = np.argmin(
        distances,
        axis=0
    )

    etaisyydet = np.min(
        distances,
        axis=0
    )

    alueet = (
        alueet
        .reshape(h, w)
        .astype(float)
    )

    etaisyydet = (
        etaisyydet
        .reshape(h, w)
    )

    alueet[
        ~np.isfinite(etaisyydet)
    ] = np.nan

    return (
        alueet,
        etaisyydet
    )


# ============================================================
# PALLONPINTAINEN SUUNTA EMOSTA EHDOKKAASEEN
# ============================================================

def pallomainen_suunta(
    parent_y,
    parent_x,
    child_y,
    child_x,
    h,
    w
):
    """
    Laskee suunnan emosta lapseen pallopinnalla.

    Palauttaa kulman radiaaneina.

    0   = itä
    pi/2 = pohjoinen
    -pi/2 = etelä
    """

    parent_lon, parent_lat = (
        pixel_to_coord(
            parent_y,
            parent_x,
            h,
            w
        )
    )

    child_lon, child_lat = (
        pixel_to_coord(
            child_y,
            child_x,
            h,
            w
        )
    )

    lon1 = np.deg2rad(
        parent_lon
    )

    lon2 = np.deg2rad(
        child_lon
    )

    lat1 = np.deg2rad(
        parent_lat
    )

    lat2 = np.deg2rad(
        child_lat
    )

    dlon = (
        lon2 - lon1
    )

    # Lyhin longitude-suunnan ero
    dlon = (
        (dlon + np.pi)
        % (2.0 * np.pi)
    ) - np.pi

    y = (
        np.sin(dlon)
        * np.cos(lat2)
    )

    x = (
        np.cos(lat1)
        * np.sin(lat2)
        -
        np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    return np.arctan2(
        y,
        x
    )


# ============================================================
# SEKTORIN PARAS EHDOKAS
# ============================================================

def valitse_sektorikeskukset(
    pisteet,
    ehdokasalue,
    parent_yx,
    h,
    w,
    lapsia=6,
    minimietaisyys=10,
    sektorin_toleranssi=np.deg2rad(35.0)
):
    """
    Valitsee emolle lapsikeskuksia kuudesta suunnasta.

    Tärkeä ero tavalliseen top-N-valintaan:

        EI:
            kuusi parasta pikseliä

        VAAN:
            kuusi hyvää pikseliä kuudesta eri suunnasta.

    Sektorit ovat kuitenkin joustavia. Siksi tulos ei
    muodosta täydellistä kuusikulmiota.
    """

    py, px = parent_yx

    yy, xx = np.where(
        ehdokasalue
    )

    if len(yy) == 0:
        return []

    arvot = pisteet[
        yy,
        xx
    ]

    kelpaa = np.isfinite(
        arvot
    )

    yy = yy[kelpaa]
    xx = xx[kelpaa]
    arvot = arvot[kelpaa]

    if len(yy) == 0:
        return []

    # --------------------------------------------------------
    # Lasketaan jokaisen ehdokkaan suunta
    # --------------------------------------------------------

    kulmat = np.array([
        pallomainen_suunta(
            py,
            px,
            cy,
            cx,
            h,
            w
        )
        for cy, cx in zip(
            yy,
            xx
        )
    ])

    # --------------------------------------------------------
    # Kuusi sektoria
    #
    # Keskitykset:
    #
    # 0°
    # 60°
    # 120°
    # 180°
    # 240°
    # 300°
    # --------------------------------------------------------

    sektorit = np.linspace(
        0,
        2.0 * np.pi,
        lapsia,
        endpoint=False
    )

    valitut = []

    for sektorikeskus in sektorit:

        # Kulmaero [-pi, pi]
        delta = np.abs(
            np.arctan2(
                np.sin(
                    kulmat -
                    sektorikeskus
                ),
                np.cos(
                    kulmat -
                    sektorikeskus
                )
            )
        )

        ehdokkaat = np.where(
            delta <=
            sektorin_toleranssi
        )[0]

        if len(ehdokkaat) == 0:
            continue

        # ----------------------------------------------------
        # Paras ehdokas sektorissa
        # ----------------------------------------------------

        jarjestys = ehdokkaat[
            np.argsort(
                arvot[ehdokkaat]
            )[::-1]
        ]

        loytyi = False

        for idx in jarjestys:

            cy = int(
                yy[idx]
            )

            cx = int(
                xx[idx]
            )

            # ------------------------------------------------
            # Etäisyys emoon
            # ------------------------------------------------

            d_parent = np.sqrt(
                (cy - py) ** 2
                +
                (cx - px) ** 2
            )

            if d_parent < minimietaisyys:
                continue

            # ------------------------------------------------
            # Etäisyys muihin lapsiin
            # ------------------------------------------------

            liian_lahella = False

            for ly, lx in valitut:

                d = np.sqrt(
                    (cy - ly) ** 2
                    +
                    (cx - lx) ** 2
                )

                if d < minimietaisyys:

                    liian_lahella = True
                    break

            if liian_lahella:
                continue

            valitut.append(
                (cy, cx)
            )

            loytyi = True
            break

        # ----------------------------------------------------
        # Jos tästä sektorista ei löytynyt mitään,
        # sitä voidaan täydentää myöhemmin.
        # ----------------------------------------------------

    # ========================================================
    # TÄYDENNYS
    # ========================================================

    if len(valitut) < lapsia:

        jarjestys = np.argsort(
            arvot
        )[::-1]

        for idx in jarjestys:

            if len(valitut) >= lapsia:
                break

            cy = int(
                yy[idx]
            )

            cx = int(
                xx[idx]
            )

            if (
                cy,
                cx
            ) in valitut:
                continue

            d_parent = np.sqrt(
                (cy - py) ** 2
                +
                (cx - px) ** 2
            )

            if d_parent < minimietaisyys:
                continue

            liian_lahella = False

            for ly, lx in valitut:

                d = np.sqrt(
                    (cy - ly) ** 2
                    +
                    (cx - lx) ** 2
                )

                if d < minimietaisyys:

                    liian_lahella = True
                    break

            if liian_lahella:
                continue

            valitut.append(
                (cy, cx)
            )

    return valitut


# ============================================================
# JUURIKESKUSTEN VALINTA
# ============================================================

def valitse_paakeskukset(
    pisteet,
    maa_maski,
    maara,
    minimietaisyys
):
    """
    Valitsee ylimmän tason keskukset.
    """

    yy, xx = np.where(
        maa_maski
    )

    if len(yy) == 0:
        return []

    arvot = pisteet[
        yy,
        xx
    ]

    jarjestys = np.argsort(
        arvot
    )[::-1]

    valitut = []

    for idx in jarjestys:

        y = int(
            yy[idx]
        )

        x = int(
            xx[idx]
        )

        if not np.isfinite(
            arvot[idx]
        ):
            continue

        liian_lahella = False

        for cy, cx in valitut:

            d = np.sqrt(
                (y - cy) ** 2
                +
                (x - cx) ** 2
            )

            if d < minimietaisyys:

                liian_lahella = True
                break

        if liian_lahella:
            continue

        valitut.append(
            (y, x)
        )

        if len(valitut) >= maara:
            break

    return valitut


# ============================================================
# HIERARKIAN KESKUS
# ============================================================

def tee_keskus(
    id_,
    taso,
    y,
    x,
    parent_id=None
):
    """
    Yksittäinen hierarkkinen keskus.
    """

    return {
        "id": int(id_),
        "taso": int(taso),
        "y": int(y),
        "x": int(x),
        "parent_id": (
            None
            if parent_id is None
            else int(parent_id)
        ),
        "children": [],
        "lon": None,
        "lat": None
    }


# ============================================================
# HIERARKIAN RAKENTAMINEN
# ============================================================

def rakenna_kaupunkihierarkia(
    graph,
    pisteet,
    root_centers,
    shape,
    tasoja=2,
    lapsia_per_keskus=6,
    minimietaisyydet=None,
    parent_area_fraction=0.70,
    sektorin_toleranssi=np.deg2rad(35.0)
):
    """
    Rakentaa sisäkkäisen kaupunkihierarkian.

    tasoja=0:
        vain pääkaupungit

    tasoja=1:
        pääkaupungit
        + 6 lasta / pääkaupunki

    tasoja=2:
        pääkaupungit
        + lapset
        + lasten lapset

    jne.

    parent_area_fraction:
        kuinka suuri osa emokeskuksen kustannusalueesta
        sallitaan lasten etsintäalueeksi.
    """

    h, w = shape

    if minimietaisyydet is None:

        minimietaisyydet = [
            30,
            15,
            7,
            4,
            2
        ]

    # --------------------------------------------------------
    # Keskukset dictionaryyn
    # --------------------------------------------------------

    kaupungit = {}

    seuraava_id = 0

    nykyinen_taso = []

    # ========================================================
    # TASO 0
    # ========================================================

    for y, x in root_centers:

        kaupunki = tee_keskus(
            seuraava_id,
            0,
            y,
            x,
            None
        )

        kaupungit[
            seuraava_id
        ] = kaupunki

        nykyinen_taso.append(
            seuraava_id
        )

        seuraava_id += 1

    # ========================================================
    # TASOT
    # ========================================================

    for taso in range(
        1,
        tasoja + 1
    ):

        if len(nykyinen_taso) == 0:
            break

        # ----------------------------------------------------
        # Edellisen tason keskukset
        # ----------------------------------------------------

        parent_centers = [
            (
                kaupungit[cid]["y"],
                kaupungit[cid]["x"]
            )
            for cid in nykyinen_taso
        ]

        # ----------------------------------------------------
        # Edellisen tason Voronoi
        # ----------------------------------------------------

        alueet, etaisyydet = (
            laske_kustannusalueet(
                graph,
                parent_centers,
                shape
            )
        )

        uudet = []

        # ----------------------------------------------------
        # Jokaiselle emolle omat lapset
        # ----------------------------------------------------

        for parent_idx, parent_id in enumerate(
            nykyinen_taso
        ):

            parent = kaupungit[
                parent_id
            ]

            py = parent["y"]
            px = parent["x"]

            # ------------------------------------------------
            # Emon alue
            # ------------------------------------------------

            oma_alue = (
                alueet ==
                parent_idx
            )

            if not np.any(
                oma_alue
            ):
                continue

            # ------------------------------------------------
            # Kustannusetäisyys emosta
            # ------------------------------------------------

            parent_node = pixel_idx(
                py,
                px,
                w
            )

            dist = dijkstra(
                csgraph=graph,
                directed=True,
                indices=parent_node
            )

            dist = dist.reshape(
                h,
                w
            )

            kelvolliset = (
                oma_alue &
                np.isfinite(dist)
            )

            if not np.any(
                kelvolliset
            ):
                continue

            # ------------------------------------------------
            # Käytetään emon alueen sisempää osaa
            #
            # Tämä estää lapsia kasaantumasta aivan
            # alueiden reunoille.
            # ------------------------------------------------

            raja = np.percentile(
                dist[kelvolliset],
                90
            )

            max_dist = (
                raja *
                parent_area_fraction
            )

            # ------------------------------------------------
            # Minimietäisyys tällä tasolla
            # ------------------------------------------------

            if (
                taso - 1
                <
                len(minimietaisyydet)
            ):

                min_dist = (
                    minimietaisyydet[
                        taso - 1
                    ]
                )

            else:

                min_dist = max(
                    2,
                    minimietaisyydet[-1]
                    /
                    (2 ** (
                        taso -
                        len(minimietaisyydet)
                    ))
                )

            # ------------------------------------------------
            # Ehdokasalue
            # ------------------------------------------------

            ehdokasalue = (
                oma_alue &
                np.isfinite(dist) &
                (dist <= max_dist)
            )

            # ------------------------------------------------
            # Valitaan kuusi suunnallisesti hajautettua lasta
            # ------------------------------------------------

            lapset = (
                valitse_sektorikeskukset(
                    pisteet,
                    ehdokasalue,
                    (py, px),
                    h,
                    w,
                    lapsia=
                        lapsia_per_keskus,
                    minimietaisyys=
                        min_dist,
                    sektorin_toleranssi=
                        sektorin_toleranssi
                )
            )

            # ------------------------------------------------
            # Rekisteröidään lapset
            # ------------------------------------------------

            for cy, cx in lapset:

                child = tee_keskus(
                    seuraava_id,
                    taso,
                    cy,
                    cx,
                    parent_id
                )

                kaupungit[
                    seuraava_id
                ] = child

                parent[
                    "children"
                ].append(
                    seuraava_id
                )

                uudet.append(
                    seuraava_id
                )

                seuraava_id += 1

        nykyinen_taso = uudet

    # ========================================================
    # WGS84-KOORDINAATIT
    # ========================================================

    for city in kaupungit.values():

        lon, lat = pixel_to_coord(
            city["y"],
            city["x"],
            h,
            w
        )

        city["lon"] = lon
        city["lat"] = lat

    for city in kaupungit.values():
        city["population"]=-999
        city["power"]=-999
    return kaupungit


# ============================================================
# HIERARKIAN ALUEET
# ============================================================



def laske_kaikkien_tasojen_alueet(
    graph,
    kaupungit,
    vaestot,
    shape,
    kaupungistumisprosentti=0.50
):
    """
    Laskee Voronoi-alueet jokaiselle hierarkiatasolle
    sekä alueiden ja kaupunkien väestöt.

    kaupungin väestö =
        alueen väestö * kaupungistumisprosentti
    """

    h, w = shape

    vaestot = np.asarray(
        vaestot,
        dtype=float
    )

    if vaestot.shape != (h, w):
        raise ValueError(
            "vaestot-rasterin koko ei vastaa shapea."
        )

    if not 0.0 <= kaupungistumisprosentti <= 1.0:
        raise ValueError(
            "kaupungistumisprosentin pitää olla "
            "välillä 0.0 ... 1.0."
        )

    tulos = {}

    tasot = sorted(
        set(
            city["taso"]
            for city in kaupungit.values()
        )
    )

    for taso in tasot:

        city_ids = [
            cid
            for cid, city in kaupungit.items()
            if city["taso"] == taso
        ]

        keskukset = [
            (
                kaupungit[cid]["y"],
                kaupungit[cid]["x"]
            )
            for cid in city_ids
        ]

        # ----------------------------------------------------
        # Voronoi
        # ----------------------------------------------------

        alueet, etaisyydet = (
            laske_kustannusalueet(
                graph,
                keskukset,
                shape
            )
        )

        # ----------------------------------------------------
        # Muutetaan Voronoi-indeksit city_id:ksi
        # ----------------------------------------------------

        alueet_id = np.full(
            (h, w),
            -1,
            dtype=int
        )

        valid = np.isfinite(
            alueet
        )

        if (
            len(city_ids) > 0
            and np.any(valid)
        ):

            alueet_int = (
                alueet[valid]
                .astype(int)
            )

            alueet_id[valid] = (
                np.asarray(city_ids)[
                    alueet_int
                ]
            )

        # ----------------------------------------------------
        # Lasketaan jokaisen alueen väestö
        # ----------------------------------------------------

        alueiden_vaesto = {}

        for city_id in city_ids:

            maski = (
                alueet_id == city_id
            )

            kelvollinen = (
                maski
                &
                np.isfinite(vaestot)
            )

            alueen_vaesto = float(
                np.sum(
                    vaestot[
                        kelvollinen
                    ]
                )
            )

            alueiden_vaesto[
                city_id
            ] = alueen_vaesto

            # ------------------------------------------------
            # Kaupungin väestö
            # ------------------------------------------------

            kaupungin_vaesto = (
                alueen_vaesto
                *
                kaupungistumisprosentti
            )

            kaupungit[
                city_id
            ]["area_population"] = (
                alueen_vaesto
            )

            kaupungit[
                city_id
            ]["population"] = (
                kaupungin_vaesto
            )

            kaupungit[
                city_id
            ]["urbanization"] = (
                kaupungistumisprosentti
            )

        tulos[taso] = {
            "alueet": alueet_id,
            "etaisyydet": etaisyydet,
            "city_ids": city_ids,
            "alueiden_vaesto": alueiden_vaesto
        }

    return tulos



# ============================================================
# REITIN PURKU
# ============================================================

def pura_reitti(
    predecessors,
    alku_idx,
    loppu_idx,
    w
):
    """
    Muuttaa Dijkstran predecessor-arrayn
    [(y,x), ...]-reitiksi.
    """

    if alku_idx == loppu_idx:

        return [
            (
                alku_idx // w,
                alku_idx % w
            )
        ]

    if predecessors[loppu_idx] < 0:
        return []

    polku = []

    nykyinen = loppu_idx

    while nykyinen != alku_idx:

        y = nykyinen // w
        x = nykyinen % w

        polku.append(
            (y, x)
        )

        nykyinen = (
            predecessors[nykyinen]
        )

        if nykyinen < 0:
            return []

    polku.append(
        (
            alku_idx // w,
            alku_idx % w
        )
    )

    return polku[::-1]


# ============================================================
# KAHDEN KESKUKSEN REITTI
# ============================================================

def etsi_reitti_verkosta(
    graph,
    alku_yx,
    loppu_yx,
    shape
):
    """
    Etsii yhden reitin.
    """

    h, w = shape

    ay, ax = alku_yx
    ly, lx = loppu_yx

    alku_idx = pixel_idx(
        ay,
        ax,
        w
    )

    loppu_idx = pixel_idx(
        ly,
        lx,
        w
    )

    dist, predecessors = (
        dijkstra(
            csgraph=graph,
            directed=True,
            indices=alku_idx,
            return_predecessors=True
        )
    )

    if not np.isfinite(
        dist[loppu_idx]
    ):
        return [], np.inf

    polku = pura_reitti(
        predecessors,
        alku_idx,
        loppu_idx,
        w
    )

    return (
        polku,
        float(dist[loppu_idx])
    )


# ============================================================
# REITIT KAUPUNKIEN VÄLILLÄ
# ============================================================

def laske_hierarkian_reitit(
    graph,
    kaupungit,
    shape,
    yhdista_saman_tason_kaupunkeja=False
):
    """
    Luo kaksi erilaista reittitasoa:

    1. lapsi -> emo
    2. haluttaessa saman tason kaikki parit

    Oletuksena yhdistetään vain hierarkkisesti
    lapsi -> emo.

    Tämä pitää verkon realistisemman kokoisena.
    """

    reitit = []

    # ========================================================
    # LAPSI -> EMO
    # ========================================================

    for city_id, city in kaupungit.items():

        parent_id = city[
            "parent_id"
        ]

        if parent_id is None:
            continue

        parent = kaupungit[
            parent_id
        ]

        polku, kustannus = (
            etsi_reitti_verkosta(
                graph,
                (
                    city["y"],
                    city["x"]
                ),
                (
                    parent["y"],
                    parent["x"]
                ),
                shape
            )
        )

        if not polku:
            continue

        reitit.append({
            "tyyppi": "hierarkkinen",
            "taso": city["taso"],
            "kaupunki_A": parent_id,
            "kaupunki_B": city_id,
            "parent_id": parent_id,
            "child_id": city_id,
            "reitti": polku,
            "kustannus": kustannus
        })

    # ========================================================
    # SAMAN TASOISTEN KAUPUNKIEN REITIT
    # ========================================================

    if yhdista_saman_tason_kaupunkeja:

        tasot = sorted(
            set(
                city["taso"]
                for city in kaupungit.values()
            )
        )

        for taso in tasot:

            ids = [
                cid
                for cid, city
                in kaupungit.items()
                if city["taso"] == taso
            ]

            for i in range(
                len(ids)
            ):

                a_id = ids[i]

                a = kaupungit[
                    a_id
                ]

                # ------------------------------------------------
                # Etsitään lähimmät saman tason naapurit.
                #
                # Ei yhdistetä kaikkia kaikkiin.
                # ------------------------------------------------

                naapurit = []

                for j in range(
                    len(ids)
                ):

                    if i == j:
                        continue

                    b_id = ids[j]

                    b = kaupungit[
                        b_id
                    ]

                    d = np.sqrt(
                        (
                            a["y"] -
                            b["y"]
                        ) ** 2
                        +
                        (
                            a["x"] -
                            b["x"]
                        ) ** 2
                    )

                    naapurit.append(
                        (
                            d,
                            b_id
                        )
                    )

                naapurit.sort(
                    key=lambda z: z[0]
                )

                # Vain 2 lähintä.
                #
                # Tällä vältetään täydellinen
                # N²-verkko.
                for _, b_id in naapurit[:2]:

                    # Estetään duplikaatit
                    if a_id > b_id:
                        continue

                    b = kaupungit[
                        b_id
                    ]

                    polku, kustannus = (
                        etsi_reitti_verkosta(
                            graph,
                            (
                                a["y"],
                                a["x"]
                            ),
                            (
                                b["y"],
                                b["x"]
                            ),
                            shape
                        )
                    )

                    if not polku:
                        continue

                    reitit.append({
                        "tyyppi":
                            "saman_tason",
                        "taso": taso,
                        "kaupunki_A":
                            a_id,
                        "kaupunki_B":
                            b_id,
                        "reitti":
                            polku,
                        "kustannus":
                            kustannus
                    })

    return reitit


# ============================================================
# PÄÄFUNKTION
# ============================================================

def rakenna_hierarkkinen_kaupunkiverkko(
    korkeuskartta,
    annual_npp,
    vaestot,
    rivers2,
    lakes2,
    meri_maski,

    # --------------------------------------------------------
    # Pääkaupunkien määrä
    # --------------------------------------------------------

    paakaupunkien_maara=5,

    # --------------------------------------------------------
    # Kaupungistumisaste
    #
    # 0.50 = 50 % vaikutusalueen väestöstä
    # kuuluu kaupunkiin.
    # --------------------------------------------------------

    kaupungistumisprosentti=0.50,

    # --------------------------------------------------------
    # Kuinka monta alatasoa luodaan?
    #
    # 0 = vain pääkaupungit
    # 1 = pääkaupungit + alikaupungit
    # 2 = pääkaupungit + alikaupungit + alialueet
    # --------------------------------------------------------

    tasoja=2,

    # --------------------------------------------------------
    # Kuinka monta lasta jokaisella keskuksella?
    # --------------------------------------------------------

    lapsia_per_keskus=6,

    # --------------------------------------------------------
    # Pääkaupunkien välinen minimietäisyys
    # --------------------------------------------------------

    root_minimietaisyys=30,

    # --------------------------------------------------------
    # Eri tasojen minimietäisyydet
    # --------------------------------------------------------

    minimietaisyydet=None,

    # --------------------------------------------------------
    # Kuinka syvälle emon alueelle lapsia sijoitetaan?
    # --------------------------------------------------------

    parent_area_fraction=0.70,

    # --------------------------------------------------------
    # Sektorin sallittu poikkeama
    # --------------------------------------------------------

    sektorin_toleranssi=np.deg2rad(35.0),

    # --------------------------------------------------------
    # Reititys
    # --------------------------------------------------------

    reittityyppi="maa",
    maa_kerroin=1.0,
    meri_kerroin=1.0,

    # --------------------------------------------------------
    # Yhdistetäänkö myös saman tason kaupunkeja?
    # --------------------------------------------------------

    yhdista_saman_tason_kaupunkeja=False
):
    """
    Rakentaa koko hierarkkisen kaupunkijärjestelmän.

    Esimerkiksi:

        paakaupunkien_maara = 5
        tasoja = 2
        lapsia_per_keskus = 6

    antaa teoriassa:

        Taso 0 = 5
        Taso 1 = 30
        Taso 2 = 180

        Yhteensä = 215

    Käytännössä jotkin lapset voivat jäädä syntymättä,
    jos emon alueella ei ole riittävästi kelvollista maata.

    Väestö:

        area_population
            = kaupungin vaikutusalueen kokonaisväestö

        population
            = area_population * kaupungistumisprosentti

    Esimerkiksi:

        kaupungistumisprosentti = 0.50

        alueen väestö = 100 000

        kaupungin väestö = 50 000
    """

    h, w = korkeuskartta.shape

    # ========================================================
    # PARAMETRIEN OLETUSARVOT
    # ========================================================

    if minimietaisyydet is None:

        minimietaisyydet = [
            30,
            15,
            7,
            4
        ]

    # --------------------------------------------------------
    # Tarkistetaan kaupungistumisprosentti
    # --------------------------------------------------------

    if not 0.0 <= kaupungistumisprosentti <= 1.0:

        raise ValueError(
            "kaupungistumisprosentin pitää olla "
            "välillä 0.0 ... 1.0"
        )

    # ========================================================
    # 1. KAUPUNKIPISTEET
    # ========================================================

    pisteet = tee_kaupunkipisteet(
        korkeuskartta,
        annual_npp,
        rivers2,
        lakes2,
        meri_maski
    )

    # ========================================================
    # 2. MAA
    # ========================================================

    maa_maski = (
        ~np.asarray(
            meri_maski
        ).astype(bool)

        &

        ~np.asarray(
            lakes2
        ).astype(bool)

        &

        ~np.asarray(
            rivers2
        ).astype(bool)
    )

    # ========================================================
    # 3. PÄÄKAUPUNGIT
    # ========================================================

    root_centers = (
        valitse_paakeskukset(
            pisteet,
            maa_maski,
            paakaupunkien_maara,
            root_minimietaisyys
        )
    )

    # ========================================================
    # 4. KUSTANNUS
    # ========================================================

    kustannus = tee_kustannusmatriisi(
        korkeuskartta,
        lakes2,
        meri_maski,
        reittityyppi=reittityyppi,
        maa_kerroin=maa_kerroin,
        meri_kerroin=meri_kerroin
    )

    # ========================================================
    # 5. YKSI YHTEINEN VERKKO
    # ========================================================

    graph = rakenna_kustannusverkko(
        kustannus,
        korkeuskartta
    )

    # ========================================================
    # 6. HIERARKIA
    # ========================================================

    kaupungit = (
        rakenna_kaupunkihierarkia(
            graph,
            pisteet,
            root_centers,
            (h, w),

            tasoja=tasoja,

            lapsia_per_keskus=
                lapsia_per_keskus,

            minimietaisyydet=
                minimietaisyydet,

            parent_area_fraction=
                parent_area_fraction,

            sektorin_toleranssi=
                sektorin_toleranssi
        )
    )

    # ========================================================
    # 7. VAIKUTUSALUEET + VÄESTÖ
    # ========================================================

    alueet = (
        laske_kaikkien_tasojen_alueet(
            graph,
            kaupungit,
            vaestot,
            (h, w),

            kaupungistumisprosentti=
                kaupungistumisprosentti
        )
    )

    # ========================================================
    # 8. REITIT
    # ========================================================

    reitit = (
        laske_hierarkian_reitit(
            graph,
            kaupungit,
            (h, w),

            yhdista_saman_tason_kaupunkeja=
                yhdista_saman_tason_kaupunkeja
        )
    )

    # ========================================================
    # 9. TASOJEN TILASTOT
    # ========================================================

    tasot = {}

    for city in kaupungit.values():

        taso = city["taso"]

        if taso not in tasot:

            tasot[taso] = []

        tasot[taso].append(
            city["id"]
        )

    # ========================================================
    # 10. PALAUTUS
    # ========================================================

    return {
        "kaupungit": kaupungit,
        "tasot": tasot,
        "alueet": alueet,
        "reitit": reitit,
        "graph": graph,
        "kustannus": kustannus,
        "pisteet": pisteet,
        "root_centers": root_centers,

        # Säilytetään myös käytetty arvo
        # tuloksessa helposti saatavilla.
        "kaupungistumisprosentti":
            kaupungistumisprosentti
    }


# ============================================================
# ESIMERKKI
# ============================================================

# tulos = rakenna_hierarkkinen_kaupunkiverkko(
#
#     korkeuskartta=korkeuskartta,
#     annual_npp=annual_npp,
#     rivers2=rivers2,
#     lakes2=lakes2,
#     meri_maski=meri_maski,
#
#     paakaupunkien_maara=5,
#     tasoja=2,
#     lapsia_per_keskus=6,
#
#     reittityyppi="maa",
#
#     yhdista_saman_tason_kaupunkeja=False
# )



# ============================================================
# HIERARKIAN TULOSTUS
# ============================================================

def tulosta_hierarkia(
    kaupungit
):
    """
    Tulostaa kaupungit hierarkkisena puuna.
    """

    roots = [
        city
        for city in kaupungit.values()
        if city["parent_id"] is None
    ]

    def tulosta(city_id, syvyys=0):

        city = kaupungit[
            city_id
        ]

        indent = "    " * syvyys

        kaupungin_vaesto = city.get(
            "population",
            0
        )

        alueen_vaesto = city.get(
            "area_population",
            0
        )

        kaupungistuminen = city.get(
            "urbanization",
            0
        )

        print(
            f"{indent}"
            f"[T{city['taso']}] "
            f"id={city['id']} | "
            f"lat={city['lat']:.2f} | "
            f"lon={city['lon']:.2f} | "
            f"kaupunki väki={kaupungin_vaesto:,.0f} | "
            f"alue väki={alueen_vaesto:,.0f} | "
            f"urban={kaupungistuminen:.0%} | "
            f"lapset={len(city['children'])}"
        )

        for child_id in city[
            "children"
        ]:

            tulosta(
                child_id,
                syvyys + 1
            )

    for root in roots:

        tulosta(
            root["id"]
        )


def tulosta_hierarkia_000(
    kaupungit
):
    """
    Tulostaa kaupungit hierarkkisena puuna.

    Näyttää jokaisesta keskuksesta:

        - hierarkiatason
        - id:n
        - koordinaatit
        - kaupungin väkiluvun
        - vaikutusalueen väkiluvun
        - lasten määrän
    """

    roots = [
        city
        for city in kaupungit.values()
        if city["parent_id"] is None
    ]

    def tulosta(city_id, syvyys=0):

        city = kaupungit[
            city_id
        ]

        indent = "    " * syvyys

        # ----------------------------------------------------
        # Väkiluvut
        # ----------------------------------------------------

        kaupungin_vaesto = city.get(
            "population",
            -1
        )

        alueen_vaesto = city.get(
            "area_population",
            -1
        )

        # ----------------------------------------------------
        # Tulostus
        # ----------------------------------------------------

        print(
            f"{indent}"
            f"[T{city['taso']}] "
            f"id={city['id']} "
            f"lat={city['lat']:.2f} "
            f"lon={city['lon']:.2f} "
            f"kaupungin väki={kaupungin_vaesto:,.0f} "
            f"alueen väki={alueen_vaesto:,.0f} "
            f"lapset={len(city['children'])}"
        )

        # ----------------------------------------------------
        # Lapset
        # ----------------------------------------------------

        for child_id in city[
            "children"
        ]:

            tulosta(
                child_id,
                syvyys + 1
            )

    # ========================================================
    # JUURET
    # ========================================================

    for root in roots:

        tulosta(
            root["id"]
        )




def muodosta_hierarkia(
    graph,
    kaupungit_yx,
    shape,
    lapsia_per_solmu=6
):
    """
    Muodostaa kahden tason hierarkian.

    Taso 0:
        nykyiset pääkaupungit

    Taso 1:
        jokaiselle pääkaupungille
        enintään 6 alikaupunkia.

    Palauttaa:
        paalueet
        alialueet
        kaupungit
    """

    h, w = shape

    # ======================================================
    # TASON 0 ALUEET
    # ======================================================

    paakaupungit = list(
        kaupungit_yx
    )

    paakaupunkien_idx = np.array(
        [
            y * w + x
            for y, x in paakaupungit
        ],
        dtype=np.int64
    )

    # Kaikkien pääkaupunkien kustannusetäisyys
    dist = dijkstra(
        graph,
        directed=True,
        indices=paakaupunkien_idx
    )

    paalueet = np.argmin(
        dist,
        axis=0
    ).reshape(h, w)

    kustannus = np.min(
        dist,
        axis=0
    ).reshape(h, w)

    paalueet[
        ~np.isfinite(kustannus)
    ] = -1

    # ======================================================
    # TASON 1 ALIALUEET
    # ======================================================

    alialueet = np.full(
        (h, w),
        -1,
        dtype=int
    )

    kaikki_kaupungit = []

    # Pääkaupungit
    for i, (y, x) in enumerate(
        paakaupungit
    ):

        kaikki_kaupungit.append({
            "id": len(kaikki_kaupungit),
            "taso": 0,
            "parent": None,
            "y": int(y),
            "x": int(x),
            "nimi": f"Pääkaupunki {i}"
        })

    # ------------------------------------------------------
    # Käydään jokainen pääalue läpi
    # ------------------------------------------------------

    seuraava_alialue_id = 0

    for paalue_id in range(
        len(paakaupungit)
    ):

        paakaupunki_yx = (
            paakaupungit[paalue_id]
        )

        # Alueen pikselit
        alue_maski = (
            paalueet == paalue_id
        )

        alue_idx = np.flatnonzero(
            alue_maski.ravel()
        )

        if len(alue_idx) == 0:
            continue

        # ==================================================
        # Etsitään alueen sisällä
        # lapsikeskukset
        # ==================================================

        lapsikeskukset = []

        # Ensimmäinen keskus ei ole aivan
        # pääkaupunki itse vaan haetaan
        # etäisyyden perusteella hajautettuja pisteitä.

        # Dijkstra pääkaupungista
        # kertoo etäisyyden pääkaupunkiin.

        parent_idx = (
            paakaupunki_yx[0] * w
            + paakaupunki_yx[1]
        )

        d_parent = dijkstra(
            graph,
            directed=True,
            indices=parent_idx
        )

        d_parent = d_parent[
            alue_idx
        ]

        # Alialueiden keskukset valitaan
        # mahdollisimman kauas toisistaan
        # greedy farthest-point -menetelmällä.

        valittavissa = alue_idx[
            np.isfinite(d_parent)
        ]

        if len(valittavissa) == 0:
            continue

        # Ensimmäinen piste:
        # melko kaukana pääkaupungista
        ensimmäinen = valittavissa[
            np.argmax(
                d_parent[
                    np.isin(
                        alue_idx,
                        valittavissa
                    )
                ]
            )
        ]

        lapsikeskukset.append(
            ensimmäinen
        )

        # --------------------------------------------------
        # Seuraavat keskukset
        # --------------------------------------------------

        while (
            len(lapsikeskukset)
            < lapsia_per_solmu
        ):

            parhaat = None
            paras_min_dist = -np.inf

            for ehdokas in valittavissa:

                if ehdokas in lapsikeskukset:
                    continue

                ey = ehdokas // w
                ex = ehdokas % w

                # Etäisyys jokaiseen jo valittuun
                min_dist = np.inf

                for keskus in lapsikeskukset:

                    cy = keskus // w
                    cx = keskus % w

                    keskus_idx = (
                        cy * w + cx
                    )

                    d = dijkstra(
                        graph,
                        directed=True,
                        indices=keskus_idx
                    )[ehdokas]

                    min_dist = min(
                        min_dist,
                        d
                    )

                if min_dist > paras_min_dist:

                    paras_min_dist = min_dist
                    parhaat = ehdokas

            if parhaat is None:
                break

            lapsikeskukset.append(
                parhaat
            )

        # ==================================================
        # Muodostetaan tämän pääalueen
        # sisäinen Voronoi-jako
        # ==================================================

        if not lapsikeskukset:
            continue

        lapsi_idx = np.array(
            lapsikeskukset,
            dtype=np.int64
        )

        lapsi_dist = dijkstra(
            graph,
            directed=True,
            indices=lapsi_idx
        )

        # Vain tämän pääalueen pikselit
        alue_dist = lapsi_dist[
            :,
            alue_idx
        ]

        lähin = np.argmin(
            alue_dist,
            axis=0
        )

        for local_id, global_pixel in zip(
            lähin,
            alue_idx
        ):

            alialueet.ravel()[
                global_pixel
            ] = (
                seuraava_alialue_id
                + local_id
            )

        # ==================================================
        # Lisätään alikaupungit
        # ==================================================

        for local_id, idx in enumerate(
            lapsikeskukset
        ):

            y = idx // w
            x = idx % w

            kaikki_kaupungit.append({
                "id": len(kaikki_kaupungit),
                "taso": 1,
                "parent": paalue_id,
                "y": int(y),
                "x": int(x),
                "nimi":
                    f"Kaupunki "
                    f"{paalue_id}.{local_id}"
            })

        seuraava_alialue_id += len(
            lapsikeskukset
        )

    return (
        paalueet,
        alialueet,
        kaikki_kaupungit
    )

def piirrä_kaupunki_reitti_alue_kartta(fig, ax, 
    karttapohja,
    piirretaan=["kaupungit", "reitit", "alueet"],
    tasolle_asti=None,
    figsize=(10, 5),
    hillshade=True,
):
    """
    Piirtää kaupungit, reitit ja/tai alueet karttapohjan päälle.

    Parametrit
    ----------
    karttapohja : 2D numpy array
        Esimerkiksi korkeuskartta.

    piirretaan : list[str]
        Mitä piirretään:
        - "kaupungit"
        - "reitit"
        - "alueet"

    tasolle_asti : int tai None
        Piirretään hierarkiatasot 0 ... tasolle_asti.
        None = kaikki tasot.

    figsize : tuple
        Kuvan koko.

    hillshade : bool
        Käytetäänkö korkeuskartan hillshade-varjostusta.
    """

    # --------------------------------------------------
    # Asetukset
    # --------------------------------------------------

    symbolit = {
        0: "s",      # ympyrä
        1: "o",      # neliö
        2: "^",      # kolmio
        3: "D",      # vinoneliö
    }

    vari = {
        0: "red",
        1: "orange",
        2: "yellow",
        3: "cyan",
    }

    koko = [6, 4, 2, 1, 0.8]

    reitti_varit = {
        0: "red",
        1: "orange",
        2: "yellow",
        3: "cyan",
    }

    reitti_leveydet = {
        0: 4,
        1: 2.5,
        2: 1.5,
        3: 1,
    }

    tasojen_varit = {
        0: "red",
        1: "blue",
        2: "lime",
    }

    # --------------------------------------------------
    # Tarkistukset
    # --------------------------------------------------

    sallitut = {"kaupungit", "reitit", "alueet"}

    ylimaaraiset = set(piirretaan) - sallitut

    if ylimaaraiset:
        raise ValueError(
            f"Tuntemattomat piirrettävät: {ylimaaraiset}. "
            f"Sallitut: {sallitut}"
        )

    # Jos tasoa ei annettu -> kaikki
    if tasolle_asti is None:
        tasolle_asti = max(tasot.keys())

    # --------------------------------------------------
    # Karttapohja
    # --------------------------------------------------

    #plt.figure(figsize=figsize)

    #if hillshade:
    #    varjo_maski = laske_hillshade(
    #        karttapohja,
    #        azimuth=315,
    #        altitude=45
    #    )

    #    ax.imshow(
    #        karttapohja,
    #        alpha=varjo_maski,
    #        cmap="terrain"
    #    )
    #else:
    #    ax.imshow(
    #        karttapohja,
    #        cmap="terrain"
    #    )

    # --------------------------------------------------
    # Alueet
    # --------------------------------------------------

    if "alueet" in piirretaan:

        for taso in alueet:

            if taso > tasolle_asti:
                continue

            A = alueet[taso]["alueet"]

            mask = ~np.isnan(A)

            if not np.any(mask):
                continue

            # Alueiden rajat
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors=tasojen_varit.get(taso, "white"),
                linewidths=2,
                zorder=3
            )

    # --------------------------------------------------
    # Reitit
    # --------------------------------------------------

    # --------------------------------------------------
    # Reitit
    # --------------------------------------------------

    if "reitit" in piirretaan:

        # Alempi taso ensin, ylempi taso viimeisenä
        jarjestetyt_reitit = sorted(
            reitit,
            key=lambda reitti: reitti["taso"]
        )

        for reitti in jarjestetyt_reitit:

            taso = reitti["taso"]

            if taso > tasolle_asti:
                continue

            pisteet = reitti["reitti"]

            y = [p[0] for p in pisteet]
            x = [p[1] for p in pisteet]
            plot_wrapped_line(ax, x,y,  color=reitti_varit.get(taso, "black"),
                lw=reitti_leveydet.get(taso, 1),
                zorder=11 - taso )
            #ax.scatter(
            #    x,
            #    y,
            #    color=reitti_varit.get(taso, "black"),
            #    s=reitti_leveydet.get(taso, 1),
            #    zorder=5 + taso
            #)


    # --------------------------------------------------
    # Kaupungit
    # --------------------------------------------------

    if "kaupungit" in piirretaan:

        for taso, ids in tasot.items():

            if taso > tasolle_asti:
                continue

            for idx in ids:

                kaupunki = kaupungit[idx]

                ax.plot(
                    kaupunki["x"],
                    kaupunki["y"],
                    marker=symbolit.get(taso, "o"),
                    color=vari.get(taso, "white"),
                    markersize=koko[taso],
                    linestyle="None",
                    zorder=10
                )

    # --------------------------------------------------
    # Ulkoasu
    # --------------------------------------------------

    ax.set_xlim(0, karttapohja.shape[1])
    ax.set_ylim(karttapohja.shape[0], 0)

    #plt.tight_layout()
    #ax.show()
    return(fig, ax)





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
        #syvin_kohta=syvin_kohta,
        #korkein_kohta=korkein_kohta,
        leveys=leveys,
        korkeus=korkeus,

        octaves=16,

        oceanfrac=oceanfrac,
    )
)


maa_maski0=np.where(korkeuskartta>0,1,0)


meri_maski=np.where(np.copy(maa_maski)==1,0,1)

maa_maski_gpu = cp.asarray(maa_maski, dtype=cp.int32)


#plt.imshow(rivers)

#plt.imshow(accumulation)
#plt.show()



#quit(-1)

	


print("Sea level:", merenpinta)

print(
    "Sea:",
    np.count_nonzero(
        ~maa_maski
    ) / maa_maski.size
)

print(
    "Land:",
    np.count_nonzero(
        maa_maski
    ) / maa_maski.size
)

#plt.imshow(korkeuskartta,cmap="terrain")

#plt.show()


#quit(-1)

korkeus_syvyyskartta2, korkeuskartta2, maa_maski2, referenssi2, merenpinta2 = (
    mapper.generoi_kartta(
        alue=(0, 90, 0, 90),

        leveys=100,
        korkeus=100,

        referenssi=referenssi,
    )
)

#korkeuskartta=apply_dem_erosion(korkeuskartta, cell_size=50.0, iterations=50, erosion_rate=0.05, 
#                      peneplain_threshold=1000.0, protected_threshold=-0.0, smoothing_sigma=1.0)
#korkeus_syvyyskartta=apply_dem_erosion(korkeus_syvyyskartta, cell_size=50, iterations=50, erosion_rate=0.05, 
#                      peneplain_threshold=1000.0, protected_threshold=-0.0, smoothing_sigma=1.0)
                      						  


## dem props

n_lats, n_lons = korkeuskartta.shape
korkeus, leveys = maa_maski.shape
lats = np.linspace(90, -90, korkeus)   # Y-akseli (pohjoisesta etelään)
lons = np.linspace(-180, 180, leveys) # X-akseli (lännestä itään)
lon_grid_deg, lat_grid_deg = np.meshgrid(lons, lats)

#lats = np.linspace(-90, 90, n_lats)
cos_weights = np.cos(np.radians(lats))
weight_grid = cos_weights[:, np.newaxis] * np.ones(n_lons)

maa_painotettu = np.sum(weight_grid[maa_maski])
meri_painotettu = np.sum(weight_grid[meri_maski])
kokonais_paino = np.sum(weight_grid)
maa_prosentti = (maa_painotettu / kokonais_paino) * 100
meri_prosentti = (meri_painotettu / kokonais_paino) * 10
maan_keskikorkeus = np.sum(korkeuskartta[maa_maski] * weight_grid[maa_maski]) /maa_painotettu



pikselien_pinta_alat_re2=laske_pikselien_pinta_alat_re2(leveys, korkeus, alue=[-180, 180, -90, 90], planeetan_sade=1)




landfrac=maa_prosentti/100
oceanfrac=meri_prosentti/100
print("Original sealevel:", merenpinta)
print("Zoom sealevel:", merenpinta2)
print("Is equal ?:", merenpinta == merenpinta2)
print(f"Land : {maa_prosentti:.2f} %")
print(f"Ocean: {meri_prosentti:.2f} %")
print(f"Mean land height: {maan_keskikorkeus:.1f} metriä")


#piirra_korkeuskartta( data=korkeuskartta, otsikko="Fractal planet alnd height m")
#piirra_korkeus_ja_syvyyskartta( data=korkeus_syvyyskartta, otsikko="Fractal planet")


#quit(-1)








# GPU-laskenta
etaisyys_meresta_gpu = (
    laske_etaisyys_rannikosta_gpu(
        maa_maski
    )
)

# GPU -> CPU / NumPy
etaisyys_meresta_km = etaisyys_meresta_gpu.get()




print("Distance to coast ..")

varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)


# =====================================================================
# 4. KUUKAUSITTAINEN LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS (12 KUUKAUTTA)
# =====================================================================
#kuukausi_lämpötilat = np.zeros((12, korkeus, leveys))
#kuukausi_sateet = np.zeros((12, korkeus, leveys))

planet_global_moisture_coeff=planet_global_moisture_coeff*f_precip_relto_earthlike(oceanfrac/0.71)
#planet_global_moisture_coeff=oceanfrac 
 
 
print("Global moisture coeff ",planet_global_moisture_coeff ) 
 


print ("Calculate climate ...")

# Ensimmäinen ajo = warm-up


## 0) Laske ilmasto

kuukausi_lämpötilat_gpu, kuukausi_sateet_gpu,kuukausi_npp_gpu, kuukausi_ptopet_gpu ,\
kuukausi_tuuli_suunta_gpu, kuukausi_tuuli_voima_gpu, \
kuukausi_merivirta_x_gpu, kuukausi_merivirta_y_gpu=laske_ilmasto_gpu(
    star_mass_msun=star_mass_msun,
    star_luminosity_lsun=star_luminosity_lsun,
    star_teff_k=star_teff_k,
    korkeus_syvyyskartta=korkeus_syvyyskartta,
    planet_mass_me=planet_mass_me,
    planet_radius_re=planet_radius_re,
    planet_tilt=planet_axis_tilt_degrees,
    planet_ecc=planet_ecc,
    planet_mvelp_deg=planet_mvelp_degrees,
    planet_orbital_period=planet_orbital_period,
    planet_rotation_hours=planet_rotation_period_hours,
    planet_atmosphere_pressure=planet_atmosphere_pressure, planet_co2_ppm=planet_co2_ppm, 
    landfrac=landfrac,
    planet_tmean=planet_tmean, ## 14.8
    global_moisture_coeff=planet_global_moisture_coeff,)



print("...")


kuukausi_lämpötilat=kuukausi_lämpötilat_gpu.get()
kuukausi_sateet=kuukausi_sateet_gpu.get()

kuukausi_tuuli_suunta=kuukausi_tuuli_suunta_gpu.get()
kuukausi_tuuli_voima=kuukausi_tuuli_voima_gpu.get()
kuukausi_merivirta_x=kuukausi_merivirta_x_gpu.get()
kuukausi_merivirta_y=kuukausi_merivirta_y_gpu.get()
kuukausi_merivirta=np.sqrt(kuukausi_merivirta_x*kuukausi_merivirta_x+kuukausi_merivirta_y*kuukausi_merivirta_y)

annual_ptopet=cp.sum(kuukausi_ptopet_gpu, axis=0).get()
annual_npp=cp.sum(kuukausi_npp_gpu, axis=0).get()

annual_npp=np.where(annual_npp<0,0, annual_npp)

#plt.imshow(annual_ptopet*maa_maski)
#plt.imshow(annual_npp*maa_maski)
#plt.imshow(pikselien_pinta_alat_re2)

 
   
jaa, merijaa_aina=laske_jaat(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuosia=200)  
    
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



#annual_npp_original_gpu=laske_npp_miami_kuukausittain(cp.asarray(tmean_annual),cp.asarray(precip_annual), vuoden_pituus_a=planet_orbital_period)

#annual_npp=annual_npp_original_gpu.get()

 


earth_npp_sum_raster=annual_npp*maa_maski*pikselien_pinta_alat_re2*6.371*6.371*1e6

human_coeff=0.01 # 0.1 ...0.01
absolute_population_max_raster=0.000005*earth_npp_sum_raster*1e6*human_coeff/1e6 ## million people!

total_world_population=np.sum(absolute_population_max_raster)

#plt.imshow( absolute_population_max_raster)

#plt.imshow(earth_npp_sum_raster)
#plt.imshow(annual_npp)
#plt.imshow(precip_annual)



npp_total_total=np.sum((earth_npp_sum_raster/1e18))


print(npp_total_total)
print(total_world_population)

#plt.show()


#quit(-1) 
    

#plot_grid_streamplot(kuukausi_merivirta_x[0], kuukausi_merivirta_y[0], alue=[-180, 180, -90, 90])


#plt.imshow(maa_maski)


#accumulation, rivers=calculate_rivers_cupy(korkeuskartta, iterations=1000, threshold=10)



#accumulation, rivers=calculate_climate_rivers(korkeuskartta, precip_annual, tmean_annual, iterations=100, threshold=15000)

#rivers, lakes=calculate_climate_rivers_and_lakes(korkeuskartta, 
#precip_annual, tmean_annual, iterations=500, river_threshold=5000, lake_threshold=10000)

rivers, lakes = calculate_climate_rivers_and_lakes( \
    korkeuskartta, precip_annual, tmean_annual, 
    iterations=50, \
    river_threshold=10e10, \
    lake_threshold=10e10 ,
    minimum_catchment_km2=1000.0 ,\
    planet_radius_re=1.0,\
    atmosphere_pressure_bar=1.0, \
    planet_gravity_earths=1.0, \
)


max1=np.max(rivers)
min1=np.min(rivers)
delt1=max1-min1
rivers=(rivers-min1)/delt1
#rivers=np.exp(rivers)/np.exp(1)
rivers=np.where(rivers>0.0,1,0)

max2=np.max(lakes)
min2=np.min(lakes)
delt2=max2-min2
lakes=(lakes-min1)/delt2
#lakes=np.exp(lakes)/np.exp(1)
lakes=np.where(lakes>0.0,1,0)

#plt.imshow(rivers)

#lt.imshow(lakes)

#plt.show()
#quit(-1)




rivers2=np.copy(rivers)
lakes2=np.copy(lakes)

rivers=np.where(rivers==0,np.nan, rivers)
lakes=np.where(lakes==0,np.nan, lakes)


rivers_distance=np.copy(rivers2)
rivers_distance=np.where(rivers_distance>0,0,1)
rivers_distance_gpu=laske_etaisyys_rannikosta_gpu(cp.asarray(rivers_distance))
lakes_distance=np.copy(lakes2)
lakes_distance=np.where(lakes_distance>1,0,1)
#plt.imshow(lakes_distance)
#lakes_distance=np.where(lakes_distance>0,0,1)

lakes_distance_gpu=laske_etaisyys_rannikosta_gpu(cp.asarray(lakes_distance))
rivers_distance_gpu=laske_etaisyys_rannikosta_gpu(cp.asarray(rivers_distance))

#etaisyys_meresta_km
#rannikko_etaisyys_gpu
# Muistiystävällinen tapa: lasketaan minimi kahdessa vaiheessa
#lyhin_etaisyys_veteen_gpu = cp.minimum(rivers_distance_gpu, lakes_distance_gpu)
#lyhin_etaisyys_veteen_gpu = cp.minimum(lyhin_etaisyys_veteen_gpu, etaisyys_meresta_gpu)
#lyhin_etaisyys_veteen_gpu = cp.minimum(rivers_distance_gpu, etaisyys_meresta_gpu)

#plt.imshow(lakes_distance_gpu.get())
#plt.imshow(rivers_distance_gpu.get())
#plt.imshow(lakes_distance_gpu.get())

## distance to 1000 m height mointains

etaisyys_1000m_vuoreen=np.copy(korkeuskartta)
etaisyys_1000m_vuoreen=np.where(etaisyys_1000m_vuoreen>1500,1,0)
etaisyys_1000m_vuoreen_gpu=laske_etaisyys_rannikosta_gpu(cp.asarray(etaisyys_1000m_vuoreen))

lahivuori=etaisyys_1000m_vuoreen_gpu.get()
lahivesi=etaisyys_meresta_gpu.get()*rivers_distance_gpu.get()

lahivuori=1/(lahivuori+1)
lahivesi=1/(lahivesi+1)



kuiva1=np.copy(precip_annual)
kuiva2=np.copy(kuiva1)

kuuma=np.copy(tmean_annual)

kuiva1=np.where(kuiva1<500,1,0)
kuiva2=np.where(kuiva2>80,1,0)
kuiva=kuiva1*kuiva2 
 
kuuma=np.where(kuuma>19,1,0)

sivilisaatio=lahivuori*lahivesi*kuiva*kuuma*maa_maski*rivers2

max2=np.max(sivilisaatio)
min2=np.min(sivilisaatio)
delt2=max2-min2
sivilisaatio=(sivilisaatio-min1)/delt2
#lakes=np.exp(lakes)/np.exp(1)
sivilisaatio=np.where(sivilisaatio>0.0,1,0)

sivilisaatio=np.where(sivilisaatio<0.1,np.nan, 1)


tulos = rakenna_hierarkkinen_kaupunkiverkko(
    korkeuskartta=korkeuskartta,
    annual_npp=annual_npp, vaestot=absolute_population_max_raster,
    rivers2=rivers2,
    lakes2=lakes2,
    meri_maski=meri_maski,

    paakaupunkien_maara=6,

    # 0 = vain pääkaupungit
    # 1 = pääkaupungit + alikaupungit
    # 2 = pääkaupungit + kaksi alempaa tasoa
    tasoja=2,

    lapsia_per_keskus=6,

    reittityyppi="maa",

    yhdista_saman_tason_kaupunkeja=True
)

# ============================================================

# tulos = rakenna_hierarkkinen_kaupunkiverkko(
#
#     korkeuskartta=korkeuskartta,
#     annual_npp=annual_npp,
#     rivers2=rivers2,
#     lakes2=lakes2,
#     meri_maski=meri_maski,
#
#     paakaupunkien_maara=5,
#     tasoja=2,
#     lapsia_per_keskus=6,
#
#     reittityyppi="maa",
#
#     yhdista_saman_tason_kaupunkeja=False
# )

#        "kaupungit": kaupungit,
#        "tasot": tasot,
#        "alueet": alueet,
#        "reitit": reitit,
#        "graph": graph,
#        "kustannus": kustannus,
#        "pisteet": pisteet,
#        "root_centers": root_centers

	
#print(tulos)



kaupungit = tulos["kaupungit"]
tasot = tulos["tasot"]
reitit = tulos["reitit"]
alueet = tulos["alueet"]


print(kaupungit)

tulosta_hierarkia(kaupungit)





fig, ax = piirra_fantasy_map(
    korkeuskartta,
    tmean_annual,
    precip_annual,
    rivers2,
    lakes2
)

fig, ax=piirrä_kaupunki_reitti_alue_kartta(fig, ax,
    karttapohja=np.where(korkeuskartta>0,1,0),
    piirretaan=["kaupungit", "reitit", "alueet"],
    tasolle_asti=None
)


plt.show()







tilasto_tulokset = laske_ilmastotilastot(
    kuukausi_lämpötilat,
    kuukausi_sateet
)





print(" Mean annual temp  Ta_mean :", round(tilasto_tulokset[ "painotettu_keskilampo"],2))
print(" Mean sum  precip  Pr_ann  :", round(tilasto_tulokset[ "keskiarvoinen_vuosisade"]))
print(" Min annual temp   T_min   :", round(tilasto_tulokset["min_lampo" ],2))
print(" Max annual temp   T_max   :", round(tilasto_tulokset[ "max_lampo"],2))



#plot_tmean_raster(korkeuskartta, tmean_annual, title="Annual mean temperature degC")

#plot_precip_raster(korkeuskartta, precip_annual, title="Annual precipitation mm")

# Ajetaan funktio esimerkiksi koordinaateilla Lon=25.0, Lat=60.0 (Helsingin suunnilla)
#piirra_ilmastodiagrammi(lon=25.0,lat=60.0,korkeuskartta=korkeuskartta, kuukausi_lampotilat=kuukausi_lämpötilat,kuukausi_sateet=kuukausi_sateet,)


#plt.imshow(merijaa_peittavyys_vuosi[6])
#plt.imshow(merijaa_paksuus_vuosi[1])

#plt.contour(maa_maski, levels=[0,1,2], colors=["red"], lw=2)
#plt.show()

piirra_ilmasto_kartta(
    koppen=koppen,
    tmean_annual=tmean_annual,
    precip_annual=precip_annual,
    korkeuskartta_m=korkeuskartta,
    tmin_annual=tmin_annual,
    tmax_annual=tmax_annual,
    precip_dry=precip_dry,
    precip_wet=precip_wet,
    korkeuskartta=korkeuskartta,
    merijaa_aina=merijaa_aina,
    rivers=rivers,
    lakes=lakes,
    jaa=jaa,
)


print(".")
