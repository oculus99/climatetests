
#############################
#
## Dimple planet climate Köppen classes
#
# Python3+CuPy - requires CUDA
#
## 22.09.2026 0000.0023.05
#
#####################

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
#seed1=999
seed1 = 53 ## hyva


#maapallon_sade_km = 6371.0
#syvin_kohta=-11000
#korkein_kohta=8848
#manner_osuus=0.30

maapallon_sade_km = 6371.0
syvin_kohta=-2000
korkein_kohta=2000
#manner_osuus=0.30
manner_osuus=0.30



#tilt_planet_axis=23.44
#ecc_planet=0.013
#mvelp_planet=102.0
# Parametrit (esimerkkinä Maan arvot)
tilt_planet_axis = 23.44*1
ecc_planet = 0.013*1
mvelp_planet = 102.0  # Perihelin pituus asteina (Maa saavuttaa perihelin tammikuun alussa)
tmax_planet=50
global_moisture_coeff=1




###########################################
#######################################
### kartta


import math
import numpy as np
from noise import pnoise3

class PlanetMap:
    def __init__(
        self,
        leveys=360,
        korkeus=180,
        syvin_kohta=-4000,
        korkein_kohta=3000,
        noise_scale=1.0,
        seed=42,
    ):
        self.leveys = leveys
        self.korkeus = korkeus
        self.syvin_kohta = syvin_kohta
        self.korkein_kohta = korkein_kohta
        self.noise_scale = noise_scale
        self.seed = float(seed)  # Varmistetaan liukuluvuksi koordinaatteja varten
        
        # Alustetaan tyhjä kartta raakakorkeuksille
        self.kartta = np.zeros((self.korkeus, self.leveys))

    def generoi_kartta(self, octaves=16, persistence=0.5, lacunarity=2.0, distortion=0.5, distortion_scale=1.5):
        """
        Generoi 3D Perlin -kohinaan perustuvan planeetan kartan domain warpingilla (vääristymällä).
        
        Parametrit:
          distortion: Kuinka voimakkaasti vuorijonot ja mantereet kaartuvat/vääntyvät (suositus: 0.1 - 1.0)
          distortion_scale: Vääristymän mittakaava. Pienempi arvo luo laajoja kaaria, suurempi arvo pientä siksakkia.
        """
        # Seed-pohjaiset siirtymät eri kohinakerroksille erottamaan ne toisistaan
        seed_offset_x = self.seed * 100.0
        seed_offset_y = self.seed * 200.0
        seed_offset_z = self.seed * 300.0

        for y in range(self.korkeus):
            lat = (y / self.korkeus) * math.pi - (math.pi / 2.0)
            
            for x in range(self.leveys):
                lon = (x / self.leveys) * 2.0 * math.pi
                
                # Alkuperäiset 3D-koordinaatit yksikköpallon pinnalla
                nx = math.cos(lat) * math.cos(lon)
                ny = math.cos(lat) * math.sin(lon)
                nz = math.sin(lat)
                
                # --- VAIHE 1: Lasketaan koordinaattien siirtymä (Offset / Domain Warping) ---
                # Haetaan kolme eri kohina-arvoa X, Y ja Z akseleille vääristymää varten.
                # Käytetään matalampaa oktaavilukua (esim. 2 tai 3), jotta kaaret ovat sulavia.
                ox = pnoise3(
                    nx * distortion_scale + seed_offset_x, 
                    ny * distortion_scale, 
                    nz * distortion_scale, 
                    octaves=3
                )
                oy = pnoise3(
                    nx * distortion_scale, 
                    ny * distortion_scale + seed_offset_y, 
                    nz * distortion_scale, 
                    octaves=3
                )
                oz = pnoise3(
                    nx * distortion_scale, 
                    ny * distortion_scale, 
                    nz * distortion_scale + seed_offset_z, 
                    octaves=3
                )
                
                # --- VAIHE 2: Sovelletaan siirtymä ja skaalataan pääkohinalle ---
                # Lisätään saatu offset alkuperäisiin koordinaatteihin ennen pääkohinan hakua
                final_x = (nx + ox * distortion) * self.noise_scale
                final_y = (ny + oy * distortion) * self.noise_scale
                final_z = (nz + oz * distortion) * self.noise_scale
                
                # --- VAIHE 3: Haetaan varsinainen maaston korkeuskohina ---
                # Käytetään base-parametrissa kokonaislukua siemeneksi, jos halutaan (esim. int(self.seed))
                kohina_arvo = pnoise3(
                    final_x, 
                    final_y, 
                    final_z,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=int(abs(self.seed))
                )
                
                # Skaalataan kohina välille [0, 1]
                kohina_norm = (kohina_arvo + 1.0) / 2.0
                kohina_norm = max(0.0, min(1.0, kohina_norm))
                
                # Skaalataan halutulle korkeusvälille
                korkeus_m = self.syvin_kohta + kohina_norm * (self.korkein_kohta - self.syvin_kohta)
                
                self.kartta[y, x] = korkeus_m
                
        # --- LUODAAN PYYDELLYT MASKIT JA MUUNNOKSET ---
        korkeuskartta_maa = np.where(self.kartta > 0, self.kartta, 0.0)
        maa_maski = np.where(self.kartta > 0, 1.0, 0.0).astype(int)
        meri_maski = np.where(self.kartta <= 0, 1.0, 0.0).astype(int)
                
        return self.kartta, korkeuskartta_maa, maa_maski, meri_maski






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
    lat,
    kk,
    tilt_planet_axis,
    ecc_planet,
    mvelp_planet,
    tmax_planet_param,
):
    """
    Laskee kuukauden:

        - rata-aseman
        - rataetäisyyden
        - säteilykertoimen
        - auringon deklinaation
        - planeetan peruslämpötilan

    Tulos on Celsius-asteina.

    Tämä EI vielä huomioi:

        - korkeutta
        - meriä
        - merivirtoja
        - sateita
        - pilviä
        - mannerilmastoa
    """

    ratakulma_deg = (
        kk * 30.0
    )

    ratakulma_rad = np.radians(
        ratakulma_deg
    )

    true_anomaly = np.radians(
        ratakulma_deg
        -
        mvelp_planet
    )

    # ------------------------------------------------------------
    # Elliptisen radan etäisyys
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Säteilyn suhteellinen voimakkuus
    # ------------------------------------------------------------

    sateily_kerroin = (
        1.0
        /
        np.sqrt(etaisyys_au)
    )

    # ------------------------------------------------------------
    # Auringon deklinaatio
    # ------------------------------------------------------------

    deklinaatio = (
        tilt_planet_axis
        *
        np.sin(
            ratakulma_rad
            -
            np.radians(60.0)
        )
    )

    # ------------------------------------------------------------
    # Planeetan referenssilämpötila
    # ------------------------------------------------------------

    tmax = cp.float32(
        tmax_planet_param
    )

    sateily_poikkeama = (
        cp.float32(
            sateily_kerroin
        )
        -
        cp.float32(
            1.0
        )
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

    # ------------------------------------------------------------
    # Auringon kulman aiheuttama lämpötilagradientti
    # ------------------------------------------------------------

    zeniitti_etaisyys = cp.abs(
        lat
        -
        cp.float32(
            deklinaatio
        )
    )

    zeniitti_etaisyys = cp.clip(
        zeniitti_etaisyys,
        cp.float32(0.0),
        cp.float32(90.0)
    )

    leveysaste_viilennys = (
        cp.float32(0.73)
        *
        zeniitti_etaisyys
    )

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


# ================================================================
# TUULIMALLI
# ================================================================

def laske_tuulet(
    lat,
    kk,
    tilt_planet_axis,
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
        0.65
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
    tmax_default=57.0,
    global_moisture_coeff=1.0,
):

    korkeus, leveys = korkeus_syvyyskartta.shape

    lapse_rate = -6.5 / 1000

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

        peruslampotulos = mittaa_gpu(
            "aurinko",
            laske_auringon_peruslampotila,
            lat_grid_deg,
            kk,
            planet_tilt,
            planet_ecc,
            planet_mvelp_deg,
            tmax_default,
            kuukausi=kk,
        )

        peruslampotila = peruslampotulos[0]

        ilmastollinen_paivantaasaaja = (
            peruslampotulos[1]
        )

        kuukausi_lämpötilat_gpu[kk] = peruslampotila

        # ========================================================
        # TUULI
        # ========================================================

        perustuulitulos = mittaa_gpu(
            "tuulet",
            laske_tuulet,
            lat_grid_deg,
            kk,
            planet_tilt,
            kuukausi=kk,
        )

        tuuli_x = perustuulitulos[0]
        tuuli_y = perustuulitulos[1]
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

        # ========================================================
        # SADEVYÖHYKKEIDEN TARKENNUS
        # ========================================================

        perussade = mittaa_gpu(
            "tarkenna_sadevyohykkeet",
            tarkenna_sadevyohykkeet,
            perussade,
            lat_grid_deg,
            ilmastollinen_paivantaasaaja,
            kuukausi=kk,
        )

        # ========================================================
        # TUULEN ETÄISYYS MERESTÄ
        # ========================================================

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

        sade_end.record()
        sade_end.synchronize()

        profiili["sateen_tallennus"] += (
            cp.cuda.get_elapsed_time(
                sade_start,
                sade_end
            ) / 1000.0
        )

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

        meri_pilvisyys_vaikutus = meri_bin * -5

        kuukausi_lämpötilat_gpu[kk] = (
            kuukausi_lämpötilat_gpu[kk]
            + meri_pilvisyys_vaikutus
            + lämpötila_vähete
        )

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
    syvin_kohta=-2000,
    korkein_kohta=2200,

    seed=seed1,
)

tulokset=mapper.generoi_kartta()



korkeus_syvyyskartta = tulokset[0]
korkeuskartta = tulokset[1]
maa_maski = tulokset[2]
meri_maski = tulokset[3]

plt.imshow(korkeuskartta, cmap="terrain")

#plt.imshow(maa_maski, cmap="terrain")

#plt.imshow(meri_maski, cmap="viridis")
#plt.show()


#quit(-1)


#print("globaali vedenpinta:",
#      tulokset["vedenpinta"])


#zoomaaja = KartanTekijaCuPy(
#    leveys=1000,
#    korkeus=1000,
#    syvin_kohta=syvin_kohta,
#    korkein_kohta=korkein_kohta,
#    manner_osuus=manner_osuus,
#    seed=seed1,
#    octaves=24,
#)

#zoomattu = zoomaaja.kartta(
#    kartan_alue=(0, 90, 0, 90),
#    leveys=400,
#    korkeus=400,
#    sealevel=tulokset["vedenpinta"],
#).tulokset(
#    cpu=True
#)



#tulokset = kartantekija.tulokset(cpu=True)


#korkeus_syvyyskartta = kartantekija.korkeus_syvyyskartta.get()
#korkeuskartta = kartantekija.korkeuskartta.get()
#maa_maski = kartantekija.maa_maski.get()
#meri_maski = kartantekija.meri_maski.get()
#zoomattu_nelio=zoomattu["kartta_nelio"]
#zoomattu_korkeus_syvyyskartta =  zoomattu["korkeus_syvyyskartta"]
#zoomattu_korkeuskartta = zoomattu["korkeuskartta"]
#zoomattu_maa_maski = zoomattu["maa_maski"]
#zoomattu_meri_maski = zoomattu["meri_maski"]

#print(korkeus_syvyyskartta.shape)
#print(korkeuskartta.shape)
#print(maa_maski.shape)
#print(meri_maski.shape)
#print("zoomattu")
#print(zoomattu["kartta_nelio"])

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

print(" Rannikko-etäisyys")

maa_maski_gpu = cp.asarray(maa_maski)

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

print(" Rannikko-etäisyys")

varjo_maski = laske_hillshade(korkeuskartta, azimuth=315, altitude=45)

#korkeus_gradientti_x = np.gradient(korkeuskartta, axis=1)
#korkeus_gradientti_y = np.gradient(korkeuskartta, axis=0)

n_lats, n_lons = korkeuskartta.shape
lats = np.linspace(-90, 90, n_lats)
cos_weights = np.cos(np.radians(lats))
weight_grid = cos_weights[:, np.newaxis] * np.ones(n_lons)
maa_painotettu = np.sum(weight_grid[maa_maski])
meri_painotettu = np.sum(weight_grid[meri_maski])
kokonais_paino = np.sum(weight_grid)
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
 
#warmup_gpu(seconds=2.0)
    
#benchmark_ilmasto(korkeus_syvyyskartta)
#benchmark_multiple(korkeus_syvyyskartta)
#quit(-1)



#print ("GPU warming up ...")

# Ensimmäinen ajo = warm-up


## 0) Laske ilmasto

kuukausi_lämpötilat_gpu, kuukausi_sateet_gpu, \
kuukausi_tuuli_suunta_gpu, kuukausi_tuuli_voima_gpu, \
kuukausi_merivirta_x_gpu, kuukausi_merivirta_y_gpu=laske_ilmasto_gpu(korkeus_syvyyskartta, planet_mass_me=1, planet_radius_re=1.0, planet_tilt=23.44, planet_ecc=0.013, planet_mvelp_deg=101.2, tmax_default=tmax_planet, global_moisture_coeff=global_moisture_coeff)

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

   

#jaa, merijaa_aina=laske_jaat(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, vuosia=200)  
    
    

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

plt.imshow(np.exp(varjo_maski), cmap="gray", alpha=0.1,extent=[-180, 180, -90, 90])

#plt.contour(merijaa_aina, levels=[0.25,1],extent=[-180, 180, 90, -90], lw=1, alpha=0.7, colors=["lightblue"])
#plt.contourf(merijaa_aina, levels=[0.25,1],extent=[-180, 180, 90, -90], lw=1, alpha=0.7, colors=["lightblue"])

#plt.contour(merijaa_vain_joskus, levels=[0.5,1],extent=[-180, 180, 90, -90], lw=1,linestyle=":", colors=["blue"])

#plt.contour(jaa, levels=[20,50,100,200],extent=[-180, 180, 90, -90], lw=2, colors=["white"])


#plt.contourf(jaa, levels=[20,50,100,200],extent=[-180, 180, 90, -90], lw=2, colors=["white"])


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
