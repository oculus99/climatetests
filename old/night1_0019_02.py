
#############################
#
## simple planet climate 
#
## 19.09.2026 0000.0019.02
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
korkeus = 180*2
leveys = 360*2
seed1 = 42

leveysasteet = np.linspace(90, -90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)

maksimi_korkeus = 2300.0  # metriä
merenpinta_kynnys = 0.01   # Perlin-kynnysraja merelle/maalle (-1..1)

k=0.75 ## sigmoidin param
x0=0.6 ## sigmoidin param




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

# KÄYTTÖESIMERKKI:
# oletetaan että 'korkeus' on 2D-numpy-taulukko ja 'vari_kuva' on RGB-kuva (0-1 float)
# varjo_maski = laske_hillshade(korkeus, azimuth=315, altitude=45)
# valmis_kuva = sekoita_soft_light(vari_kuva, varjo_maski)



# Pyydetään noise-kirjastoa jos saatavilla, muuten käytetään vaihtoehtoista generaattoria
try:
    from noise import pnoise3
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
   
   
###HAS_NOISE = False  
##merenpinta_kynnys=0.03
# =====================================================================
# 2. MAASTON KORKEUSKARTAN GENEROINTI (Perlin 3D / Palloprojektio)
# =====================================================================
korkeuskartta = np.zeros((korkeus, leveys))


# =====================================================================
# 2. LUONNOLLISEN MAASTON GENEROINTI (Ridged Perlin 3D)
# =====================================================================

korkeuskartta = np.zeros((korkeus, leveys))

if HAS_NOISE:
    scale = 1
    octaves = 64

    lat_rads = np.radians(leveysasteet)
    lon_rads = np.radians(pituusasteet)

    cos_lat = np.cos(lat_rads)
    sin_lat = np.sin(lat_rads)
    cos_lon = np.cos(lon_rads)
    sin_lon = np.sin(lon_rads)

    for i in range(korkeus):
        c_lat = cos_lat[i]
        s_lat = sin_lat[i]

        for j in range(leveys):
            x = c_lat * cos_lon[j]
            y = c_lat * sin_lon[j]
            z = s_lat

            # ---------------------------------------------------------
            # 1. Domain warp
            # ---------------------------------------------------------
            warp_x = pnoise3(
                (x + 31.7) * 0.8,
                (y + 17.2) * 0.8,
                (z + 9.4) * 0.8,
                octaves=3,
                base=seed1 + 10
            )

            warp_y = pnoise3(
                (x - 12.4) * 0.8,
                (y + 51.8) * 0.8,
                (z + 3.1) * 0.8,
                octaves=3,
                base=seed1 + 20
            )

            # Väännetään pääkohinan koordinaatteja
            wx = x + warp_x * 0.45
            wy = y + warp_y * 0.45
            wz = z

            # ---------------------------------------------------------
            # 2. Varsinainen maastokohina
            # ---------------------------------------------------------
            n = pnoise3(
                (wx + 13.37) * scale,
                (wy + 42.42) * scale,
                (wz + 7.77) * scale,
                octaves=octaves,
                base=seed1
            )

            # ---------------------------------------------------------
            # 3. Ridged noise
            # ---------------------------------------------------------
            ridge = 1.0 - abs(n)

            # Leveät vuoristot
            ridge = ridge ** 2.0

            # ---------------------------------------------------------
            # 4. Suuri mannerkohina
            # ---------------------------------------------------------
            macro_large = pnoise3(
                (x + 91.2) * 0.18*6,
                (y + 47.1) * 0.18*6,
                (z + 13.8) * 0.18*6,
                octaves=12,
                base=seed1 + 30
            )

            # ---------------------------------------------------------
            # 5. Keskikokoinen mannerkohina
            # ---------------------------------------------------------
            macro_medium = pnoise3(
                (x - 37.4) * 0.32*4,
                (y + 82.1) * 0.32*4,
                (z + 24.6) * 0.32*4,
                octaves=12,
                base=seed1 + 40
            )

            macro_large = (macro_large + 1.0) * 0.5
            macro_medium = (macro_medium + 1.0) * 0.5

            # Yhdistetään suuret ja keskikokoiset manneralueet
            macro = (
                0.55 * macro_large +
                0.45 * macro_medium
            )

            # ---------------------------------------------------------
            # 6. Muodostetaan useita erillisiä manneralueita
            # ---------------------------------------------------------
            land = np.clip(
                (macro - 0.47) / 0.18,
                0.0,
                1.0
            )

            # Pehmeät rannikot
            land = (
                land *
                land *
                (3.0 - 2.0 * land)
            )

            # ---------------------------------------------------------
            # 7. Vuoristoalueiden maski
            # ---------------------------------------------------------
            mountain_mask = np.clip(
                (macro - 0.50) / 0.40,
                0.0,
                1.0
            )

            # Smoothstep
            mountain_mask = (
                mountain_mask *
                mountain_mask *
                (3.0 - 2.0 * mountain_mask)
            )

            # ---------------------------------------------------------
            # 8. Tasangot
            # ---------------------------------------------------------
            plateau = (
                0.08 +
                0.18 * macro
            )

            # ---------------------------------------------------------
            # 9. Yhdistetään maasto
            # ---------------------------------------------------------
            terrain = (
                plateau +
                ridge * mountain_mask * 0.65
            )

            # ---------------------------------------------------------
            # 10. Merenpohja / meri
            # ---------------------------------------------------------
            ocean = 0.015 + 0.025 * macro

            # Maa nostetaan meren yläpuolelle
            korkeuskartta[i, j] = (
                ocean * (1.0 - land) +
                terrain * land
            )



else:
    lon_grid, lat_grid = np.meshgrid(
        np.radians(pituusasteet),
        np.radians(leveysasteet)
    )

    # =========================================================
    # Pallopinnan koordinaatit
    # =========================================================
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    # =========================================================
    # Deterministinen pseudo-random noise
    # =========================================================
    def value_noise_3d(x, y, z, seed=0, grid_size=32):
        """
        Vektoroitu 3D value noise.
        Ei käytä pnoise3:a.

        Sama input + sama seed = sama tulos.
        """

        # Koordinaatit ruudukkoon
        px = x * grid_size
        py = y * grid_size
        pz = z * grid_size

        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        z0 = np.floor(pz).astype(np.int64)

        fx = px - x0
        fy = py - y0
        fz = pz - z0

        # Smoothstep
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        fz = fz * fz * (3.0 - 2.0 * fz)

        def random_value(ix, iy, iz):
            """
            Deterministinen hash.
            Palauttaa arvot väliltä -1...1.
            """
            h = (
                ix * 374761393 +
                iy * 668265263 +
                iz * 2147483647 +
                seed * 1274126177
            )

            h = h ^ (h >> 13)
            h = h * 1274126177
            h = h ^ (h >> 16)

            # Muutetaan unsigned-tyyppiseksi
            h = h & 0xffffffff

            return h.astype(np.float64) / 2147483647.5 - 1.0

        # Kahdeksan kulmapistettä
        c000 = random_value(x0,     y0,     z0)
        c100 = random_value(x0 + 1, y0,     z0)
        c010 = random_value(x0,     y0 + 1, z0)
        c110 = random_value(x0 + 1, y0 + 1, z0)

        c001 = random_value(x0,     y0,     z0 + 1)
        c101 = random_value(x0 + 1, y0,     z0 + 1)
        c011 = random_value(x0,     y0 + 1, z0 + 1)
        c111 = random_value(x0 + 1, y0 + 1, z0 + 1)

        # Interpolointi X
        nx00 = c000 * (1.0 - fx) + c100 * fx
        nx10 = c010 * (1.0 - fx) + c110 * fx
        nx01 = c001 * (1.0 - fx) + c101 * fx
        nx11 = c011 * (1.0 - fx) + c111 * fx

        # Interpolointi Y
        nxy0 = nx00 * (1.0 - fy) + nx10 * fy
        nxy1 = nx01 * (1.0 - fy) + nx11 * fy

        # Interpolointi Z
        return nxy0 * (1.0 - fz) + nxy1 * fz

    # =========================================================
    # Fractal Brownian Motion
    # =========================================================
    def fbm(x, y, z, seed, octaves=6):
        value = np.zeros_like(x)

        amplitude = 1.0
        frequency = 1.0

        amplitude_sum = 0.0

        for octave in range(octaves):
            value += amplitude * value_noise_3d(
                x * frequency,
                y * frequency,
                z * frequency,
                seed=seed + octave * 101,
                grid_size=4
            )

            amplitude_sum += amplitude

            frequency *= 2.0
            amplitude *= 0.5

        return value / amplitude_sum

    # =========================================================
    # 1. Suuret manneralueet
    # =========================================================
    continent = fbm(
        x,
        y,
        z,
        seed1,
        octaves=5
    )

    # -1...1 -> 0...1
    continent = (continent + 1.0) * 0.5

    # =========================================================
    # 2. Muodostetaan mantereet
    # =========================================================
    # Kynnysarvo määrää kuinka paljon planeetasta on maata.
    land = np.clip(
        (continent - 0.43) / 0.20,
        0.0,
        1.0
    )

    # Smoothstep tekee rannikoista luonnollisemmat
    land = land * land * (3.0 - 2.0 * land)

    # =========================================================
    # 3. Keskikokoinen maaston vaihtelu
    # =========================================================
    terrain = fbm(
        x * 2.0,
        y * 2.0,
        z * 2.0,
        seed1 + 1000,
        octaves=5
    )

    terrain = (terrain + 1.0) * 0.5

    # =========================================================
    # 4. Vuoristokohina
    # =========================================================
    mountain_noise = fbm(
        x * 3.0,
        y * 3.0,
        z * 3.0,
        seed1 + 2000,
        octaves=4
    )

    # Ridged noise
    mountain = 1.0 - np.abs(mountain_noise)

    # Korostetaan vain korkeimpia ridgejä
    mountain = mountain ** 3.0

    # =========================================================
    # 5. Yhdistetään eri mittakaavat
    # =========================================================
    korkeuskartta = (
        0.05
        + land * (
            0.35 * terrain +
            0.65 * mountain
        )
    )

    # Meret lähes tasaisiksi / mataliksi
    korkeuskartta *= (
        0.12 + 0.88 * land
    )

    # =========================================================
    # 6. Lopullinen normalisointi
    # =========================================================
    korkeuskartta -= korkeuskartta.min()
    korkeuskartta /= korkeuskartta.max()

    # Hieman luonnollisempi jakauma:
    # paljon matalaa maastoa, vähemmän todella korkeita vuoria.
    korkeuskartta = korkeuskartta ** 1.35





# Normaalistetaan kartta välille 0-1
min0 = np.min(korkeuskartta)
max0 = np.max(korkeuskartta)
if (max0 - min0) > 0:
    korkeuskartta = (korkeuskartta - min0) / (max0 - min0)

# KOODIMUUTOS: Luonnollisempi kontrastin muotoilu potenssifunktiolla sigmoidin sijaan.
# Arvo > 1.0 laajentaa tasankoja/meriä ja työntää vuoret omiksi jonoikseen.
korkeuskartta = korkeuskartta ** 1.5

# Määritetään maa_maski Perlin-kynnyksen perusteella
maa_maski = korkeuskartta > merenpinta_kynnys




# Skaalataan maa-alueiden korkeudet välille 0 ... maksimi_korkeus
korkeus_metreinä = np.zeros_like(korkeuskartta)

if np.any(maa_maski):
    min_maa = korkeuskartta[maa_maski].min()
    max_maa = korkeuskartta[maa_maski].max()
    if max_maa > min_maa:
        korkeus_metreinä[maa_maski] = (korkeuskartta[maa_maski] - min_maa) / (max_maa - min_maa) * maksimi_korkeus

# =====================================================================
# 3. MERIETÄISYYSRASATERIN LASKENTA
# =====================================================================
lon_grid_deg, lat_grid_deg = np.meshgrid(pituusasteet, leveysasteet)

# Lasketaan etäisyys mereen pikseleinä
#etaisyys_pikseleina = distance_transform_edt(maa_maski)
# Muutetaan pikselietäisyys kilometreiksi huomioiden napojen kapeneminen (kosini)
#aste_km = 111.0
#leveysaste_korjaus = np.cos(np.radians(np.abs(lat_grid_deg)))
#pikselin_koko_km = (aste_km * leveysaste_korjaus*leveys)/360
#etaisyys_meresta_km = etaisyys_pikseleina * pikselin_koko_km

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
maapallon_sade_km = 6371.0
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
korkeus_gradientti_x = np.gradient(korkeus_metreinä, axis=1)

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

korkeus_gradientti_x = np.gradient(korkeus_metreinä, axis=1)



for kk in range(12):
    # =================================================================
    # 1. AURINKO JA ILMASTOLLINEN PÄIVÄNTASAAJA (ITCZ)
    # =================================================================
    # Auringon deklinaatio (akselin kallistuma)
    kulma = 2.0 * np.pi * (kk - 5.5) / 12.0
    akselin_kallistuma = 23.44 * np.cos(kulma)
    
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
    
    # =================================================================
    # 3. LÄMPÖTILA
    # =================================================================
    etäisyys_auringosta = np.abs(lat_grid_deg - akselin_kallistuma)
    kk_temp_ka = 32.0 - 0.73 * etäisyys_auringosta - (korkeus_metreinä / 1000.0) * 6.5
    kk_temp_ka += kk_meri_anomalia
    kuukausi_lämpötilat[kk, :, :] = kk_temp_ka + 12
    
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
    efektiivinen_etaisyys_km += np.where(korkeus_metreinä > 1200.0, 1500.0, 0.0)
    
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
    alavuus_kerroin = np.clip(1.3 - (korkeus_metreinä / 800.0), 0.4, 1.4)

    # --- EFREKTIIVINEN ETÄISYYS JA RANNIKKOKERROIN ---
    efektiivinen_etaisyys_km = etaisyys_meresta_km.copy()
    efektiivinen_etaisyys_km += np.where(korkeus_metreinä > 1200.0, 1500.0, 0.0)
    
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
    korkeussade_pohja = np.clip(korkeus_metreinä / 150.0, 0, 80.0)
    tuuli_orografia_vaikutus = kk_orografinen_pakote * 2.5 
    dynaaminen_orografia = np.clip(korkeussade_pohja + tuuli_orografia_vaikutus, 0.0, 250.0) * lopullinen_rannikko_kerroin * 0.1   
    
    # Tallennetaan kuukauden sademäärä
    kuukausi_sateet[kk, :, :] = perus_sade + dynaaminen_orografia







meri_anomalia=np.mean(kk_meri_anomalia, axis=0)


lumi, jaa, kokonais_korkeus = aja_jaatikko_malli(korkeus_metreinä, kuukausi_lämpötilat, kuukausi_sateet, vuodet=100)

#merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeuskartta, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia*0+1,lat_grid_deg)

merijaa_paksuus_vuosi, merijaa_peittavyys_vuosi=mallinna_merijaa(korkeus_metreinä, maa_maski, kuukausi_lämpötilat, kuukausi_sateet, tuuli_suunta_base, meri_anomalia, lat_grid_deg)

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
    "#afafaf"   # 14: EF (Napavyöhyke jäätikkö tai polaariaavikko)
]

tarkat_nimet = [
    "Meri", "Sademetsä (Af)", "Monsuuni (Am)", "Savanni (Aw)", 
    "Kuuma aavikko (BWh)", "Kylmä aavikko (BWk)", "Kuuma aro (BSh)", "Kylmä aro (BSk)",
    "Välimerenilmasto (Csa/Csb)", "Meri-ilmasto (Cfb)", "Subtrooppinen kostea (Cfa)",
    "Mannerilmasto (Dfa/Dfb)", "Taiga / Havumetsä (Dfc/Dfd)", "Tundra (ET)", "Polaariaavikko (EF)"
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
    "maaston_korkeus_m": korkeus_metreinä,
}
variable_names=[
        "keski_lampotila",
        "keski_sademaara",
        "maaston_korkeus_m",
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

varjo_maski = laske_hillshade(korkeus_metreinä, azimuth=315, altitude=45)

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

plt.title("Tarkennettu Köppen-ilmastoluokitus (Dynaamiset tuulet, merivirrat ja sateenvarjot)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Pituusaste")
plt.ylabel("Leveysaste")
plt.grid(color='black', linestyle='--', alpha=0.1) # Vaihdettu valkoinen ruudukko mustaksi, jotta se näkyy navoilla
plt.tight_layout()

# Näytetään ikkuna
plt.show()

quit(-1)

#plt.imshow(lumi)
#plt.imshow(jaa)

#plt.imshow(huippu)
  
    
#plt.imshow(korkeus_metreinä)
# ==========================================
# RASTERIKUVAN PIIRTÄMINEN (MATPLOTLIB)
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Vasen kuva: Kallioperän muoto (korkeuskartta)
im1 = ax[0].imshow(korkeus_metreinä, cmap='terrain', origin='lower')
ax[0].set_title('Kallioperän korkeus (m)')
fig.colorbar(im1, ax=ax[0], label='Metriä merenpinnasta')

# Oikea kuva: Jään paksuus rasterina
# Käytetään 'Blues'-värikarttaa, jotta jää erottuu selkeästi
im2 = ax[1].imshow(jaa, cmap='Blues', origin='lower')
ax[1].set_title('Jään paksuus n vuoden jälkeen (m)')
fig.colorbar(im2, ax=ax[1], label='Jään paksuus metreinä')

plt.tight_layout()
plt.show()
