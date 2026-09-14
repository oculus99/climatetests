
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. FAKTUAALINEN PINTA
# ============================================================
def luo_fraktaalipinta(n=77, seed=42, manner_kynnys=0.68):
    """
    Luo satunnaisen mutta skaalautuvan manner/meri-pinnan.

    Palauttaa:
        korkeus       : normalisoitu 0...1
        onko_manner   : bool-maski
        korkeus_m     : mannerkorkeus metreinä
    """
    rng = np.random.default_rng(seed)

    korkeus = np.zeros((n, n), dtype=float)

    for oktiivi in range(1, 8):
        freq = 2 ** oktiivi
        amp = 1.0 / freq

        kohina = rng.normal(0.0, 1.0, (n, n))

        # Pienemmillä skaaloilla vähemmän tasoitusta.
        sigma = max(1, n // (4 * freq))
        kohina = gaussian_filter(kohina, sigma=sigma)

        kohina -= kohina.mean()

        std = kohina.std()
        if std > 0:
            kohina /= std

        korkeus += amp * kohina

    # Normalisointi
    korkeus -= korkeus.min()
    korkeus /= (korkeus.max() + 1e-12)

    # Manner/meri
    onko_manner = korkeus > manner_kynnys

    # Mannerkorkeus 0...4000 m
    korkeus_m = np.where(
        onko_manner,
        (korkeus - manner_kynnys)
        / (1.0 - manner_kynnys + 1e-12)
        * 4000.0,
        0.0
    )

    return korkeus, onko_manner, korkeus_m


# ============================================================
# 2. GLOBAALI KOSINI-PAINOTUS
# ============================================================
def globaali_keskiarvo(X, LAT):
    """
    Pallon pinta-alalla painotettu keskiarvo.

    Leveysasteilla pinta-ala ~ cos(lat).
    """
    paino = np.cos(LAT)
    return np.sum(X * paino) / (np.sum(paino) + 1e-12)


def globaali_mannerosuus(onko_manner, LAT):
    """
    Cos(lat)-painotettu mannerosuus.
    """
    paino = np.cos(LAT)
    return (
        np.sum(onko_manner.astype(float) * paino)
        / (np.sum(paino) + 1e-12)
    )


# ============================================================
# 3. AURINKO JA INSOLAATIO
# ============================================================
def aurinko_deklinaatio(paiva_vuodesta, tilt, vuoden_pituus_paivina):
    """
    Auringon deklinaatio.

    paiva_vuodesta:
        0...vuoden_pituus_paivina

    tilt:
        akselikallistus radiaaneina.
    """
    return tilt * np.sin(
        2.0 * np.pi
        * (paiva_vuodesta - 80.0)
        / vuoden_pituus_paivina
    )


def paivittainen_insolaatio_vektori(
    lat_array,
    dekl,
    S,
    n_tuntia=96
):
    """
    Päivittäinen TOA-insolaatio W/m².

    Integroi päivän yli useammalla kuin alkuperäisellä
    48 pisteellä. 96 pistettä = 15 minuutin aikaväli.

    Tämä on ennen albedoa oleva saapuva aurinkoenergia.
    """
    if n_tuntia < 24:
        raise ValueError("n_tuntia pitää olla vähintään 24.")

    I_sum = np.zeros_like(lat_array, dtype=float)

    dt_tunti = 24.0 / n_tuntia

    for h in range(n_tuntia):
        tunti = (h + 0.5) * dt_tunti

        # Tuntikulma
        omega = np.deg2rad(
            (tunti - 12.0) * 15.0
        )

        sin_alt = (
            np.sin(lat_array) * np.sin(dekl)
            + np.cos(lat_array)
            * np.cos(dekl)
            * np.cos(omega)
        )

        # Aurinko horisontin yläpuolella
        I_sum += np.where(
            sin_alt > 0.0,
            S * sin_alt * dt_tunti,
            0.0
        )

    return I_sum / 24.0


# ============================================================
# 4. CO2-PAKOTE
# ============================================================
def co2_sateilyvoimakkuus(p_co2_ppm, co2_ref_ppm=280.0):
    """
    CO2:n säteilypakote suhteessa referenssipitoisuuteen.

    Palauttaa W/m².

    ΔF = 5.35 ln(C/C0)

    Huom:
    Tämä on globaali efektiivinen pakotekaava,
    ei spektrisesti ratkaistu säteilymalli.
    """
    C = np.maximum(np.asarray(p_co2_ppm, dtype=float), 1.0)

    return 5.35 * np.log(
        C / co2_ref_ppm
    )


# ============================================================
# 5. KASVIHUONEEN TEHOKAS LÄMPÖTILAVAIKUTUS
# ============================================================
def kasvihuone_lampotilaero(
    p_atm,
    p_co2_ppm,
    perus_gh_k=33.0,
    lambda_co2=0.76,
    co2_ref_ppm=280.0
):
    """
    Yksinkertaistettu efektiivinen kasvihuoneparametrisointi.

    Earth-like vertailussa:
        1 atm
        280 ppm CO2
        -> noin 33 K kasvihuone-ero.

    CO2:n lisävaikutus:
        ΔT = lambda * ΔF_CO2

    lambda_co2 ~ 0.76 K/(W/m²) antaa noin 3 K vasteen
    CO2:n kaksinkertaistumiselle.

    HUOM:
    Tämä ei ole varsinainen säteilytransfer-malli.
    Se on tarkoituksella yksinkertainen planeettamallin
    parametrisaatio.
    """

    p = max(float(p_atm), 0.0)

    # Paineen vaikutus kasvihuoneen tehokkuuteen.
    #
    # sqrt käyttäytyy järkevästi toy-mallissa:
    #   0 atm -> 0
    #   1 atm -> 1
    #   2 atm -> 1.41
    #
    # Ei pidä tulkita tarkaksi fysiikaksi.
    painekerroin = np.sqrt(p)

    F_co2 = co2_sateilyvoimakkuus(
        p_co2_ppm,
        co2_ref_ppm=co2_ref_ppm
    )

    delta_T = (
        painekerroin
        * (
            perus_gh_k
            + lambda_co2 * F_co2
        )
    )

    return max(delta_T, 0.0)


# ============================================================
# 6. ALBEDO
# ============================================================
def albedo_pinnan_mukaan(
    onko_manner,
    korkeus_m,
    T,
    pilvisyys=None,
    pilvi_albedo=0.50
):
    """
    Pinnan + lumen/jään + pilvien tehokas albedo.

    Tärkeä muutos:
    Pilvi ei enää yksinkertaisesti lisää albedoon
        +0.20 * pilvisyys

    vaan pilvisyys sekoittaa pinta-albedoa kohti
    pilven tehokasta albedoa.

    Tämä estää albedon kasvamasta liian aggressiivisesti
    pilvisyyden mukana.
    """

    # Peruspinta
    albedo_pinta = np.where(
        onko_manner,
        0.20,       # manner
        0.06        # avomeri
    )

    # Lumi mantereella
    lumi = (
        onko_manner
        & (T < 273.15)
    )

    albedo_pinta = np.where(
        lumi,
        0.65,
        albedo_pinta
    )

    # Merijää
    meri_jaa = (
        (~onko_manner)
        & (T < 271.15)
    )

    albedo_pinta = np.where(
        meri_jaa,
        0.55,
        albedo_pinta
    )

    # Korkeuden pieni albedovaikutus mantereella
    albedo_pinta = np.where(
        onko_manner,
        albedo_pinta
        + 0.05 * np.clip(korkeus_m / 4000.0, 0.0, 1.0),
        albedo_pinta
    )

    # Pilvien vaikutus
    if pilvisyys is not None:
        f = np.clip(pilvisyys, 0.0, 1.0)

        # Pilvi voi kasvattaa albedoa vain, jos se on pintaa
        # heijastavampi. Näin jää ei muutu pilven vuoksi
        # epäfysikaalisesti tummemmaksi.
        # Pilvet eivät saa nostaa albedoa suoraan kohti 0.50:ta.
        # Muuten pilvi-albedopalaute jäähdyttää mallin liian voimakkaasti.
        pilvi_lisa = np.minimum(
            0.12,
            np.maximum(pilvi_albedo - albedo_pinta, 0.0)
        )
        albedo_pinta = albedo_pinta + f * pilvi_lisa

    return np.clip(
        albedo_pinta,
        0.05,
        0.90
    )


# ============================================================
# 7. ENERGIATASAPAINO
# ============================================================
def energia_tasapaino(
    I_abs,
    albedo,
    p_atm,
    p_co2_ppm,
    sigma=5.670374419e-8,
    min_tehokas_lampotila=150.0
):
    """
    Yksinkertainen TOA-energiatasapaino.

    Vaiheet:

        1. Saapuva aurinkoenergia
        2. Albedon jälkeen absorboitu energia
        3. Stefan-Boltzmann -> efektiivinen säteilylämpötila
        4. Kasvihuone lisää pintalämpötilaa

    Tärkeä ero vanhaan malliin:

        VANHA:
            I_abs * (1-albedo) + 200 W/m²

        UUSI:
            T_eff = [(I_abs*(1-albedo))/sigma]^(1/4)
            T_surface = T_eff + ΔT_greenhouse

    Näin kasvihuone ei luo energiaa tyhjästä.
    """

    I = np.maximum(
        np.asarray(I_abs, dtype=float),
        0.0
    )

    alb = np.clip(
        np.asarray(albedo, dtype=float),
        0.0,
        1.0
    )

    # TOA:n absorboitu aurinkoenergia
    absorboitu = I * (1.0 - alb)

    # Efektiivinen säteilylämpötila
    #
    # Minimiraja on numeerinen suojaraja napa-yössä.
    # Todellinen napa-alueen lämpötila syntyy myöhemmin
    # lämpökapasiteetin ja kuljetuksen kautta.
    T_eff = np.where(
        absorboitu > 0.0,
        (absorboitu / sigma) ** 0.25,
        min_tehokas_lampotila
    )

    T_eff = np.maximum(
        T_eff,
        min_tehokas_lampotila
    )

    # Kasvihuone
    delta_T_gh = kasvihuone_lampotilaero(
        p_atm=p_atm,
        p_co2_ppm=p_co2_ppm
    )

    T_uusi = T_eff + delta_T_gh

    return np.clip(
        T_uusi,
        50.0,
        1000.0
    )


# ============================================================
# 8. ENERGIATARKISTUKSET
# ============================================================
def tarkista_energiabudjetti(
    I,
    albedo,
    LAT,
    otsikko=""
):
    """
    Tulostaa hyödylliset energiabudjetin diagnostiset arvot.
    """

    I_glob = globaali_keskiarvo(
        I,
        LAT
    )

    absorboitu = I * (1.0 - albedo)

    A_glob = globaali_keskiarvo(
        albedo,
        LAT
    )

    absorbed_glob = globaali_keskiarvo(
        absorboitu,
        LAT
    )

    print(f"\n=== Energiabudjetti {otsikko} ===")
    print(
        f"  TOA-insolaatio:       "
        f"{I_glob:7.2f} W/m²"
    )
    print(
        f"  Keskimääräinen albedo:"
        f" {A_glob:7.3f}"
    )
    print(
        f"  Absorboitu energia:   "
        f"{absorbed_glob:7.2f} W/m²"
    )
    print(
        f"  Heijastunut energia:  "
        f"{I_glob - absorbed_glob:7.2f} W/m²"
    )

    return {
        "I_globaali": I_glob,
        "albedo_globaali": A_glob,
        "absorboitu_globaali": absorbed_glob,
    }


# ============================================================
# 4. ILMAKEHÄN SOLUT JA LÄMMÖN KULJETUS
# ============================================================

def solujen_maara(omega_suhteessa_maahan):
    """
    Arvioitu Hadley/Ferrel/Polar-solujen lukumäärä.

    Earth-like:
        omega = 1 -> 3 solua / pallonpuolisko

    Tämä on edelleen parametrisaatio, ei dynaaminen GCM.
    """
    omega = max(float(omega_suhteessa_maahan), 0.01)

    return max(
        1,
        int(round(
            3.0 + np.log2(omega)
        ))
    )


def tuulikentta(LAT, omega_suhteessa_maahan):
    """
    Yksinkertaistettu planeetan meridionaalinen/zonaalinen
    tuulikenttä.

    Palauttaa:
        u = itä-länsisuuntainen tuuli
        v = pohjois-eteläsuuntainen tuuli
        N = solujen määrä
    """

    N = solujen_maara(
        omega_suhteessa_maahan
    )

    lat_norm = LAT / (np.pi / 2.0)

    # Zonaalinen tuuli.
    # Maksimi ~30 m/s Earth-like tapauksessa.
    u = (
        30.0
        * np.sin(N * np.pi * lat_norm)
        * np.cos(lat_norm)
    )

    # Meridionaalinen komponentti.
    v = (
        5.0
        * np.cos(N * np.pi * lat_norm)
        * np.cos(lat_norm)
    )

    # Napa-alueella kosini -> 0
    u = np.where(
        np.abs(lat_norm) <= 1.0,
        u,
        0.0
    )

    v = np.where(
        np.abs(lat_norm) <= 1.0,
        v,
        0.0
    )

    return u, v, N


def vakaa_diffuusio_1d(T, kappa, akseli=0):
    """
    Yksi eksplisiittinen diffuusioaskel.

    Käytetään rajattua kertoimella:
        0 <= kappa <= 0.25

    jotta diffuusio ei muutu numeerisesti epävakaaksi.

    akseli=0:
        meridionaalinen

    akseli=1:
        zonaalinen
    """

    T = np.asarray(T, dtype=float)
    kappa = np.clip(
        np.asarray(kappa, dtype=float),
        0.0,
        0.25
    )

    U = T.copy()

    if akseli == 0:

        # Sisäiset leveysasteet
        U[1:-1, :] += kappa[1:-1, :] * (
            T[2:, :]
            - 2.0 * T[1:-1, :]
            + T[:-2, :]
        )

        # Napa-alueet:
        # Ne eivät saa diffundoida "tyhjän ruudukon" läpi.
        U[0, :] = T[0, :]
        U[-1, :] = T[-1, :]

    else:

        # Periodinen longitude
        T_left = np.roll(T, 1, axis=1)
        T_right = np.roll(T, -1, axis=1)

        U += kappa * (
            T_right
            - 2.0 * T
            + T_left
        )

    return U


def ilmakehan_kuljetus(
    T,
    onko_manner,
    omega_suhteessa_maahan,
    vahvuus=0.08,
    n_iter=3
):
    """
    Ilmakehän lämpökuljetus.

    TÄRKEÄ:
    Tämä ei lisää eikä poista globaalia energiaa.

    Se vain tasaa alueellisia lämpötilaeroja.

    Mallissa on:
        - perussekoitus
        - Hadley
        - Ferrel
        - Polar
        - zonaalinen sekoitus

    Kuljetuskerroin on rajoitettu numeerisesti.
    """

    n = T.shape[0]

    lat = np.linspace(
        -np.pi / 2.0,
        np.pi / 2.0,
        n
    )

    LAT, _ = np.meshgrid(
        lat,
        np.arange(n),
        indexing="ij"
    )

    u, v, N = tuulikentta(
        LAT,
        omega_suhteessa_maahan
    )

    T_uusi = T.copy()

    lat_abs = np.abs(
        np.rad2deg(LAT)
    )

    # --------------------------------------------------------
    # Meridionaalinen diffuusio
    # --------------------------------------------------------

    kappa_merid = np.full_like(
        T,
        float(vahvuus)
    )

    # Tuulen vahvistama kuljetus
    kappa_merid += (
        0.025
        * np.abs(v) / 5.0
    )

    # Hadley
    kappa_merid += (
        0.035
        * np.exp(
            -(lat_abs / 20.0) ** 2
        )
    )

    # Ferrel
    kappa_merid += (
        0.080
        * np.exp(
            -((lat_abs - 30.0) / 25.0) ** 2
        )
    )

    # Polar
    kappa_merid += (
        0.025
        * np.exp(
            -((lat_abs - 75.0) / 15.0) ** 2
        )
    )

    # Varmistetaan numeerinen vakaus
    kappa_merid = np.clip(
        kappa_merid,
        0.0,
        0.20
    )

    for _ in range(max(1, int(n_iter))):

        T_uusi = vakaa_diffuusio_1d(
            T_uusi,
            kappa_merid,
            akseli=0
        )

    # --------------------------------------------------------
    # ZONAAALINEN SEKOITUS
    # --------------------------------------------------------

    kappa_zon = (
        0.015
        * (1.0 + np.abs(u) / 30.0)
        * float(vahvuus / 0.08)
    )

    kappa_zon = np.clip(
        kappa_zon,
        0.0,
        0.10
    )

    T_uusi = vakaa_diffuusio_1d(
        T_uusi,
        kappa_zon,
        akseli=1
    )

    # --------------------------------------------------------
    # Kuljetus ei saa muuttaa globaalia keskilämpötilaa.
    # --------------------------------------------------------

    paino = np.cos(LAT)

    ennen = (
        np.sum(T * paino)
        / (np.sum(paino) + 1e-12)
    )

    jalkeen = (
        np.sum(T_uusi * paino)
        / (np.sum(paino) + 1e-12)
    )

    # Korjaa mahdollinen pieni numeerinen energiavirhe.
    T_uusi += ennen - jalkeen

    return T_uusi, u, v, N


# ============================================================
# 5. MERIVIRRAT
# ============================================================

def merivirrat(
    T_meri,
    onko_manner,
    korkeus_m,
    omega_suhteessa_maahan,
    tuuli_u=None,
    tuuli_v=None,
    merivirran_vahvuus=1.0
):
    """
    Yksinkertaistettu valtamerikuljetus.

    Sisältää:

        1. Ekman-tyyppisen tuulivasteen
        2. Termohaliinisen lämpögradienttivasteen
        3. Rajatun meridionaalisen lämpöadvektion
        4. Diffuusion

    Tärkeä ero alkuperäiseen:
    nopeutta ei kerrota suoraan lämpötilalla.

    Sen sijaan nopeus muunnetaan kuljetusosuuden
    parametriksi, jolloin yksi askel ei voi muuttaa
    lämpötilaa epärealistisesti.
    """

    n = T_meri.shape[0]

    lat = np.linspace(
        -np.pi / 2.0,
        np.pi / 2.0,
        n
    )

    LAT, _ = np.meshgrid(
        lat,
        np.arange(n),
        indexing="ij"
    )

    # --------------------------------------------------------
    # 5.1 Ekman
    # --------------------------------------------------------

    if (
        tuuli_u is not None
        and tuuli_v is not None
    ):

        # Pohjoisella pallonpuoliskolla
        # Ekman-suunta oikealle tuulesta.
        #
        # Eteläisellä vasemmalle.
        merkki = np.where(
            LAT >= 0.0,
            1.0,
            -1.0
        )

        ekman_kulma = (
            merkki
            * np.deg2rad(30.0)
        )

        u_ekman = (
            0.025
            * (
                tuuli_u
                * np.cos(ekman_kulma)
                - tuuli_v
                * np.sin(ekman_kulma)
            )
        )

        v_ekman = (
            0.025
            * (
                tuuli_u
                * np.sin(ekman_kulma)
                + tuuli_v
                * np.cos(ekman_kulma)
            )
        )

    else:

        u_ekman = np.zeros_like(
            T_meri
        )

        v_ekman = np.zeros_like(
            T_meri
        )

    # --------------------------------------------------------
    # 5.2 Termohaliininen komponentti
    # --------------------------------------------------------

    R = 6.371e6

    dlat = np.pi / max(n - 1, 1)

    dy = R * dlat

    T_zonaali = T_meri.mean(
        axis=1
    )

    # np.gradient palauttaa K / gridpoint
    grad_T = np.gradient(
        T_zonaali,
        dy
    )

    # Tyypillinen Earth-like gradient
    # 15 K / 6371 km
    grad_ref = (
        15.0
        / 6.371e6
    )

    grad_norm = (
        grad_T
        / (grad_ref + 1e-12)
    )

    # 0.01 on alkuperäisen mallin termohaliininen
    # "vahvuus", mutta nopeus rajoitetaan.
    v_termo_1d = (
        -0.01
        * grad_norm
    )

    v_termo_1d = np.clip(
        v_termo_1d,
        -0.10,
        0.10
    )

    v_termo = (
        v_termo_1d[:, None]
        * np.ones((1, n))
    )

    # --------------------------------------------------------
    # 5.3 Kokonaisvirta
    # --------------------------------------------------------

    u_meri = (
        merivirran_vahvuus
        * u_ekman
    )

    v_meri = (
        merivirran_vahvuus
        * (
            v_ekman
            + v_termo
        )
    )

    # Merellä vain
    u_meri = np.where(
        onko_manner,
        0.0,
        u_meri
    )

    v_meri = np.where(
        onko_manner,
        0.0,
        v_meri
    )

    # Jäätynyt meri ei kuljeta tässä
    # yksinkertaistetussa mallissa.
    jaata = (
        T_meri < 271.15
    )

    u_meri = np.where(
        jaata,
        0.0,
        u_meri
    )

    v_meri = np.where(
        jaata,
        0.0,
        v_meri
    )

    # --------------------------------------------------------
    # 5.4 Lämpötilan advektio
    # --------------------------------------------------------
    #
    # Käytetään rajoitettua sekoitusosuutta:
    #
    #   alpha = dt * velocity / dy
    #
    # mutta tässä alpha parametrisoidaan suoraan.
    #
    # Tämä on paljon vakaampi kuin:
    #
    #   T -= 0.05 * velocity * gradient
    #

    T_uusi = T_meri.copy()

    # Meridionaalinen kuljetus
    for i in range(1, n - 1):

        v_i = v_meri[i, :]

        # Kuljetusosuus:
        alpha = np.clip(
            0.08
            * np.abs(v_i) / 0.10,
            0.0,
            0.15
        )

        T_yla = T_meri[i + 1, :]
        T_ala = T_meri[i - 1, :]
        T_nyt = T_meri[i, :]

        # Upwind-tyyppinen ero
        delta = np.where(
            v_i > 0.0,
            T_ala - T_nyt,
            T_yla - T_nyt
        )

        T_uusi[i, :] += (
            alpha * delta
        )

    # --------------------------------------------------------
    # 5.5 Meridionaalinen diffuusio
    # --------------------------------------------------------

    kappa = (
        0.015
        * (
            1.0
            + np.abs(v_meri) / 0.10
        )
    )

    kappa = np.clip(
        kappa,
        0.0,
        0.10
    )

    T_diff = vakaa_diffuusio_1d(
        T_uusi,
        kappa,
        akseli=0
    )

    # Meri vain
    T_diff = np.where(
        onko_manner,
        T_meri,
        T_diff
    )

    # --------------------------------------------------------
    # 5.6 Energiaa ei saa syntyä merivirroista
    # --------------------------------------------------------

    paino = np.cos(LAT)

    ennen = (
        np.sum(
            T_meri * paino
        )
        / (
            np.sum(paino)
            + 1e-12
        )
    )

    jalkeen = (
        np.sum(
            T_diff * paino
        )
        / (
            np.sum(paino)
            + 1e-12
        )
    )

    # Pieni korjaus koko planeetan lämpötilakeskiarvoon.
    # Koska kuljetus on sisäinen prosessi, sen ei pidä
    # muuttaa globaalia energiaa.
    T_diff += (
        ennen - jalkeen
    )

    return (
        T_diff,
        u_meri,
        v_meri
    )


# ============================================================
# 6. LÄMPÖKAPASITEETTI
# ============================================================

def lampokapasiteetti(
    onko_manner,
    korkeus_m,
    meren_syvyys=10.0
):
    """
    Tehollinen pintakerroksen lämpökapasiteetti.

    Meri:
        4.0e6 J/(m² K) per metri vettä

    10 m:
        4.0e7 J/(m² K)

    Manner:
        2 m aktiivista maaperää
        ~2.0e6 J/(m² K) per kokonaiskerros

    Tämä on tarkoituksellinen climate-box-model
    parametrisaatio.
    """

    C_meri = (
        4.0e6
        * max(float(meren_syvyys), 1.0)
    )

    C_manner = (
        2.0e6
        * 2.0
    )

    # Korkeus vaikuttaa vain hieman mannermaan
    # lämpökapasiteettiin.
    korkeuskerroin = (
        1.0
        - 0.10
        * np.clip(
            korkeus_m / 4000.0,
            0.0,
            1.0
        )
    )

    C_manner *= korkeuskerroin

    return np.where(
        onko_manner,
        C_manner,
        C_meri
    )


# ============================================================
# 7. LÄMPÖTILAN AIKAINTEGRAATIO
# ============================================================

def paivita_lampotila(
    T,
    T_tavoite,
    C,
    dt,
    max_muutos_K=3.0
):
    """
    Päivittää pintalämpötilan kohti kuljetuksen ja
    säteilyn määräämää tavoitetta.

    Muutos rajataan yhdellä aikastepillä.

    Tämä estää napa-alueita tai pieniä mannerlaikkuja
    hyppäämästä epärealistisesti kymmeniä asteita.
    """

    # Käytetään yksinkertaista relaxaatiota.
    #
    # τ = C / (4 sigma T³)
    #
    # Tämä on Planck-tyyppinen lineaaristettu
    # säteilyaikaskaala.

    sigma = 5.670374419e-8

    T_safe = np.maximum(
        T,
        150.0
    )

    tau = (
        C
        / (
            4.0
            * sigma
            * T_safe ** 3
            + 1e-12
        )
    )

    alpha = np.clip(
        dt / tau,
        0.0,
        1.0
    )

    delta = (
        T_tavoite - T
    )

    # Fysikaalisesti karkea mutta numeerisesti
    # turvallinen kuukausitason rajoitus.
    delta = np.clip(
        delta,
        -max_muutos_K,
        max_muutos_K
    )

    T_uusi = (
        T
        + alpha * delta
    )

    return np.clip(
        T_uusi,
        50.0,
        1000.0
    )


# ============================================================
# 6. KOSTEUS, PILVET JA SADE
# ============================================================

def saturoitunut_vesihoyrypaine(T):
    """
    Clausius-Clapeyron-tyyppinen kylläisen vesihöyryn paine.

    T kelvineinä.
    Palauttaa Pa.

    Kaava on tarkoitettu planeettamallin parametrisaatioksi,
    ei tarkaksi ilmakehän kosteustermodynamiikaksi.
    """

    T = np.clip(
        np.asarray(T, dtype=float),
        180.0,
        350.0
    )

    return (
        611.2
        * np.exp(
            17.67
            * (T - 273.15)
            / (T - 29.65)
        )
    )


def saturoitunut_sekoitussuhde(
    T,
    p_atm,
    epsilon=0.622
):
    """
    Kylläisen vesihöyryn sekoitussuhde kg/kg.

    r_s = epsilon * e_s / (p-e_s)

    Paine annetaan tässä atm-yksikköinä,
    mutta muutetaan sisäisesti Pa:ksi.
    """

    p = max(
        float(p_atm),
        0.05
    ) * 101325.0

    e_s = saturoitunut_vesihoyrypaine(T)

    # Estetään e_s >= p
    e_s = np.minimum(
        e_s,
        0.95 * p
    )

    r_s = (
        epsilon
        * e_s
        / np.maximum(
            p - e_s,
            1.0
        )
    )

    return np.maximum(
        r_s,
        0.0
    )


def ilmakehan_vesihoyryn_kapasiteetti(
    T,
    p_atm,
    vesimassa_kerroin=12.0
):
    """
    Ilmakehän tehollinen vesihöyrykapasiteetti kg/m².

    Käytetään yksinkertaista:
        q_s * vesimassa_kerroin

    parametrisaatiota.

    Earth-like:
        lämpimässä tropiikissa kapasiteetti on suuri,
        kylmässä napa-alueella pieni.

    Tämä ei ole koko troposfäärin tarkka integraali.
    """

    q_s = saturoitunut_sekoitussuhde(
        T,
        p_atm
    )

    W_sat = (
        vesimassa_kerroin
        * q_s
    )

    return np.clip(
        W_sat,
        0.01,
        100.0
    )


def haihtuminen(
    T,
    I_abs,
    onko_manner,
    korkeus_m,
    p_atm,
    W=None,
    tuuli_u=None,
    tuuli_v=None,
    merivirta_u=None,
    merivirta_v=None,
    dt_paivia=30.0,
    latentti_energia_kerroin=0.08
):
    """
    Laskee pinnalta ilmakehään siirtyvän vesimäärän.

    Palauttaa:
        E_vesi : kg/m² kuukaudessa

    Perustuu:
        - lämpötilaan
        - saatavilla olevaan energiaan
        - pinta-alaan
        - tuuleen
        - merivirtaan

    Tärkeä:
    Haihtuminen EI ole enää suoraan:
        W/m² -> mm

    vaan ensin muodostetaan energiasta vesivuoksi
    latenttilämmön avulla.
    """

    T = np.asarray(
        T,
        dtype=float
    )

    I = np.maximum(
        np.asarray(I_abs, dtype=float),
        0.0
    )

    # --------------------------------------------------------
    # Latenttilämmön höyrystymisenergia
    # --------------------------------------------------------

    # n. 2.45 MJ/kg lämpimässä vedessä.
    L_v = 2.45e6

    # --------------------------------------------------------
    # Lämpötilakerroin
    # --------------------------------------------------------

    f_T = np.clip(
        (T - 250.0) / 35.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Pinta-alan saatavuus
    # --------------------------------------------------------

    # Meri saa suurimman haihtumispotentiaalin.
    f_meri = np.where(
        ~onko_manner,
        1.0,
        0.0
    )

    # Mannermaan haihtuminen pienenee korkealla.
    f_manner = np.where(
        onko_manner,
        0.55
        * (
            1.0
            - 0.70
            * np.clip(
                korkeus_m / 4000.0,
                0.0,
                1.0
            )
        ),
        0.0
    )

    f_pinta = (
        f_meri
        + f_manner
    )

    # --------------------------------------------------------
    # Ilman suhteellinen kosteus
    # --------------------------------------------------------

    if W is None:
        f_kosteus = np.ones_like(T)
    else:
        W_sat = ilmakehan_vesihoyryn_kapasiteetti(
            T,
            p_atm
        )
        RH = np.clip(
            np.asarray(W, dtype=float)
            / np.maximum(W_sat, 1e-6),
            0.0,
            1.0
        )
        # Haihtuminen hidastuu ilman lähestyessä kylläisyyttä,
        # mutta ei mene täysin nollaan.
        f_kosteus = 0.20 + 0.80 * (1.0 - RH)

    # --------------------------------------------------------
    # Tuulen vaikutus
    # --------------------------------------------------------

    f_tuuli = np.ones_like(T)

    if tuuli_u is not None:
        f_tuuli += (
            0.10
            * np.clip(
                np.abs(tuuli_u) / 30.0,
                0.0,
                1.0
            )
        )

    if tuuli_v is not None:
        f_tuuli += (
            0.10
            * np.clip(
                np.abs(tuuli_v) / 5.0,
                0.0,
                1.0
            )
        )

    # --------------------------------------------------------
    # Merivirran vaikutus
    # --------------------------------------------------------

    f_meri_virta = np.ones_like(T)

    if merivirta_u is not None:
        f_meri_virta += (
            0.05
            * np.clip(
                np.abs(merivirta_u) / 0.1,
                0.0,
                1.0
            )
        )

    if merivirta_v is not None:
        f_meri_virta += (
            0.05
            * np.clip(
                np.abs(merivirta_v) / 0.1,
                0.0,
                1.0
            )
        )

    # --------------------------------------------------------
    # Käytettävissä oleva energia
    # --------------------------------------------------------

    # Haihtumiseen käytetään vain pieni osa pintaan
    # tulevasta energiasta.
    #
    # Tämä on parametrisaatio, jolla vältetään se,
    # että koko nettosäteily muuttuisi vedeksi.
    F_evap = (
        latentti_energia_kerroin
        * I
        * f_T
        * f_pinta
        * f_kosteus
        * f_tuuli
        * f_meri_virta
    )

    F_evap = np.maximum(
        F_evap,
        0.0
    )

    # --------------------------------------------------------
    # W/m² -> kg/m²/kk
    # --------------------------------------------------------

    E = (
        F_evap
        * dt_paivia
        * 86400.0
        / L_v
    )

    # 1 kg/m² vettä = 1 mm sadetta.
    #
    # Haihtumisen ei kuitenkaan anneta ylittää
    # kohtuullista kuukausiarvoa.
    E = np.clip(
        E,
        0.0,
        250.0
    )

    return E


def kosteuden_kuljetus(
    W,
    tuuli_v=None,
    n_iter=4,
    diffuusio=0.08
):
    """
    Kuljettaa vesihöyryä meridionaalisesti ja diffundoivasti.

    W:
        kg/m²

    Tärkeä:
        kokonaisvesimäärää ei kasvateta.

    Raja-arvot estävät eksplisiittisen diffuusion
    numeerisen epävakauden.
    """

    W = np.maximum(
        np.asarray(W, dtype=float),
        0.0
    )

    n = W.shape[0]

    W_uusi = W.copy()

    # --------------------------------------------------------
    # Tuulen aiheuttama meridionaalinen kuljetus
    # --------------------------------------------------------

    if tuuli_v is not None:

        for _ in range(
            max(1, int(n_iter))
        ):

            W_vanha = W_uusi.copy()
            W_uusi = W_vanha.copy()

            for i in range(1, n - 1):

                v_i = tuuli_v[i, :]

                # Kuljetusosuus
                alpha = np.clip(
                    0.04
                    * np.abs(v_i) / 5.0,
                    0.0,
                    0.08
                )

                delta = np.where(
                    v_i > 0.0,
                    W_vanha[i - 1, :]
                    - W_vanha[i, :],

                    W_vanha[i + 1, :]
                    - W_vanha[i, :]
                )

                W_uusi[i, :] += (
                    alpha * delta
                )

    # --------------------------------------------------------
    # Diffuusio
    # --------------------------------------------------------

    kappa = np.clip(
        float(diffuusio),
        0.0,
        0.20
    )

    for _ in range(
        max(1, int(n_iter))
    ):

        W_vanha = W_uusi.copy()

        W_uusi = vakaa_diffuusio_1d(
            W_vanha,
            np.full_like(
                W_vanha,
                kappa
            ),
            akseli=0
        )

        W_uusi = np.maximum(
            W_uusi,
            0.0
        )

    return W_uusi


def kosteuden_pinnan_rajoitus(
    W,
    T,
    p_atm,
    maa_maski=None
):
    """
    Rajoittaa vesihöyryn kylläiseen kapasiteettiin.

    Ylimenevä vesi palautetaan sateeksi.

    Palauttaa:
        W_rajoitettu
        sade_kg_m2
    """

    W = np.maximum(
        np.asarray(W, dtype=float),
        0.0
    )

    W_sat = ilmakehan_vesihoyryn_kapasiteetti(
        T,
        p_atm
    )

    ylimaarainen = np.maximum(
        W - W_sat,
        0.0
    )

    W_rajoitettu = np.minimum(
        W,
        W_sat
    )

    # Jos maa_maski annetaan, sallitaan hieman
    # suurempi paikallinen kosteus ennen sadetta.
    if maa_maski is not None:

        W_rajoitettu = np.where(
            maa_maski,
            np.minimum(
                W,
                1.15 * W_sat
            ),
            W_rajoitettu
        )

        ylimaarainen = np.maximum(
            W - W_rajoitettu,
            0.0
        )

    return (
        W_rajoitettu,
        ylimaarainen
    )


def pilvisyys_vesihoyrysta(
    W,
    T,
    onko_manner,
    p_atm=1.0
):
    """
    Muuttaa vesihöyrymäärän pilvisyydeksi 0...1.

    Perustaso:
        meri     ~0.45
        manner  ~0.30

    Kosteus nostaa pilvisyyttä.

    Pilvisyys ei kuitenkaan kasva lineaarisesti
    rajatta vesimäärän mukana.
    """

    W = np.maximum(
        W,
        0.0
    )

    W_sat = ilmakehan_vesihoyryn_kapasiteetti(
        T,
        p_atm
    )

    RH = np.clip(
        W / np.maximum(
            W_sat,
            1e-6
        ),
        0.0,
        1.5
    )

    # Suhteellinen kosteus
    RH_eff = np.clip(
        RH,
        0.0,
        1.0
    )

    # Peruspilvisyys käyttäjän alkuperäisen mallin
    # mukaisesti.
    perustaso = np.where(
        onko_manner,
        0.30,
        0.45
    )

    # Pilvien kasvu kosteuden mukana
    pilvisyys = (
        perustaso
        + 0.45
        * RH_eff
        * np.clip(
            (T - 245.0) / 45.0,
            0.0,
            1.0
        )
    )

    return np.clip(
        pilvisyys,
        0.0,
        1.0
    )


def sade_vesihoyrysta(
    ylimaarainen,
    W,
    T,
    onko_manner,
    korkeus_m,
    pilvisyys,
    sade_tehokkuus=1.0
):
    """
    Muuttaa ylimääräisen vesihöyryn sateeksi.

    Sade syntyy pääasiassa silloin, kun:
        W > W_sat

    Lisäksi pilvisyys tehostaa kondensaatiota.

    Palauttaa:
        sade kg/m²/kk = mm/kk
    """

    ylimaarainen = np.maximum(
        ylimaarainen,
        0.0
    )

    # Pilvisyyden vaikutus
    pilvi_kerroin = (
        0.35
        + 0.65
        * np.clip(
            pilvisyys,
            0.0,
            1.0
        )
    )

    # Orogrfia
    oro = np.where(
        onko_manner,
        1.0
        + 0.20
        * np.clip(
            korkeus_m / 4000.0,
            0.0,
            1.0
        ),
        1.0
    )

    # Kylmä ilma ei pysty tuottamaan suurta
    # hetkellistä sadetta samalla tavalla kuin lämmin.
    f_T = np.clip(
        (T - 245.0) / 35.0,
        0.05,
        1.0
    )

    sade = (
        ylimaarainen
        * sade_tehokkuus
        * pilvi_kerroin
        * oro
        * f_T
    )

    return np.clip(
        sade,
        0.0,
        500.0
    )


def kosteus_pilvet_sade(
    T,
    I_abs,
    onko_manner,
    korkeus_m,
    p_atm,
    tuuli_v=None,
    merivirta_v=None,
    merivirta_u=None,
    edellinen_vesihoyry=None,
    dt_paivia=30.0,
    latentti_energia_kerroin=0.08
):
    """
    Koko vesikierron yksi kuukausiaskel.

    Prosessi:

        1. Haihtuminen
        2. Vesihöyryn lisäys ilmakehään
        3. Tuulikuljetus
        4. Diffuusio
        5. Kylläisyysrajoitus
        6. Pilvisyys
        7. Sade

    Palauttaa:

        sade        : mm/kk
        pilvisyys   : 0...1
        vesihoyry   : kg/m²
        haihtuminen : mm/kk
    """

    n = T.shape[0]

    # --------------------------------------------------------
    # 1. Haihtuminen
    # --------------------------------------------------------

    E = haihtuminen(
        T=T,
        I_abs=I_abs,
        onko_manner=onko_manner,
        korkeus_m=korkeus_m,
        p_atm=p_atm,
        W=edellinen_vesihoyry,
        tuuli_u=None,
        tuuli_v=tuuli_v,
        merivirta_u=merivirta_u,
        merivirta_v=merivirta_v,
        dt_paivia=dt_paivia,
        latentti_energia_kerroin=latentti_energia_kerroin
    )

    # --------------------------------------------------------
    # 2. Alkuperäinen vesihöyry
    # --------------------------------------------------------

    if edellinen_vesihoyry is None:

        # Aloitus suhteellisen kuivasta ilmakehästä.
        W = (
            0.35
            * ilmakehan_vesihoyryn_kapasiteetti(
                T,
                p_atm
            )
        )

    else:

        W = np.maximum(
            edellinen_vesihoyry,
            0.0
        )

    W += E

    # --------------------------------------------------------
    # 3. Kuljetus
    # --------------------------------------------------------

    W = kosteuden_kuljetus(
        W,
        tuuli_v=tuuli_v,
        n_iter=4,
        diffuusio=0.08
    )

    # --------------------------------------------------------
    # 4. Kylläisyys
    # --------------------------------------------------------

    W, ylimaarainen = (
        kosteuden_pinnan_rajoitus(
            W,
            T,
            p_atm,
            maa_maski=onko_manner
        )
    )

    # --------------------------------------------------------
    # 5. Pilvet
    # --------------------------------------------------------

    pilvisyys = pilvisyys_vesihoyrysta(
        W,
        T,
        onko_manner,
        p_atm=p_atm
    )

    # --------------------------------------------------------
    # 6. Sade
    # --------------------------------------------------------

    sade = sade_vesihoyrysta(
        ylimaarainen=ylimaarainen,
        W=W,
        T=T,
        onko_manner=onko_manner,
        korkeus_m=korkeus_m,
        pilvisyys=pilvisyys
    )

    # --------------------------------------------------------
    # 7. Sade poistaa vettä ilmakehästä
    # --------------------------------------------------------

    W -= sade

    W = np.maximum(
        W,
        0.0
    )

    return (
        sade,
        pilvisyys,
        W,
        E
    )

# ============================================================
# 6. KOSTEUS, PILVET JA SADE
# ============================================================

def saturoitunut_vesihoyrypaine(T):
    """
    Clausius-Clapeyron-tyyppinen kylläisen vesihöyryn paine.

    T kelvineinä.
    Palauttaa Pa.

    Kaava on tarkoitettu planeettamallin parametrisaatioksi,
    ei tarkaksi ilmakehän kosteustermodynamiikaksi.
    """

    T = np.clip(
        np.asarray(T, dtype=float),
        180.0,
        350.0
    )

    return (
        611.2
        * np.exp(
            17.67
            * (T - 273.15)
            / (T - 29.65)
        )
    )


def saturoitunut_sekoitussuhde(
    T,
    p_atm,
    epsilon=0.622
):
    """
    Kylläisen vesihöyryn sekoitussuhde kg/kg.

    r_s = epsilon * e_s / (p-e_s)

    Paine annetaan tässä atm-yksikköinä,
    mutta muutetaan sisäisesti Pa:ksi.
    """

    p = max(
        float(p_atm),
        0.05
    ) * 101325.0

    e_s = saturoitunut_vesihoyrypaine(T)

    # Estetään e_s >= p
    e_s = np.minimum(
        e_s,
        0.95 * p
    )

    r_s = (
        epsilon
        * e_s
        / np.maximum(
            p - e_s,
            1.0
        )
    )

    return np.maximum(
        r_s,
        0.0
    )


def ilmakehan_vesihoyryn_kapasiteetti(
    T,
    p_atm,
    vesimassa_kerroin=12.0
):
    """
    Ilmakehän tehollinen vesihöyrykapasiteetti kg/m².

    Käytetään yksinkertaista:
        q_s * vesimassa_kerroin

    parametrisaatiota.

    Earth-like:
        lämpimässä tropiikissa kapasiteetti on suuri,
        kylmässä napa-alueella pieni.

    Tämä ei ole koko troposfäärin tarkka integraali.
    """

    q_s = saturoitunut_sekoitussuhde(
        T,
        p_atm
    )

    W_sat = (
        vesimassa_kerroin
        * q_s
    )

    return np.clip(
        W_sat,
        0.01,
        100.0
    )


def haihtuminen(
    T,
    I_abs,
    onko_manner,
    korkeus_m,
    p_atm,
    W=None,
    tuuli_u=None,
    tuuli_v=None,
    merivirta_u=None,
    merivirta_v=None,
    dt_paivia=30.0,
    latentti_energia_kerroin=0.08
):
    """
    Laskee pinnalta ilmakehään siirtyvän vesimäärän.

    Palauttaa:
        E_vesi : kg/m² kuukaudessa

    Perustuu:
        - lämpötilaan
        - saatavilla olevaan energiaan
        - pinta-alaan
        - tuuleen
        - merivirtaan

    Tärkeä:
    Haihtuminen EI ole enää suoraan:
        W/m² -> mm

    vaan ensin muodostetaan energiasta vesivuoksi
    latenttilämmön avulla.
    """

    T = np.asarray(
        T,
        dtype=float
    )

    I = np.maximum(
        np.asarray(I_abs, dtype=float),
        0.0
    )

    # --------------------------------------------------------
    # Latenttilämmön höyrystymisenergia
    # --------------------------------------------------------

    # n. 2.45 MJ/kg lämpimässä vedessä.
    L_v = 2.45e6

    # --------------------------------------------------------
    # Lämpötilakerroin
    # --------------------------------------------------------

    f_T = np.clip(
        (T - 250.0) / 35.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Pinta-alan saatavuus
    # --------------------------------------------------------

    # Meri saa suurimman haihtumispotentiaalin.
    f_meri = np.where(
        ~onko_manner,
        1.0,
        0.0
    )

    # Mannermaan haihtuminen pienenee korkealla.
    f_manner = np.where(
        onko_manner,
        0.55
        * (
            1.0
            - 0.70
            * np.clip(
                korkeus_m / 4000.0,
                0.0,
                1.0
            )
        ),
        0.0
    )

    f_pinta = (
        f_meri
        + f_manner
    )

    # --------------------------------------------------------
    # Ilman suhteellinen kosteus
    # --------------------------------------------------------

    if W is None:
        f_kosteus = np.ones_like(T)
    else:
        W_sat = ilmakehan_vesihoyryn_kapasiteetti(
            T,
            p_atm
        )
        RH = np.clip(
            np.asarray(W, dtype=float)
            / np.maximum(W_sat, 1e-6),
            0.0,
            1.0
        )
        # Haihtuminen hidastuu ilman lähestyessä kylläisyyttä,
        # mutta ei mene täysin nollaan.
        f_kosteus = 0.20 + 0.80 * (1.0 - RH)

    # --------------------------------------------------------
    # Tuulen vaikutus
    # --------------------------------------------------------

    f_tuuli = np.ones_like(T)

    if tuuli_u is not None:
        f_tuuli += (
            0.10
            * np.clip(
                np.abs(tuuli_u) / 30.0,
                0.0,
                1.0
            )
        )

    if tuuli_v is not None:
        f_tuuli += (
            0.10
            * np.clip(
                np.abs(tuuli_v) / 5.0,
                0.0,
                1.0
            )
        )

    # --------------------------------------------------------
    # Merivirran vaikutus
    # --------------------------------------------------------

    f_meri_virta = np.ones_like(T)

    if merivirta_u is not None:
        f_meri_virta += (
            0.05
            * np.clip(
                np.abs(merivirta_u) / 0.1,
                0.0,
                1.0
            )
        )

    if merivirta_v is not None:
        f_meri_virta += (
            0.05
            * np.clip(
                np.abs(merivirta_v) / 0.1,
                0.0,
                1.0
            )
        )

    # --------------------------------------------------------
    # Käytettävissä oleva energia
    # --------------------------------------------------------

    # Haihtumiseen käytetään vain pieni osa pintaan
    # tulevasta energiasta.
    #
    # Tämä on parametrisaatio, jolla vältetään se,
    # että koko nettosäteily muuttuisi vedeksi.
    F_evap = (
        latentti_energia_kerroin
        * I
        * f_T
        * f_pinta
        * f_kosteus
        * f_tuuli
        * f_meri_virta
    )

    F_evap = np.maximum(
        F_evap,
        0.0
    )

    # --------------------------------------------------------
    # W/m² -> kg/m²/kk
    # --------------------------------------------------------

    E = (
        F_evap
        * dt_paivia
        * 86400.0
        / L_v
    )

    # 1 kg/m² vettä = 1 mm sadetta.
    #
    # Haihtumisen ei kuitenkaan anneta ylittää
    # kohtuullista kuukausiarvoa.
    E = np.clip(
        E,
        0.0,
        250.0
    )

    return E


def kosteuden_kuljetus(
    W,
    tuuli_v=None,
    n_iter=4,
    diffuusio=0.08
):
    """
    Kuljettaa vesihöyryä meridionaalisesti ja diffundoivasti.

    W:
        kg/m²

    Tärkeä:
        kokonaisvesimäärää ei kasvateta.

    Raja-arvot estävät eksplisiittisen diffuusion
    numeerisen epävakauden.
    """

    W = np.maximum(
        np.asarray(W, dtype=float),
        0.0
    )

    n = W.shape[0]

    W_uusi = W.copy()

    # --------------------------------------------------------
    # Tuulen aiheuttama meridionaalinen kuljetus
    # --------------------------------------------------------

    if tuuli_v is not None:

        for _ in range(
            max(1, int(n_iter))
        ):

            W_vanha = W_uusi.copy()
            W_uusi = W_vanha.copy()

            for i in range(1, n - 1):

                v_i = tuuli_v[i, :]

                # Kuljetusosuus
                alpha = np.clip(
                    0.04
                    * np.abs(v_i) / 5.0,
                    0.0,
                    0.08
                )

                delta = np.where(
                    v_i > 0.0,
                    W_vanha[i - 1, :]
                    - W_vanha[i, :],

                    W_vanha[i + 1, :]
                    - W_vanha[i, :]
                )

                W_uusi[i, :] += (
                    alpha * delta
                )

    # --------------------------------------------------------
    # Diffuusio
    # --------------------------------------------------------

    kappa = np.clip(
        float(diffuusio),
        0.0,
        0.20
    )

    for _ in range(
        max(1, int(n_iter))
    ):

        W_vanha = W_uusi.copy()

        W_uusi = vakaa_diffuusio_1d(
            W_vanha,
            np.full_like(
                W_vanha,
                kappa
            ),
            akseli=0
        )

        W_uusi = np.maximum(
            W_uusi,
            0.0
        )

    return W_uusi


def kosteuden_pinnan_rajoitus(
    W,
    T,
    p_atm,
    maa_maski=None
):
    """
    Rajoittaa vesihöyryn kylläiseen kapasiteettiin.

    Ylimenevä vesi palautetaan sateeksi.

    Palauttaa:
        W_rajoitettu
        sade_kg_m2
    """

    W = np.maximum(
        np.asarray(W, dtype=float),
        0.0
    )

    W_sat = ilmakehan_vesihoyryn_kapasiteetti(
        T,
        p_atm
    )

    ylimaarainen = np.maximum(
        W - W_sat,
        0.0
    )

    W_rajoitettu = np.minimum(
        W,
        W_sat
    )

    # Jos maa_maski annetaan, sallitaan hieman
    # suurempi paikallinen kosteus ennen sadetta.
    if maa_maski is not None:

        W_rajoitettu = np.where(
            maa_maski,
            np.minimum(
                W,
                1.15 * W_sat
            ),
            W_rajoitettu
        )

        ylimaarainen = np.maximum(
            W - W_rajoitettu,
            0.0
        )

    return (
        W_rajoitettu,
        ylimaarainen
    )


def pilvisyys_vesihoyrysta(
    W,
    T,
    onko_manner,
    p_atm=1.0
):
    """
    Muuttaa vesihöyrymäärän pilvisyydeksi 0...1.

    Perustaso:
        meri     ~0.45
        manner  ~0.30

    Kosteus nostaa pilvisyyttä.

    Pilvisyys ei kuitenkaan kasva lineaarisesti
    rajatta vesimäärän mukana.
    """

    W = np.maximum(
        W,
        0.0
    )

    W_sat = ilmakehan_vesihoyryn_kapasiteetti(
        T,
        p_atm
    )

    RH = np.clip(
        W / np.maximum(
            W_sat,
            1e-6
        ),
        0.0,
        1.5
    )

    # Suhteellinen kosteus
    RH_eff = np.clip(
        RH,
        0.0,
        1.0
    )

    # Peruspilvisyys käyttäjän alkuperäisen mallin
    # mukaisesti.
    perustaso = np.where(
        onko_manner,
        0.30,
        0.45
    )

    # Pilvien kasvu kosteuden mukana
    pilvisyys = (
        perustaso
        + 0.45
        * RH_eff
        * np.clip(
            (T - 245.0) / 45.0,
            0.0,
            1.0
        )
    )

    return np.clip(
        pilvisyys,
        0.0,
        1.0
    )


def sade_vesihoyrysta(
    ylimaarainen,
    W,
    T,
    onko_manner,
    korkeus_m,
    pilvisyys,
    sade_tehokkuus=1.0
):
    """
    Muuttaa ylimääräisen vesihöyryn sateeksi.

    Sade syntyy pääasiassa silloin, kun:
        W > W_sat

    Lisäksi pilvisyys tehostaa kondensaatiota.

    Palauttaa:
        sade kg/m²/kk = mm/kk
    """

    ylimaarainen = np.maximum(
        ylimaarainen,
        0.0
    )

    # Pilvisyyden vaikutus
    pilvi_kerroin = (
        0.35
        + 0.65
        * np.clip(
            pilvisyys,
            0.0,
            1.0
        )
    )

    # Orogrfia
    oro = np.where(
        onko_manner,
        1.0
        + 0.20
        * np.clip(
            korkeus_m / 4000.0,
            0.0,
            1.0
        ),
        1.0
    )

    # Kylmä ilma ei pysty tuottamaan suurta
    # hetkellistä sadetta samalla tavalla kuin lämmin.
    f_T = np.clip(
        (T - 245.0) / 35.0,
        0.05,
        1.0
    )

    sade = (
        ylimaarainen
        * sade_tehokkuus
        * pilvi_kerroin
        * oro
        * f_T
    )

    return np.clip(
        sade,
        0.0,
        500.0
    )


def kosteus_pilvet_sade(
    T,
    I_abs,
    onko_manner,
    korkeus_m,
    p_atm,
    tuuli_v=None,
    merivirta_v=None,
    merivirta_u=None,
    edellinen_vesihoyry=None,
    dt_paivia=30.0,
    latentti_energia_kerroin=0.08
):
    """
    Koko vesikierron yksi kuukausiaskel.

    Prosessi:

        1. Haihtuminen
        2. Vesihöyryn lisäys ilmakehään
        3. Tuulikuljetus
        4. Diffuusio
        5. Kylläisyysrajoitus
        6. Pilvisyys
        7. Sade

    Palauttaa:

        sade        : mm/kk
        pilvisyys   : 0...1
        vesihoyry   : kg/m²
        haihtuminen : mm/kk
    """

    n = T.shape[0]

    # --------------------------------------------------------
    # 1. Haihtuminen
    # --------------------------------------------------------

    E = haihtuminen(
        T=T,
        I_abs=I_abs,
        onko_manner=onko_manner,
        korkeus_m=korkeus_m,
        p_atm=p_atm,
        W=edellinen_vesihoyry,
        tuuli_u=None,
        tuuli_v=tuuli_v,
        merivirta_u=merivirta_u,
        merivirta_v=merivirta_v,
        dt_paivia=dt_paivia,
        latentti_energia_kerroin=latentti_energia_kerroin
    )

    # --------------------------------------------------------
    # 2. Alkuperäinen vesihöyry
    # --------------------------------------------------------

    if edellinen_vesihoyry is None:

        # Aloitus suhteellisen kuivasta ilmakehästä.
        W = (
            0.35
            * ilmakehan_vesihoyryn_kapasiteetti(
                T,
                p_atm
            )
        )

    else:

        W = np.maximum(
            edellinen_vesihoyry,
            0.0
        )

    W += E

    # --------------------------------------------------------
    # 3. Kuljetus
    # --------------------------------------------------------

    W = kosteuden_kuljetus(
        W,
        tuuli_v=tuuli_v,
        n_iter=4,
        diffuusio=0.08
    )

    # --------------------------------------------------------
    # 4. Kylläisyys
    # --------------------------------------------------------

    W, ylimaarainen = (
        kosteuden_pinnan_rajoitus(
            W,
            T,
            p_atm,
            maa_maski=onko_manner
        )
    )

    # --------------------------------------------------------
    # 5. Pilvet
    # --------------------------------------------------------

    pilvisyys = pilvisyys_vesihoyrysta(
        W,
        T,
        onko_manner,
        p_atm=p_atm
    )

    # --------------------------------------------------------
    # 6. Sade
    # --------------------------------------------------------

    sade = sade_vesihoyrysta(
        ylimaarainen=ylimaarainen,
        W=W,
        T=T,
        onko_manner=onko_manner,
        korkeus_m=korkeus_m,
        pilvisyys=pilvisyys
    )

    # --------------------------------------------------------
    # 7. Sade poistaa vettä ilmakehästä
    # --------------------------------------------------------

    W -= sade

    W = np.maximum(
        W,
        0.0
    )

    return (
        sade,
        pilvisyys,
        W,
        E
    )


# ============================================================
# 8. PÄÄOHJELMA
# ============================================================

def aja_malli(
    n=32,
    S=1361.0,
    mvelp=1.0,
    tilt=np.deg2rad(23.44),
    vuoden_pituus_maan_vuosina=1.0,
    p_atm=1.0,
    p_co2_ppm=280.0,
    rp_earths=None,
    omega_suhteessa_maahan=1.0,
    seed=42,
    vuosia=20,
    kuljetus_vahvuus=0.08,
    latentti_kerroin=0.08,
    n_iter=3,
    debug=False,
):
    """
    Planeetan ilmastoboksimalli.

    Tärkeimmät tilamuuttujat:

        T              pintalämpötila K
        W              ilmakehän vesihöyry kg/m²
        pilvisyys      0...1

    Mallin aikasteppi:
        noin 30 päivää.

    Viimeinen vuosi tallennetaan tarkempaa analyysiä varten.
    """

    if rp_earths is None:
        rp_earths = mvelp ** 0.27

    # --------------------------------------------------------
    # Pinta
    # --------------------------------------------------------

    _, onko_manner, korkeus_m = (
        luo_fraktaalipinta(
            n=n,
            seed=seed
        )
    )

    # --------------------------------------------------------
    # Leveysasteet
    # --------------------------------------------------------

    lat = np.linspace(
        -np.pi / 2.0,
        np.pi / 2.0,
        n
    )

    LAT, _ = np.meshgrid(
        lat,
        np.arange(n),
        indexing="ij"
    )

    paino_cos = np.cos(LAT)

    # --------------------------------------------------------
    # Aika
    # --------------------------------------------------------

    vuoden_pituus_paivina = (
        365.25
        * vuoden_pituus_maan_vuosina
    )

    dt_paivia = (
        30.0
        * vuoden_pituus_maan_vuosina
    )

    dt = (
        dt_paivia
        * 86400.0
    )

    # --------------------------------------------------------
    # Alkutila
    # --------------------------------------------------------

    T = np.full(
        (n, n),
        288.0,
        dtype=float
    )

    pilvisyys = np.where(
        onko_manner,
        0.30,
        0.45
    ).astype(float)

    W = (
        0.35
        * ilmakehan_vesihoyryn_kapasiteetti(
            T,
            p_atm
        )
    )

    C = lampokapasiteetti(
        onko_manner,
        korkeus_m
    )

    # --------------------------------------------------------
    # Tallennus
    # --------------------------------------------------------

    kk_lampotilat = []
    kk_sateet = []
    kk_pilvisyys = []
    kk_tuuli_u = []
    kk_tuuli_v = []
    kk_merivirta_u = []
    kk_merivirta_v = []
    kk_vesihoyry = []
    kk_haihtuminen = []
    kk_soluja = []

    vuosi_keskilampo = []
    vuosi_sademaarat = []
    vuosi_pilvisyys = []

    kuukausia_yhteensa = (
        int(round(vuosia * 12))
    )

    # --------------------------------------------------------
    # Aikasilmukka
    # --------------------------------------------------------

    for kk in range(
        kuukausia_yhteensa
    ):

        kk_vuodessa = kk % 12

        # Kuukauden keskikohta vuoden sisällä
        paiva = (
            (kk_vuodessa + 0.5)
            * vuoden_pituus_paivina
            / 12.0
        )

        dekl = aurinko_deklinaatio(
            paiva,
            tilt,
            vuoden_pituus_paivina
        )

        # ----------------------------------------------------
        # Päivittäinen keskimääräinen insolaation voimakkuus
        # ----------------------------------------------------

        I_kk = (
            paivittainen_insolaatio_vektori(
                LAT,
                dekl,
                S,
                n_tuntia=96
            )
        )

        # HUOM:
        # Latenttia lämpöä ei enää vähennetä suoraan
        # päiväntasaajan insolaatiosta.
        #
        # Se käsitellään myöhemmin haihtumisena.

        if (
            debug
            and kk == 0
        ):
            tarkista_energiabudjetti(
                I_kk,
                albedo_pinnan_mukaan(
                    onko_manner,
                    korkeus_m,
                    T,
                    pilvisyys
                ),
                LAT,
                otsikko="1. kuukausi"
            )

        # ----------------------------------------------------
        # Säteilyn + albedon iterointi
        # ----------------------------------------------------

        T_sateily = T.copy()

        for _ in range(6):

            albedo = (
                albedo_pinnan_mukaan(
                    onko_manner,
                    korkeus_m,
                    T_sateily,
                    pilvisyys
                )
            )

            T_uusi = np.zeros_like(
                T
            )

            for i in range(n):
                for j in range(n):

                    T_uusi[i, j] = (
                        energia_tasapaino(
                            I_kk[i, j],
                            albedo[i, j],
                            p_atm,
                            p_co2_ppm
                        )
                    )

            ero = np.max(
                np.abs(
                    T_uusi - T_sateily
                )
            )

            # Rentoutus estää albedo-lämpötilapalautteen
            # värähtelyä.
            T_sateily = (
                0.60 * T_sateily
                + 0.40 * T_uusi
            )

            if ero < 0.10:
                break

        # ----------------------------------------------------
        # Ilmakehän lämpökuljetus
        # ----------------------------------------------------

        T_atm, u_tuuli, v_tuuli, N_soluja = (
            ilmakehan_kuljetus(
                T_sateily,
                onko_manner,
                omega_suhteessa_maahan,
                vahvuus=kuljetus_vahvuus,
                n_iter=n_iter
            )
        )

        # ----------------------------------------------------
        # Merivirrat
        # ----------------------------------------------------

        T_meri, u_meri, v_meri = (
            merivirrat(
                T_atm,
                onko_manner,
                korkeus_m,
                omega_suhteessa_maahan,
                tuuli_u=u_tuuli,
                tuuli_v=v_tuuli,
                merivirran_vahvuus=1.0
            )
        )

        # ----------------------------------------------------
        # Ilmakehä + meri
        # ----------------------------------------------------
        #
        # Kuljetusten jälkeen muodostetaan paikallinen
        # tavoitelämpötila.
        #
        # Maa reagoi hieman nopeammin,
        # meri hieman hitaammin.

        T_kok = (
            0.50 * T_sateily
            + 0.30 * T_atm
            + 0.20 * T_meri
        )

        # Meri ei kuitenkaan saa hallita manneralueita.
        T_kok = np.where(
            onko_manner,
            0.70 * T_sateily
            + 0.30 * T_atm,
            T_kok
        )

        # ----------------------------------------------------
        # Lämpökapasiteetti
        # ----------------------------------------------------

        T = paivita_lampotila(
            T,
            T_kok,
            C,
            dt,
            max_muutos_K=2.5
        )

        # ----------------------------------------------------
        # Vesikierto
        # ----------------------------------------------------

        (
            sade_kk,
            pilvisyys_uusi,
            W,
            haihtuminen_kk
        ) = kosteus_pilvet_sade(
            T=T,
            I_abs=I_kk,
            onko_manner=onko_manner,
            korkeus_m=korkeus_m,
            p_atm=p_atm,
            tuuli_v=v_tuuli,
            merivirta_v=v_meri,
            merivirta_u=u_meri,
            edellinen_vesihoyry=W,
            dt_paivia=dt_paivia,
            latentti_energia_kerroin=latentti_kerroin
        )

        # ----------------------------------------------------
        # Pilvien ajallinen inertia
        # ----------------------------------------------------

        pilvisyys = (
            0.70 * pilvisyys
            + 0.30 * pilvisyys_uusi
        )

        pilvisyys = np.clip(
            pilvisyys,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Viimeisen vuoden tallennus
        # ----------------------------------------------------

        if kk >= (
            kuukausia_yhteensa - 12
        ):

            kk_lampotilat.append(
                T.copy()
            )

            kk_sateet.append(
                sade_kk.copy()
            )

            kk_pilvisyys.append(
                pilvisyys.copy()
            )

            kk_tuuli_u.append(
                u_tuuli.copy()
            )

            kk_tuuli_v.append(
                v_tuuli.copy()
            )

            kk_merivirta_u.append(
                u_meri.copy()
            )

            kk_merivirta_v.append(
                v_meri.copy()
            )

            kk_vesihoyry.append(
                W.copy()
            )

            kk_haihtuminen.append(
                haihtuminen_kk.copy()
            )

            kk_soluja.append(
                N_soluja
            )

        # ----------------------------------------------------
        # Vuosikeskiarvot
        # ----------------------------------------------------

        if (
            (kk + 1) % 12 == 0
        ):

            T_vuosi = (
                globaali_keskiarvo(
                    T,
                    LAT
                )
            )

            sade_vuosi = (
                globaali_keskiarvo(
                    sade_kk,
                    LAT
                )
            )

            pilvi_vuosi = (
                globaali_keskiarvo(
                    pilvisyys,
                    LAT
                )
            )

            vuosi_keskilampo.append(
                T_vuosi - 273.15
            )

            vuosi_sademaarat.append(
                sade_vuosi
            )

            vuosi_pilvisyys.append(
                pilvi_vuosi
            )

    # --------------------------------------------------------
    # Muodostetaan numpy-taulukot
    # --------------------------------------------------------

    return {
        "T": np.array(
            kk_lampotilat
        ),

        "sade": np.array(
            kk_sateet
        ),

        "pilvisyys": np.array(
            kk_pilvisyys
        ),

        "vesihoyry": np.array(
            kk_vesihoyry
        ),

        "haihtuminen": np.array(
            kk_haihtuminen
        ),

        "tuuli_u": np.array(
            kk_tuuli_u
        ),

        "tuuli_v": np.array(
            kk_tuuli_v
        ),

        "merivirta_u": np.array(
            kk_merivirta_u
        ),

        "merivirta_v": np.array(
            kk_merivirta_v
        ),

        "soluja": np.array(
            kk_soluja
        ),

        "onko_manner":
            onko_manner,

        "korkeus_m":
            korkeus_m,

        "lat":
            lat,

        "paino_cos":
            paino_cos,

        "LAT":
            LAT,

        "rp_earths":
            rp_earths,

        "omega":
            omega_suhteessa_maahan,

        "vuosi_keskilampo":
            np.array(
                vuosi_keskilampo
            ),

        "vuosi_sade":
            np.array(
                vuosi_sademaarat
            ),

        "vuosi_pilvisyys":
            np.array(
                vuosi_pilvisyys
            ),
    }


# ============================================================
# 9. TULOSTEN ANALYYSI
# ============================================================

def analysoi_tulokset(tulos):
    """
    Laskee ja näyttää viimeisen vuoden tärkeimmät globaalit,
    maa-/merialuekohtaiset sekä leveysastekohtaiset suureet.
    """

    T = tulos["T"]                  # [12, n, n], K
    sade = tulos["sade"]            # [12, n, n]
    pilvisyys = tulos["pilvisyys"]  # [12, n, n]
    W = tulos["vesihoyry"]
    haihtuminen = tulos["haihtuminen"]

    LAT = tulos["LAT"]
    lat = tulos["lat"]
    paino = tulos["paino_cos"]

    onko_manner = tulos["onko_manner"]
    korkeus_m = tulos["korkeus_m"]

    # --------------------------------------------------------
    # Apufunktio: globaali pinta-alapainotettu keskiarvo
    # --------------------------------------------------------

    def globaali(taulukko):
        """
        Cos(lat)-painotettu globaali keskiarvo.
        """
        w = paino
        return np.sum(
            taulukko * w
        ) / np.sum(w)

    # --------------------------------------------------------
    # Apufunktio: maa / meri
    # --------------------------------------------------------

    def aluekeskiarvo(taulukko, maski):
        """
        Cos(lat)-painotettu keskiarvo annetulla maskilla.
        """
        w = paino * maski

        nimittaja = np.sum(w)

        if nimittaja <= 0:
            return np.nan

        return np.sum(
            taulukko * w
        ) / nimittaja

    # --------------------------------------------------------
    # Viimeisen vuoden kuukausikeskiarvot
    # --------------------------------------------------------

    T_vuosi = np.mean(T, axis=0)
    sade_vuosi = np.mean(sade, axis=0)
    pilvi_vuosi = np.mean(pilvisyys, axis=0)
    W_vuosi = np.mean(W, axis=0)
    haihtuminen_vuosi = np.mean(
        haihtuminen,
        axis=0
    )

    # --------------------------------------------------------
    # Globaalit arvot
    # --------------------------------------------------------

    keskilampotila_C = (
        globaali(T_vuosi)
        - 273.15
    )

    keskisade = globaali(
        sade_vuosi
    )

    keskipilvisyys = globaali(
        pilvi_vuosi
    )

    keskivesihoyry = globaali(
        W_vuosi
    )

    keskittainen_haihtuminen = globaali(
        haihtuminen_vuosi
    )

    # --------------------------------------------------------
    # Maa / meri -maskit
    # --------------------------------------------------------

    maa = onko_manner.astype(bool)
    meri = ~maa

    # --------------------------------------------------------
    # Maa- ja merialueiden lämpötilat
    # --------------------------------------------------------

    maa_lampotila_C = (
        aluekeskiarvo(
            T_vuosi,
            maa
        )
        - 273.15
    )

    meri_lampotila_C = (
        aluekeskiarvo(
            T_vuosi,
            meri
        )
        - 273.15
    )

    maa_sade = aluekeskiarvo(
        sade_vuosi,
        maa
    )

    meri_sade = aluekeskiarvo(
        sade_vuosi,
        meri
    )

    maa_pilvisyys = aluekeskiarvo(
        pilvi_vuosi,
        maa
    )

    meri_pilvisyys = aluekeskiarvo(
        pilvi_vuosi,
        meri
    )

    maa_vesihoyry = aluekeskiarvo(
        W_vuosi,
        maa
    )

    meri_vesihoyry = aluekeskiarvo(
        W_vuosi,
        meri
    )

    maa_haihtuminen = aluekeskiarvo(
        haihtuminen_vuosi,
        maa
    )

    meri_haihtuminen = aluekeskiarvo(
        haihtuminen_vuosi,
        meri
    )

    # --------------------------------------------------------
    # Maan pinta-alaosuus
    # --------------------------------------------------------

    maa_osuus = (
        np.sum(
            paino * maa
        )
        / np.sum(paino)
    )

    meri_osuus = 1.0 - maa_osuus

    # --------------------------------------------------------
    # Maan korkeustiedot
    # --------------------------------------------------------

    maan_keskikorkeus_m = np.mean(
        korkeus_m[maa]
    )

    maan_min_korkeus_m = np.min(
        korkeus_m[maa]
    )

    maan_max_korkeus_m = np.max(
        korkeus_m[maa]
    )

    # --------------------------------------------------------
    # Leveysasteanalyysi
    # --------------------------------------------------------
    #
    # Halutut leveysasteet:
    # 0°, ±30°, ±60°, ±90°
    #
    # Otetaan lähin mallin leveysaste.
    # --------------------------------------------------------

    halutut_lat = [
        -90,
        -60,
        -30,
        0,
        30,
        60,
        90
    ]

    leveysasteet = []

    for lat_deg in halutut_lat:

        tavoite = np.deg2rad(
            lat_deg
        )

        i = np.argmin(
            np.abs(
                lat - tavoite
            )
        )

        # Rivi kyseiseltä leveysasteelta
        T_lat = T_vuosi[i, :]
        sade_lat = sade_vuosi[i, :]
        pilvi_lat = pilvi_vuosi[i, :]
        W_lat = W_vuosi[i, :]

        maa_lat = maa[i, :]
        meri_lat = meri[i, :]

        # Koska rivi on yhdellä leveysasteella,
        # käytetään tavallista keskiarvoa pituusasteiden yli.
        T_koko_lat = np.mean(T_lat) - 273.15
        sade_koko_lat = np.mean(sade_lat)
        pilvi_koko_lat = np.mean(pilvi_lat)

        # Maa / meri kyseisellä leveysasteella
        if np.any(maa_lat):
            T_maa_lat = (
                np.mean(
                    T_lat[maa_lat]
                ) - 273.15
            )

            sade_maa_lat = np.mean(
                sade_lat[maa_lat]
            )

            pilvi_maa_lat = np.mean(
                pilvi_lat[maa_lat]
            )
        else:
            T_maa_lat = np.nan
            sade_maa_lat = np.nan
            pilvi_maa_lat = np.nan

        if np.any(meri_lat):
            T_meri_lat = (
                np.mean(
                    T_lat[meri_lat]
                ) - 273.15
            )

            sade_meri_lat = np.mean(
                sade_lat[meri_lat]
            )

            pilvi_meri_lat = np.mean(
                pilvi_lat[meri_lat]
            )
        else:
            T_meri_lat = np.nan
            sade_meri_lat = np.nan
            pilvi_meri_lat = np.nan

        leveysasteet.append({
            "pyydetty_lat": lat_deg,
            "mallin_lat": np.rad2deg(
                lat[i]
            ),

            "lampotila_C":
                T_koko_lat,

            "sade":
                sade_koko_lat,

            "pilvisyys":
                pilvi_koko_lat,

            "maa_lampotila_C":
                T_maa_lat,

            "maa_sade":
                sade_maa_lat,

            "maa_pilvisyys":
                pilvi_maa_lat,

            "meri_lampotila_C":
                T_meri_lat,

            "meri_sade":
                sade_meri_lat,

            "meri_pilvisyys":
                pilvi_meri_lat,
        })

    # --------------------------------------------------------
    # Vuosien lämpötilakehitys
    # --------------------------------------------------------

    vuosilampotilat = tulos[
        "vuosi_keskilampo"
    ]

    vuosisateet = tulos[
        "vuosi_sade"
    ]

    vuosipilvisyys = tulos[
        "vuosi_pilvisyys"
    ]

    # --------------------------------------------------------
    # Tulostetaan yhteenveto
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ILMASTOMALLIN TULOSTEN ANALYYSI")
    print("=" * 70)

    print()
    print("GLOBAALI")
    print("-" * 70)

    print(
        f"Keskilämpötila:       "
        f"{keskilampotila_C:8.2f} °C"
    )

    print(
        f"Keskisademäärä:       "
        f"{keskisade:8.3f}"
    )

    print(
        f"Keskipilvisyys:       "
        f"{100*keskipilvisyys:8.2f} %"
    )

    print(
        f"Keskivesihöyry:       "
        f"{keskivesihoyry:8.4f}"
    )

    print(
        f"Keskihaihtuminen:     "
        f"{keskittainen_haihtuminen:8.4f}"
    )

    print()
    print("MAA / MERI")
    print("-" * 70)

    print(
        f"Maan pinta-alaosuus:  "
        f"{100*maa_osuus:8.2f} %"
    )

    print(
        f"Merien pinta-alaosuus: "
        f"{100*meri_osuus:7.2f} %"
    )

    print(
        f"Maan keskilämpötila:  "
        f"{maa_lampotila_C:8.2f} °C"
    )

    print(
        f"Merien keskilämpötila: "
        f"{meri_lampotila_C:7.2f} °C"
    )

    print(
        f"Maan keskisade:       "
        f"{maa_sade:8.3f}"
    )

    print(
        f"Merien keskisade:     "
        f"{meri_sade:8.3f}"
    )

    print(
        f"Maan keskipilvisyys:  "
        f"{100*maa_pilvisyys:8.2f} %"
    )

    print(
        f"Merien keskipilvisyys:"
        f" {100*meri_pilvisyys:7.2f} %"
    )

    print(
        f"Maan vesihöyry:       "
        f"{maa_vesihoyry:8.4f}"
    )

    print(
        f"Merien vesihöyry:     "
        f"{meri_vesihoyry:8.4f}"
    )

    print(
        f"Maan haihtuminen:     "
        f"{maa_haihtuminen:8.4f}"
    )

    print(
        f"Merien haihtuminen:   "
        f"{meri_haihtuminen:8.4f}"
    )

    print()
    print("MAAN KORKEUS")
    print("-" * 70)

    print(
        f"Keskikorkeus:         "
        f"{maan_keskikorkeus_m:8.1f} m"
    )

    print(
        f"Minimikorkeus:        "
        f"{maan_min_korkeus_m:8.1f} m"
    )

    print(
        f"Maksimikorkeus:       "
        f"{maan_max_korkeus_m:8.1f} m"
    )

    print()
    print("LEVEYSASTEET")
    print("-" * 70)

    print(
        f"{'Lat':>7} "
        f"{'T °C':>9} "
        f"{'Sade':>9} "
        f"{'Pilvi':>9} "
        f"{'Maa T':>9} "
        f"{'Meri T':>9}"
    )

    for x in leveysasteet:

        print(
            f"{x['mallin_lat']:7.1f} "
            f"{x['lampotila_C']:9.2f} "
            f"{x['sade']:9.3f} "
            f"{100*x['pilvisyys']:8.2f}% "
            f"{x['maa_lampotila_C']:9.2f} "
            f"{x['meri_lampotila_C']:9.2f}"
        )

    print()
    print("VUOSITTAINEN KEHITYS")
    print("-" * 70)

    print(
        f"{'Vuosi':>6} "
        f"{'T °C':>10} "
        f"{'Sade':>10} "
        f"{'Pilvi':>10}"
    )

    for i in range(
        len(vuosilampotilat)
    ):

        print(
            f"{i+1:6d} "
            f"{vuosilampotilat[i]:10.2f} "
            f"{vuosisateet[i]:10.3f} "
            f"{100*vuosipilvisyys[i]:9.2f}%"
        )

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # Palautetaan kaikki analyysitulokset
    # --------------------------------------------------------

    return {
        "keskilampotila_C":
            keskilampotila_C,

        "keskisade":
            keskisade,

        "keskipilvisyys":
            keskipilvisyys,

        "keskivesihoyry":
            keskivesihoyry,

        "keskihaihtuminen":
            keskittainen_haihtuminen,

        "maa_osuus":
            maa_osuus,

        "meri_osuus":
            meri_osuus,

        "maa_lampotila_C":
            maa_lampotila_C,

        "meri_lampotila_C":
            meri_lampotila_C,

        "maa_sade":
            maa_sade,

        "meri_sade":
            meri_sade,

        "maa_pilvisyys":
            maa_pilvisyys,

        "meri_pilvisyys":
            meri_pilvisyys,

        "maa_vesihoyry":
            maa_vesihoyry,

        "meri_vesihoyry":
            meri_vesihoyry,

        "maa_haihtuminen":
            maa_haihtuminen,

        "meri_haihtuminen":
            meri_haihtuminen,

        "maan_keskikorkeus_m":
            maan_keskikorkeus_m,

        "maan_min_korkeus_m":
            maan_min_korkeus_m,

        "maan_max_korkeus_m":
            maan_max_korkeus_m,

        "leveysasteet":
            leveysasteet,

        "vuosi_keskilampo":
            vuosilampotilat,

        "vuosi_sade":
            vuosisateet,

        "vuosi_pilvisyys":
            vuosipilvisyys,
    }



# ============================================================
# 10. TULOSTEN PLOTTAUS
# ============================================================

def plottaa_tulokset(tulos):
    """
    Piirtää ilmastomallin tärkeimmät tulokset.

    Kuvaajat:
        1. Globaalin keskilämpötilan kehitys
        2. Globaalin sademäärän kehitys
        3. Globaalin pilvisyyden kehitys
        4. Viimeisen vuoden lämpötila leveysasteittain
        5. Viimeisen vuoden sade leveysasteittain
        6. Viimeisen vuoden pilvisyys leveysasteittain
        7. Viimeisen vuoden vesihöyry leveysasteittain
        8. Viimeisen vuoden lämpötilakartta
        9. Viimeisen vuoden sademääräkartta
       10. Viimeisen vuoden pilvisyyskartta
    """

    import numpy as np
    import matplotlib.pyplot as plt

    T = tulos["T"]                  # [12, lat, lon]
    sade = tulos["sade"]
    pilvisyys = tulos["pilvisyys"]
    W = tulos["vesihoyry"]

    LAT = tulos["LAT"]
    lat = tulos["lat"]

    onko_manner = tulos["onko_manner"]

    vuosilampo = tulos["vuosi_keskilampo"]
    vuosisade = tulos["vuosi_sade"]
    vuosipilvi = tulos["vuosi_pilvisyys"]

    # --------------------------------------------------------
    # Viimeisen vuoden keskiarvot
    # --------------------------------------------------------

    T_vuosi = np.mean(T, axis=0)
    sade_vuosi = np.mean(sade, axis=0)
    pilvi_vuosi = np.mean(pilvisyys, axis=0)
    W_vuosi = np.mean(W, axis=0)

    T_C = T_vuosi - 273.15

    # --------------------------------------------------------
    # Leveysasteprofiilit
    # --------------------------------------------------------

    T_lat = np.mean(T_C, axis=1)
    sade_lat = np.mean(sade_vuosi, axis=1)
    pilvi_lat = np.mean(pilvi_vuosi, axis=1)
    W_lat = np.mean(W_vuosi, axis=1)

    # Maa / meri leveysasteittain
    maa = onko_manner.astype(bool)
    meri = ~maa

    T_maa_lat = np.full(len(lat), np.nan)
    T_meri_lat = np.full(len(lat), np.nan)

    sade_maa_lat = np.full(len(lat), np.nan)
    sade_meri_lat = np.full(len(lat), np.nan)

    pilvi_maa_lat = np.full(len(lat), np.nan)
    pilvi_meri_lat = np.full(len(lat), np.nan)

    for i in range(len(lat)):

        if np.any(maa[i]):
            T_maa_lat[i] = np.mean(
                T_C[i, maa[i]]
            )

            sade_maa_lat[i] = np.mean(
                sade_vuosi[i, maa[i]]
            )

            pilvi_maa_lat[i] = np.mean(
                pilvi_vuosi[i, maa[i]]
            )

        if np.any(meri[i]):
            T_meri_lat[i] = np.mean(
                T_C[i, meri[i]]
            )

            sade_meri_lat[i] = np.mean(
                sade_vuosi[i, meri[i]]
            )

            pilvi_meri_lat[i] = np.mean(
                pilvi_vuosi[i, meri[i]]
            )

    lat_deg = np.rad2deg(lat)

    # --------------------------------------------------------
    # KUVA 1: vuosikehitys
    # --------------------------------------------------------

    vuodet = np.arange(
        1,
        len(vuosilampo) + 1
    )

    fig, ax = plt.subplots(
        3, 1,
        figsize=(10, 11),
        sharex=True
    )

    ax[0].plot(
        vuodet,
        vuosilampo,
        color="firebrick",
        linewidth=2
    )

    ax[0].set_ylabel(
        "Lämpötila (°C)"
    )

    ax[0].set_title(
        "Globaalin ilmaston kehitys"
    )

    ax[0].grid(alpha=0.3)

    ax[1].plot(
        vuodet,
        vuosisade,
        color="royalblue",
        linewidth=2
    )

    ax[1].set_ylabel(
        "Sade"
    )

    ax[1].grid(alpha=0.3)

    ax[2].plot(
        vuodet,
        vuosipilvi * 100,
        color="gray",
        linewidth=2
    )

    ax[2].set_ylabel(
        "Pilvisyys (%)"
    )

    ax[2].set_xlabel(
        "Vuosi"
    )

    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # KUVA 2: leveysasteprofiilit
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        2, 2,
        figsize=(12, 9)
    )

    # Lämpötila
    ax[0, 0].plot(
        lat_deg,
        T_lat,
        color="black",
        linewidth=2,
        label="Kaikki"
    )

    ax[0, 0].plot(
        lat_deg,
        T_maa_lat,
        color="saddlebrown",
        linewidth=2,
        label="Maa"
    )

    ax[0, 0].plot(
        lat_deg,
        T_meri_lat,
        color="royalblue",
        linewidth=2,
        label="Meri"
    )

    ax[0, 0].set_title(
        "Lämpötila leveysasteittain"
    )

    ax[0, 0].set_ylabel(
        "°C"
    )

    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    # Sade
    ax[0, 1].plot(
        lat_deg,
        sade_lat,
        color="royalblue",
        linewidth=2,
        label="Kaikki"
    )

    ax[0, 1].plot(
        lat_deg,
        sade_maa_lat,
        color="saddlebrown",
        linewidth=2,
        label="Maa"
    )

    ax[0, 1].plot(
        lat_deg,
        sade_meri_lat,
        color="teal",
        linewidth=2,
        label="Meri"
    )

    ax[0, 1].set_title(
        "Sademäärä leveysasteittain"
    )

    ax[0, 1].set_ylabel(
        "Sade"
    )

    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    # Pilvisyys
    ax[1, 0].plot(
        lat_deg,
        pilvi_lat * 100,
        color="gray",
        linewidth=2,
        label="Kaikki"
    )

    ax[1, 0].plot(
        lat_deg,
        pilvi_maa_lat * 100,
        color="saddlebrown",
        linewidth=2,
        label="Maa"
    )

    ax[1, 0].plot(
        lat_deg,
        pilvi_meri_lat * 100,
        color="royalblue",
        linewidth=2,
        label="Meri"
    )

    ax[1, 0].set_title(
        "Pilvisyys leveysasteittain"
    )

    ax[1, 0].set_ylabel(
        "%"
    )

    ax[1, 0].set_xlabel(
        "Leveysaste (°)"
    )

    ax[1, 0].legend()
    ax[1, 0].grid(alpha=0.3)

    # Vesihöyry
    ax[1, 1].plot(
        lat_deg,
        W_lat,
        color="darkcyan",
        linewidth=2
    )

    ax[1, 1].set_title(
        "Vesihöyry leveysasteittain"
    )

    ax[1, 1].set_ylabel(
        "Vesihöyry"
    )

    ax[1, 1].set_xlabel(
        "Leveysaste (°)"
    )

    ax[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # KUVA 3: kartat
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )

    extent = [
        0,
        360,
        -90,
        90
    ]

    # Lämpötila
    im0 = ax[0].imshow(
        T_C,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="RdYlBu_r"
    )

    ax[0].set_title(
        "Lämpötila (°C)"
    )

    ax[0].set_xlabel(
        "Pituusaste (°)"
    )

    ax[0].set_ylabel(
        "Leveysaste (°)"
    )

    plt.colorbar(
        im0,
        ax=ax[0],
        label="°C"
    )

    # Sade
    im1 = ax[1].imshow(
        sade_vuosi,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Blues"
    )

    ax[1].set_title(
        "Sademäärä"
    )

    ax[1].set_xlabel(
        "Pituusaste (°)"
    )

    plt.colorbar(
        im1,
        ax=ax[1],
        label="Sade"
    )

    # Pilvisyys
    im2 = ax[2].imshow(
        pilvi_vuosi * 100,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=100
    )

    ax[2].set_title(
        "Pilvisyys (%)"
    )

    ax[2].set_xlabel(
        "Pituusaste (°)"
    )

    plt.colorbar(
        im2,
        ax=ax[2],
        label="%"
    )

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # KUVA 4: maa / meri lämpötila
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        lat_deg,
        T_maa_lat,
        color="saddlebrown",
        linewidth=2.5,
        label="Maa"
    )

    ax.plot(
        lat_deg,
        T_meri_lat,
        color="royalblue",
        linewidth=2.5,
        label="Meri"
    )

    ax.plot(
        lat_deg,
        T_lat,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Kaikki"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.5
    )

    ax.set_title(
        "Maa- ja merialueiden lämpötila"
    )

    ax.set_xlabel(
        "Leveysaste (°)"
    )

    ax.set_ylabel(
        "Lämpötila (°C)"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


tulos = aja_malli(
    vuosia=50, p_co2_ppm=30000,
    n=32,
    seed=42
)

analyysi = analysoi_tulokset(tulos)

plottaa_tulokset(tulos)

