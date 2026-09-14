
"""
Planeetan energia-tasapainomalli - täysi versio (viimeistelty)
- Fraktaalipinta (meri/manner) 3D-pallokoordinaateissa (pnoise3)
- Ilmakehän solut: Hadley + Ferrel (0.40/30°) + Polar (0.05/75°)
- Merivirrat (Ekman + termohaliininen 0.01)
- Kasvihuone (F_perus 175)
- Latentti lämpö ekvaattorilla (-12%)
- Pilvisyys: 0.55 meri / 0.40 manner + ITCZ-piikki
- Kosteus: 40 diffuusioaskelta + poistuma + evapotranspiraatio
- Mantereen sade skaalattu 40 mm/kk maksimiin
- Meren sade kerroin 0.0013
- Globaalit keskiarvot cos(lat)-painotettuina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from noise import pnoise3


# ============================================================
# 0. GLOBAALIT VAKIOT
# ============================================================
KORKEUS_MAX_M = 2000.0
MEREN_SYVYYS_M = 10.0
MANTEREEN_SADE_MAX = 40.0
MEREN_SADE_KERROIN = 0.0013
EVP_KERROIN = 0.15


# ============================================================
# 1. FAKTUAALINEN PINTA
# ============================================================
def luo_fraktaalipinta(n=77, seed=42, manner_kynnys=0.60,
                       korkeus_max_m=KORKEUS_MAX_M):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    korkeus = np.zeros((n, n), dtype=float)

    oktaavit = 6
    lakunaarisuus = 2.0
    pysyvyys = 0.5
    aloitus_skaala = 1.5

    for i in range(n):
        for j in range(n):
            x = np.sin(v[j]) * np.cos(u[i]) * aloitus_skaala
            y = np.sin(v[j]) * np.sin(u[i]) * aloitus_skaala
            z = np.cos(v[j]) * aloitus_skaala
            korkeus[i, j] = pnoise3(
                x, y, z,
                octaves=oktaavit,
                lacunarity=lakunaarisuus,
                persistence=pysyvyys,
                base=seed
            )

    korkeus -= korkeus.min()
    korkeus /= (korkeus.max() + 1e-12)
    onko_manner = korkeus > manner_kynnys
    korkeus_m = np.where(
        onko_manner,
        (korkeus - manner_kynnys) / (1.0 - manner_kynnys + 1e-12) * korkeus_max_m,
        0.0
    )
    return korkeus, onko_manner, korkeus_m


# ============================================================
# 2. SÄTEILY
# ============================================================
def paivittainen_insolaatio_vektori(lat_array, dekl, S, n_tuntia=48):
    I_sum = np.zeros_like(lat_array)
    dt_tunti = 24.0 / n_tuntia
    for h in range(n_tuntia):
        tunti = (h + 0.5) * dt_tunti
        omega = np.deg2rad((tunti - 12.0) * 15.0)
        sin_alt = (np.sin(lat_array) * np.sin(dekl) +
                   np.cos(lat_array) * np.cos(dekl) * np.cos(omega))
        I_sum += np.where(sin_alt > 0, S * sin_alt * dt_tunti, 0.0)
    return I_sum / 24.0


def aurinko_deklinaatio(paiva_vuodesta, tilt, vuoden_pituus_paivina):
    return tilt * np.sin(2 * np.pi * (paiva_vuodesta - 80) / vuoden_pituus_paivina)


# ============================================================
# 3. KASVIHUONE JA ENERGIATASAPAINO
# ============================================================
def kasvihuone_pakote(T_pinta, p_atm, p_co2_ppm):
    F_perus = 175.0 * (p_atm / 1.0)   # MUUTOS 1: 190 → 175
    F_co2 = 5.35 * np.log(np.maximum(p_co2_ppm, 1) / 280.0)
    return np.maximum(F_perus + F_co2, 0.0)


def energia_tasapaino(T0, I_abs, albedo, p_atm, p_co2_ppm,
                      sigma=5.67e-8, k_max=200, tol=1e-3):
    T = max(T0, 50.0)
    for _ in range(k_max):
        F_gh = kasvihuone_pakote(T, p_atm, p_co2_ppm)
        nettosateily = I_abs * (1 - albedo) + F_gh
        if nettosateily <= 0:
            T_uusi = 50.0
        else:
            T_uusi = (nettosateily / sigma) ** 0.25
        if abs(T_uusi - T) < tol:
            return T_uusi
        T = 0.5 * T + 0.5 * T_uusi
    return T


def albedo_pinnan_mukaan(onko_manner, korkeus_m, T, pilvisyys=None):
    albedo = np.where(onko_manner, 0.20, 0.06)
    lumi = (T < 278.15) & onko_manner
    albedo = np.where(lumi, 0.65, albedo)
    meri_jaa = (T < 271.15) & ~onko_manner
    albedo = np.where(meri_jaa, 0.55, albedo)
    korkeus_norm = korkeus_m / KORKEUS_MAX_M
    albedo = np.where(onko_manner, albedo + korkeus_norm * 0.05, albedo)
    if pilvisyys is not None:
        albedo = albedo + 0.20 * pilvisyys
    return np.clip(albedo, 0.05, 0.90)


# ============================================================
# 4. ILMAKEHÄN SOLUT JA KULJETUS
# ============================================================
def solujen_maara(omega_suhteessa_maahan):
    return max(1, int(round(3.0 + np.log2(max(omega_suhteessa_maahan, 0.01)))))


def tuulikentta(LAT, omega_suhteessa_maahan):
    N = solujen_maara(omega_suhteessa_maahan)
    lat_norm = LAT / (np.pi / 2)
    u = 30.0 * np.sin(N * np.pi * lat_norm) * np.cos(lat_norm)
    v = 5.0 * np.cos(N * np.pi * lat_norm) * np.cos(lat_norm)
    return u, v, N


def ilmakehan_kuljetus(T, onko_manner, omega_suhteessa_maahan, vahvuus=0.08,
                       n_iter=3):
    n = T.shape[0]
    lat = np.linspace(-np.pi/2, np.pi/2, n)
    LAT, _ = np.meshgrid(lat, np.arange(n), indexing='ij')
    u, v, N = tuulikentta(LAT, omega_suhteessa_maahan)

    T_uusi = T.copy()
    kappa_merid = vahvuus * (1 + np.abs(v) / 5.0)
    lat_abs = np.abs(np.rad2deg(LAT))
    kappa_merid += 0.10 * np.exp(-(lat_abs / 20)**2)          # Hadley
    kappa_merid += 0.40 * np.exp(-((lat_abs - 30) / 25)**2)   # Ferrel
    # MUUTOS 4: Polar 0.10 → 0.05
    kappa_merid += 0.05 * np.exp(-((lat_abs - 75) / 15)**2)

    for _ in range(n_iter):
        for i in range(1, n - 1):
            T_uusi[i, :] += kappa_merid[i, :] * (
                T_uusi[i+1, :] - 2*T_uusi[i, :] + T_uusi[i-1, :]
            )

    kappa_zon = 0.1 * vahvuus * (1 + np.abs(u) / 30.0)
    T_z = T_uusi.copy()
    for j in range(n):
        jp = (j + 1) % n
        jm = (j - 1) % n
        T_uusi[:, j] += kappa_zon[:, j] * (T_z[:, jp] - 2*T_z[:, j] + T_z[:, jm])

    return T_uusi, u, v, N


# ============================================================
# 5. MERIVIRRAT
# ============================================================
def merivirrat(T_meri, onko_manner, korkeus_m, omega_suhteessa_maahan,
               tuuli_u=None, tuuli_v=None):
    n = T_meri.shape[0]
    lat = np.linspace(-np.pi/2, np.pi/2, n)
    LAT, _ = np.meshgrid(lat, np.arange(n), indexing='ij')

    if tuuli_u is not None and tuuli_v is not None:
        ekman_kulma = np.where(LAT >= 0, np.deg2rad(30), -np.deg2rad(30))
        u_ekman = 0.03 * (tuuli_u * np.cos(ekman_kulma) - tuuli_v * np.sin(ekman_kulma))
        v_ekman = 0.03 * (tuuli_u * np.sin(ekman_kulma) + tuuli_v * np.cos(ekman_kulma))
    else:
        u_ekman = np.zeros_like(T_meri)
        v_ekman = np.zeros_like(T_meri)

    R = 6.371e6
    dlat = np.pi / n
    dy = R * dlat
    T_zonaali = T_meri.mean(axis=1)
    grad_T = np.gradient(T_zonaali) / dy
    grad_tyyp = 15.0 / 6.371e6
    v_termo = -0.01 * (grad_T / grad_tyyp)[:, None] * np.ones((1, n))
    v_termo = np.clip(v_termo, -0.5, 0.5)

    u_meri = u_ekman
    v_meri = v_ekman + v_termo

    u_meri = np.where(onko_manner, 0.0, u_meri)
    v_meri = np.where(onko_manner, 0.0, v_meri)
    jaata = T_meri < 271.0
    u_meri = np.where(jaata, 0.0, u_meri)
    v_meri = np.where(jaata, 0.0, v_meri)

    T_uusi = T_meri.copy()
    for i in range(1, n - 1):
        v_i = v_meri[i, :]
        advektio = np.where(v_i > 0,
                            v_i * (T_meri[i, :] - T_meri[i-1, :]),
                            v_i * (T_meri[i+1, :] - T_meri[i, :]))
        T_uusi[i, :] -= 0.05 * advektio

    kappa = 0.05 * (1 + np.abs(v_meri) / 0.1)
    for i in range(1, n - 1):
        T_uusi[i, :] += kappa[i, :] * (T_meri[i+1, :] - 2*T_meri[i, :] + T_meri[i-1, :])

    T_uusi = np.where(onko_manner, T_meri, T_uusi)
    return T_uusi, u_meri, v_meri


# ============================================================
# 6. KOSTEUS, PILVET JA SADE
# ============================================================
def kosteus_pilvet_sade(T, I_abs, onko_manner, korkeus_m, p_atm, LAT,
                        tuuli_v=None, merivirta_v=None, merivirta_u=None):
    """
    Kosteus, pilvisyys ja sade.
    MUUTOS 2: perustaso 0.40 manner / 0.55 meri
    MUUTOS 3: ITCZ-pilvisyyspiikki ekvaattorilla
    """
    e_s = 611 * np.exp(17.27 * (T - 273.15) / (T - 35.85))
    E_meri = np.where(~onko_manner, MEREN_SADE_KERROIN * I_abs * (e_s / 1000), 0.0)

    if merivirta_v is not None:
        E_meri *= (1 + 0.5 * np.clip(merivirta_v, -0.5, 0.5))
    if merivirta_u is not None:
        E_meri *= (1 + 0.1 * np.abs(np.clip(merivirta_u, -0.5, 0.5)))

    n = E_meri.shape[0]
    meri_mask = ~onko_manner

    if tuuli_v is not None:
        E_meri_siirretty = E_meri.copy()
        for _ in range(3):
            E_uusi = E_meri_siirretty.copy()
            for i in range(1, n - 1):
                v_i = tuuli_v[i, 0]
                if v_i > 0.5:
                    E_uusi[i, :] = E_meri_siirretty[i, :] + 0.3 * v_i * (
                        E_meri_siirretty[i-1, :] - E_meri_siirretty[i, :])
                elif v_i < -0.5:
                    E_uusi[i, :] = E_meri_siirretty[i, :] + 0.3 * (-v_i) * (
                        E_meri_siirretty[i+1, :] - E_meri_siirretty[i, :])
            E_meri_siirretty = E_uusi
    else:
        E_meri_siirretty = E_meri.copy()

    # Kosteuden diffuusio mantereelle + poistuma
    K = E_meri_siirretty.copy()
    for _ in range(40):
        K_uusi = K.copy()
        K_uusi[:, 1:-1] += 0.30 * (K[:, :-2] - 2*K[:, 1:-1] + K[:, 2:])
        K_uusi[:, 0] += 0.30 * (K[:, -1] - 2*K[:, 0] + K[:, 1])
        K_uusi[:, -1] += 0.30 * (K[:, -2] - 2*K[:, -1] + K[:, 0])
        K_uusi[1:-1, :] += 0.30 * (K[:-2, :] - 2*K[1:-1, :] + K[2:, :])
        K_uusi = np.maximum(K_uusi, 0.0)
        K_uusi[meri_mask] = E_meri_siirretty[meri_mask]
        K_uusi = np.where(onko_manner, K_uusi * 0.90, K_uusi)
        K = K_uusi

    # Pilvisyys
    K_meri_max = K[meri_mask].max() + 1e-9
    K_norm = np.clip(K / K_meri_max, 0, 1)
    T_paino = np.clip((T - 250) / 40, 0, 1)

    # MUUTOS 2: perustaso 0.40 manner / 0.55 meri
    perustaso = np.where(onko_manner, 0.40, 0.55)
    pilvisyys = perustaso + 0.35 * K_norm * T_paino

    # MUUTOS 3: ITCZ-pilvisyyspiikki ekvaattorilla (|lat| < 10°)
    lat_abs_deg = np.abs(np.rad2deg(LAT))
    itcz_paino = np.exp(-(lat_abs_deg / 10)**2)
    pilvisyys = pilvisyys + 0.15 * itcz_paino * T_paino

    pilvisyys = np.clip(pilvisyys, 0, 1)

    # Normalisoitu korkeus
    korkeus_norm = korkeus_m / KORKEUS_MAX_M

    # Paikallinen evapotranspiraatio
    E_manner_paikallinen = np.where(
        onko_manner,
        0.001 * EVP_KERROIN * I_abs * (e_s / 1000) * (1 - korkeus_norm),
        0.0
    )
    E_manner_paikallinen *= np.clip((T - 273.15) / 20, 0, 1)

    # Mantereen sade
    E_manner = np.where(
        onko_manner,
        MANTEREEN_SADE_MAX * K_norm * (1 - 0.3 * korkeus_norm) + E_manner_paikallinen,
        0.0
    )
    oro = np.where(onko_manner, 1 + 0.5 * korkeus_norm, 1.0)
    E_manner *= oro
    E_manner *= np.clip((T - 250) / 40, 0, 1)

    sade = E_meri + E_manner
    sade = np.maximum(sade, 0.0)

    return sade, pilvisyys, K


# ============================================================
# 7. LÄMPÖKAPASITEETTI
# ============================================================
def lampokapasiteetti(onko_manner, korkeus_m, meren_syvyys=MEREN_SYVYYS_M):
    C_meri = 4.0e6 * meren_syvyys
    C_manner = 2.0e6 * 2.0
    return np.where(onko_manner, C_manner, C_meri)


# ============================================================
# 8. PÄÄOHJELMA
# ============================================================
def aja_malli(
    n=32, S=1361, mvelp=1.0,
    tilt=np.deg2rad(23.44),
    vuoden_pituus_maan_vuosina=1.0,
    p_atm=1.0, p_co2_ppm=400, rp_earths=None,
    omega_suhteessa_maahan=1.0, seed=42,
    vuosia=20, kuljetus_vahvuus=0.08,
    latentti_kerroin=0.12, n_iter=3,
    debug=False,
):
    if rp_earths is None:
        rp_earths = mvelp ** 0.27

    _, onko_manner, korkeus_m = luo_fraktaalipinta(n=n, seed=seed)

    lat = np.linspace(-np.pi/2, np.pi/2, n)
    LAT, _ = np.meshgrid(lat, np.arange(n), indexing='ij')
    paino_cos = np.cos(LAT)

    vuoden_pituus_paivina = 365.25 * vuoden_pituus_maan_vuosina
    T = np.full((n, n), 288.0)
    pilvisyys = np.full((n, n), 0.5)
    C = lampokapasiteetti(onko_manner, korkeus_m)
    dt = 30 * 24 * 3600

    kk_lampotilat = []
    kk_sateet = []
    kk_pilvisyys = []
    kk_tuuli_u = []
    kk_tuuli_v = []
    kk_merivirta_u = []
    kk_merivirta_v = []
    kk_soluja = []
    vuosi_keskilampo = []

    kuukausia_yhteensa = int(vuosia * 12)

    for kk in range(kuukausia_yhteensa):
        kk_vuodessa = kk % 12
        paiva = (kk_vuodessa + 0.5) * vuoden_pituus_paivina / 12
        dekl = aurinko_deklinaatio(paiva, tilt, vuoden_pituus_paivina)

        I_kk = paivittainen_insolaatio_vektori(LAT, dekl, S, n_tuntia=48)

        lat_abs = np.abs(np.rad2deg(LAT))
        ekvaattori_paino = np.exp(-(lat_abs / 20)**2)
        I_kk = I_kk * (1 - latentti_kerroin * ekvaattori_paino)

        if debug and kk == 0:
            I_globaali = (I_kk * paino_cos).sum() / paino_cos.sum()
            print(f"\nDEBUG: Ensimmäinen kuukausi")
            print(f"  I_kk globaali (cos-painotettu): {I_globaali:.1f} W/m² (pitäisi ~340)")

        for _ in range(5):
            albedo = albedo_pinnan_mukaan(onko_manner, korkeus_m, T, pilvisyys)
            T_sateily = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    T_sateily[i, j] = energia_tasapaino(
                        T[i, j], I_kk[i, j], albedo[i, j], p_atm, p_co2_ppm
                    )
            if np.max(np.abs(T_sateily - T)) < 0.5:
                break
            T = 0.5 * T + 0.5 * T_sateily

        T_atm, u_tuuli, v_tuuli, N_soluja = ilmakehan_kuljetus(
            T_sateily, onko_manner, omega_suhteessa_maahan,
            vahvuus=kuljetus_vahvuus, n_iter=n_iter
        )

        T_meri, u_meri, v_meri = merivirrat(
            T_atm, onko_manner, korkeus_m, omega_suhteessa_maahan,
            tuuli_u=u_tuuli, tuuli_v=v_tuuli
        )

        T_kok = 0.5 * T_sateily + 0.5 * T_meri

        tau = C / (4 * 5.67e-8 * np.maximum(T, 100)**3)
        paino_dt = np.clip(dt / tau, 0, 1)
        T = T + (T_kok - T) * paino_dt
        T = np.clip(T, 50, 1000)

        sade_kk, pilvisyys_uusi, _ = kosteus_pilvet_sade(
            T, I_kk, onko_manner, korkeus_m, p_atm, LAT,
            tuuli_v=v_tuuli, merivirta_v=v_meri, merivirta_u=u_meri
        )
        sade_kk = sade_kk * 30

        pilvisyys = 0.7 * pilvisyys + 0.3 * pilvisyys_uusi

        if kk >= kuukausia_yhteensa - 12:
            kk_lampotilat.append(T.copy())
            kk_sateet.append(sade_kk.copy())
            kk_pilvisyys.append(pilvisyys.copy())
            kk_tuuli_u.append(u_tuuli.copy())
            kk_tuuli_v.append(v_tuuli.copy())
            kk_merivirta_u.append(u_meri.copy())
            kk_merivirta_v.append(v_meri.copy())
            kk_soluja.append(N_soluja)

        if (kk + 1) % 12 == 0:
            T_globaali_vuosi = (T * paino_cos).sum() / paino_cos.sum()
            vuosi_keskilampo.append(T_globaali_vuosi - 273.15)

    return {
        'T': np.array(kk_lampotilat),
        'sade': np.array(kk_sateet),
        'pilvisyys': np.array(kk_pilvisyys),
        'tuuli_u': np.array(kk_tuuli_u),
        'tuuli_v': np.array(kk_tuuli_v),
        'merivirta_u': np.array(kk_merivirta_u),
        'merivirta_v': np.array(kk_merivirta_v),
        'soluja': np.array(kk_soluja),
        'onko_manner': onko_manner,
        'korkeus_m': korkeus_m,
        'lat': lat,
        'paino_cos': paino_cos,
        'rp_earths': rp_earths,
        'omega': omega_suhteessa_maahan,
        'vuosi_keskilampo': np.array(vuosi_keskilampo),
    }


# ============================================================
# 9. DEBUG-TULOSTUKSET
# ============================================================
def tulosta_zonaalit(tulos, otsikko=""):
    T = tulos['T']
    sade = tulos['sade']
    lat = tulos['lat']

    T_zonaali = T.mean(axis=2)
    sade_zonaali = sade.mean(axis=2)

    lat_deg = np.rad2deg(lat)

    print(f"\n{otsikko}")
    print(f"  {'Lat':>5} {'T_avg':>8} {'T_min':>8} {'T_max':>8} "
          f"{'Sade_avg':>10} {'Sade_min':>10} {'Sade_max':>10}")
    print("  " + "-" * 66)
    for aste in range(-80, 81, 10):
        idx = np.argmin(np.abs(lat_deg - aste))
        T_ka = T_zonaali[:, idx].mean() - 273.15
        T_min = T_zonaali[:, idx].min() - 273.15
        T_max = T_zonaali[:, idx].max() - 273.15
        S_ka = sade_zonaali[:, idx].mean()
        S_min = sade_zonaali[:, idx].min()
        S_max = sade_zonaali[:, idx].max()
        print(f"  {aste:>5} {T_ka:>8.2f} {T_min:>8.2f} {T_max:>8.2f} "
              f"{S_ka:>10.1f} {S_min:>10.1f} {S_max:>10.1f}")


def manner_meri_osuus(onko_manner, paino_cos):
    manner_paino = (onko_manner * paino_cos).sum() / paino_cos.sum()
    meri_paino = ((~onko_manner) * paino_cos).sum() / paino_cos.sum()
    return manner_paino, meri_paino


# ============================================================
# 10. VISUALISOINTI
# ============================================================
def piirra_tulokset(tulos, otsikko=""):
    T = tulos['T']
    sade = tulos['sade']
    pilvisyys = tulos['pilvisyys']
    lat = tulos['lat']

    T_zonaali = T.mean(axis=2)
    sade_zonaali = sade.mean(axis=2)
    pilvisyys_zonaali = pilvisyys.mean(axis=2)

    fig, ax = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(otsikko, fontsize=14, fontweight='bold')

    im1 = ax[0, 0].imshow(T.mean(axis=0) - 273.15, origin='lower',
                          extent=[-180, 180, -90, 90], cmap='RdYlBu_r')
    ax[0, 0].set_title('Keskilämpötila (°C)')
    plt.colorbar(im1, ax=ax[0, 0])

    im2 = ax[0, 1].imshow(pilvisyys.mean(axis=0), origin='lower',
                          extent=[-180, 180, -90, 90], cmap='Greys',
                          vmin=0, vmax=1)
    ax[0, 1].set_title('Keskimääräinen pilvisyys (0-1)')
    plt.colorbar(im2, ax=ax[0, 1])

    for kk in range(0, 12, 3):
        ax[1, 0].plot(np.rad2deg(lat), T_zonaali[kk] - 273.15, label=f'kk {kk+1}')
    ax[1, 0].set_xlabel('Leveysaste (°)')
    ax[1, 0].set_ylabel('T (°C)')
    ax[1, 0].set_title('Zonaalinen lämpötila (viimeinen vuosi)')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    for kk in range(0, 12, 3):
        ax[1, 1].plot(np.rad2deg(lat), sade_zonaali[kk], label=f'kk {kk+1}')
    ax[1, 1].set_xlabel('Leveysaste (°)')
    ax[1, 1].set_ylabel('Sade (mm/kk)')
    ax[1, 1].set_title('Zonaalinen sademäärä (viimeinen vuosi)')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    ax[2, 0].plot(tulos['vuosi_keskilampo'], 'k-', lw=2)
    ax[2, 0].set_xlabel('Vuosi')
    ax[2, 0].set_ylabel('Globaali keskilämpötila (°C)')
    ax[2, 0].set_title('Spin-up: cos-painotettu globaali T')
    ax[2, 0].grid(True)

    ax[2, 1].plot(np.rad2deg(lat), pilvisyys_zonaali.mean(axis=0), 'b-', lw=2)
    ax[2, 1].set_xlabel('Leveysaste (°)')
    ax[2, 1].set_ylabel('Pilvisyys')
    ax[2, 1].set_title('Pilvisyys vs. leveysaste')
    ax[2, 1].set_ylim(0, 1)
    ax[2, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. AJO
# ============================================================
if __name__ == "__main__":
    print("=== Maan kaltainen planeetta - täysi versio ===\n")
    tulos = aja_malli(
        n=77, S=1361, mvelp=1.0,
        tilt=np.deg2rad(23.44),
        vuoden_pituus_maan_vuosina=1.0,
        p_atm=1.0, p_co2_ppm=280,
        omega_suhteessa_maahan=1.0, seed=42,
        vuosia=10, kuljetus_vahvuus=0.08,
        latentti_kerroin=0.12, n_iter=3,
        debug=True,
    )

    print("\nVuosikeskilämpötila kehitys (joka 2. vuosi):")
    for i in range(0, len(tulos['vuosi_keskilampo']), 2):
        print(f"  Vuosi {i+1:3d}: {tulos['vuosi_keskilampo'][i]:7.2f} °C")
    print(f"  Vuosi {len(tulos['vuosi_keskilampo']):3d}: "
          f"{tulos['vuosi_keskilampo'][-1]:7.2f} °C")

    T_vika = tulos['T']
    sade_vika = tulos['sade']
    pilvisyys_vika = tulos['pilvisyys']
    paino_cos = tulos['paino_cos']
    onko_manner = tulos['onko_manner']
    korkeus_m = tulos['korkeus_m']

    korkeus_cos = (korkeus_m * paino_cos).sum() / (paino_cos.sum() + 1e-12)
    korkeus_mantereen_cos = (
        (korkeus_m * onko_manner * paino_cos).sum() /
        ((onko_manner * paino_cos).sum() + 1e-12)
    )
    korkeus_aritmeettinen = korkeus_m[onko_manner].mean() if onko_manner.any() else 0.0

    print(f"\n  Korkeus (koko planeetta, cos-painotettu): {korkeus_cos:.1f} m")
    print(f"  Mantereen keskikorkeus (cos-painotettu):  {korkeus_mantereen_cos:.1f} m")
    print(f"  Mantereen keskikorkeus (aritmeettinen):   {korkeus_aritmeettinen:.1f} m")
    print(f"  KORKEUS_MAX_M:                            {KORKEUS_MAX_M:.1f} m")

    T_globaali = (T_vika * paino_cos).sum() / (paino_cos.sum() * T_vika.shape[0])
    sade_globaali = (sade_vika * paino_cos).sum() / (paino_cos.sum() * sade_vika.shape[0])
    pilvisyys_globaali = (pilvisyys_vika * paino_cos).sum() / (paino_cos.sum() * pilvisyys_vika.shape[0])

    tulosta_zonaalit(tulos, "Viimeisen vuoden zonaaliset keskiarvot:")

    manner_osuus, meri_osuus = manner_meri_osuus(onko_manner, paino_cos)
    print(f"\n  Mantereiden osuus (cos-painotettu): {manner_osuus*100:.1f} %")
    print(f"  Merien osuus (cos-painotettu):       {meri_osuus*100:.1f} %")
    print(f"  Mantereiden osuus (aritmeettinen):   {onko_manner.mean()*100:.1f} %")

    sade_mean = sade_vika.mean(axis=0)
    pilvisyys_mean = pilvisyys_vika.mean(axis=0)

    print(f"\n  Globaali keskilämpötila: {T_globaali-273.15:.2f} °C")
    print(f"  Globaali sade (cos): {sade_globaali:.1f} mm/kk")
    print(f"  Sade merellä: {sade_mean[~onko_manner].mean():.1f} mm/kk")
    print(f"  Sade mantereella: {sade_mean[onko_manner].mean():.1f} mm/kk")
    print(f"  Pilvisyys merellä: {pilvisyys_mean[~onko_manner].mean():.2f}")
    print(f"  Pilvisyys mantereella: {pilvisyys_mean[onko_manner].mean():.2f}")
    print(f"  Soluja/pallonpuolisko: {tulos['soluja'][0]}")

    piirra_tulokset(tulos, otsikko="Maan kaltainen planeetta - täysi versio")
