
import numpy as np

# Pyydetään noise-kirjastoa jos saatavilla, muuten käytetään vaihtoehtoista generaattoria
try:
    from noise import pnoise2
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False

# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus = 180
leveys = 360

leveysasteet = np.linspace(90, -90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)

maksimi_korkeus = 2000.0  # metriä
merenpinta_kynnys = 0.0   # Perlin-kynnysraja merelle/maalle (-1..1)

# =====================================================================
# 2. MAASTON KORKEUSKARTAN GENEROINTI (Perlin / Sini-yhdistelmä)
# =====================================================================
korkeuskartta = np.zeros((korkeus, leveys))

if HAS_NOISE:
    scale = 0.02
    octaves = 4
    for i in range(korkeus):
        for j in range(leveys):
            korkeuskartta[i, j] = pnoise2(i * scale, j * scale, octaves=octaves)
else:
    # Varajärjestelmä (jos noise-kirjastoa ei ole asennettu)
    lon_grid, lat_grid = np.meshgrid(np.radians(pituusasteet), np.radians(leveysasteet))
    korkeuskartta = (
        0.5 * np.sin(2 * lat_grid) * np.cos(3 * lon_grid) +
        0.3 * np.cos(5 * lat_grid) +
        0.2 * np.sin(7 * lon_grid)
    )

# Maa/meri-maski ja skaalattu korkeus metreinä
landmask = (korkeuskartta > merenpinta_kynnys).astype(int)

korkeus_metreinä = np.zeros_like(korkeuskartta)
maa_ehdot = korkeuskartta > merenpinta_kynnys

# Skaalataan vain maa-alueiden korkeudet välille 0 ... maksimi_korkeus
if np.any(maa_ehdot):
    min_maa = korkeuskartta[maa_ehdot].min()
    max_maa = korkeuskartta[maa_ehdot].max()
    if max_maa > min_maa:
        korkeus_metreinä[maa_ehdot] = (korkeuskartta[maa_ehdot] - min_maa) / (max_maa - min_maa) * maksimi_korkeus

# =====================================================================
# 3. LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS
# =====================================================================
# Lämpötilaprofiili leveysasteittain (päiväntasaaja lämmin, navat kylmät)
lat_abs = np.abs(leveysasteet)[:, np.newaxis]

lämpötila_ka = 30.0 - 0.7 * lat_abs - (korkeus_metreinä / 1000.0) * 6.5
lämpötila_min = lämpötila_ka - (10.0 + 0.3 * lat_abs)
lämpötila_max = lämpötila_ka + (10.0 + 0.3 * lat_abs)

# Kuukausittainen sademäärä mm/kk (tietokonemalli ITCZ-vyöhykkeellä ja vuoristoerolla)
itcz_vaikutus = np.exp(-((lat_abs - 5.0)**2) / 200.0) * 150.0
orografinen_sade = np.clip(korkeus_metreinä / 100.0, 0, 50.0)
sademäärä_ka = 20.0 + itcz_vaikutus + orografinen_sade

# =====================================================================
# 4. PINTA-ALOJEN PAINOTUKSET (Pallopinta-alat 2D-muodossa)
# =====================================================================
d_lat = np.radians(180.0 / korkeus)
d_lon = np.radians(360.0 / leveys)
lat_rad = np.radians(leveysasteet)

# Lasketaan jokaisen ruudun pinta-alat 2D-taulukoksi (korkeus, leveys)
pinta_ala_solut_1d = np.cos(lat_rad)[:, np.newaxis] * d_lat * d_lon
pinta_ala_solut_2d = np.broadcast_to(pinta_ala_solut_1d, (korkeus, leveys))
pinta_ala_normalisoitu = pinta_ala_solut_2d / np.sum(pinta_ala_solut_2d)

# =====================================================================
# 5. DEBUG-TULOSTUS: GLOBAALIT TUNNUSLUVUT JA LEVEYSASTEET
# =====================================================================
# 1. Vuotuinen sademäärä kartalle (12 kk)
vuosi_sademäärä_kartta = sademäärä_ka * 12.0

maa_maski = (landmask == 1)

# 2. Maa- ja meriosuudet
maa_osuus_kartta = maa_maski.astype(float)

# 3. Pinta-alapainotetut globaalit arvot
globaali_lämpötila_ka = np.sum(lämpötila_ka * pinta_ala_normalisoitu)
globaali_sadesumma_ka = np.sum(vuosi_sademäärä_kartta * pinta_ala_normalisoitu)

globaali_maa_prosentti = np.sum(maa_osuus_kartta * pinta_ala_normalisoitu) * 100.0
globaali_meri_prosentti = 100.0 - globaali_maa_prosentti

# 4. Maan keskikorkeus (merenpinnan yläpuolella)
globaali_maan_pinta_ala = np.sum(pinta_ala_normalisoitu[maa_maski])

if globaali_maan_pinta_ala > 0:
    globaali_maan_keskikorkeus = np.sum(korkeus_metreinä[maa_maski] * pinta_ala_normalisoitu[maa_maski]) / globaali_maan_pinta_ala
else:
    globaali_maan_keskikorkeus = 0.0

askel = max(1, korkeus // 18)

print("\n" + "="*85)
print("PLANEETAN GLOBAALIT TUNNUSMERKIT (PINTA-ALAPAINOTETTU)")
print("="*85)
print(f" Globaali keskilämpötila : {globaali_lämpötila_ka:+.2f} °C")
print(f" Globaali sadesumma      : {globaali_sadesumma_ka:.1f} mm/vuosi")
print(f" Pinta-alan jakauma      : Maa {globaali_maa_prosentti:.1f}% | Meri {globaali_meri_prosentti:.1f}%")
print(f" Maan keskikorkeus       : {globaali_maan_keskikorkeus:.0f} m m.p.y.")
print("="*85)

print("\n" + "="*85)
print("LEVEYSASTEITTAISET KESKILÄMPÖTILAT, SADEMÄÄRÄT JA KORKEUDET (DEBUG)")
print("="*85)
print(f"{'Leveys':^8} | {'Min (°C)':^8} | {'Keski (°C)':^9} | {'Max (°C)':^8} | {'Sade (mm/v)':^11} | {'Maa-%':^7} | {'Kork. (m)':^8} | Visualisointi")
print("-" * 85)

for i in range(0, korkeus, askel):
    lat = leveysasteet[i]
    t_min = np.mean(lämpötila_min[i, :])
    t_ka  = np.mean(lämpötila_ka[i, :])
    t_max = np.mean(lämpötila_max[i, :])
    sade_v = np.mean(vuosi_sademäärä_kartta[i, :])
    
    # Leveysasteen maa-% ja maaston keskikorkeus
    rivi_maa_maski = maa_maski[i, :]
    maa_p = np.mean(maa_osuus_kartta[i, :]) * 100.0
    
    if np.any(rivi_maa_maski):
        maan_korkeus_rivi = np.mean(korkeus_metreinä[i, rivi_maa_maski])
    else:
        maan_korkeus_rivi = 0.0  # Pelkkää merta
    
    korkeus_tuloste = int(round(maan_korkeus_rivi))
    
    palkki_pituus = int(np.clip((t_ka + 40) / 2, 0, 20))
    palkki = "█" * palkki_pituus
    print(f"{lat:^+8.1f} | {t_min:^+8.1f} | {t_ka:^+9.1f} | {t_max:^+8.1f} | {sade_v:^11.0f} | {maa_p:^6.1f}% | {korkeus_tuloste:^8d} | {palkki}")

print("="*85)
