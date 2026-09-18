

## simple delta-t rainfall pre-set planet climate 
## 18.09. 0000.00018.03


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Pyydetään noise-kirjastoa jos saatavilla, muuten käytetään vaihtoehtoista generaattoria
try:
    from noise import pnoise3
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False

# =====================================================================
# 1. PARAMETRIT JA ASETUKSET
# =====================================================================
korkeus = 180
leveys = 360
seed1 = 12

leveysasteet = np.linspace(90, -90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)

maksimi_korkeus = 1000.0  # metriä
merenpinta_kynnys = 0.1   # Perlin-kynnysraja merelle/maalle (-1..1)

# =====================================================================
# 2. MAASTON KORKEUSKARTAN GENEROINTI (Perlin 3D / Palloprojektio)
# =====================================================================
korkeuskartta = np.zeros((korkeus, leveys))

if HAS_NOISE:
    scale = 0.5 
    octaves = 16

    for i in range(korkeus):
        lat_rad = np.radians(leveysasteet[i])
        for j in range(leveys):
            lon_rad = np.radians(pituusasteet[j])
            
            # 3D-piste pallon pinnalta poistaa napojen venymisen ja tekee saumattoman itä-länsi-rajapinnan
            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)

            korkeuskartta[i, j] = pnoise3(
                (x + 100.0) * scale, 
                (y + 100.0) * scale, 
                (z + 100.0) * scale, 
                octaves=octaves, base=seed1
            )
else:
    # Varajärjestelmä (jos noise-kirjastoa ei ole asennettu)
    lon_grid, lat_grid = np.meshgrid(np.radians(pituusasteet), np.radians(leveysasteet))
    korkeuskartta = (
        0.5 * np.sin(2 * lat_grid) * np.cos(3 * lon_grid) +
        0.3 * np.cos(5 * lat_grid) +
        0.2 * np.sin(7 * lon_grid)
    )

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
etaisyys_pikseleina = distance_transform_edt(maa_maski)

# Muutetaan pikselietäisyys kilometreiksi huomioiden napojen kapeneminen (kosini)
aste_km = 111.0
leveysaste_korjaus = np.cos(np.radians(np.abs(lat_grid_deg)))
pikselin_koko_km = aste_km * leveysaste_korjaus
etaisyys_meresta_km = etaisyys_pikseleina * pikselin_koko_km

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
if np.any(maa_maski):
    virta_pakote = np.sin(np.radians(lat_grid_deg * 2.0)) * np.sign(tuuli_suunta_base)
    # Suodatetaan vaikutus vain rannikoiden läheisyyteen (maks 400km merestä)
    meri_anomalia = virta_pakote * np.exp(-etaisyys_meresta_km / 200.0) * 4.0

# =====================================================================
# 4. KUUKAUSITTAINEN LÄMPÖTILA- JA SADEMÄÄRÄMALLINNUS (12 KUUKAUTTA)
# =====================================================================
kuukausi_lämpötilat = np.zeros((12, korkeus, leveys))
kuukausi_sateet = np.zeros((12, korkeus, leveys))

for kk in range(12):
    # Auringon deklinaatio (akselin kallistuma)
    kulma = 2.0 * np.pi * (kk - 5.5) / 12.0  # Heinäkuu (5.5) maksimi pohjoisessa
    akselin_kallistuma = 23.44 * np.cos(kulma)
    
    # 4.1 LÄMPÖTILA (Sisältää dynaamisen auringonpaikan, korkeuden ja merivirrat)
    etäisyys_auringosta = np.abs(lat_grid_deg - akselin_kallistuma)
    kk_temp_ka = 32.0 - 0.73 * etäisyys_auringosta - (korkeus_metreinä / 1000.0) * 6.5
    kk_temp_ka += meri_anomalia
    kuukausi_lämpötilat[kk, :, :] = kk_temp_ka+12
    
    # 4.2 SADEMÄÄRÄ (Ilmastovyöhykkeet vaeltavat, tuulet osuvat vuoristoon)
    # ITCZ (Päiväntasaajan sade) seuraa aurinkoa hieman viiveellä
    itcz_paikka = akselin_kallistuma * 0.8
    #itcz_vaikutus = np.exp(-((lat_grid_deg - itcz_paikka)**2) / 100.0) * 450.0
    #itcz_vaikutus
    itcz_vaikutus = np.exp(-((lat_grid_deg - itcz_paikka)**2) / 50.0) * 300.0
    #itcz_vaikutus    
    # Naparintama (Lauhkean vyöhykkeen rintamasateet)
    polar_pohjoinen = 50.0 + akselin_kallistuma * 0.2
    polar_etelä = -50.0 + akselin_kallistuma * 0.2
    polar_front_vaikutus = (
        np.exp(-((lat_grid_deg - polar_pohjoinen)**2) / 200.0) * 180.0 +
        np.exp(-((lat_grid_deg - polar_etelä)**2) / 200.0) * 180.0
    )
    
    # Subtrooppinen korkeapaine (Aavikkokuivuus siirtyy ITCZ:n molemmin puolin)
    kuiva_pohjoinen = 25.0 + akselin_kallistuma * 0.5
    kuiva_etelä = -25.0 + akselin_kallistuma * 0.5
    subtropiikki_kuivuus = 1.0 - 0.65 * (
        np.exp(-((lat_grid_deg - kuiva_pohjoinen)**2) / 80.0) +
        np.exp(-((lat_grid_deg - kuiva_etelä)**2) / 80.0)
    )
    subtropiikki_kuivuus = np.clip(subtropiikki_kuivuus, 0.15, 1.0)
    
    # Maantieteellinen rannikkokerroin (kosteus kantaa 1500 km sisämaahan)
    rannikko_kerroin = np.exp(-etaisyys_meresta_km / 1500.0)
    perus_sade = ((25.0 + itcz_vaikutus + polar_front_vaikutus) * subtropiikki_kuivuus) * rannikko_kerroin
    
    # DYNAAMINEN VUORISTOSADE JA SATEENVARJO
    # Jos tuuli puhaltaa nousevaan rinteeseen = sataa, jos laskevaan = kuiva sateenvarjo
    korkeussade_pohja = np.clip(korkeus_metreinä / 150.0, 0, 80.0)
    tuuli_orografia_vaikutus = orografinen_pakote * 2.5 
    dynaaminen_orografia = np.clip(korkeussade_pohja + tuuli_orografia_vaikutus, 0.0, 250.0)
    
    # Kuukauden lopullinen sademäärä
    kuukausi_sateet[kk, :, :] = perus_sade + dynaaminen_orografia

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
maski_Dfc_Dfd = pohja_D & (lämpötila_max < 22.0)

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
ilmastovyohyke_tarkka[maski_E] = 13

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
    "#ffffff"   # 13: E (Napavyöhyke - PUHDAS VALKOINEN)
]

tarkat_nimet = [
    "Meri", "Sademetsä (Af)", "Monsuuni (Am)", "Savanni (Aw)", 
    "Kuuma aavikko (BWh)", "Kylmä aavikko (BWk)", "Kuuma aro (BSh)", "Kylmä aro (BSk)",
    "Välimerenilmasto (Csa/Csb)", "Meri-ilmasto (Cfb)", "Subtrooppinen kostea (Cfa)",
    "Mannerilmasto (Dfa/Dfb)", "Taiga / Havumetsä (Dfc/Dfd)", "Napavyöhyke (E)"
]

cmap_tarkka = ListedColormap(koppen_tarkat_varit)
rajat_tarkka = np.arange(-0.5, 14.5, 1.0)
norm_tarkka = BoundaryNorm(rajat_tarkka, cmap_tarkka.N)

# PAKOTETAAN MATPLOTLIB AVAAMAAN UUSI IKKUNA (Estää koodisolun sisäisen lukituksen)
fig = plt.figure("Köppen-ilmastokartta", figsize=(14, 7), dpi=100)

im = plt.imshow(ilmastovyohyke_tarkka, cmap=cmap_tarkka, norm=norm_tarkka, extent=[-180, 180, -90, 90])

cbar = plt.colorbar(im, ticks=list(range(14)), orientation='horizontal', pad=0.15, shrink=0.9)
cbar.ax.set_xticklabels(tarkat_nimet, rotation=25, ha='right')
cbar.ax.tick_params(labelsize=8)

plt.title("Tarkennettu Köppen-ilmastoluokitus (Dynaamiset tuulet, merivirrat ja sateenvarjot)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Pituusaste")
plt.ylabel("Leveysaste")
plt.grid(color='black', linestyle='--', alpha=0.1) # Vaihdettu valkoinen ruudukko mustaksi, jotta se näkyy navoilla
plt.tight_layout()

# Näytetään ikkuna
plt.show()
