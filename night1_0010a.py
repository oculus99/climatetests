
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from noise import pnoise2
from scipy.ndimage import gaussian_filter

# =====================================================================
# 1. PLANEETAN PARAMETRIT 
# =====================================================================
kasvihuone_ilmiö = 1.55   
aurinkovakio_perus = 1361 
maksimi_korkeus = 2000   

orbital_period = 365.25  
tilt = 23.5             
mvelp = 102.9           
ecc = 0.0167            
rotation_speed = 1.0    

leveys, korkeus = 180, 90 
#leveys, korkeus = 360, 180
# =====================================================================
# 2. MAASTON GENERUOINTI
# =====================================================================
print("Generoidaan maastoa...")

import numpy as np
from noise import pnoise3  # Huom: käytetään 3D-kohinaa 2D:n sijaan

korkeuskartta = np.zeros((korkeus, leveys))
sade = 1.0  # Pallon säde kohina-avaruudessa (vaikuttaa "tiheyteen", toimii kuten skaala)

for y in range(korkeus):
    # Leveysaste (latitude) -90 ja +90 asteen välillä (radiaaneina -pi/2 ... pi/2)
    pii_y = np.pi * (y / korkeus - 0.5)
    
    for x in range(leveys):
        # Pituusaste (longitude) 0 ja 360 asteen välillä (radiaaneina 0 ... 2*pi)
        pii_x = 2 * np.pi * (x / leveys)
        
        # Muutetaan pallokoordinaatit (lat, lon) 3D-pisteeksi (X, Y, Z) pallon pinnalla
        # Tämän ansiosta kartta kiertyy saumattomasti ja navat sulkeutuvat siististi
        nx = sade * np.cos(pii_y) * np.cos(pii_x)
        ny = sade * np.cos(pii_y) * np.sin(pii_x)
        nz = sade * np.sin(pii_y)
        
        # Haetaan kohina 3D-pisteestä
        val = pnoise3(nx, ny, nz, octaves=6, persistence=0.5, lacunarity=2.0)
        
        korkeuskartta[y, x] = val



korkeuskartta=np.exp(korkeuskartta)/np.exp(1)
korkeuskartta=np.exp(korkeuskartta)/np.exp(1)

korkeuskartta = (korkeuskartta - korkeuskartta.min()) / (korkeuskartta.max() - korkeuskartta.min()) * 2 - 1
landmask = np.where(korkeuskartta >= 0, 1, 0)
korkeus_metreinä = np.where(landmask == 1, korkeuskartta * maksimi_korkeus, 0)

albedo_perus = np.where(landmask == 1, 0.18, 0.08)

# Lasketaan alustava mannermaisuusindeksi (etäisyys merestä) korkeapaineita varten
etäisyys_merestä = gaussian_filter(landmask.astype(float), sigma=6) * landmask

# =====================================================================
# 3. MERIVIRTOJEN SIMULOINTI (Lämpimät & Kylmät virrat)
# =====================================================================
print("Lasketaan merivirtoja...")
merivirta_lämpö = np.zeros((korkeus, leveys))
for y in range(korkeus):
    lat = -90.0 + (y / korkeus) * 180.0
    for x in range(leveys):
        if landmask[y, x] == 0:
            # Lämpimät virrat (Länsiosat valtamerta / Länsirannikot korkeilla leveyksillä)
            if 0 < lat < 60:
                merivirta_lämpö[y, x] += 8.0 * np.sin(np.radians(lat)) * (1.0 - x/leveys)
            elif -60 < lat < 0:
                merivirta_lämpö[y, x] += 8.0 * np.sin(np.radians(abs(lat))) * (1.0 - x/leveys)
            
            # Kylmät virrat (Subtrooppiset länsirannikot / Valtameren itäosat, esim. Atacama)
            if 10 < abs(lat) < 40:
                itaisuus = x / leveys
                kylmyys = -6.0 * np.cos(np.radians((abs(lat) - 25) * 6)) * itaisuus
                merivirta_lämpö[y, x] += kylmyys

merivirta_lämpö = gaussian_filter(merivirta_lämpö, sigma=3) * (1.0 - landmask)

# =====================================================================
# 4. 12 KUUKAUDEN SIMULAATIO
# =====================================================================
print("Simuloidaan vuosi 12 kuukauden jaksossa...")
leveysasteet = np.linspace(-90, 90, korkeus)
pituusasteet = np.linspace(-180, 180, leveys)
L_rad, l_rad = np.meshgrid(np.radians(pituusasteet), np.radians(leveysasteet))

kuukausi_paivat = np.linspace(15, orbital_period - 15, 12)

lämpötilat_vuosi = []
sateet_vuosi = []
jääkartat_vuosi = []

korkeus_gradientti_x = np.gradient(gaussian_filter(korkeus_metreinä, sigma=1), axis=1)

for kk_indeksi, paiva in enumerate(kuukausi_paivat):
    keskianomalia = 2 * np.pi * (paiva / orbital_period)
    r = (1 - ecc**2) / (1 + ecc * np.cos(keskianomalia - np.radians(mvelp)))
    aurinkovakio_nykyinen = aurinkovakio_perus / (r**2)
    deklinaatio = np.radians(tilt * np.cos(keskianomalia))

    insolaatio = np.zeros_like(l_rad)
    for i, lat in enumerate(leveysasteet):
        lat_rad = np.radians(lat)
        cos_h0 = -np.tan(lat_rad) * np.tan(deklinaatio)
        if cos_h0 < -1: h0 = np.pi  
        elif cos_h0 > 1: h0 = 0     
        else: h0 = np.arccos(cos_h0)
        insolaatio[i, :] = (aurinkovakio_nykyinen / np.pi) * (h0 * np.sin(lat_rad) * np.sin(deklinaatio) + np.cos(lat_rad) * np.cos(deklinaatio) * np.sin(h0))
    insolaatio = np.clip(insolaatio, 0, None)
    
    T_alustava = ((insolaatio * (1 - albedo_perus)) / (4 * 5.67e-8))**0.25 * kasvihuone_ilmiö - 273.15
    
    albedo_dynaaminen = np.copy(albedo_perus)
    albedo_dynaaminen[(landmask == 0) & (T_alustava < -2)] = 0.55  
    albedo_dynaaminen[(landmask == 1) & (T_alustava < -5)] = 0.45  
    
    jäämaski = np.where(albedo_dynaaminen > 0.4, 2, 1)
    jääkartat_vuosi.append(jäämaski)

    T_rad_pohja = ((insolaatio * (1 - albedo_dynaaminen)) / (4 * 5.67e-8))**0.25 * kasvihuone_ilmiö - 273.15
    T_tasattu = gaussian_filter(T_rad_pohja, sigma=[8, 4]) 
    
    T_lopullinen = 0.3 * T_rad_pohja + 0.7 * T_tasattu + merivirta_lämpö - (korkeus_metreinä * 0.0065)
    lämpötilat_vuosi.append(T_lopullinen)

    # ITCZ:n kausittainen asento ja sisämaan monsooni
    itcz_coeff=0.5
    itcz_perus = np.degrees(deklinaatio)*itcz_coeff
    itcz_kartta = np.full_like(T_lopullinen, itcz_perus)
    itcz_kartta += landmask * (itcz_perus * 0.5)

    # MANNERMAISET TALVIKORKEAPAINEET (esim. Siperia/Mongolia)
    kylmyys_efekti = np.clip((-5.0 - T_lopullinen) / 20.0, 0, 1)
    korkeapaine_voimakkuus = etäisyys_merestä * kylmyys_efekti
    #korkeapaine_kuivuus = 1.0 - 0.8 * korkeapaine_voimakkuus  # Leikkaa sateita jopa 80%
    korkeapaine_kuivuus = 1.0 - 0.5 * korkeapaine_voimakkuus  # Leikkaa sateita jopa 80%
    # SADEVYÖHYKKEET GAUSSIN JAKAUMALLA
    S_pohja = np.zeros_like(T_lopullinen)
    hadley_leveys = 30.0 / (rotation_speed ** 0.5)
    
    for i, lat in enumerate(leveysasteet):
        for j in range(leveys):
            itcz_p = itcz_kartta[i, j]
            
            # 1. Trooppiset kaatosateet (Kapea ITCZ + Vahva päiväntasaajapohja)
            g_itcz = 220.0 * np.exp(-0.5 * ((lat - itcz_p) / 6.0) ** 2)
            g_equator = 120.0 * np.exp(-0.5 * (lat / 10.0) ** 2)
            
            # 2. Diffuusit lauhkean vyöhykkeen sateet (~55°)
            g_midlat_n = 95.0 * np.exp(-0.5 * ((lat - 55.0) / 16.0) ** 2)
            g_midlat_s = 95.0 * np.exp(-0.5 * ((lat - (-55.0)) / 16.0) ** 2)
            
            # 3. Subtrooppiset kuivat vyöhykkeet (Hadley-solu)
            #kuivuus_pohj = np.exp(-0.5 * ((lat - (itcz_p + hadley_leveys)) / 9.0) ** 2)
            #kuivuus_etel = np.exp(-0.5 * ((lat - (itcz_p - hadley_leveys)) / 9.0) ** 2)
            kuivuus_pohj = np.exp(-0.5 * ((lat - (itcz_p + hadley_leveys)) / 5.0) ** 2)
            kuivuus_etel = np.exp(-0.5 * ((lat - (itcz_p - hadley_leveys)) / 5.0) ** 2)
            kuivuus_kerroin = 1.0 - 0.65 * (kuivuus_pohj + kuivuus_etel)
            
            perussadanta = (max(g_itcz, g_equator) + g_midlat_n + g_midlat_s + 15.0) * kuivuus_kerroin
            S_pohja[i, j] = perussadanta * korkeapaine_kuivuus[i, j]

    # JATKUVA TUULI- JA KOSTEUSKULJETUS
    S_lopullinen_paiva = np.zeros_like(S_pohja)
    for i, lat in enumerate(leveysasteet):
        lat_rad = np.radians(lat)
        
        tuulen_voimakkuus = -np.sin(3 * lat_rad)
        tuulen_suunta = 1 if tuulen_voimakkuus >= 0 else -1
        kosteuden_kantama = 0.980 + 0.010 * abs(tuulen_voimakkuus)

        kosteus_maa = np.zeros(leveys)
        pituudet = range(leveys) if tuulen_suunta == 1 else range(leveys-1, -1, -1)
        nykyinen_kosteus = 70.0
        
        for x in pituudet:
            if landmask[i, x] == 0:
                nykyinen_kosteus = 130.0 + merivirta_lämpö[i, x] * 4.0
                nykyinen_kosteus = max(30.0, nykyinen_kosteus)
            else:
                kosteus_maa[x] = nykyinen_kosteus
                nykyinen_kosteus *= kosteuden_kantama

        rinne_kohtaaminen = tuulen_voimakkuus * korkeus_gradientti_x[i, :]
        S_orografinen = np.clip(rinne_kohtaaminen * 0.2, -35, 85)
        
        S_lopullinen_paiva[i, :] = S_pohja[i, :] + (kosteus_maa * 0.35 * landmask[i, :]) + S_orografinen

    sateet_vuosi.append(np.clip(S_lopullinen_paiva, 5, None))

jääkartat_vuosi = np.array(jääkartat_vuosi)

lämpötila_ka = np.mean(lämpötilat_vuosi, axis=0)
lämpötila_max = np.max(lämpötilat_vuosi, axis=0)
lämpötila_min = np.min(lämpötilat_vuosi, axis=0)
sademäärä_ka = np.mean(sateet_vuosi, axis=0)

jää_kuukausi_summa = np.sum(np.where(jääkartat_vuosi == 2, 1, 0), axis=0)
jäävyöhykkeet = np.zeros_like(lämpötila_ka)
jäävyöhykkeet[jää_kuukausi_summa > 0] = 1   
jäävyöhykkeet[jää_kuukausi_summa == 12] = 2 

# =====================================================================
# 5. BIOMILUOKITUS
# =====================================================================
biomikartta = np.zeros_like(lämpötila_ka) + 7.0
maa = (landmask == 1)

biomikartta[maa] = 3  # Oletus: Lauhkea metsä

# Kylmät biomit
biomikartta[maa & (lämpötila_max < 10)] = 5                                     # Tundra
biomikartta[maa & (lämpötila_max < -2)] = 6                                     # Ikuinen jäätikkö
biomikartta[maa & (lämpötila_max >= 10) & (lämpötila_min < -3) & (lämpötila_ka < 8)] = 4 # Taiga

# Kuivat biomit
biomikartta[maa & (sademäärä_ka < 20) & (lämpötila_max > 10) & (lämpötila_ka > 8)] = 0   # Lämmin aavikko
biomikartta[maa & (sademäärä_ka < 35) & (lämpötila_max > 10) & (lämpötila_ka <= 8)] = 1  # Kylmä aro/savanni

# Puolikuivat biomit
biomikartta[maa & (sademäärä_ka >= 35) & (sademäärä_ka < 80) & (lämpötila_max > 10)] = 1 # Savanni/Aro

# Trooppiset biomit
biomikartta[maa & (lämpötila_min >= 10) & (sademäärä_ka >= 115)] = 2            # Sademetsä

# =====================================================================
# 6. VISUALISOINTI
# =====================================================================
print("Luodaan kuvaajia...")
biomi_värit = ['#f4a261', '#e9c46a', '#2a9d8f', '#264653', '#1d3557', '#a8dadc', '#f1faee', '#0b1d3a']
biomi_cmap = LinearSegmentedColormap.from_list('biomit', biomi_värit, N=8)

fig = plt.figure(figsize=(15, 10))

# --- Lokero 1: Biomit ---
ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(biomikartta, cmap=biomi_cmap, extent=[-180,180,-90,90], aspect='auto', origin='lower', vmin=0, vmax=7)
ax1.set_title("Biomit (Saumattomat ilmastomuuttujat & Korkeapaineet)")
ax1.set_ylabel("Leveysaste")
cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0.43, 1.31, 2.18, 3.06, 3.93, 4.81, 5.68, 6.56])
cbar1.ax.set_yticklabels(['Aavikko', 'Savanni/Aro', 'Sademetsä', 'Lauhkea metsä', 'Taiga', 'Tundra', 'Ikuinen jää', 'Valtameri'])

# --- Lokero 2: Vuotuinen sademäärä ---
ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(sademäärä_ka, cmap='YlGnBu', extent=[-180,180,-90,90], aspect='auto', origin='lower')
ax2.set_title("Vuotuinen keskisademäärä (mm/kk)")
fig.colorbar(im2, ax=ax2)

# --- Lokero 3: Merivirrat ---
ax3 = fig.add_subplot(2, 2, 3)
im3 = ax3.imshow(merivirta_lämpö, cmap='coolwarm', extent=[-180,180,-90,90], aspect='auto', origin='lower')
ax3.set_title("Merivirtojen lämpötilavaikutus (°C)")
fig.colorbar(im3, ax=ax3)

# --- Lokero 4: Lämpötilaprofiili ---
ax4 = fig.add_subplot(2, 2, 4)
keski_x = leveys // 2
ax4.plot(leveysasteet, lämpötila_max[:, keski_x], color='crimson', label='Max (Lämpimin kk)', linewidth=2)
ax4.plot(leveysasteet, lämpötila_ka[:, keski_x], color='black', linestyle='--', label='Vuoden keskiarvo', linewidth=1.5)
ax4.plot(leveysasteet, lämpötila_min[:, keski_x], color='royalblue', label='Min (Kylmin kk)', linewidth=2)
ax4.axhline(0, color='silver', linestyle=':')
ax4.set_title("12 kuukauden lämpötilat")
ax4.set_xlabel("Leveysaste")
ax4.set_ylabel("Lämpötila (°C)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
