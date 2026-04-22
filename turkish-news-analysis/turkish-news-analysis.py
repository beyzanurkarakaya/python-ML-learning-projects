# Türkçe Haber Analizi

import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib3
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import os
if not os.path.exists(nltk.data.find('corpora/stopwords')):
    nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# SSL uyarısını kapat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# --- 1. VERİ ÇEKME ---
# Hürriyet
url = "https://www.hurriyet.com.tr/gundem/"
response = requests.get(url, headers=headers, verify=False)
soup = BeautifulSoup(response.text, "html.parser")

haberler = []
for link in soup.find_all("a", href=True):
    text = link.get_text(strip=True)
    if len(text) > 20:
        haberler.append(text)

df_hur = pd.DataFrame(haberler, columns=["baslik"])
print(f"Hürriyet toplam haber: {len(df_hur)}")

# NTV
url_ntv = "https://www.ntv.com.tr/gundem"
response_ntv = requests.get(url_ntv, headers=headers, verify=False)
soup_ntv = BeautifulSoup(response_ntv.text, "html.parser")

haberler_ntv = []
for link in soup_ntv.find_all("a", href=True):
    text = link.get_text(strip=True)
    if len(text) > 20:
        haberler_ntv.append(text)

df_ntv = pd.DataFrame(haberler_ntv, columns=["baslik"])
print(f"NTV toplam haber: {len(df_ntv)}")

# Cumhuriyet
url_cum = "https://www.cumhuriyet.com.tr/turkiye"
response_cum = requests.get(url_cum, headers=headers, verify=False)
soup_cum = BeautifulSoup(response_cum.text, "html.parser")

haberler_cum = []
for link in soup_cum.find_all("a", href=True):
    text = link.get_text(strip=True)
    if len(text) > 20:
        haberler_cum.append(text)

df_cum = pd.DataFrame(haberler_cum, columns=["baslik"])
print(f"Cumhuriyet toplam haber: {len(df_cum)}")

# Posta
url_pos = "https://www.posta.com.tr/gundem/"
response_pos = requests.get(url_pos, headers=headers, verify=False)
soup_pos = BeautifulSoup(response_pos.text, "html.parser")

haberler_pos = []
for link in soup_pos.find_all("a", href=True):
    text = link.get_text(strip=True)
    if len(text) > 20:
        haberler_pos.append(text)

df_pos = pd.DataFrame(haberler_pos, columns=["baslik"])
print(f"Posta toplam haber: {len(df_pos)}")

# --- 2. METİN TEMİZLEME ---
stop_words = set(stopwords.words('turkish'))
stop_words.update({
    "bir", "bu", "şu", "o", "ben", "sen", "biz", "siz", "onlar",
    "ve", "ile", "ama", "fakat", "ancak", "ya", "veya", "hem", "ki",
    "için", "gibi", "kadar", "göre", "karşı", "önce", "sonra", "üzere",
    "çok", "daha", "en", "hiç", "her", "bile", "da", "de", "mi", "mu", "mü",
    "edildi", "oldu", "dedi", "alındı", "yapıldı", "geldi", "gitti",
    "bulundu", "açıkladı", "belirtti", "ifade", "ettiği", "olan",
    "yeni", "ilk", "son", "tüm", "bin", "gün", "yıl",
    "tarafından", "gelen", "ayrıca", "ancak", "üzerinden", "ilçesinde",
    "adliyeye", "sevk", "hakkında", "hayatını",
    "yaşındaki", "antalyada", "teslim", "kaybetti", "gencin"
})

def temizle(metin):
    metin = re.sub(r"#\w+:", "", metin)
    metin = metin.lower()
    metin = re.sub(r"[^\w\s]", "", metin)
    metin = re.sub(r"\d+", "", metin)
    kelimeler = metin.split()
    kelimeler = [k for k in kelimeler if k not in stop_words and len(k) > 2]
    return kelimeler

# Hürriyet kelimeleri
tum_kelimeler_hur = []
for baslik in df_hur["baslik"]:
    tum_kelimeler_hur.extend(temizle(baslik))
sayac_hur = Counter(tum_kelimeler_hur)

# NTV kelimeleri
tum_kelimeler_ntv = []
for baslik in df_ntv["baslik"]:
    tum_kelimeler_ntv.extend(temizle(baslik))
sayac_ntv = Counter(tum_kelimeler_ntv)

# Cumhuriyet kelimeleri
tum_kelimeler_cum = []
for baslik in df_cum["baslik"]:
    tum_kelimeler_cum.extend(temizle(baslik))
sayac_cum = Counter(tum_kelimeler_cum)

# Posta kelimeleri
tum_kelimeler_pos = []
for baslik in df_pos["baslik"]:
    tum_kelimeler_pos.extend(temizle(baslik))
sayac_pos = Counter(tum_kelimeler_pos)

# --- 3. HÜRRİYET KELİME BULUTU ---
wc_hur = WordCloud(width=800, height=400, background_color="white",
               colormap="Blues", max_words=100).generate(" ".join(tum_kelimeler_hur))

plt.figure(figsize=(12, 6))
plt.imshow(wc_hur, interpolation="bilinear")
plt.axis("off")
plt.title("Hürriyet Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 4. HÜRRİYET EN SIK KELİMELER ---
en_cok_hur = pd.DataFrame(sayac_hur.most_common(15), columns=["kelime", "sayi"])

plt.figure(figsize=(10, 6))
sns.barplot(data=en_cok_hur, x="sayi", y="kelime", hue="kelime", palette="Blues_r", legend=False)
plt.title("Hürriyet Gündem — En Sık Geçen 15 Kelime", fontsize=14)
plt.xlabel("Frekans")
plt.ylabel("")
plt.tight_layout()
plt.show()

# --- 5. NTV KELİME BULUTU ---
wc_ntv = WordCloud(width=800, height=400, background_color="white",
                   colormap="Oranges", max_words=100).generate(" ".join(tum_kelimeler_ntv))

plt.figure(figsize=(12, 6))
plt.imshow(wc_ntv, interpolation="bilinear")
plt.axis("off")
plt.title("NTV Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 6. NTV EN SIK KELİMELER ---
en_cok_ntv = pd.DataFrame(sayac_ntv.most_common(15), columns=["kelime", "sayi"])

plt.figure(figsize=(10, 6))
sns.barplot(data=en_cok_ntv, x="sayi", y="kelime", hue="kelime", palette="Oranges_r", legend=False)
plt.title("NTV Gündem — En Sık Geçen 15 Kelime", fontsize=14)
plt.xlabel("Frekans")
plt.ylabel("")
plt.tight_layout()
plt.show()

# --- 7. CUMHURİYET KELİME BULUTU ---
wc_cum = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens", max_words=100).generate(" ".join(tum_kelimeler_cum))

plt.figure(figsize=(12, 6))
plt.imshow(wc_cum, interpolation="bilinear")
plt.axis("off")
plt.title("Cumhuriyet Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 8. CUMHURİYET EN SIK KELİMELER ---
en_cok_cum = pd.DataFrame(sayac_cum.most_common(15), columns=["kelime", "sayi"])

plt.figure(figsize=(10, 6))
sns.barplot(data=en_cok_cum, x="sayi", y="kelime", hue="kelime", palette="Greens_r", legend=False)
plt.title("Cumhuriyet Gündem — En Sık Geçen 15 Kelime", fontsize=14)
plt.xlabel("Frekans")
plt.ylabel("")
plt.tight_layout()
plt.show()

# --- 9. POSTA KELİME BULUTU ---
wc_pos = WordCloud(width=800, height=400, background_color="white",
                   colormap="Purples", max_words=100).generate(" ".join(tum_kelimeler_pos))

plt.figure(figsize=(12, 6))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Posta Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 10. POSTA EN SIK KELİMELER ---
en_cok_pos = pd.DataFrame(sayac_pos.most_common(15), columns=["kelime", "sayi"])

plt.figure(figsize=(10, 6))
sns.barplot(data=en_cok_pos, x="sayi", y="kelime", hue="kelime", palette="Purples_r", legend=False)
plt.title("Posta Gündem — En Sık Geçen 15 Kelime", fontsize=14)
plt.xlabel("Frekans")
plt.ylabel("")
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# --- 11. KELİME KARŞILAŞTIRMASI ---
hur_kelimeler = dict(sayac_hur.most_common(15))
ntv_kelimeler = dict(sayac_ntv.most_common(15))
cum_kelimeler = dict(sayac_cum.most_common(15))
pos_kelimeler = dict(sayac_pos.most_common(15))
tum_kelimeler_set = list(set(list(hur_kelimeler.keys()) + list(ntv_kelimeler.keys()) + list(cum_kelimeler.keys()) + list(pos_kelimeler.keys())))
hur_degerler = [hur_kelimeler.get(k, 0) for k in tum_kelimeler_set]
ntv_degerler = [ntv_kelimeler.get(k, 0) for k in tum_kelimeler_set]
cum_degerler = [cum_kelimeler.get(k, 0) for k in tum_kelimeler_set]
pos_degerler = [pos_kelimeler.get(k, 0) for k in tum_kelimeler_set]

x = range(len(tum_kelimeler_set))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar([i - width/2 for i in x], hur_degerler, width, label="Hürriyet", color="steelblue")
ax.bar([i + width/2 for i in x], ntv_degerler, width, label="NTV", color="peru")
ax.bar([i + width/2 for i in x], cum_degerler, width, label="Cumhuriyet", color="olivedrab")
ax.bar([i + width/2 for i in x], pos_degerler, width, label="Posta", color="rebeccapurple")
ax.set_title("Hürriyet / NTV / Cumhuriyet / Posta — Kelime Karşılaştırması", fontsize=14)
ax.set_xticks(list(x))
ax.set_xticklabels(tum_kelimeler_set, rotation=45, ha="right")
ax.set_ylabel("Frekans")
ax.legend()
plt.tight_layout()
plt.show()

# --- 12. ML TABANLI KÜMELEME ---
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df_hur["baslik"])
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
df_hur["kume"] = kmeans.fit_predict(X)

kume_isimleri = {
    0: "Kültür & Yaşam",
    1: "Asayiş & Operasyon",
    2: "Yerel Haberler",
    3: "Genel & Karışık",
    4: "Adliye & Güvenlik",
    5: "Dünya & Savunma",
    6: "Suç & Yargı"
}

df_hur["kume_adi"] = df_hur["kume"].map(kume_isimleri)

X_ntv = vectorizer.transform(df_ntv["baslik"])
df_ntv["kume"] = kmeans.predict(X_ntv)
df_ntv["kume_adi"] = df_ntv["kume"].map(kume_isimleri)

X_cum = vectorizer.transform(df_cum["baslik"])
df_cum["kume"] = kmeans.predict(X_cum)
df_cum["kume_adi"] = df_cum["kume"].map(kume_isimleri)

X_pos = vectorizer.transform(df_pos["baslik"])
df_pos["kume"] = kmeans.predict(X_pos)
df_pos["kume_adi"] = df_pos["kume"].map(kume_isimleri)

# --- 13. KÜME KARŞILAŞTIRMASI ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Hürriyet — sol üst
df_hur["kume_adi"].value_counts().plot(kind="bar", ax=axes[0, 0], color="steelblue", edgecolor="white")
axes[0, 0].set_title("Hürriyet — Kategori Dağılımı", fontsize=13)
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("Haber Sayısı")
axes[0, 0].tick_params(axis="x", rotation=30)

# NTV — sağ üst
df_ntv["kume_adi"].value_counts().plot(kind="bar", ax=axes[0, 1], color="peru", edgecolor="white")
axes[0, 1].set_title("NTV — Kategori Dağılımı", fontsize=13)
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("Haber Sayısı")
axes[0, 1].tick_params(axis="x", rotation=30)

# Cumhuriyet — sol alt
df_cum["kume_adi"].value_counts().plot(kind="bar", ax=axes[1, 0], color="olivedrab", edgecolor="white")
axes[1, 0].set_title("Cumhuriyet — Kategori Dağılımı", fontsize=13)
axes[1, 0].set_xlabel("")
axes[1, 0].set_ylabel("Haber Sayısı")
axes[1, 0].tick_params(axis="x", rotation=30)

# Sağ alt — boş
df_pos["kume_adi"].value_counts().plot(kind="bar", ax=axes[1, 1], color="rebeccapurple", edgecolor="white")
axes[1, 1].set_title("Posta — Kategori Dağılımı", fontsize=13)
axes[1, 1].set_xlabel("")
axes[1, 1].set_ylabel("Haber Sayısı")
axes[1, 1].tick_params(axis="x", rotation=30)

plt.suptitle("Hürriyet / NTV / Cumhuriyet / Posta — Kategori Karşılaştırması", fontsize=15)
plt.tight_layout()
plt.show()

# --- 14. KELİME ÇEŞİTLİLİĞİ ANALİZİ (Type-Token Ratio) ---

def ttr_hesapla(kelimeler):
    if len(kelimeler) == 0:
        return 0
    benzersiz = len(set(kelimeler))
    toplam = len(kelimeler)
    ttr = benzersiz / toplam
    return round(ttr, 4)

sonuclar = {
    "Hürriyet": ttr_hesapla(tum_kelimeler_hur),
    "NTV":      ttr_hesapla(tum_kelimeler_ntv),
    "Cumhuriyet": ttr_hesapla(tum_kelimeler_cum),
    "Posta":    ttr_hesapla(tum_kelimeler_pos)
}

print("\nKelime Çeşitliliği Skoru (TTR):")
for site, skor in sonuclar.items():
    print(f"  {site}: {skor}")

# Görselleştir
plt.figure(figsize=(8, 5))
plt.bar(sonuclar.keys(), sonuclar.values(), 
        color=["steelblue", "peru", "olivedrab", "rebeccapurple"],
        edgecolor="white")
plt.title("Haber Siteleri — Kelime Çeşitliliği (TTR)", fontsize=14)
plt.ylabel("TTR Skoru (0-1)")
plt.ylim(0, 1)
for i, (site, skor) in enumerate(sonuclar.items()):
    plt.text(i, skor + 0.01, str(skor), ha="center", fontsize=11)
plt.tight_layout()
plt.show()

# --- 15. Kİ-KARE TESTİ ---
from scipy.stats import chi2_contingency

# Her sitenin kategori dağılımını sayısal tabloya çevir
kategoriler_listesi = list(kume_isimleri.values())

hur_sayilar = [df_hur[df_hur["kume_adi"] == k].shape[0] for k in kategoriler_listesi]
ntv_sayilar = [df_ntv[df_ntv["kume_adi"] == k].shape[0] for k in kategoriler_listesi]
cum_sayilar = [df_cum[df_cum["kume_adi"] == k].shape[0] for k in kategoriler_listesi]
pos_sayilar = [df_pos[df_pos["kume_adi"] == k].shape[0] for k in kategoriler_listesi]

# Contingency table oluştur
contingency_table = pd.DataFrame(
    [hur_sayilar, ntv_sayilar, cum_sayilar, pos_sayilar],
    index=["Hürriyet", "NTV", "Cumhuriyet", "Posta"],
    columns=kategoriler_listesi
)

print("Kategori Dağılım Tablosu:")
print(contingency_table)

# Ki-kare testi uygula
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nKi-kare istatistiği : {chi2:.4f}")
print(f"p-değeri            : {p:.4f}")
print(f"Serbestlik derecesi : {dof}")

if p < 0.05:
    print("\n✅ Sonuç: Siteler arasında istatistiksel olarak anlamlı bir fark var (p < 0.05)")
else:
    print("\n❌ Sonuç: Siteler arasında istatistiksel olarak anlamlı bir fark yok (p >= 0.05)")

# Görselleştir — ısı haritası
plt.figure(figsize=(12, 5))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="OrRd",
            linewidths=0.5, linecolor="gray")
plt.title("Haber Siteleri — Kategori Dağılımı Isı Haritası", fontsize=14)
plt.xlabel("Kategori")
plt.ylabel("Site")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# --- 16. POST-HOC ANALİZ ---
from itertools import combinations
from scipy.stats import chi2_contingency
import numpy as np

siteler = {
    "Hürriyet": df_hur,
    "NTV": df_ntv,
    "Cumhuriyet": df_cum,
    "Posta": df_pos
}

print("İkili Site Karşılaştırmaları (Post-hoc Ki-kare):\n")

sonuc_listesi = []

for (site1, df1), (site2, df2) in combinations(siteler.items(), 2):
    sayilar1 = [df1[df1["kume_adi"] == k].shape[0] for k in kategoriler_listesi]
    sayilar2 = [df2[df2["kume_adi"] == k].shape[0] for k in kategoriler_listesi]
    
    # Sıfır olan sütunları temizle
    tablo_df = pd.DataFrame([sayilar1, sayilar2], columns=kategoriler_listesi)
    tablo_df = tablo_df.loc[:, (tablo_df != 0).any(axis=0)]
    tablo = tablo_df.values

    try:
        chi2, p, dof, _ = chi2_contingency(tablo)
        sonuc_listesi.append({
            "Karşılaştırma": f"{site1} vs {site2}",
            "Chi2": round(chi2, 4),
            "p-değeri": round(p, 4),
            "Anlamlı mı?": "✅ Evet" if p < 0.05 else "❌ Hayır"
        })
    except:
        pass

sonuc_df = pd.DataFrame(sonuc_listesi)
print(sonuc_df.to_string(index=False))

# Bonferroni düzeltmesi
n_test = len(sonuc_listesi)
bonferroni_esik = 0.05 / n_test
print(f"\nBonferroni düzeltmesi ile eşik: {bonferroni_esik:.4f}")
print("\nBonferroni'ye göre anlamlı olanlar:")
for _, row in sonuc_df.iterrows():
    p_val = row["p-değeri"]
    if p_val < bonferroni_esik:
        print(f"  ✅ {row['Karşılaştırma']} (p={p_val})")
    else:
        print(f"  ❌ {row['Karşılaştırma']} (p={p_val})")

# Görselleştir — p değerleri ısı haritası
siteler_listesi = list(siteler.keys())
p_matrix = pd.DataFrame(np.ones((4, 4)), 
                         index=siteler_listesi, 
                         columns=siteler_listesi)

for _, row in sonuc_df.iterrows():
    s1, s2 = row["Karşılaştırma"].split(" vs ")
    p_matrix.loc[s1, s2] = row["p-değeri"]
    p_matrix.loc[s2, s1] = row["p-değeri"]

plt.figure(figsize=(8, 6))
sns.heatmap(p_matrix, annot=True, fmt=".4f", cmap="RdYlGn_r",
            linewidths=0.5, vmin=0, vmax=0.05)
plt.title("Post-hoc Analiz — p Değerleri Isı Haritası\n(Yeşil = anlamlı fark yok, Kırmızı = anlamlı fark var)", fontsize=13)
plt.tight_layout()
plt.show()


# --- 17. DUYGU ANALİZİ (Lexicon Tabanlı) ---
# Türkçe duygu sözlüğü
olumlu_kelimeler = {
    "güzel", "iyi", "başarı", "başarılı", "kazandı", "sevindi", "mutlu",
    "harika", "mükemmel", "olumlu", "gelişme", "ilerleme", "kazanç",
    "zafer", "sevgi", "umut", "destek", "yardım", "çözüm", "barış",
    "büyüme", "artış", "onay", "kutlama", "ödül", "birinci", "rekor",
    "kurtarıldı", "sağlıklı", "iyileşti", "açıldı", "sevinç", "övgü"
}

olumsuz_kelimeler = {
    "öldü", "öldürüldü", "hayatını kaybetti", "yaralandı", "tutuklandı",
    "gözaltı", "gözaltına", "yakalandı", "kaçtı", "saldırı", "kavga",
    "kaza", "yangın", "patlama", "bomba", "tehlike", "kriz", "suç",
    "şüpheli", "cinayet", "hırsız", "dolandırıcı", "ölü", "ölüm",
    "deprem", "sel", "fırtına", "enkaz", "mahkum", "ceza", "hapis",
    "iflas", "zarar", "kayıp", "düşüş", "kötü", "başarısız", "sorun",
    "endişe", "korku", "tehdit", "savaş", "çatışma", "işgal"
}

def duygu_skoru(df, site_adi):
    olumlu = 0
    olumsuz = 0
    notr = 0

    for baslik in df["baslik"]:
        baslik_lower = baslik.lower()
        kelimeler = set(baslik_lower.split())
        
        ol_var = any(k in baslik_lower for k in olumlu_kelimeler)
        ols_var = any(k in baslik_lower for k in olumsuz_kelimeler)
        
        if ol_var and not ols_var:
            olumlu += 1
        elif ols_var and not ol_var:
            olumsuz += 1
        else:
            notr += 1

    toplam = len(df)
    sonuc = {
        "Olumlu":  round(olumlu / toplam * 100, 1),
        "Olumsuz": round(olumsuz / toplam * 100, 1),
        "Nötr":    round(notr / toplam * 100, 1)
    }

    print(f"\n{site_adi}:")
    for etiket, yuzde in sonuc.items():
        print(f"  {etiket}: %{yuzde}")

    return sonuc

print("Duygu Analizi Sonuçları:")
hur_duygu = duygu_skoru(df_hur, "Hürriyet")
ntv_duygu = duygu_skoru(df_ntv, "NTV")
cum_duygu = duygu_skoru(df_cum, "Cumhuriyet")
pos_duygu = duygu_skoru(df_pos, "Posta")

# --- Görselleştir ---
siteler_duygu = ["Hürriyet", "NTV", "Cumhuriyet", "Posta"]
olumlu_list = [hur_duygu["Olumlu"], ntv_duygu["Olumlu"], cum_duygu["Olumlu"], pos_duygu["Olumlu"]]
olumsuz_list = [hur_duygu["Olumsuz"], ntv_duygu["Olumsuz"], cum_duygu["Olumsuz"], pos_duygu["Olumsuz"]]
notr_list = [hur_duygu["Nötr"], ntv_duygu["Nötr"], cum_duygu["Nötr"], pos_duygu["Nötr"]]

x = range(len(siteler_duygu))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar([i - width for i in x], olumlu_list, width, label="Olumlu", color="mediumseagreen", edgecolor="white")
ax.bar([i for i in x], olumsuz_list, width, label="Olumsuz", color="tomato", edgecolor="white")
ax.bar([i + width for i in x], notr_list, width, label="Nötr", color="steelblue", edgecolor="white")

ax.set_title("Haber Siteleri — Genel Duygu Tonu (%)", fontsize=14)
ax.set_xticks(list(x))
ax.set_xticklabels(siteler_duygu)
ax.set_ylabel("Yüzde (%)")
ax.set_ylim(0, 100)
ax.legend()

for i in range(len(siteler_duygu)):
    ax.text(i - width, olumlu_list[i] + 1, f"%{olumlu_list[i]}", ha="center", fontsize=9)
    ax.text(i, olumsuz_list[i] + 1, f"%{olumsuz_list[i]}", ha="center", fontsize=9)
    ax.text(i + width, notr_list[i] + 1, f"%{notr_list[i]}", ha="center", fontsize=9)

plt.tight_layout()
plt.show()
