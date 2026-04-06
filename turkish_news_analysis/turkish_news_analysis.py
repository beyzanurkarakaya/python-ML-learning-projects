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

# --- 5. CUMHURİYET KELİME BULUTU ---
wc_cum = WordCloud(width=800, height=400, background_color="white",
                   colormap="Greens", max_words=100).generate(" ".join(tum_kelimeler_cum))

plt.figure(figsize=(12, 6))
plt.imshow(wc_cum, interpolation="bilinear")
plt.axis("off")
plt.title("Cumhuriyet Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 6. CUMHURİYET EN SIK KELİMELER ---
en_cok_cum = pd.DataFrame(sayac_cum.most_common(15), columns=["kelime", "sayi"])

plt.figure(figsize=(10, 6))
sns.barplot(data=en_cok_cum, x="sayi", y="kelime", hue="kelime", palette="Greens_r", legend=False)
plt.title("Cumhuriyet Gündem — En Sık Geçen 15 Kelime", fontsize=14)
plt.xlabel("Frekans")
plt.ylabel("")
plt.tight_layout()
plt.show()

# --- 5. POSTA KELİME BULUTU ---
wc_pos = WordCloud(width=800, height=400, background_color="white",
                   colormap="Purples", max_words=100).generate(" ".join(tum_kelimeler_pos))

plt.figure(figsize=(12, 6))
plt.imshow(wc_pos, interpolation="bilinear")
plt.axis("off")
plt.title("Posta Gündem — Kelime Bulutu", fontsize=14)
plt.tight_layout()
plt.show()

# --- 6. POSTA EN SIK KELİMELER ---
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

print(sayac_pos.most_common(5))

# --- 7. KELİME KARŞILAŞTIRMASI ---
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

# --- 8. ML TABANLI KÜMELEME ---
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

# --- 9. KÜME KARŞILAŞTIRMASI ---
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