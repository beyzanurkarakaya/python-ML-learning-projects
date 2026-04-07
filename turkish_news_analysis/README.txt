# 📰 Türkçe Haber Sitesi Analizi

Bu proje, Türkiye'nin önde gelen haber sitelerinin gündem içeriklerini web scraping, 
doğal dil işleme (NLP) ve istatistiksel yöntemlerle karşılaştırmalı olarak analiz etmektedir.

## 📌 Analiz Edilen Siteler
- Hürriyet
- NTV
- Cumhuriyet
- Posta

## 🔍 Yapılan Analizler

### 1. Veri Toplama
- BeautifulSoup ile anlık haber başlıklarının çekilmesi
- Metin temizleme (stop words, noktalama, rakam temizliği)

### 2. Kelime Analizi
- En sık kullanılan 15 kelime (bar chart)
- Kelime bulutu görselleştirmesi
- Siteler arası kelime karşılaştırması

### 3. Kelime Çeşitliliği (TTR)
- Type-Token Ratio ile her sitenin dil zenginliği ölçümü

### 4. Makine Öğrenimi ile Kümeleme
- TF-IDF vektörizasyonu
- KMeans algoritması ile 7 kategoriye otomatik sınıflandırma
- Kategoriler: Kültür & Yaşam, Asayiş & Operasyon, Yerel Haberler, 
  Genel & Karışık, Adliye & Güvenlik, Dünya & Savunma, Suç & Yargı

### 5. İstatistiksel Analiz
- Ki-kare testi ile siteler arası kategori farklılıklarının doğrulanması
- Post-hoc analiz ile ikili site karşılaştırmaları
- Bonferroni düzeltmesi

## 📊 Temel Bulgular
- Tüm siteler arasında istatistiksel olarak anlamlı gündem farkı tespit edildi (p < 0.05)
- Hürriyet, diğer sitelerden en belirgin şekilde ayrışan site oldu
- NTV, Cumhuriyet ve Posta birbirine benzer gündem öncelikleri izledi
- Posta en yüksek kelime çeşitliliği skoruna sahip (TTR: ~0.90)

## 🛠️ Kullanılan Teknolojiler

| Kütüphane | Kullanım Amacı |
|---|---|
| `requests` + `BeautifulSoup` | Web scraping |
| `pandas` | Veri işleme |
| `nltk` | Türkçe stop words |
| `wordcloud` | Kelime bulutu |
| `matplotlib` + `seaborn` | Görselleştirme |
| `scikit-learn` | TF-IDF, KMeans |
| `scipy` | Ki-kare testi |

## ⚙️ Kurulum
```bash
git clone https://github.com/beyzanurkarakaya/turkish-news-analysis.git
cd turkish-news-analysis
pip install -r requirements.txt
```

## 🚀 Kullanım
```bash
python turkish_news_analysis.py
```

Kod çalıştırıldığında sırasıyla şu görseller üretilir:
1. Her site için kelime bulutu
2. Her site için en sık kelimeler bar chart
3. Siteler arası kelime karşılaştırması
4. TTR skoru bar chart
5. Siteler arası kategori ısı haritası
6. Post-hoc analiz ısı haritası

## ⚠️ Notlar
- Veriler her çalıştırmada anlık olarak çekilir, sonuçlar güne göre değişebilir
- KMeans kümeleme rastgelelik içerdiğinden kategori dağılımları her çalıştırmada 
  küçük farklılıklar gösterebilir
- SSL doğrulaması kapatılmıştır (verify=False), sadece lokal geliştirme amaçlıdır

## 👤 Yazar
GitHub: [@beyzanurkarakaya](https://github.com/beyzanurkarakaya)
