# WordFish-Advanced-Scrabble-Solver
WordFish, Kelimelik ve Scrabble türevi oyunlar için geliştirilmiş, uçtan uca çalışan yüksek performanslı bir hamle öneri motorudur. Görüntü işlemeden (Computer Vision) başlayarak, GADDAG veri yapısı üzerinde derinlemesine kelime arama yapan ve Android otomasyonu ile entegre çalışan bir mühendislik projesidir.

## 🚀 Öne Çıkan Özellikler

### 1. Gelişmiş Görüntü İşleme (Computer Vision)
* **Dinamik Izgara Tespiti:** Reklam kaymaları veya farklı ekran çözünürlüklerinden bağımsız olarak, **"Tarak Filtresi" (Comb Filter)** ve **Morfolojik Operatörler** kullanarak 15x15 oyun tahtasını milimetrik hassasiyetle tespit eder.
* **Hibrit Karakter Tanıma (OCR):** OpenCV şablon eşleştirme (Template Matching) yöntemini; "I/İ" ve "T/I" gibi kritik harf karışıklıklarını önleyen **geometrik "Tie-Breaker" algoritmalarıyla** birleştirir.
* **Otomatik Renk Analizi:** Karelerin (Y3, K2 vb.) ve joker taşların tespiti için HSV renk uzayı tabanlı segmentasyon uygular.

### 2. GADDAG Tabanlı Kelime Motoru
* **Yüksek Performans:** Standart sözlük aramaları yerine GADDAG veri yapısını kullanarak saniyeler içinde on binlerce olasılığı tarar.
* **Esnek Yerleştirme:** Mevcut tahta üzerindeki harfleri köprü olarak kullanarak yatay ve dikey tüm geçerli hamleleri puan değerleriyle hesaplar.
* **Türkçe Desteği:** Türkçe karakter setine ve oyunun özel puanlama kurallarına (Y3, K3 noktaları vb.) tam uyumludur.

### 3. Otomasyon ve Test Araçları
* **ADB (Android Debug Bridge) Entegrasyonu:** Telefon USB ile bağlıyken tek tuşla (Enter) anlık ekran görüntüsü alır, bilgisayara aktarır ve analiz eder.
* **Watcher (Gözcü) Modu:** Belirlenen bir klasörü izleyerek yeni eklenen ekran görüntülerini otomatik olarak işler.

## 🛠️ Kurulum

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/kullaniciadin/WordFish.git](https://github.com/kullaniciadin/WordFish.git)
    cd WordFish
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install opencv-python numpy
    ```
3.  ADB araçlarının bilgisayarınızda yüklü ve PATH'e ekli olduğundan emin olun.

## 🎮 Kullanım

### Hızlı Başlangıç (ADB ile)
Telefonunuzu bağlayın, USB Hata Ayıklama'yı açın ve şunu çalıştırın:
```bash
python adb_grabber.py
```
Hamle sırası size geldiğinde terminalde `ENTER` tuşuna basmanız yeterlidir.

### Manuel Test
```bash
python main.py ornek_ekran_goruntusu.jpg
```

## 🏗️ Proje Yapısı
* `vision.py`: Görüntü işleme, tahta tespiti ve tile OCR mantığı.
* `engine.py`: GADDAG kelime bulma motoru ve puanlama sistemi.
* `main.py`: Vizyon ve motor modüllerini birleştiren ana kontrolcü.
* `adb_grabber.py`: Android cihazlar için canlı görüntü yakalama aracı.
* `watcher.py`: Klasör izleme tabanlı otomasyon betiği.

## 🗺️ Yol Haritası (Roadmap)
- [ ] **Stratejik Değerlendirme:** Sadece en yüksek puanı değil, rakibe fırsat vermeyen (savunma odaklı) hamlelerin puanlanması.
- [ ] **Rack-Leave Analizi:** Elimizde kalan harflerin istatistiksel değerine göre hamle ağırlıklandırma.
- [ ] **Monte Carlo Tree Search:** Eksik bilgi (torbadaki harfler) durumunda olasılık tabanlı hamle tahminleme.

## ⚖️ Lisans
Bu proje kişisel gelişim amaçlı geliştirilmiştir. Kullanım sorumluluğu kullanıcıya aittir.
