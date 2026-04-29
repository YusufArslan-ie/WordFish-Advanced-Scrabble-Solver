import os
import time
import subprocess

# İzlenecek klasörün adı. 
# İstersen buraya doğrudan bilgisayarındaki Google Drive/OneDrive klasörünün tam yolunu da yazabilirsin.
# Örn: WATCH_DIR = r"C:\Users\SeninAdin\Google Drive\KelimelikTest"
WATCH_DIR = "test_resimleri"

def get_image_files(directory):
    """Klasördeki jpg ve png dosyalarını listeler."""
    files = []
    if not os.path.exists(directory):
        return files
        
    for f in os.listdir(directory):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            files.append(os.path.join(directory, f))
    return files

def main():
    # Klasör yoksa oluştur
    if not os.path.exists(WATCH_DIR):
        os.makedirs(WATCH_DIR)
        
    print("═" * 60)
    print(f"👀 GÖZCÜ AKTİF: '{WATCH_DIR}' klasörü izleniyor...")
    print("   Telefondan aldığın SS'leri bu klasöre atman yeterli.")
    print("   Durdurmak için terminalde Ctrl+C yapabilirsin.")
    print("═" * 60)

    # Betik ilk çalıştığında klasörde zaten var olan dosyaları "işlenmiş" say
    processed_files = set(get_image_files(WATCH_DIR))

    try:
        while True:
            current_files = set(get_image_files(WATCH_DIR))
            new_files = current_files - processed_files

            for new_file in new_files:
                # Bulut senkronizasyonunun (Drive/OneDrive) dosyayı indirmeyi bitirmesi için ufak bir pay
                time.sleep(1.5) 
                
                print(f"\n📸 YENİ GÖRÜNTÜ YAKALANDI: {os.path.basename(new_file)}")
                
                # main.py'yi bu yeni dosya ile tetikle!
                subprocess.run(["python", "main.py", new_file])

                # Dosyayı işlenmişler listesine ekle ki bir daha taramasın
                processed_files.add(new_file)
                
                print("\n👀 Gözcü yeni resim bekliyor...")

            # İşlemciyi yormamak için her döngüde 1 saniye dinlen
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Gözcü durduruldu.")

if __name__ == "__main__":
    main()