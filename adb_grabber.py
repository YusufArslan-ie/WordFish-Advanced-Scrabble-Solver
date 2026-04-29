import os
import subprocess
import time

# YAPILANDIRMA
PC_SAVE_DIR = "test_resimleri"  # Resmin kaydedileceği yer
REMOTE_PATH = "/sdcard/scrabble_capture.png" # Telefondaki geçici yol
MAIN_SCRIPT = "main.py"

def run_command(command):
    """Sistem komutlarını çalıştırır ve çıktıyı döndürür."""
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result

def main():
    if not os.path.exists(PC_SAVE_DIR):
        os.makedirs(PC_SAVE_DIR)

    print("🚀 ADB OTOMASYONU BAŞLATILDI")
    print("--------------------------------------------------")
    print("Komut: Bilgisayardan 'Enter'a bastığında:")
    print("1. Telefondan anlık SS alınacak.")
    print("2. Resim bilgisayara çekilecek.")
    print("3. main.py otomatik çalıştırılacak.")
    print("--------------------------------------------------")

    try:
        while True:
            input("\n📸 Hamle sırası sende mi? Ekran görüntüsü almak için ENTER'a bas... (Çıkış için Ctrl+C)")
            
            timestamp = int(time.time())
            filename = f"adb_snap_{timestamp}.jpg"
            local_path = os.path.join(PC_SAVE_DIR, filename)

            print("📡 Telefonla iletişim kuruluyor...")
            
            # 1. Telefondan ekran görüntüsü al (Sessizce)
            run_command(f"adb shell screencap -p {REMOTE_PATH}")
            
            # 2. Resmi bilgisayara çek
            pull_res = run_command(f"adb pull {REMOTE_PATH} {local_path}")
            
            if "pulled" in pull_res.stdout or pull_res.returncode == 0:
                print(f"✅ Görüntü alındı: {filename}")
                
                # 3. main.py'yi hemen çalıştır
                print("🧠 Analiz başlıyor...")
                subprocess.run(["python", MAIN_SCRIPT, local_path])
            else:
                print("❌ HATA: Telefon bağlantısı kurulamadı. USB kablosunu kontrol et!")

    except KeyboardInterrupt:
        print("\n🛑 Otomasyon durduruldu.")

if __name__ == "__main__":
    main()