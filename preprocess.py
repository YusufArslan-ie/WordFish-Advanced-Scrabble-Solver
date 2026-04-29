import csv

def preprocess_tdk_csv(input_csv, output_file):
    VALID_LETTERS = set("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
    valid_words = set()

    def turkish_upper(word):
        word = word.replace('i', 'İ')
        word = word.replace('ı', 'I')
        return word.upper()

    print(f"'{input_csv}' dosyası okunuyor, lütfen bekleyin...")

    try:
        with open(input_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                word = row.get('madde', '')
                
                if not word:
                    continue
                    
                word = word.strip()
                
                # Birden fazla kelimeden oluşanları atla
                if ' ' in word:
                    continue

                upper_word = turkish_upper(word)

                # Kelimedeki tüm harfler 29 harflik alfabemizde varsa kümeye ekle
                if all(char in VALID_LETTERS for char in upper_word):
                    valid_words.add(upper_word)

        # Kelimeleri alfabetik sırala
        sorted_words = sorted(list(valid_words))

        # Temizlenmiş veriyi yeni txt dosyasına yaz
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in sorted_words:
                f.write(word + '\n')

        print("İşlem başarıyla tamamlandı!")
        print(f"Toplam {len(sorted_words)} adet geçerli kelime '{output_file}' dosyasına kaydedildi.")

    except FileNotFoundError:
        print(f"Hata: '{input_csv}' bulunamadı. Dosya adını kontrol et.")

preprocess_tdk_csv('tdk_word_data.csv', 'scrabble_words_tr.txt')