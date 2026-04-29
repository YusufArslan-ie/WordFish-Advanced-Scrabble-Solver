"""
Scrabble TR — Vision Entegreli Hamle Önericisi

Kullanım:
    1) Oyunun ekran görüntüsünü klasöre at (örn: ornek1.jpg).
    2) `python main.py` çalıştır.
    3) Sistem görüntüyü okur, tahtayı analiz eder ve en iyi hamleleri bulur.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- VİZYON MODÜLÜ ---
from vision import extract_game_data
# ---------------------

from engine import ScrabbleEngine
from board import Direction
from gaddag import TurkishAlphabet


# ═══════════════════════════════════════════════════════════════════
#  DİNAMİK YAPILANDIRMA
# ═══════════════════════════════════════════════════════════════════

IMAGE_PATH: str = sys.argv[1] if len(sys.argv) > 1 else "ornek1.jpg"
TEMPLATES_DIR: str = "templates"
DICTIONARY_PATH: str = "scrabble_words_tr.txt"
TOP_N: int = 15

# ═══════════════════════════════════════════════════════════════════


# -------------------------------------------------------------------
#  Yardımcı işlemler
# -------------------------------------------------------------------

_ALPHA_ORDER = {c: i for i, c in enumerate(TurkishAlphabet.LETTERS)}
_ALPHA_ORDER["?"] = 99

_TR_LOWER = {
    'A': 'a', 'B': 'b', 'C': 'c', 'Ç': 'ç', 'D': 'd', 'E': 'e',
    'F': 'f', 'G': 'g', 'Ğ': 'ğ', 'H': 'h', 'I': 'ı', 'İ': 'i',
    'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'O': 'o',
    'Ö': 'ö', 'P': 'p', 'R': 'r', 'S': 's', 'Ş': 'ş', 'T': 't',
    'U': 'u', 'Ü': 'ü', 'V': 'v', 'Y': 'y', 'Z': 'z',
}

def _tr_lower(ch: str) -> str:
    return _TR_LOWER.get(ch, ch.lower())

def _tr_sort(s: str) -> str:
    return "".join(sorted(s, key=lambda c: _ALPHA_ORDER.get(c, 100)))

def consumed_letters(move) -> str:
    chars = ["?" if p.is_blank else p.letter for p in move.placements]
    return _tr_sort("".join(chars))

def rack_leave(rack: str, move) -> str:
    counts: dict[str, int] = {}
    for c in rack:
        counts[c] = counts.get(c, 0) + 1
    for p in move.placements:
        key = "?" if p.is_blank else p.letter
        if counts.get(key, 0) > 0:
            counts[key] -= 1
    leftover = "".join(c * n for c, n in counts.items())
    return _tr_sort(leftover)

def word_display_and_start(move, board) -> tuple[str, tuple[int, int]]:
    p_idx = {(p.row, p.col): p for p in move.placements}
    if move.direction == Direction.HORIZONTAL:
        row = move.placements[0].row
        cols = [p.col for p in move.placements]
        lo, hi = min(cols), max(cols)
        c = lo - 1
        while c >= 0 and board.is_filled(row, c):
            lo = c; c -= 1
        c = hi + 1
        while c < board.size and board.is_filled(row, c):
            hi = c; c += 1
        chars: list[str] = []
        for c in range(lo, hi + 1):
            if (row, c) in p_idx:
                p = p_idx[(row, c)]
                chars.append(_tr_lower(p.letter) if p.is_blank else p.letter)
            else:
                chars.append(board.letter_at(row, c))
        return "".join(chars), (row, lo)
    else:
        col = move.placements[0].col
        rows = [p.row for p in move.placements]
        lo, hi = min(rows), max(rows)
        r = lo - 1
        while r >= 0 and board.is_filled(r, col):
            lo = r; r -= 1
        r = hi + 1
        while r < board.size and board.is_filled(r, col):
            hi = r; r += 1
        chars = []
        for r in range(lo, hi + 1):
            if (r, col) in p_idx:
                p = p_idx[(r, col)]
                chars.append(_tr_lower(p.letter) if p.is_blank else p.letter)
            else:
                chars.append(board.letter_at(r, col))
        return "".join(chars), (lo, col)

# -------------------------------------------------------------------
#  Tablo render
# -------------------------------------------------------------------

def render_table(rows: list[list[str]], headers: list[str], aligns: list[str]) -> str:
    n = len(headers)
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(n)
    ]

    def hline(L, M, R):
        return L + M.join("─" * (w + 2) for w in widths) + R

    def fmt(values):
        cells = []
        for i, v in enumerate(values):
            if aligns[i] == "R":
                cells.append(f" {v:>{widths[i]}} ")
            else:
                cells.append(f" {v:<{widths[i]}} ")
        return "│" + "│".join(cells) + "│"

    lines = [hline("┌", "┬", "┐"), fmt(headers), hline("├", "┼", "┤")]
    lines.extend(fmt(r) for r in rows)
    lines.append(hline("└", "┴", "┘"))
    return "\n".join(lines)


# -------------------------------------------------------------------
#  Ana akış
# -------------------------------------------------------------------

def main() -> None:
    print("═" * 64)
    print("  Scrabble TR — Yapay Zeka Hamle Önericisi")
    print("═" * 64)

    dict_path = Path(DICTIONARY_PATH)
    if not dict_path.exists():
        print(f"\n✗ HATA: Sözlük dosyası bulunamadı → {DICTIONARY_PATH}")
        sys.exit(1)

    print(f"\n► Sözlük yükleniyor : {DICTIONARY_PATH}")
    engine = ScrabbleEngine.from_dictionary(dict_path)
    print(f"  → {len(engine.gaddag):,} kelime, {engine.gaddag.node_count:,} düğüm")

    print(f"\n► Görüntü Analiz Ediliyor: {IMAGE_PATH}")
    # Vizyon modülünden veriyi çekiyoruz!
    game_data = extract_game_data(IMAGE_PATH, TEMPLATES_DIR)
    
    board_matrix = game_data["board"]
    rack_string = "".join(game_data["rack"])
    y3_pos = game_data["y3_pos"]

    print("► Tahtaya yerleştiriliyor...")
    # Matrisi tarayıp bulduğumuz harfleri 1 harflik kelimeler olarak motora yerleştiriyoruz
    placed_count = 0
    for r in range(15):
        for c in range(15):
            char = board_matrix[r][c]
            if char not in (".", "★"):
                
                # --- YENİ EKLENEN KISIM ---
                is_joker = "*" in char
                letter = char.replace("*", "")  # Yıldızı temizle, sadece saf harfi al
                # -------------------------
                
                blanks = [0] if is_joker else []
                try:
                    engine.place_word(letter, row=r, col=c, direction=Direction.HORIZONTAL, blank_indices=blanks)
                    placed_count += 1
                except Exception as e:
                    print(f"  ✗ HATA: '{letter}' harfi ({r},{c}) konumuna yerleştirilemedi: {e}")
                    sys.exit(1)

    print(f"  ✓ Toplam {placed_count} taş başarıyla yerleştirildi.")

    if y3_pos:
        engine.set_y3(y3_pos)

    print("\nGüncel tahta:")
    for line in engine.board_str().splitlines():
        print("  " + line)

    print(f"\nİstaka       : {rack_string}  ({len(rack_string)} taş)")
    print(f"Y3 konumu    : {y3_pos if y3_pos else '(yok)'}")
    print(f"Listelenecek : ilk {TOP_N} hamle")

    print("\n► Hamle uzayı taranıyor...")
    suggestions = engine.suggest_moves(rack_string, top_n=TOP_N)

    if not suggestions:
        print("\nGeçerli hiçbir hamle bulunamadı.")
        return

    headers = ["#", "PUAN", "KELİME (YÖN @ SATIR,SÜTUN)", "HARCANAN", "KALAN"]
    aligns = ["R", "R", "L", "L", "L"]
    rows: list[list[str]] = []
    
    for i, (score, move) in enumerate(suggestions, start=1):
        word_str, (sr, sc) = word_display_and_start(move, engine.board)
        
        # --- KULLANICI DOSTU FORMATLAMA ---
        # 1. Yönleri Türkçeleştir
        yon_tr = "Yatay" if move.direction == Direction.HORIZONTAL else "Dikey"
        
        # 2. Bilgisayar indeksini (0-14) İnsan indeksine (1-15) çevir (+1 ekleyerek)
        insan_satir = sr + 1
        insan_sutun = sc + 1
        # ----------------------------------

        rows.append([
            str(i),
            str(score),
            f"{word_str}  ({yon_tr} @ {insan_satir},{insan_sutun})",
            consumed_letters(move) or "—",
            rack_leave(rack_string, move) or "—",
        ])

    print(f"\nEn iyi {len(rows)} hamle:\n")
    print(render_table(rows, headers, aligns))
    print()


if __name__ == "__main__":
    main()