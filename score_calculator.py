"""
Scrabble TR — Modül D: ScoreCalculator

Bir CandidateMove'un (Modül C çıktısı) tam puanını hesaplar:
  • Ana kelime + tüm yan (cross) kelimelerin puanları
  • Harf çarpanları (H2/H3) → her etkilediği kelimede uygulanır
  • Kelime çarpanları (K2/K3) → bu sırayla uygulanır (önce H, sonra K)
  • Çarpanlar SADECE bu turda yerleştirilen taşlar için aktif
  • Joker → harf puanı 0; ama K2/K3 hücresinde kelime çarpanı yine geçerli
  • Bingo (7 taş) → +50
  • Y3 hücresi → tüm toplama en sonda +25 (bir kez)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from board import ScrabbleBoard, Direction
from move_generator import CandidateMove, Placement


# =========================================================================
# Sabitler
# =========================================================================

LETTER_VALUES: dict[str, int] = {
    'A': 1, 'B': 3, 'C': 4, 'Ç': 4, 'D': 3, 'E': 1, 'F': 7, 'G': 5,
    'Ğ': 8, 'H': 5, 'I': 2, 'İ': 1, 'J': 10, 'K': 1, 'L': 1, 'M': 2,
    'N': 1, 'O': 2, 'Ö': 7, 'P': 5, 'R': 1, 'S': 2, 'Ş': 4, 'T': 1,
    'U': 2, 'Ü': 3, 'V': 7, 'Y': 3, 'Z': 4,
}

BINGO_BONUS = 50
BINGO_THRESHOLD = 7
Y3_BONUS = 25
DEFAULT_BOARD_SIZE = 15


class Bonus(Enum):
    NONE = 0
    H2 = 1   # Harf x2
    H3 = 2   # Harf x3
    K2 = 3   # Kelime x2
    K3 = 4   # Kelime x3


# =========================================================================
# Statik tahta çarpan haritası
# =========================================================================

def _xy_to_rc(x: int, y: int) -> tuple[int, int]:
    """1-tabanlı (x, y) → 0-tabanlı (row, col).  row = y-1, col = x-1."""
    return (y - 1, x - 1)


def _build_bonus_map() -> dict[tuple[int, int], Bonus]:
    """Kullanıcının verdiği 1-tabanlı (x,y) listelerini (row,col)→Bonus haritasına çevir."""
    K3 = [(3, 1), (13, 1), (1, 3), (15, 3), (1, 13), (15, 13), (3, 15), (13, 15)]
    K2 = [(8, 3), (4, 4), (12, 4), (3, 8), (13, 8), (4, 12), (12, 12), (8, 13)]
    H3 = [(2, 2), (14, 2), (5, 5), (11, 5), (5, 11), (11, 11), (2, 14), (14, 14)]
    H2 = [
        (6, 1), (10, 1), (7, 2), (9, 2), (1, 6), (6, 6), (10, 6), (15, 6),
        (2, 7), (7, 7), (9, 7), (14, 7), (2, 9), (7, 9), (9, 9), (14, 9),
        (1, 10), (6, 10), (10, 10), (15, 10), (7, 14), (9, 14), (6, 15), (10, 15),
    ]
    out: dict[tuple[int, int], Bonus] = {}
    for xy in K3: out[_xy_to_rc(*xy)] = Bonus.K3
    for xy in K2: out[_xy_to_rc(*xy)] = Bonus.K2
    for xy in H3: out[_xy_to_rc(*xy)] = Bonus.H3
    for xy in H2: out[_xy_to_rc(*xy)] = Bonus.H2
    return out


BONUS_MAP: dict[tuple[int, int], Bonus] = _build_bonus_map()


# =========================================================================
# ScoreCalculator
# =========================================================================

class ScoreCalculator:
    """
    score_move(move, board) -> int

    `board` parametresi, hamlenin OYNANMASINDAN ÖNCEKİ tahta durumu olmalıdır
    (Modül C zaten bu varsayımla çalışır).
    """

    def __init__(
        self,
        size: int = DEFAULT_BOARD_SIZE,
        y3_position: Optional[tuple[int, int]] = (0, 0),
    ) -> None:
        self.size = size
        self.y3_position = y3_position   # None = Y3 yok
        self.bonus_map = BONUS_MAP

    # ---------- Ortak API ----------

    def score_move(self, move: CandidateMove, board: ScrabbleBoard) -> int:
        placements = move.placements
        direction = move.direction
        placements_idx = {(p.row, p.col): p for p in placements}

        # 1) Ana kelime
        main_cells = self._main_word_cells(placements, direction, board, placements_idx)
        total = self._score_word(main_cells)

        # 2) Yan (cross) kelimeler
        perp = Direction.VERTICAL if direction == Direction.HORIZONTAL else Direction.HORIZONTAL
        for p in placements:
            cw = self._cross_word_cells(p, perp, board)
            if len(cw) >= 2:                # tek harfli "kelime" sayılmaz
                total += self._score_word(cw)

        # 3) Bingo
        if len(placements) == BINGO_THRESHOLD:
            total += BINGO_BONUS

        # 4) Y3 (tüm toplamın en sonunda, yalnızca bir kez)
        if self.y3_position is not None and any(
            (p.row, p.col) == self.y3_position for p in placements
        ):
            total += Y3_BONUS

        return total

    # ---------- Hücre dizilerini kur ----------

    def _main_word_cells(
        self, placements, direction, board, placements_idx,
    ) -> list[tuple[int, int, str, bool, bool]]:
        """
        (row, col, letter, is_blank, is_new_placement) listesi döner.
        Forced (eski) taşlar da listededir; bonusları skorlamada uygulanmaz.
        """
        if direction == Direction.HORIZONTAL:
            row = placements[0].row
            cols = [p.col for p in placements]
            lo, hi = min(cols), max(cols)
            # Sol/sağ uzantı (sadece eski/forced taşlar)
            c = lo - 1
            while c >= 0 and board.is_filled(row, c):
                lo = c; c -= 1
            c = hi + 1
            while c < self.size and board.is_filled(row, c):
                hi = c; c += 1
            return [self._make_cell(row, c, board, placements_idx) for c in range(lo, hi + 1)]
        else:
            col = placements[0].col
            rows = [p.row for p in placements]
            lo, hi = min(rows), max(rows)
            r = lo - 1
            while r >= 0 and board.is_filled(r, col):
                lo = r; r -= 1
            r = hi + 1
            while r < self.size and board.is_filled(r, col):
                hi = r; r += 1
            return [self._make_cell(r, col, board, placements_idx) for r in range(lo, hi + 1)]

    def _cross_word_cells(
        self, p: Placement, perp: Direction, board,
    ) -> list[tuple[int, int, str, bool, bool]]:
        """p'den geçen, perpendicular yöndeki kelime hücreleri."""
        r0, c0 = p.row, p.col
        # Yan kelimedeki forced taşlar tamamen tahtadan; tek 'new' = p'nin kendisi.
        if perp == Direction.VERTICAL:
            top, bot = r0, r0
            while top - 1 >= 0 and board.is_filled(top - 1, c0): top -= 1
            while bot + 1 < self.size and board.is_filled(bot + 1, c0): bot += 1
            cells = []
            for r in range(top, bot + 1):
                if r == r0:
                    cells.append((r, c0, p.letter, p.is_blank, True))
                else:
                    t = board.tile_at(r, c0)
                    cells.append((r, c0, t.letter, t.is_blank, False))
            return cells
        else:
            left, right = c0, c0
            while left - 1 >= 0 and board.is_filled(r0, left - 1): left -= 1
            while right + 1 < self.size and board.is_filled(r0, right + 1): right += 1
            cells = []
            for c in range(left, right + 1):
                if c == c0:
                    cells.append((r0, c, p.letter, p.is_blank, True))
                else:
                    t = board.tile_at(r0, c)
                    cells.append((r0, c, t.letter, t.is_blank, False))
            return cells

    @staticmethod
    def _make_cell(r, c, board, placements_idx):
        if (r, c) in placements_idx:
            p = placements_idx[(r, c)]
            return (r, c, p.letter, p.is_blank, True)
        t = board.tile_at(r, c)
        return (r, c, t.letter, t.is_blank, False)

    # ---------- Tek kelime puanı ----------

    def _score_word(self, cells) -> int:
        """
        Önce harf çarpanları, sonra kelime çarpanları.
        Çarpanlar SADECE is_new=True hücreler için uygulanır.
        Joker harfin yalın puanı 0; kelime çarpanı yine kelimeye uygulanır.
        """
        word_mult = 1
        word_sum = 0
        for (r, c, letter, is_blank, is_new) in cells:
            base = 0 if is_blank else LETTER_VALUES[letter]
            if is_new:
                bonus = self.bonus_map.get((r, c), Bonus.NONE)
                if bonus == Bonus.H2:
                    base *= 2
                elif bonus == Bonus.H3:
                    base *= 3
                elif bonus == Bonus.K2:
                    word_mult *= 2
                elif bonus == Bonus.K3:
                    word_mult *= 3
            word_sum += base
        return word_sum * word_mult


# =========================================================================
# Hızlı sınama
# =========================================================================

if __name__ == "__main__":
    from gaddag import Gaddag
    from board import ScrabbleBoard, Placement
    from move_generator import MoveGenerator, Rack, CandidateMove

    g = Gaddag()
    for w in [
        "AT", "ATA", "ATEŞ", "ARABA", "BAR", "BAL", "AB", "AL", "RA",
        "EV", "EVET", "KEDİ",
    ]:
        g.add_word(w)

    sc = ScoreCalculator(size=15, y3_position=None)

    # ---------- T1: Boş tahtada ARABA, merkez (7,7)'den yatay ----------
    board = ScrabbleBoard(g, size=15)
    move = CandidateMove(
        placements=tuple([
            Placement(7, 7, "A"), Placement(7, 8, "R"), Placement(7, 9, "A"),
            Placement(7, 10, "B"), Placement(7, 11, "A"),
        ]),
        main_word="ARABA",
        direction=Direction.HORIZONTAL,
        anchor=(7, 7),
    )
    s = sc.score_move(move, board)
    # Hücreler: (7,7)A, (7,8)R, (7,9)A, (7,10)B, (7,11)A
    # Bonus haritasında (7,*) hücrelerinden hiçbiri yok → çarpan yok
    # Toplam: 1+1+1+3+1 = 7
    print(f"T1 ARABA boş tahta: {s} (beklenen 7)")
    assert s == 7

    # ---------- T2: ARABA + (8,7)T (9,7)E (10,7)Ş = ATEŞ dikey ----------
    board.commit_move(list(move.placements))
    move2 = CandidateMove(
        placements=tuple([
            Placement(8, 7, "T"), Placement(9, 7, "E"), Placement(10, 7, "Ş"),
        ]),
        main_word="ATEŞ",
        direction=Direction.VERTICAL,
        anchor=(8, 7),
    )
    s = sc.score_move(move2, board)
    # Ana ATEŞ: A(forced)=1 + T=1 + E=1 + Ş=4 = 7. Bonus haritasında (8,7),(9,7),(10,7) yok.
    # Yan kelimeler: tek harf → yok.
    print(f"T2 ATEŞ dikey: {s} (beklenen 7)")
    assert s == 7
    board.undo_move(list(move.placements))   # tahtayı sıfırla

    # ---------- T3: H2 hücresi etkisi (yeni taş) ----------
    # (5,0) row=5, col=0 → kullanıcı (1,6) → (row=5, col=0) H2.
    # AB kelimesini (5,0)..(5,1) yatay oynayalım.
    move3 = CandidateMove(
        placements=tuple([Placement(5, 0, "A"), Placement(5, 1, "B")]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(5, 0),
    )
    # Aslında anchor (5,0) → boş tahtada anchor merkez (7,7) olmalı; ama
    # ScoreCalculator MoveGenerator'ı denetlemez, sadece skor verir. Skor testi:
    # A(yeni, H2)=1*2=2 + B(yeni, bonus yok)=3 → 5
    s = sc.score_move(move3, board)
    print(f"T3 H2 etkisi: {s} (beklenen 5)")
    assert s == 5

    # ---------- T4: K2 hücresi etkisi ----------
    # (3,3) row=3, col=3 → kullanıcı (4,4) → (row=3, col=3) K2.
    # AB → (3,3)A (3,4)B → ana kelime "AB". A(K2 yeni)=1, B(yeni)=3 → 4 * 2 = 8
    move4 = CandidateMove(
        placements=tuple([Placement(3, 3, "A"), Placement(3, 4, "B")]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(3, 3),
    )
    s = sc.score_move(move4, board)
    print(f"T4 K2 etkisi: {s} (beklenen 8)")
    assert s == 8

    # ---------- T5: H3 hücresi (1,1) row=1,col=1 ----------
    # AB → (1,1)A (1,2)B. A(H3)=1*3=3, B=3 → 6
    move5 = CandidateMove(
        placements=tuple([Placement(1, 1, "A"), Placement(1, 2, "B")]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(1, 1),
    )
    s = sc.score_move(move5, board)
    print(f"T5 H3 etkisi: {s} (beklenen 6)")
    assert s == 6

    # ---------- T6: K3 köşe ----------
    # (0,2) → (1,3) (x=1,y=3) K3. AB (0,2)A (0,3)B → A=1, B=3 → 4 * 3 = 12
    move6 = CandidateMove(
        placements=tuple([Placement(0, 2, "A"), Placement(0, 3, "B")]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(0, 2),
    )
    s = sc.score_move(move6, board)
    print(f"T6 K3 etkisi: {s} (beklenen 12)")
    assert s == 12

    # ---------- T7: Joker H3 hücresinde (joker = 0, çarpan etkisiz) ----------
    # (1,1) H3'e joker A → 0*3 = 0; B(yeni, bonus yok) = 3 → 3
    move7 = CandidateMove(
        placements=tuple([
            Placement(1, 1, "A", is_blank=True),
            Placement(1, 2, "B"),
        ]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(1, 1),
    )
    s = sc.score_move(move7, board)
    print(f"T7 Joker H3: {s} (beklenen 3)")
    assert s == 3

    # ---------- T8: Joker K2 hücresinde (kelime çarpanı yine işler) ----------
    # (3,3) K2'ye joker A → 0; B = 3 → toplam 3 * 2 = 6
    move8 = CandidateMove(
        placements=tuple([
            Placement(3, 3, "A", is_blank=True),
            Placement(3, 4, "B"),
        ]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(3, 3),
    )
    s = sc.score_move(move8, board)
    print(f"T8 Joker K2: {s} (beklenen 6)")
    assert s == 6

    # ---------- T9: Forced taş üzerinde bonus İPTAL ----------
    # ARABA tahtaya (7,7..11) konsun. (7,7) bonusu olsa bile forced; ama
    # bizim haritada (7,7) bonus yok zaten. Onun yerine: önce K2 olan (3,3)'e
    # AB yerleştir, sonra B'yi forced kabul ederek (3,4)'ten devam eden başka
    # bir kelime kuralım. (Sözlüğümüzde uzun kelime yok; sentetik test:
    # AB tahtada → BA dikey kuralım: (3,4) B forced, yeni (4,4) A → (4,4) bonus
    # haritasında "(5,5)"=row4,col4 H3. Ama (4,4) → (x=5,y=5) → H3.
    # Yan kelime YOK çünkü (3,4)'ün üstü/altı boş ve (4,4)'ten dikey "B-A" → 'BA'
    # sözlükte değil. Test sentetik, sadece ScoreCalculator'ı denetliyoruz.
    board.commit_move([Placement(3, 3, "A"), Placement(3, 4, "B")])
    # Yeni hamle: (4, 4) "A" — main word VERTICAL: (3,4)B forced + (4,4)A new
    move9 = CandidateMove(
        placements=tuple([Placement(4, 4, "A")]),
        main_word="BA",
        direction=Direction.VERTICAL,
        anchor=(4, 4),
    )
    s = sc.score_move(move9, board)
    # Hücreler: (3,4)B forced=3 (bonusu yok zaten),
    #           (4,4)A new → bonus map'te (4,4) H3 → 1*3=3
    # word_mult = 1 → toplam = 3 + 3 = 6
    print(f"T9 forced + H3 yeni: {s} (beklenen 6)")
    assert s == 6
    board.undo_move([Placement(3, 3, "A"), Placement(3, 4, "B")])

    # ---------- T10: Bingo +50 ----------
    move10 = CandidateMove(
        placements=tuple([
            Placement(7, 4, "A"), Placement(7, 5, "R"), Placement(7, 6, "A"),
            Placement(7, 7, "B"), Placement(7, 8, "A"), Placement(7, 9, "L"),
            Placement(7, 10, "A"),
        ]),
        main_word="ARABALA",   # sözlük denetlemiyoruz, sadece skor
        direction=Direction.HORIZONTAL,
        anchor=(7, 7),
    )
    # Hücre bonusları: (7,7) H2 değil mi? → (x=8,y=8) H2 listesinde değil. Yok.
    # Hiçbiri (7,*) bonus haritasında yok → harfler yalın.
    # 1+1+1+3+1+1+1 = 9, bingo +50 → 59
    s = sc.score_move(move10, board)
    print(f"T10 Bingo: {s} (beklenen 59)")
    assert s == 59

    # ---------- T11: Y3 +25 ----------
    sc_y3 = ScoreCalculator(size=15, y3_position=(0, 0))
    move11 = CandidateMove(
        placements=tuple([Placement(0, 0, "A"), Placement(0, 1, "B")]),
        main_word="AB",
        direction=Direction.HORIZONTAL,
        anchor=(0, 0),
    )
    # (0,0) bonus haritada yok. A=1+B=3=4. Y3 var → +25. Toplam 29.
    s = sc_y3.score_move(move11, board)
    print(f"T11 Y3: {s} (beklenen 29)")
    assert s == 29

    # ---------- T12: Yan kelime puanı eklenir ----------
    # AB tahtada (3,3)..(3,4). Üstüne (2,3) "B" + (2,4) "A" yani BA yatay.
    # Ana BA: (2,3)B + (2,4)A → 3+1 = 4. Bonus: (2,3)?(x=4,y=3) yok; (2,4)?(x=5,y=3) yok.
    # Yan kelimeler: (2,3)'ten dikey → (3,3)A, "BA" 2 harfli. B=3 + A(forced)=1 = 4.
    #   Yeni taş B'nin bonusu yok → çarpan yok.
    # Yan (2,4): (3,4)B forced ile dikey "AB" → A(yeni)=1, B(forced)=3 = 4.
    # Toplam: 4 + 4 + 4 = 12
    board.commit_move([Placement(3, 3, "A"), Placement(3, 4, "B")])
    move12 = CandidateMove(
        placements=tuple([Placement(2, 3, "B"), Placement(2, 4, "A")]),
        main_word="BA",
        direction=Direction.HORIZONTAL,
        anchor=(2, 3),
    )
    s = sc.score_move(move12, board)
    print(f"T12 Yan kelimeler: {s} (beklenen 12)")
    assert s == 12
    board.undo_move([Placement(3, 3, "A"), Placement(3, 4, "B")])

    print("\n--- ScoreCalculator tüm testler geçti ---")
