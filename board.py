"""
Scrabble TR — Modül B: Tahta Durumu ve İstihbarat

ScrabbleBoard, oyun tahtasının tek doğru kaynağıdır (single source of truth)
ve Modül C (Hamle Üretici) için körü körüne aramayı engelleyen üç filtreyi
sağlar:

  1) Anchor hücreleri   — kelimenin temas etmek zorunda olduğu çıpa kareler
  2) Cross-check maske  — bir hücreye konabilecek harfler (29-bit bitmask)
  3) Left budget        — anchor'dan sola serbest uzatma payı (tekrar önleme)

Transpose Trick:
  Yatay ve dikey kelime aramaları için iki ayrı algoritma yazmak yerine,
  tahtayı iki BoardView olarak tutuyoruz: HORIZONTAL (kanonik) ve VERTICAL
  (90° transpose edilmiş). Her iki view de "soldan sağa yerleştirme" gibi
  davranır. Modül C tek bir algoritmayı her iki view üzerinde de çalıştırarak
  tüm hamleleri (yatay + dikey) bulur. Sonuç koordinatları view'in
  to_canonical() metodu ile global koordinata çevrilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator, Optional

from gaddag import Gaddag, TurkishAlphabet


# =========================================================================
# Yardımcı tipler
# =========================================================================

class Direction(str, Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"


@dataclass(frozen=True)
class Tile:
    """Tahtaya yerleşmiş bir taş."""
    letter: str            # Tahtada görünen büyük Türkçe harf
    is_blank: bool = False # Joker olarak yerleştirildi mi (puanı 0)?

    def __post_init__(self):
        if not TurkishAlphabet.is_valid_letter(self.letter):
            raise ValueError(f"Geçersiz harf: {self.letter!r}")


@dataclass(frozen=True)
class Placement:
    """Bir hamle içerisindeki tek taş yerleştirmesi (kanonik koord.)."""
    row: int
    col: int
    letter: str
    is_blank: bool = False


# =========================================================================
# Tek yönlü tahta görünümü
# =========================================================================

class BoardView:
    """
    Tahtanın tek bir yöndeki "soldan sağa yerleştirme" görünümü.

    HORIZONTAL view: kanonik koordinatların aynısı (r, c).
    VERTICAL  view : kanonik (R, C) → view (C, R) eşlemesiyle saklanır.
                     Böylece dikey kanonik kelimeler, view'in satırlarında
                     yatay kelime gibi görünür ve aynı algoritma kullanılır.

    Modül C için sorgu API'si (her iki view'de de aynı):
        size                          : Kenar uzunluğu (ör. 15)
        in_bounds(r, c)               : Sınır kontrolü
        is_filled / is_empty(r, c)    : Hücre dolu mu?
        letter_at(r, c)               : Hücredeki harf (yoksa None)
        tile_at(r, c)                 : Hücredeki Tile nesnesi (joker bilgisi)
        is_anchor(r, c)               : Anchor mı?
        anchors()                     : Anchor (r, c) iterator'u
        cross_check_mask(r, c)        : 29-bit izinli harf maskesi
        left_budget(r, c)             : Anchor'dan sola serbest pay
        is_empty_board()              : İlk hamle mi?
        starting_square               : (center, center)
        to_canonical(r, c)            : View koord. → kanonik koord.
    """

    def __init__(self, gaddag: Gaddag, size: int, direction: Direction):
        self.gaddag = gaddag
        self.size = size
        self.direction = direction

        # None ya da Tile
        self._cells: list[list[Optional[Tile]]] = [
            [None] * size for _ in range(size)
        ]
        # Boş hücreler için 29-bit cross-check maskesi
        self._cross_masks: list[list[int]] = [
            [TurkishAlphabet.FULL_MASK] * size for _ in range(size)
        ]
        # Anchor flag matrisi (boş + dolu komşulu hücreler)
        self._anchor_flags: list[list[bool]] = [
            [False] * size for _ in range(size)
        ]
        # Hızlı iterasyon için anchor listesi
        self._anchor_list: list[tuple[int, int]] = []
        self._tile_count: int = 0

    # --------- Sınır / koordinat ---------

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def to_canonical(self, r: int, c: int) -> tuple[int, int]:
        """View koordinatını kanonik (yatay) koordinata çevir."""
        if self.direction == Direction.HORIZONTAL:
            return r, c
        return c, r  # transpose

    # --------- Hücre erişimi ---------

    def is_filled(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self._cells[r][c] is not None

    def is_empty(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self._cells[r][c] is None

    def letter_at(self, r: int, c: int) -> Optional[str]:
        if not self.in_bounds(r, c):
            return None
        tile = self._cells[r][c]
        return tile.letter if tile else None

    def tile_at(self, r: int, c: int) -> Optional[Tile]:
        if not self.in_bounds(r, c):
            return None
        return self._cells[r][c]

    # --------- İstihbarat sorguları (Modül C buradan besleniyor) ---------

    def is_anchor(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self._anchor_flags[r][c]

    def anchors(self) -> Iterator[tuple[int, int]]:
        """Mevcut anchor (r, c) çiftlerini yield eder (view koord.)."""
        return iter(self._anchor_list)

    def cross_check_mask(self, r: int, c: int) -> int:
        """
        Boş bir hücreye konabilecek harflerin 29-bit maskesi.
        Hücre doluysa ya da sınır dışıysa 0 döner.

        Modül C bunu rack maskesi ve GADDAG edge maskesi ile AND'leyerek
        canlı dalları tek bit-işleminde belirler.
        """
        if not self.in_bounds(r, c) or self._cells[r][c] is not None:
            return 0
        return self._cross_masks[r][c]

    def left_budget(self, r: int, c: int) -> int:
        """
        Anchor (r, c)'den sola; başka bir anchor'a, dolu hücreye veya tahta
        kenarına çarpmadan açılabilecek serbest hücre sayısı.

        Modül C en fazla bu kadar rack harfini sola yerleştirip ardından
        SEP'i geçmek zorundadır. (Sol komşu doluysa 0 döner; o durumda
        Modül C zorunlu sol parçayı `letter_at()` ile yürür.)
        """
        if c == 0 or not self.is_empty(r, c - 1):
            return 0
        count = 0
        col = c - 1
        # (r, c-1) boş; başka bir anchor'a/dolu/edge'e çarpana kadar say.
        while (
            col >= 0
            and self._cells[r][col] is None
            and not self._anchor_flags[r][col]
        ):
            count += 1
            col -= 1
        return count

    # --------- Empty board / first move ---------

    def is_empty_board(self) -> bool:
        return self._tile_count == 0

    @property
    def starting_square(self) -> tuple[int, int]:
        c = self.size // 2
        return (c, c)

    # --------- ScrabbleBoard tarafından çağrılan iç mutasyonlar ---------

    def _set_tile(self, r: int, c: int, tile: Tile) -> None:
        if self._cells[r][c] is None:
            self._tile_count += 1
        self._cells[r][c] = tile

    def _clear_tile(self, r: int, c: int) -> None:
        if self._cells[r][c] is not None:
            self._tile_count -= 1
        self._cells[r][c] = None

    def _recompute_all(self) -> None:
        """Tüm anchor + cross-check matrisini sıfırdan oluştur."""
        self._anchor_list.clear()
        empty_board = self._tile_count == 0
        center_r, center_c = self.starting_square
        for r in range(self.size):
            for c in range(self.size):
                if self._cells[r][c] is not None:
                    self._anchor_flags[r][c] = False
                    self._cross_masks[r][c] = 0
                    continue
                # Boş hücre
                if empty_board:
                    is_anc = (r == center_r and c == center_c)
                else:
                    is_anc = self._has_filled_neighbor(r, c)
                self._anchor_flags[r][c] = is_anc
                if is_anc:
                    self._anchor_list.append((r, c))
                self._cross_masks[r][c] = self._compute_cross_mask(r, c)

    def _has_filled_neighbor(self, r: int, c: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < self.size
                and 0 <= nc < self.size
                and self._cells[nr][nc] is not None
            ):
                return True
        return False

    def _compute_cross_mask(self, r: int, c: int) -> int:
        """
        Hücreye konacak harfin perpendicular (sütun) yönünde komşu
        harflerle birlikte sözlükte var olan bir kelime oluşturmasını
        sağlayan harflerin 29-bit maskesi.
        """
        # Yukarıdaki ardışık dolu harfler (sırayla yukarıdan aşağıya)
        upper: list[str] = []
        rr = r - 1
        while rr >= 0 and self._cells[rr][c] is not None:
            upper.append(self._cells[rr][c].letter)
            rr -= 1
        upper.reverse()

        # Aşağıdaki ardışık dolu harfler (yukarıdan aşağıya)
        lower: list[str] = []
        rr = r + 1
        while rr < self.size and self._cells[rr][c] is not None:
            lower.append(self._cells[rr][c].letter)
            rr += 1

        if not upper and not lower:
            return TurkishAlphabet.FULL_MASK

        upper_str = "".join(upper)
        lower_str = "".join(lower)
        mask = 0
        for i in range(TurkishAlphabet.SIZE):
            L = TurkishAlphabet.letter(i)
            if self.gaddag.contains(upper_str + L + lower_str):
                mask |= 1 << i
        return mask

    # --------- Pretty print ---------

    def __str__(self) -> str:
        rows = []
        for r in range(self.size):
            buf = []
            for c in range(self.size):
                tile = self._cells[r][c]
                if tile is None:
                    buf.append("·")
                else:
                    buf.append(
                        tile.letter.lower() if tile.is_blank else tile.letter
                    )
            rows.append(" ".join(buf))
        return "\n".join(rows)


# =========================================================================
# Üst seviye facade
# =========================================================================

class ScrabbleBoard:
    """
    Tahtanın iki senkron view'ını tutan ana sınıf.

    Modül C kullanım örüntüsü:

        for view in (board.horizontal, board.vertical):
            for (r, c) in view.anchors():
                mask = view.cross_check_mask(r, c)
                budget = view.left_budget(r, c)
                # ... GADDAG ile arama ...
                # Bulunan hamlenin koordinatlarını kanonik'e çevir:
                gr, gc = view.to_canonical(r, c)

    Mutasyon:
        board.commit_move([Placement(...), ...])   # tüm taşlar + refresh
        board.undo_move([...])                     # tahtayı geri al
    """

    DEFAULT_SIZE = 15

    def __init__(self, gaddag: Gaddag, size: int = DEFAULT_SIZE):
        self.gaddag = gaddag
        self.size = size
        self.horizontal = BoardView(gaddag, size, Direction.HORIZONTAL)
        self.vertical = BoardView(gaddag, size, Direction.VERTICAL)
        self._refresh()

    # --------- Kanonik tahta sorguları (kısa yollar) ---------

    def is_filled(self, r: int, c: int) -> bool:
        return self.horizontal.is_filled(r, c)

    def letter_at(self, r: int, c: int) -> Optional[str]:
        return self.horizontal.letter_at(r, c)

    def tile_at(self, r: int, c: int) -> Optional[Tile]:
        return self.horizontal.tile_at(r, c)

    def is_empty_board(self) -> bool:
        return self.horizontal.is_empty_board()

    # --------- Mutasyon ---------

    def place_tile(
        self, r: int, c: int, letter: str, is_blank: bool = False
    ) -> None:
        """Tek taş yerleştir (refresh tetiklemez — toplu işlem için)."""
        if not self.horizontal.in_bounds(r, c):
            raise IndexError(f"({r},{c}) tahta dışı")
        if self.is_filled(r, c):
            raise ValueError(f"({r},{c}) zaten dolu")
        tile = Tile(letter=letter, is_blank=is_blank)
        self.horizontal._set_tile(r, c, tile)
        self.vertical._set_tile(c, r, tile)   # transpose

    def remove_tile(self, r: int, c: int) -> None:
        """Tek taşı kaldır (refresh tetiklemez)."""
        self.horizontal._clear_tile(r, c)
        self.vertical._clear_tile(c, r)

    def commit_move(self, placements: Iterable[Placement]) -> None:
        """Hamledeki tüm taşları yerleştir ve istihbaratı yenile."""
        for p in placements:
            self.place_tile(p.row, p.col, p.letter, p.is_blank)
        self._refresh()

    def undo_move(self, placements: Iterable[Placement]) -> None:
        for p in placements:
            self.remove_tile(p.row, p.col)
        self._refresh()

    def _refresh(self) -> None:
        """Anchor + cross-check verisini her iki view için yeniden hesapla."""
        self.horizontal._recompute_all()
        self.vertical._recompute_all()

    # --------- Bilgi ---------

    def stats(self) -> dict:
        return {
            "size": self.size,
            "tiles_on_board": self.horizontal._tile_count,
            "h_anchors": len(self.horizontal._anchor_list),
            "v_anchors": len(self.vertical._anchor_list),
        }

    def __str__(self) -> str:
        return str(self.horizontal)


# =========================================================================
# Hızlı sınama
# =========================================================================

if __name__ == "__main__":
    # Mini sözlük
    g = Gaddag()
    for w in [
        "AT", "ATA", "ARABA", "ARI", "BAL", "BALIK",
        "EV", "EVE", "EVET", "KEDİ", "ÇİĞ", "ATEŞ", "RA", "AB",
    ]:
        g.add_word(w)

    board = ScrabbleBoard(g, size=15)

    # 1) Boş tahta: tek anchor merkezde
    assert board.is_empty_board()
    assert list(board.horizontal.anchors()) == [(7, 7)]
    assert list(board.vertical.anchors()) == [(7, 7)]
    print("✓ Boş tahta: tek anchor merkezde.")

    # 2) ARABA yatay olarak (7, 7..11) oynanıyor
    board.commit_move([
        Placement(7, 7, "A"),
        Placement(7, 8, "R"),
        Placement(7, 9, "A"),
        Placement(7, 10, "B"),
        Placement(7, 11, "A"),
    ])

    print("\nTahta:")
    print(board)
    print(board.stats())

    # 3) Anchor sayıları her iki view'de eşit olmalı (aynı pozisyonlar)
    h_set = set(board.horizontal.anchors())
    v_set_canonical = {
        board.vertical.to_canonical(r, c) for (r, c) in board.vertical.anchors()
    }
    assert h_set == v_set_canonical, "Yatay/dikey anchor pozisyonları farklı!"
    print(f"\n✓ Yatay ve dikey view aynı kanonik anchor kümesini üretiyor: {len(h_set)} adet.")

    # 4) Cross-check: (6, 7) hücresine konacak X için "X" + "A" sözlükte olmalı
    #    Sözlüğümüzde 'AB' var; ama 'AB' = X+A yazıldığında X='A', kelime='AA' (yok).
    #    Tek harfli üst boş, alt 'A' var → kelime "X"+"A" = 2 harfli. Sözlükte
    #    'A' ile biten 2 harfli: yok aslında ama 'RA' var → X='R' geçerli.
    mask_67 = board.horizontal.cross_check_mask(6, 7)
    allowed_67 = TurkishAlphabet.mask_to_letters(mask_67)
    print(f"\n(6,7) izinli harfler: {sorted(allowed_67)}")
    assert "R" in allowed_67, "RA sözlükte var → R izinli olmalı"

    # 5) Doğrulama: (6, 8) hücresinde alt komşu 'R'. Sözlükte X+R yok → boş maske
    mask_68 = board.horizontal.cross_check_mask(6, 8)
    allowed_68 = TurkishAlphabet.mask_to_letters(mask_68)
    print(f"(6,8) izinli harfler: {sorted(allowed_68)}")
    assert allowed_68 == frozenset(), "X+R kelimesi sözlükte yok"

    # 6) left_budget: (7, 6) anchor → sol komşu boş, sola kadar boş, edge'e
    #    kadar tüm hücreler anchor değil → budget = 6 (kolonlar 0..5)
    assert board.horizontal.is_anchor(7, 6)
    bud = board.horizontal.left_budget(7, 6)
    print(f"\nleft_budget(7, 6) = {bud} (beklenen: 6)")
    assert bud == 6

    # 7) left_budget: (7, 12) anchor → sol komşu (7, 11) DOLU → budget = 0
    assert board.horizontal.is_anchor(7, 12)
    assert board.horizontal.left_budget(7, 12) == 0
    print("✓ left_budget(7, 12) = 0 (sol komşu dolu, Case B).")

    # 8) (8, 7) anchor → ARABA'nın altında; sol (8, 6) boş ama (8, 6) anchor mı?
    #    (8, 6) sol/sağ/üst/alt: (8,5) boş, (8,7) anchor=evet ama bu dolu komşu
    #    değil; (7,6) boş; (9,6) boş. Yani (8, 6) DOLU komşusu yok → anchor değil.
    #    Yani (8, 7)'den sola gidersek budget = 7 (kolonlar 0..6 hepsi anchor
    #    olmadan boş — aslında değil, bakalım).
    if board.horizontal.is_anchor(8, 7):
        bud_87 = board.horizontal.left_budget(8, 7)
        print(f"left_budget(8, 7) = {bud_87}")

    # 9) Transpose tutarlılığı: ARABA dikey view'de "sütun" olarak görünür,
    #    yani v.cells[7..11][7] dolu (her biri bir tile).
    for c_v in range(7, 12):
        assert board.vertical.is_filled(c_v, 7), f"v[{c_v}][7] dolu olmalı"
    # v(7, 8) → kanonik (8, 7) → boş
    assert board.vertical.is_empty(7, 8)
    print("\n✓ Transpose tutarlı: vertical view canonical'ın transpose'u.")

    # 10) Undo testi
    board.undo_move([
        Placement(7, 7, "A"), Placement(7, 8, "R"), Placement(7, 9, "A"),
        Placement(7, 10, "B"), Placement(7, 11, "A"),
    ])
    assert board.is_empty_board()
    assert list(board.horizontal.anchors()) == [(7, 7)]
    print("✓ Undo sonrası tahta boş, anchor yine merkezde.")

    print("\n--- Modül B sınamaları geçti ---")
