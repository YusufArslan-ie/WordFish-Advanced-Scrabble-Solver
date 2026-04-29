"""
Scrabble TR — Modül C: Hamle Üretici (Move Generator)

Modül A (Gaddag) + Modül B (ScrabbleBoard) üzerinde GADDAG tabanlı tam-uzay
arama yapar. Verili rack ve mevcut tahta için TÜM yasal aday hamleleri üretir.
Skor hesaplamaz — bu Modül D'nin işidir.

Algoritma (Gordon, 1994):
  Her view × her anchor için:
    1) Anchor hücresine rack'ten/joker'den bir harf yerleştir
       (cross-check & GADDAG edge & rack maskeleri AND'lenir).
    2) GADDAG'da reverse-prefix kısmında SOLA uzat:
         - Boş hücrede: rack/joker harf yerleştir (left_budget'i azalt)
         - Dolu hücrede: mevcut harfi GADDAG'da zorunlu olarak tüket
         - Her noktada SEP'i geçip Adım 3'e atlama seçeneği var
    3) SEP geçildikten sonra SAĞA uzat:
         - Boş hücre + terminal node + en az 1 rack taşı = aday hamle
         - Dolu hücrede: zorunlu, devam et
  Tekrar önleme: left_budget Modül B tarafından "leftmost-anchor-owns"
  kuralına göre hesaplanır.

Joker (?), sözlük yapısına gömülmez; sadece arama anında "rack'te olmayan
harfi de oynayabilirim" branch'i olarak ele alınır. Çapraz-kontrol ve
GADDAG edge maskeleri joker için de aynen geçerlidir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from gaddag import Gaddag, TurkishAlphabet
from board import ScrabbleBoard, BoardView, Direction, Placement


# =========================================================================
# Rack — oyuncunun istakası (consume/restore mantığı)
# =========================================================================

class Rack:
    """29 harf sayacı + joker sayacı. Recursion'da consume/restore edilir."""

    JOKER_CHAR = "?"

    def __init__(self, letters: Iterable[str] = (), blanks: int = 0) -> None:
        self._counts: list[int] = [0] * TurkishAlphabet.SIZE
        self._blanks: int = blanks
        for L in letters:
            self._counts[TurkishAlphabet.index(L)] += 1

    @classmethod
    def from_string(cls, s: str) -> "Rack":
        """'AYK?' gibi: '?' joker olarak yorumlanır; diğerleri TR harfi."""
        letters: list[str] = []
        blanks = 0
        for ch in s:
            if ch == cls.JOKER_CHAR:
                blanks += 1
            else:
                letters.append(ch)
        return cls(letters, blanks=blanks)

    def __len__(self) -> int:
        return sum(self._counts) + self._blanks

    def is_empty(self) -> bool:
        return len(self) == 0

    # Recursion içinde sıcak yol — saf int işlemleri:
    def has_letter(self, idx: int) -> bool:
        return self._counts[idx] > 0

    def has_blank(self) -> bool:
        return self._blanks > 0

    def consume_letter(self, idx: int) -> None:
        self._counts[idx] -= 1

    def restore_letter(self, idx: int) -> None:
        self._counts[idx] += 1

    def consume_blank(self) -> None:
        self._blanks -= 1

    def restore_blank(self) -> None:
        self._blanks += 1

    def available_mask(self) -> int:
        """Rack'teki gerçek harflerin 29-bit maskesi (joker hariç)."""
        mask = 0
        for i in range(TurkishAlphabet.SIZE):
            if self._counts[i] > 0:
                mask |= 1 << i
        return mask

    def __repr__(self) -> str:
        chars: list[str] = []
        for i, c in enumerate(self._counts):
            chars.extend([TurkishAlphabet.letter(i)] * c)
        return f"Rack('{''.join(chars)}{'?' * self._blanks}')"


# =========================================================================
# CandidateMove — bir aday hamle (skoru Modül D koyar)
# =========================================================================

@dataclass(frozen=True)
class CandidateMove:
    placements: tuple[Placement, ...]   # kanonik koordinat
    main_word: str                       # placement yönündeki ana kelime
    direction: Direction                 # H veya V
    anchor: tuple[int, int]              # kanonik anchor koord.

    def placement_key(self) -> frozenset:
        """Aynı fiziksel hamleyi farklı yön etiketiyle yakalamak için anahtar."""
        return frozenset(
            (p.row, p.col, p.letter, p.is_blank) for p in self.placements
        )

    def __repr__(self) -> str:
        cells = ",".join(f"({p.row},{p.col}){p.letter}{'*' if p.is_blank else ''}"
                         for p in self.placements)
        return f"<{self.direction.value} '{self.main_word}' [{cells}]>"


# =========================================================================
# MoveGenerator — ana arama motoru
# =========================================================================

class MoveGenerator:

    def __init__(self, gaddag: Gaddag, board: ScrabbleBoard) -> None:
        self.gaddag = gaddag
        self.board = board

    # --------- Public API ---------

    def generate(self, rack: Rack) -> list[CandidateMove]:
        """Verili rack için tüm yasal hamleleri üret (dedupe edilmiş)."""
        if rack.is_empty():
            return []
        moves: list[CandidateMove] = []
        # Yatay yön
        self._generate_in_view(self.board.horizontal, rack, moves)
        # Dikey yön (transpose trick: aynı algoritma, transpose edilmiş tahta)
        self._generate_in_view(self.board.vertical, rack, moves)
        return self._dedupe(moves)

    # --------- View içi sürücü ---------

    def _generate_in_view(
        self, view: BoardView, rack: Rack, moves: list[CandidateMove]
    ) -> None:
        for (ar, ac) in view.anchors():
            self._search_anchor(view, ar, ac, rack, moves)

    # --------- Anchor başlangıcı ---------

    def _search_anchor(
        self, view: BoardView, ar: int, ac: int,
        rack: Rack, moves: list[CandidateMove],
    ) -> None:
        cross_mask = view.cross_check_mask(ar, ac)
        if cross_mask == 0:
            return  # bu hücreye hiçbir harf konamaz

        budget = view.left_budget(ar, ac)
        root = self.gaddag.get_root()
        edge_mask = self.gaddag.edge_mask(root)
        candidate_mask = cross_mask & edge_mask
        if candidate_mask == 0:
            return  # hiçbir harf işe yaramaz (joker bile)

        rack_mask = rack.available_mask()
        placements: list[Placement] = []

        # 1) Gerçek harf ile anchor'a yerleş
        for L_idx in self._iter_bits(candidate_mask & rack_mask):
            L = TurkishAlphabet.letter(L_idx)
            next_node = self.gaddag.traverse(root, L)
            if next_node is None:
                continue
            rack.consume_letter(L_idx)
            placements.append(self._mk_placement(view, ar, ac, L, False))
            self._extend_left(
                view, ar, ac, ac - 1, next_node,
                budget, rack, placements, moves,
            )
            placements.pop()
            rack.restore_letter(L_idx)

        # 2) Joker ile anchor'a yerleş
        if rack.has_blank():
            for L_idx in self._iter_bits(candidate_mask):
                L = TurkishAlphabet.letter(L_idx)
                next_node = self.gaddag.traverse(root, L)
                if next_node is None:
                    continue
                rack.consume_blank()
                placements.append(self._mk_placement(view, ar, ac, L, True))
                self._extend_left(
                    view, ar, ac, ac - 1, next_node,
                    budget, rack, placements, moves,
                )
                placements.pop()
                rack.restore_blank()

    # --------- SOLA uzat (GADDAG reverse-prefix) ---------

    def _extend_left(
        self, view: BoardView, ar: int, ac: int, current_c: int,
        node, remaining_budget: int,
        rack: Rack, placements: list[Placement],
        moves: list[CandidateMove],
    ) -> None:
        # Seçenek A: SEP'i geçip sağa uzatmaya başla.
        # KRİTİK: SEP yalnızca sol parça gerçekten bittiyse geçilebilir.
        # Aksi halde, forced sol kuyruğu (örn. ARABA'nın AAAB'sı) hiç sözlükte
        # aranmadığı halde reconstruct'ta ana kelimeye yapışır → false-positive.
        can_cross_separator = (
            current_c < 0 or view.is_empty(ar, current_c)
        )
        if can_cross_separator:
            sep_node = self.gaddag.cross_separator(node)
            if sep_node is not None:
                self._extend_right(
                    view, ar, ac, ac + 1, sep_node, rack, placements, moves,
                )

        # Seçenek B: bir adım daha sola git
        if current_c < 0:
            return

        if view.is_filled(ar, current_c):
            # Zorunlu: tahtadaki harfi tüket (budget harcanmaz)
            existing = view.letter_at(ar, current_c)
            forced_node = self.gaddag.traverse(node, existing)
            if forced_node is not None:
                self._extend_left(
                    view, ar, ac, current_c - 1, forced_node,
                    remaining_budget, rack, placements, moves,
                )
            return

        # Boş hücre — budget yoksa sola devam edemeyiz
        if remaining_budget <= 0:
            return

        cross_mask = view.cross_check_mask(ar, current_c)
        if cross_mask == 0:
            return
        edge_mask = self.gaddag.edge_mask(node)
        candidate_mask = cross_mask & edge_mask
        if candidate_mask == 0:
            return
        rack_mask = rack.available_mask()

        # Gerçek harf
        for L_idx in self._iter_bits(candidate_mask & rack_mask):
            L = TurkishAlphabet.letter(L_idx)
            next_node = self.gaddag.traverse(node, L)
            if next_node is None:
                continue
            rack.consume_letter(L_idx)
            placements.append(self._mk_placement(view, ar, current_c, L, False))
            self._extend_left(
                view, ar, ac, current_c - 1, next_node,
                remaining_budget - 1, rack, placements, moves,
            )
            placements.pop()
            rack.restore_letter(L_idx)

        # Joker
        if rack.has_blank():
            for L_idx in self._iter_bits(candidate_mask):
                L = TurkishAlphabet.letter(L_idx)
                next_node = self.gaddag.traverse(node, L)
                if next_node is None:
                    continue
                rack.consume_blank()
                placements.append(self._mk_placement(view, ar, current_c, L, True))
                self._extend_left(
                    view, ar, ac, current_c - 1, next_node,
                    remaining_budget - 1, rack, placements, moves,
                )
                placements.pop()
                rack.restore_blank()

    # --------- SAĞA uzat (GADDAG suffix, SEP sonrası) ---------

    def _extend_right(
        self, view: BoardView, ar: int, ac: int, current_c: int,
        node, rack: Rack, placements: list[Placement],
        moves: list[CandidateMove],
    ) -> None:
        # Kelime burada bitebilir mi? (current_c boş veya sınır dışı)
        can_terminate = (
            current_c >= view.size or view.is_empty(ar, current_c)
        )
        if (
            can_terminate
            and self.gaddag.is_terminal(node)
            and len(placements) > 0
        ):
            self._record(view, ar, ac, placements, moves)

        if current_c >= view.size:
            return

        if view.is_filled(ar, current_c):
            # Zorunlu: tahtadaki harfi tüket
            existing = view.letter_at(ar, current_c)
            forced_node = self.gaddag.traverse(node, existing)
            if forced_node is not None:
                self._extend_right(
                    view, ar, ac, current_c + 1, forced_node,
                    rack, placements, moves,
                )
            return

        # Boş hücre — rack'ten / joker'den harf yerleştir
        cross_mask = view.cross_check_mask(ar, current_c)
        if cross_mask == 0:
            return
        edge_mask = self.gaddag.edge_mask(node)
        candidate_mask = cross_mask & edge_mask
        if candidate_mask == 0:
            return
        rack_mask = rack.available_mask()

        # Gerçek harf
        for L_idx in self._iter_bits(candidate_mask & rack_mask):
            L = TurkishAlphabet.letter(L_idx)
            next_node = self.gaddag.traverse(node, L)
            if next_node is None:
                continue
            rack.consume_letter(L_idx)
            placements.append(self._mk_placement(view, ar, current_c, L, False))
            self._extend_right(
                view, ar, ac, current_c + 1, next_node,
                rack, placements, moves,
            )
            placements.pop()
            rack.restore_letter(L_idx)

        # Joker
        if rack.has_blank():
            for L_idx in self._iter_bits(candidate_mask):
                L = TurkishAlphabet.letter(L_idx)
                next_node = self.gaddag.traverse(node, L)
                if next_node is None:
                    continue
                rack.consume_blank()
                placements.append(self._mk_placement(view, ar, current_c, L, True))
                self._extend_right(
                    view, ar, ac, current_c + 1, next_node,
                    rack, placements, moves,
                )
                placements.pop()
                rack.restore_blank()

    # --------- Yardımcılar ---------

    @staticmethod
    def _iter_bits(mask: int) -> Iterator[int]:
        """29-bit maskedeki set bit indekslerini ver."""
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    @staticmethod
    def _mk_placement(
        view: BoardView, vr: int, vc: int, letter: str, is_blank: bool,
    ) -> Placement:
        """View koord. → kanonik koord. Placement."""
        gr, gc = view.to_canonical(vr, vc)
        return Placement(gr, gc, letter, is_blank)

    def _record(
        self, view: BoardView, ar: int, ac: int,
        placements: list[Placement], moves: list[CandidateMove],
    ) -> None:
        """Bir kelimeyi listeye CandidateMove olarak ekle."""
        word, anchor_canon = self._reconstruct_word(view, ar, placements)
        moves.append(CandidateMove(
            placements=tuple(placements),
            main_word=word,
            direction=view.direction,
            anchor=view.to_canonical(ar, ac),
        ))

    def _reconstruct_word(
        self, view: BoardView, ar: int, placements: list[Placement],
    ) -> tuple[str, tuple[int, int]]:
        """
        Tahta + placements'tan placement yönündeki ana kelimeyi oku.
        Forced (mevcut) ve placed (yeni) hücreleri birleştirerek string üretir.
        """
        # Placements kanonik; view-local sütunlara çevir
        if view.direction == Direction.HORIZONTAL:
            placed_cols = [p.col for p in placements]
            placed_letter = {p.col: p.letter for p in placements}
        else:
            placed_cols = [p.row for p in placements]   # transpose
            placed_letter = {p.row: p.letter for p in placements}

        leftmost = min(placed_cols)
        rightmost = max(placed_cols)
        # Sola, sağa varsa forced kuyruğu uzat
        c = leftmost - 1
        while c >= 0 and view.is_filled(ar, c):
            leftmost = c
            c -= 1
        c = rightmost + 1
        while c < view.size and view.is_filled(ar, c):
            rightmost = c
            c += 1

        chars: list[str] = []
        for c in range(leftmost, rightmost + 1):
            if c in placed_letter:
                chars.append(placed_letter[c])
            else:
                ch = view.letter_at(ar, c)
                assert ch is not None, f"Beklenmedik boş hücre ({ar},{c})"
                chars.append(ch)
        return "".join(chars), (ar, leftmost)  # ikinci parametre kullanılmıyor

    @staticmethod
    def _dedupe(moves: list[CandidateMove]) -> list[CandidateMove]:
        """
        Aynı fiziksel yerleştirmeyi farklı yönde üreten dups'ları kaldır.
        Tipik durum: tek-harf hamleler hem H hem V'de bulunabilir.
        """
        seen: set[frozenset] = set()
        unique: list[CandidateMove] = []
        for m in moves:
            key = m.placement_key()
            if key in seen:
                continue
            seen.add(key)
            unique.append(m)
        return unique


# =========================================================================
# Hızlı sınama
# =========================================================================

if __name__ == "__main__":
    # Mini sözlük
    g = Gaddag()
    for w in [
        "AT", "ATA", "ATEŞ", "ARABA", "ARI", "ARILAR",
        "BAL", "BALIK", "BALİ", "BAR",
        "EV", "EVE", "EVET", "EVLİ",
        "KEDİ", "KEL", "KELE",
        "ÇAY", "ÇİĞ", "ÇEK",
        "RA", "AB", "AL", "ALA", "ALAN", "ELE",
    ]:
        g.add_word(w)

    board = ScrabbleBoard(g, size=15)
    gen = MoveGenerator(g, board)

    # ---------------- 1) Boş tahta ----------------
    print("=== TEST 1: Boş tahta, rack='ARABA' ===")
    rack = Rack.from_string("ARABA")
    moves = gen.generate(rack)
    main_words = sorted({m.main_word for m in moves})
    print(f"Bulunan ana kelimeler ({len(main_words)}): {main_words}")
    # ARABA, AB, BAR, RA bekleniyor (rack: A,R,A,B,A)
    assert "ARABA" in main_words, "ARABA bulunmalı"
    assert "BAR" in main_words, "BAR bulunmalı"

    # Tüm hamleler boş tahtada merkez (7,7)'yi kapsamalı
    for m in moves:
        cols = [p.col for p in m.placements] if m.direction == Direction.HORIZONTAL else None
        rows = [p.row for p in m.placements] if m.direction == Direction.VERTICAL else None
        if m.direction == Direction.HORIZONTAL:
            assert any(p.row == 7 and p.col == 7 for p in m.placements), \
                f"H hamle merkezi kapsamıyor: {m}"
        else:
            assert any(p.row == 7 and p.col == 7 for p in m.placements), \
                f"V hamle merkezi kapsamıyor: {m}"
    print(f"✓ Tüm {len(moves)} hamle merkezi kapsıyor.\n")

    # ---------------- 2) ARABA oynandı, rack ile devam ----------------
    print("=== TEST 2: ARABA tahtada, rack='TEŞ' (ATEŞ kurmak için) ===")
    board.commit_move([
        Placement(7, 7, "A"), Placement(7, 8, "R"), Placement(7, 9, "A"),
        Placement(7, 10, "B"), Placement(7, 11, "A"),
    ])
    rack = Rack.from_string("TEŞ")
    moves = gen.generate(rack)
    main_words = sorted({m.main_word for m in moves})
    print(f"Bulunan ana kelimeler ({len(main_words)}): {main_words}")
    # En azından ATEŞ (dikey, A bağlı)
    assert "ATEŞ" in main_words, "ATEŞ bulunmalı"
    print("✓ ATEŞ bulundu.\n")

    # ---------------- 3) Cross-check filtresi ----------------
    print("=== TEST 3: Geçersiz çapraz kelime üreten hamleler reddedilmeli ===")
    # Üretilen her hamle için, oluşan TÜM çapraz kelimeleri sözlükten doğrula.
    # Burada ilk hamle setinde ARABA tahtada; (6,7)'ye konacak X için "X"+"A"
    # 2-harfli geçerli kelime olmalı (sözlüğümüzde 'RA' var → R kabul edilir).
    # Doğrudan kontrol: rack='R' ile (6,7)'ye R yerleştirme bir hamle olarak
    # bulunmalı (RA dikey kelimesi).
    rack = Rack.from_string("R")
    moves = gen.generate(rack)
    found_ra = any(
        m.main_word == "RA" and any(
            p.row == 6 and p.col == 7 for p in m.placements
        )
        for m in moves
    )
    print(f"R tek harfini (6,7)'ye koyup RA oluşturma bulundu mu? {found_ra}")
    assert found_ra, "RA dikey hamlesi bulunmalı"
    print("✓ Cross-check filtresi RA'ya izin verdi.\n")

    # ---------------- 4) Joker ----------------
    print("=== TEST 4: Joker ile anchor harfi tutturma ===")
    # Tahtayı sıfırla, ARABA'yı sil
    board.undo_move([
        Placement(7, 7, "A"), Placement(7, 8, "R"), Placement(7, 9, "A"),
        Placement(7, 10, "B"), Placement(7, 11, "A"),
    ])
    assert board.is_empty_board()

    # Rack: '?' tek joker. ARI, AB, AL, AT, RA, EV, ... gibi 2-harfliler
    # joker tek başınayken... 1 joker ile 2-harfli kelime kurmak için
    # sadece anchor (merkez) hücrede 1 harf konabilir → 1-harfli kelime
    # gerek (yok). Bu yüzden joker tek başına hiçbir hamle üretmemeli.
    rack = Rack.from_string("?")
    moves = gen.generate(rack)
    print(f"Sadece joker ile bulunan hamleler: {len(moves)} (beklenen: 0)")
    assert len(moves) == 0

    # Rack: 'A?' → joker bir harf yerine geçerek 2-harfli AB/AT/AR... kurabilir
    rack = Rack.from_string("A?")
    moves = gen.generate(rack)
    main_words = sorted({m.main_word for m in moves})
    print(f"Rack='A?' ile ana kelimeler: {main_words}")
    # AT, AB, AL en az birinin joker ile kurulması gerekir (T/B/L joker)
    assert any(w in main_words for w in ("AT", "AB", "AL")), \
        "Joker ile en az bir 2-harfli kurulmalı"
    # Joker kullanan hamle var mı?
    has_blank_move = any(any(p.is_blank for p in m.placements) for m in moves)
    assert has_blank_move, "Joker bayrağı taşıyan hamle bulunmalı"
    print("✓ Joker arama uzayını doğru biçimde genişletti.\n")

    # ---------------- 5) Tekrar önleme ----------------
    print("=== TEST 5: Aynı hamle iki kez üretilmiyor ===")
    rack = Rack.from_string("ARI")
    moves = gen.generate(rack)
    keys = [m.placement_key() for m in moves]
    assert len(keys) == len(set(keys)), "Dedupe başarısız"
    print(f"✓ {len(moves)} hamlenin hepsi benzersiz.\n")

    # ---------------- 6) Yatay + dikey (transpose) ----------------
    print("=== TEST 6: Hem yatay hem dikey hamleler bulunuyor ===")
    rack = Rack.from_string("BAL")
    moves = gen.generate(rack)
    h_count = sum(1 for m in moves if m.direction == Direction.HORIZONTAL)
    v_count = sum(1 for m in moves if m.direction == Direction.VERTICAL)
    print(f"Yatay: {h_count}, Dikey: {v_count}")
    # Boş tahtada her yatay hamle için aynı kelimeyi dikey de üretebilir
    # (single-cell merkez örtüşmesi hariç dedupe sonrası farklı sayılar
    # da olabilir ama her ikisinin de >0 olması gerekir)
    assert h_count > 0 and v_count > 0
    print("✓ Hem H hem V hamleleri üretiliyor.\n")

    print("--- Modül C sınamaları geçti ---")

    # ---------------- 7) Sağlamlık: hiçbir bulunan kelime sözlük dışı olmasın ----------------
    print("\n=== TEST 7 (sağlamlık): Üretilen tüm ana kelimeler sözlükte mi? ===")
    # Tahtayı temizle ve birkaç senaryoda kapsamlı test yap
    board2 = ScrabbleBoard(g, size=15)
    gen2 = MoveGenerator(g, board2)

    scenarios = [
        # (rack, [yerleştirilecek hamleler])
        ("ARABA", []),
        ("BAL", []),
        ("ÇİĞ", []),
        ("EVET", []),
    ]
    for rack_str, pre_moves in scenarios:
        # Pre-moves'u oyna
        for pl in pre_moves:
            board2.commit_move([pl])
        ms = gen2.generate(Rack.from_string(rack_str))
        invalid = [m for m in ms if not g.contains(m.main_word)]
        if invalid:
            print(f"❌ rack='{rack_str}': sözlük dışı kelimeler:", [m.main_word for m in invalid[:5]])
        else:
            print(f"✓ rack='{rack_str}': {len(ms)} hamle, hepsi sözlükte.")
        assert not invalid, f"Sözlük dışı kelime üretildi: {invalid[0]}"
        # Tahtayı sıfırla
        for pl in pre_moves:
            board2.undo_move([pl])

    # ARABA tahtadayken TEŞ ile devam senaryosu
    board3 = ScrabbleBoard(g, size=15)
    board3.commit_move([
        Placement(7, 7, "A"), Placement(7, 8, "R"), Placement(7, 9, "A"),
        Placement(7, 10, "B"), Placement(7, 11, "A"),
    ])
    gen3 = MoveGenerator(g, board3)
    ms = gen3.generate(Rack.from_string("TEŞ"))
    invalid = [m for m in ms if not g.contains(m.main_word)]
    main_words = sorted({m.main_word for m in ms})
    print(f"✓ ARABA + rack='TEŞ': ana kelimeler = {main_words}")
    assert not invalid, f"Sözlük dışı: {invalid[:3]}"
    assert "ATEŞ" in main_words, "ATEŞ kaybolmamalı"

    print("\n--- Tüm sağlamlık testleri geçti ---")
