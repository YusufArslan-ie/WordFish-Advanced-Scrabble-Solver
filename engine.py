"""
Scrabble TR — Engine (tüm modülleri birleştiren facade)

Kullanım:
    engine = ScrabbleEngine.from_dictionary("scrabble_words_tr.txt")

    # Tahtadaki mevcut kelimeleri yerleştir:
    engine.place_word("ARABA", row=7, col=7, direction="H")
    engine.place_word("ATEŞ",  row=7, col=7, direction="V")

    # (Opsiyonel) Y3 hücresinin konumu:
    engine.set_y3((3, 4))   # ya da engine.set_y3(None)

    # Hamle önerileri (yüksek skordan düşük skora):
    suggestions = engine.suggest_moves(rack="TEŞK?", top_n=10)
    for score, move in suggestions:
        print(f"{score:>4}  {move.main_word}  {move.direction.value}  {move.placements}")

Joker:
    rack içinde '?' karakteri = joker.
    Mevcut kelimelerde joker belirtmek için place_word(..., blank_indices=[i, ...]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from gaddag import Gaddag
from board import ScrabbleBoard, Placement, Direction
from move_generator import MoveGenerator, Rack, CandidateMove
from score_calculator import ScoreCalculator


def _parse_direction(d: Union[str, Direction]) -> Direction:
    if isinstance(d, Direction):
        return d
    s = d.upper()
    if s in ("H", "HORIZONTAL", "YATAY"): return Direction.HORIZONTAL
    if s in ("V", "VERTICAL", "DİKEY", "DIKEY"): return Direction.VERTICAL
    raise ValueError(f"Bilinmeyen yön: {d!r}")


class ScrabbleEngine:

    DEFAULT_SIZE = 15

    def __init__(self, gaddag: Gaddag, size: int = DEFAULT_SIZE) -> None:
        self.gaddag = gaddag
        self.size = size
        self.board = ScrabbleBoard(gaddag, size=size)
        self.score_calc = ScoreCalculator(size=size, y3_position=(0, 0))
        self.move_gen = MoveGenerator(gaddag, self.board)

    # ---------- Sözlüğü yükle (cache'li) ----------

    @classmethod
    def from_dictionary(
        cls,
        dict_path: Union[str, Path],
        *,
        size: int = DEFAULT_SIZE,
        use_cache: bool = True,
    ) -> "ScrabbleEngine":
        dict_path = Path(dict_path)
        cache = dict_path.with_suffix(".gaddag.pkl")
        if use_cache and cache.exists():
            try:
                gaddag = Gaddag.load(cache)
            except Exception:
                gaddag = Gaddag.from_file(dict_path)
                gaddag.save(cache)
        else:
            gaddag = Gaddag.from_file(dict_path)
            if use_cache:
                gaddag.save(cache)
        return cls(gaddag, size=size)

    # ---------- Tahtayı kur ----------

    def place_word(
        self,
        word: str,
        row: int,
        col: int,
        direction: Union[str, Direction],
        blank_indices: Iterable[int] = (),
    ) -> None:
        """Tahtaya bir kelime yerleştirir (zaten dolu hücrelerin üzerinden geçebilir; eşleşmiyorsa hata)."""
        d = _parse_direction(direction)
        blanks = set(blank_indices)
        new_placements: list[Placement] = []
        for i, ch in enumerate(word):
            r, c = (row, col + i) if d == Direction.HORIZONTAL else (row + i, col)
            if not (0 <= r < self.size and 0 <= c < self.size):
                raise ValueError(f"({r},{c}) tahta dışı")
            existing = self.board.letter_at(r, c)
            if existing is None:
                new_placements.append(Placement(r, c, ch, is_blank=(i in blanks)))
            elif existing != ch:
                raise ValueError(
                    f"({r},{c}) hücresinde '{existing}' var, '{ch}' yerleştirilemez"
                )
            # eşleşiyorsa zaten oradalar — atla
        if new_placements:
            self.board.commit_move(new_placements)

    def clear_board(self) -> None:
        self.board = ScrabbleBoard(self.gaddag, size=self.size)
        self.move_gen = MoveGenerator(self.gaddag, self.board)

    def set_y3(self, position: Optional[tuple[int, int]]) -> None:
        """(row, col) ya da None."""
        self.score_calc.y3_position = position

    # ---------- Öneri ----------

    def suggest_moves(
        self,
        rack: Union[str, Rack],
        top_n: Optional[int] = None,
    ) -> list[tuple[int, CandidateMove]]:
        """
        Yasal tüm hamleleri üretir, puanlar, yüksekten düşüğe sıralar.
        Dönüş: [(score, move), ...]
        """
        if isinstance(rack, str):
            rack = Rack.from_string(rack)
        moves = self.move_gen.generate(rack)
        scored = [
            (self.score_calc.score_move(m, self.board), m)
            for m in moves
        ]
        scored.sort(key=lambda x: (-x[0], len(x[1].placements), x[1].main_word))
        if top_n is not None:
            scored = scored[:top_n]
        return scored

    # ---------- Tanı ----------

    def board_str(self) -> str:
        return str(self.board)

    def stats(self) -> dict:
        s = self.board.stats()
        s["dictionary_words"] = len(self.gaddag)
        s["y3_position"] = self.score_calc.y3_position
        return s


# =========================================================================
# Uçtan uca sınama
# =========================================================================

if __name__ == "__main__":
    g = Gaddag()
    for w in [
        "AT", "ATA", "ATEŞ", "ARABA", "BAR", "BAL", "BALIK", "BALİ",
        "AB", "AL", "ALA", "ALAN", "RA", "EV", "EVE", "EVET", "EVLİ",
        "KEDİ", "KEL", "KELE", "ÇAY", "ÇİĞ", "ÇEK", "ELE", "EL",
        "TAŞ", "TAS", "TAT", "ATEŞE",
    ]:
        g.add_word(w)

    engine = ScrabbleEngine(g, size=15)

    # ---- Senaryo 1: Boş tahta ----
    print("=== Senaryo 1: Boş tahta, rack='ARABA' ===")
    sugg = engine.suggest_moves("ARABA", top_n=5)
    for s, m in sugg:
        cells = ",".join(f"({p.row},{p.col})" for p in m.placements)
        print(f"  {s:>4}  [{m.direction.value}] {m.main_word}  @ {cells}")
    assert len(sugg) > 0
    top_score = sugg[0][0]
    print(f"  En yüksek puan: {top_score}")
    print()

    # ---- Senaryo 2: ARABA tahtada, rack='TEŞ' ----
    print("=== Senaryo 2: ARABA yatay (7,7), rack='TEŞ' ===")
    engine.place_word("ARABA", row=7, col=7, direction="H")
    sugg = engine.suggest_moves("TEŞ", top_n=10)
    main_words = {m.main_word for _, m in sugg}
    print(f"  Bulunan ana kelimeler: {sorted(main_words)}")
    for s, m in sugg[:5]:
        cells = ",".join(f"({p.row},{p.col}){p.letter}" for p in m.placements)
        print(f"  {s:>4}  [{m.direction.value}] {m.main_word}  @ {cells}")
    assert "ATEŞ" in main_words
    print()

    # ---- Senaryo 3: Joker ----
    print("=== Senaryo 3: rack='AB?' (joker dahil) ===")
    engine.clear_board()
    sugg = engine.suggest_moves("AB?", top_n=8)
    has_blank = any(any(p.is_blank for p in m.placements) for _, m in sugg)
    for s, m in sugg[:5]:
        cells = ",".join(f"({p.row},{p.col}){p.letter}{'*' if p.is_blank else ''}" for p in m.placements)
        print(f"  {s:>4}  [{m.direction.value}] {m.main_word}  @ {cells}")
    assert has_blank, "Joker kullanan en az bir hamle olmalı"
    print()

    # ---- Senaryo 4: Y3 etkisi ----
    print("=== Senaryo 4: Y3 = (7,7) merkez; rack='ARABA' ===")
    engine.clear_board()
    engine.set_y3((7, 7))
    sugg = engine.suggest_moves("ARABA", top_n=3)
    for s, m in sugg:
        has_y3 = any((p.row, p.col) == (7, 7) for p in m.placements)
        marker = " ★Y3" if has_y3 else ""
        print(f"  {s:>4}{marker}  [{m.direction.value}] {m.main_word}")
    # En iyi hamle Y3'ü kapsamalı (merkezi kapsamayan ilk hamle yok zaten)
    assert any((p.row, p.col) == (7, 7) for p in sugg[0][1].placements)
    print()

    # ---- Senaryo 5: K3 köşesinden faydalanma ----
    print("=== Senaryo 5: ARABA + (0,2) K3 köşesi senaryosu ===")
    engine.clear_board()
    engine.set_y3(None)
    engine.place_word("ARABA", row=0, col=2, direction="H")   # (0,2) K3
    # Hücre çarpan analizi:
    #   (0,2) A → K3   → A=1, word_mult = 3
    #   (0,3) R → yok  → R=1
    #   (0,4) A → yok  → A=1
    #   (0,5) B → H2   → B=3*2=6
    #   (0,6) A → yok  → A=1
    # word_sum = 1+1+1+6+1 = 10, * word_mult 3 = 30
    test_move = CandidateMove(
        placements=tuple([
            Placement(0, 2, "A"), Placement(0, 3, "R"), Placement(0, 4, "A"),
            Placement(0, 5, "B"), Placement(0, 6, "A"),
        ]),
        main_word="ARABA",
        direction=Direction.HORIZONTAL,
        anchor=(0, 2),
    )
    sc_test = ScoreCalculator(size=15, y3_position=None)
    empty_board = ScrabbleBoard(g, size=15)
    s = sc_test.score_move(test_move, empty_board)
    print(f"  ARABA @(0,2) K3+H2 puanı: {s} (beklenen 30)")
    assert s == 30
    print()

    # ---- Senaryo 6: Tam pipeline — kullanıcının asıl kullanım kalıbı ----
    print("=== Senaryo 6: Tam pipeline ===")
    engine.clear_board()
    engine.set_y3((0, 0))
    engine.place_word("ARABA", row=7, col=7, direction="H")
    engine.place_word("ATEŞ",  row=7, col=7, direction="V")   # A çakışıyor
    print("Tahta:")
    print(engine.board_str())
    print()
    sugg = engine.suggest_moves("KEDİL?", top_n=8)
    print(f"İlk 8 öneri (rack='KEDİL?'):")
    for s, m in sugg:
        cells = ",".join(
            f"({p.row},{p.col}){p.letter}{'*' if p.is_blank else ''}"
            for p in m.placements
        )
        print(f"  {s:>4}  [{m.direction.value}] {m.main_word}  @ {cells}")
    print(f"\nÖneri sayısı: {len(sugg)}")
    print(f"Engine stats: {engine.stats()}")

    print("\n--- Engine entegrasyon testleri geçti ---")
