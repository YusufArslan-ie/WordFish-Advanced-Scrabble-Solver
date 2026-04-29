"""
Scrabble TR — Modül A: Sözlük (GADDAG)

Türkçe Scrabble motoru için GADDAG tabanlı sözlük modülü.
Modül C (Hamle Üretici), tahta üzerinde verimli kelime araması yaparken
buradaki API'yi kullanır.

GADDAG mantığı (Gordon, 1994):
    Her w kelimesi için n = len(w) adet yol eklenir:
        reverse(w[0..=i]) + SEP + w[i+1..]   ,   i = 0, 1, ..., n-1
    Bu sayede tahtadaki herhangi bir harften (anchor) başlayarak önce
    sola (ters ön ek), sonra SEP'i geçip sağa doğru tek geçişte arama
    yapılabilir.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Optional


# =========================================================================
# Türkçe alfabe yardımcısı
# (Modül C de cross-check ve istaka için aynı bitmask mantığını kullanır)
# =========================================================================

class TurkishAlphabet:
    """29 harfli Türkçe Scrabble alfabesi ve bitmask yardımcıları."""

    LETTERS: str = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    SIZE: int = 29
    _INDEX: dict[str, int] = {c: i for i, c in enumerate(LETTERS)}
    FULL_MASK: int = (1 << SIZE) - 1

    @classmethod
    def index(cls, ch: str) -> int:
        return cls._INDEX[ch]

    @classmethod
    def letter(cls, i: int) -> str:
        return cls.LETTERS[i]

    @classmethod
    def is_valid_letter(cls, ch: str) -> bool:
        return ch in cls._INDEX

    @classmethod
    def is_valid_word(cls, word: str) -> bool:
        return bool(word) and all(c in cls._INDEX for c in word)

    @classmethod
    def letter_to_mask(cls, ch: str) -> int:
        return 1 << cls._INDEX[ch]

    @classmethod
    def letters_to_mask(cls, letters: Iterable[str]) -> int:
        mask = 0
        for c in letters:
            mask |= 1 << cls._INDEX[c]
        return mask

    @classmethod
    def mask_to_letters(cls, mask: int) -> frozenset:
        return frozenset(
            cls.LETTERS[i] for i in range(cls.SIZE) if mask & (1 << i)
        )


# =========================================================================
# GADDAG düğümü
# =========================================================================

class GaddagNode:
    """Tek bir GADDAG düğümü: çıkan kenarlar (harf -> düğüm) + terminal flag."""

    __slots__ = ("edges", "is_terminal")

    def __init__(self) -> None:
        self.edges: dict[str, "GaddagNode"] = {}
        self.is_terminal: bool = False

    def __repr__(self) -> str:
        keys = "".join(sorted(self.edges.keys()))
        mark = "*" if self.is_terminal else ""
        return f"<GaddagNode edges='{keys}'{mark}>"


# =========================================================================
# GADDAG ana yapısı
# =========================================================================

class Gaddag:
    """
    Türkçe Scrabble için GADDAG sözlük yapısı.

    Modül C için temel sorgu API'si:
        get_root()              -> Aramaya başlangıç düğümü
        traverse(node, ch)      -> Bir kenarı takip et (yoksa None)
        cross_separator(node)   -> SEP'i geç, sol→sağ yön değişimi
        has_separator(node)     -> SEP kenarı var mı?
        is_terminal(node)       -> Burada geçerli kelime biter mi?
        edge_letters(node)      -> Düğümden çıkan harfler (SEP hariç, frozenset)
        edge_mask(node)         -> Aynı bilgi 29-bit bitmask (cross-check için)
        contains(word)          -> Yardımcı: kelime sözlükte var mı?
    """

    SEPARATOR: str = "+"   # Alfabede olmayan herhangi bir karakter

    def __init__(self) -> None:
        self.root: GaddagNode = GaddagNode()
        self.word_count: int = 0    # Eklenen benzersiz kelime sayısı
        self.node_count: int = 1    # Toplam düğüm sayısı (root dahil)
        self._known: set[str] = set()
        self._build_stats: dict = {}

    # -------------------------------------------------------------------
    # Yapılandırma
    # -------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path,
        *,
        min_length: int = 2,
        encoding: str = "utf-8",
    ) -> "Gaddag":
        """
        Tek-kelime-per-satır sözlük dosyasından GADDAG kur.
        Geçersiz harf içeren / kısa satırlar sessizce atlanır.
        """
        gaddag = cls()
        skipped_invalid = 0
        skipped_short = 0
        with open(path, "r", encoding=encoding) as f:
            for raw in f:
                word = raw.strip()
                if not word:
                    continue
                if len(word) < min_length:
                    skipped_short += 1
                    continue
                if not TurkishAlphabet.is_valid_word(word):
                    skipped_invalid += 1
                    continue
                gaddag.add_word(word)
        gaddag._build_stats = {
            "source": str(path),
            "skipped_invalid": skipped_invalid,
            "skipped_short": skipped_short,
        }
        return gaddag

    def add_word(self, word: str) -> bool:
        """
        Kelimeyi tüm GADDAG yollarıyla yapıya ekle.
        Zaten eklenmişse no-op. True dönerse yeni eklendi demektir.
        """
        if word in self._known:
            return False
        self._known.add(word)
        self.word_count += 1
        n = len(word)
        for i in range(n):
            left_reversed = word[: i + 1][::-1]   # reverse(w[0..=i])
            right = word[i + 1:]                  # w[i+1..]
            self._insert(left_reversed + self.SEPARATOR + right)
        return True

    def _insert(self, sequence: str) -> None:
        """Düğüm zincirini gez/oluştur, son düğümü terminal işaretle."""
        node = self.root
        for ch in sequence:
            nxt = node.edges.get(ch)
            if nxt is None:
                nxt = GaddagNode()
                node.edges[ch] = nxt
                self.node_count += 1
            node = nxt
        node.is_terminal = True

    # -------------------------------------------------------------------
    # Modül C için sorgu API'si
    # -------------------------------------------------------------------

    def get_root(self) -> GaddagNode:
        return self.root

    @staticmethod
    def traverse(node: Optional[GaddagNode], ch: str) -> Optional[GaddagNode]:
        """node üzerinden ch kenarını takip et; yoksa None."""
        if node is None:
            return None
        return node.edges.get(ch)

    @classmethod
    def cross_separator(cls, node: Optional[GaddagNode]) -> Optional[GaddagNode]:
        """SEP kenarını takip et: yön değişimi (sol → sağ)."""
        if node is None:
            return None
        return node.edges.get(cls.SEPARATOR)

    @classmethod
    def has_separator(cls, node: Optional[GaddagNode]) -> bool:
        return node is not None and cls.SEPARATOR in node.edges

    @staticmethod
    def is_terminal(node: Optional[GaddagNode]) -> bool:
        return node is not None and node.is_terminal

    @classmethod
    def edge_letters(cls, node: Optional[GaddagNode]) -> frozenset:
        """node'dan çıkan harfler (SEP hariç)."""
        if node is None:
            return frozenset()
        return frozenset(c for c in node.edges if c != cls.SEPARATOR)

    @classmethod
    def edge_mask(cls, node: Optional[GaddagNode]) -> int:
        """
        node'dan çıkan harflerin 29-bit bitmask'i.
        Modül C bunu cross-check ve istaka maskeleriyle AND'leyerek
        canlı dalları tek bit-işleminde belirler.
        """
        if node is None:
            return 0
        mask = 0
        idx = TurkishAlphabet._INDEX
        for c in node.edges:
            if c != cls.SEPARATOR:
                mask |= 1 << idx[c]
        return mask

    def contains(self, word: str) -> bool:
        """Kelime sözlükte var mı? (GADDAG i=0 yolu üzerinden kontrol)"""
        if not word:
            return False
        node = self.root.edges.get(word[0])
        if node is None:
            return False
        node = node.edges.get(self.SEPARATOR)
        if node is None:
            return False
        for ch in word[1:]:
            node = node.edges.get(ch)
            if node is None:
                return False
        return node.is_terminal

    # -------------------------------------------------------------------
    # Diskten/diske (yeniden inşaya gerek kalmasın)
    # -------------------------------------------------------------------

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "Gaddag":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Beklenen {cls.__name__}, bulunan {type(obj).__name__}"
            )
        return obj

    # -------------------------------------------------------------------
    # Bilgi
    # -------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "word_count": self.word_count,
            "node_count": self.node_count,
            "alphabet_size": TurkishAlphabet.SIZE,
            "separator": self.SEPARATOR,
            **self._build_stats,
        }

    def __len__(self) -> int:
        return self.word_count

    def __contains__(self, word: str) -> bool:
        return self.contains(word)

    def __repr__(self) -> str:
        return f"Gaddag(words={self.word_count:,}, nodes={self.node_count:,})"


# =========================================================================
# Doğrudan çalıştırıldığında: mini sınama + isteğe bağlı dosya yükleme
# =========================================================================

if __name__ == "__main__":
    import sys

    # 1) Mini sınama
    g = Gaddag()
    for w in ["AT", "ATA", "ARABA", "ARI", "ARILAR", "KEDİ", "ÇİĞ"]:
        g.add_word(w)

    assert g.contains("ARABA")
    assert g.contains("KEDİ")
    assert g.contains("ÇİĞ")
    assert not g.contains("ARAB")
    assert not g.contains("KEDIM")

    # Modül C tarzı bir gezinti: "ARABA" — A anchor'ı 3. konum (i=2)
    # Yol: reverse("ARA") + SEP + "BA" = "ARA" + SEP + "BA"
    node = g.get_root()
    for ch in "ARA":
        node = g.traverse(node, ch)
        assert node is not None, f"Sol parça kesildi: {ch}"
    node = g.cross_separator(node)
    assert node is not None, "SEP geçişi başarısız"
    for ch in "BA":
        node = g.traverse(node, ch)
        assert node is not None, f"Sağ parça kesildi: {ch}"
    assert g.is_terminal(node), "Terminal değil"

    # edge_mask cross-check için tek-işlem budamayı destekliyor mu?
    root_mask = g.edge_mask(g.get_root())
    assert root_mask & TurkishAlphabet.letter_to_mask("A")
    assert root_mask & TurkishAlphabet.letter_to_mask("Ç")  # ÇİĞ'in son harfi
    print("Mini sınama OK.")
    print(g.stats())

    # 2) Komut satırından dosya verildiyse onu yükle ve özet bas
    if len(sys.argv) > 1:
        src = sys.argv[1]
        print(f"\n{src} yükleniyor...")
        big = Gaddag.from_file(src)
        print(big)
        print(big.stats())
        # Hızlı erişim için pickle'a kaydet
        cache = Path(src).with_suffix(".gaddag.pkl")
        big.save(cache)
        print(f"Önbellek kaydedildi: {cache}")
