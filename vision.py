"""

py vision.py ornek1.jpg templates


Scrabble TR — Vision Modülü v3

Saf OpenCV + numpy boru hattı:
    Screenshot ─► TileGenerator ─► (board_tiles, rack_tiles)
                                         │
                                         ▼
                                    TileReader ─► {'letter', 'is_joker', ...}

Tasarım kararları:
  • Tahta tespiti: gri zemin + sıcak (sarı/turuncu/kahve) + soğuk (mavi/mor)
    renkleri toplanır, koyu mavi istaka bandı çıkartılır, yatay+dikey
    projeksiyonla bbox bulunur.
  • Hücre sınıflandırma: merkez %60'ın HSV ortancası ile
        S < 25                → BOŞ (gri tahta zemini)
        Y3_BG + STAR pikseli  → Y3
        H 85-110, S<130       → H2 (mavi bonus, BOŞ)
        H 130-175             → H3 (mor bonus, BOŞ)
        H 35-85               → K2 (yeşil bonus, BOŞ)
        H 5-25, V<230, S<130  → K3 (kahve bonus, BOŞ)
        diğer (sarı/turuncu)  → DOLU TAŞ
  • Harf izolasyonu: kenar inset → grayscale → Otsu (INV) → sayı bölgesi
    sıfırla → en büyük connected component'i tut, küçük bağımsız parçalardan
    yalnızca harfin yakınında olanları (Ç çengeli, İ noktası, Ö/Ü noktaları)
    koru → bbox + padded square → resize 64×64. Hem şablon hem hücre AYNI
    pipeline'dan geçer.
  • Joker tespiti: sol-üst 30×30 piksellik ROI'de koyu kırmızı
    (H 0-10|170-179, S 150+, V 30-160) yeterince varsa joker.
    joker.jpg örneği ile kalibre edildi.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
#  Sabitler
# =========================================================================

HEX_COLORS = {
    "BOARD_TILE_BG":    "#f9e69a",
    "BOARD_TEXT_COLOR": "#805838",
    "RACK_TILE_BG":     "#ecc426",
    "RACK_TEXT_COLOR":  "#b38f2a",
    "RACK_EMPTY_BG":    "#187096",
    "Y3_BG":            "#fddb66",
    "Y3_STAR_COLOR":    "#fb903e",
    "JOKER_RED":        "#881412",
}

TURKISH_LETTERS = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
CONFIDENCE_THRESHOLD = 0.50


# =========================================================================
#  HSV yardımcıları
# =========================================================================

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def hex_to_hsv(hex_color: str) -> np.ndarray:
    bgr = np.uint8([[list(hex_to_bgr(hex_color))]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]


def hsv_range(
    hex_color: str, h_tol: int = 10, s_tol: int = 70, v_tol: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = hex_to_hsv(hex_color).astype(int)
    lo = np.array([
        max(0, hsv[0] - h_tol), max(0, hsv[1] - s_tol), max(0, hsv[2] - v_tol),
    ], dtype=np.uint8)
    hi = np.array([
        min(179, hsv[0] + h_tol), min(255, hsv[1] + s_tol), min(255, hsv[2] + v_tol),
    ], dtype=np.uint8)
    return lo, hi


def mask_in_hsv(image_bgr: np.ndarray, hex_color: str, **tol) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lo, hi = hsv_range(hex_color, **tol)
    return cv2.inRange(hsv, lo, hi)


def red_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Joker noktasının koyu kırmızısı için özel maske (H wraparound)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 150, 30]), np.array([10, 255, 160]))
    m2 = cv2.inRange(hsv, np.array([170, 150, 30]), np.array([179, 255, 160]))
    return cv2.bitwise_or(m1, m2)


# =========================================================================
#  TileGenerator
# =========================================================================

@dataclass
class GridResult:
    board_tiles: list[tuple[int, int, np.ndarray]]
    rack_tiles: list[tuple[int, np.ndarray]]
    board_warped: np.ndarray
    rack_strip: Optional[np.ndarray]
    board_bbox: tuple[int, int, int, int]


class TileGenerator:
    BOARD_DIM = 15
    RACK_SLOTS = 7
    BOARD_OUTPUT_PX = 900
    TILE_INSET_FRAC = 0.05

    def __init__(self, *, debug_dir: Optional[Path] = None) -> None:
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Public ----------

    def _detect_dynamic_board(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """
        Morfolojik Filtre destekli Kusursuz Tarak Filtresi.
        Yazıları ve logoları yok edip sadece gerçek ızgara çizgilerine odaklanır.
        """
        H, W = img.shape[:2]
        
        # 1. Görüntüyü siyah beyaza çevirip kenarları bul
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # --- YENİ EKLENEN MORFOLOJİK SİLAH ---
        # 2. Sadece yatay olarak çok uzun çizgileri hayatta bırak
        # Çizginin en az ekranın %10'u kadar kesintisiz devam etmesi lazım
        line_length = max(10, int(W * 0.10))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
        
        # cv2.MORPH_OPEN ile kısa kenarları (yazı, logo, harf) tamamen siliyoruz
        clean_horiz_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        # -------------------------------------
        
        # Artık gürültüler gitti, sadece saf yatay çizgilerin yoğunluğunu ölçüyoruz
        row_density = np.sum(clean_horiz_lines, axis=1)
        
        best_y = 0
        max_score = -1
        tile_size = W / 15.0
        
        start_y = 0
        end_y = max(1, H - W + 10)
        
        for y in range(start_y, min(end_y, H)):
            score = 0
            for i in range(16):
                line_y = int(y + i * tile_size)
                if 0 <= line_y < H:
                    y0 = max(0, line_y - 1)
                    y1 = min(H, line_y + 2)
                    score += np.max(row_density[y0:y1])
            
            if score > max_score:
                max_score = score
                best_y = y
                
        if best_y == 0 and max_score == 0:
            best_y = int(H * 0.2)
            
        return (0, best_y, W, W)

    def process(self, image: Union[str, Path, np.ndarray]) -> GridResult:
        img = self._load(image)
        bbox = self._detect_dynamic_board(img)
        x, y, w, h = bbox
        side = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        H_img, W_img = img.shape[:2]
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(W_img, x0 + side)
        y1 = min(H_img, y0 + side)
        crop = img[y0:y1, x0:x1]
        warped = cv2.resize(
            crop, (self.BOARD_OUTPUT_PX, self.BOARD_OUTPUT_PX),
            interpolation=cv2.INTER_AREA,
        )
        board_tiles = list(self._slice_grid(warped))
        rack_strip, rack_tiles = self._extract_rack(img, bbox)

        if self.debug_dir:
            cv2.imwrite(str(self.debug_dir / "board_warped.png"), warped)
            if rack_strip is not None:
                cv2.imwrite(str(self.debug_dir / "rack_strip.png"), rack_strip)
            self._save_grid_overlay(warped)

        return GridResult(
            board_tiles=board_tiles,
            rack_tiles=rack_tiles,
            board_warped=warped,
            rack_strip=rack_strip,
            board_bbox=bbox,
        )

    @staticmethod
    def _load(image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        path = str(image)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {path}")
        return img

    # ---------- Tahta bbox tespiti ----------

    def _find_board_bbox(self, img: np.ndarray) -> tuple[int, int, int, int]:
        """
        Kelimelik/Scrabble mobil ekranları için optimize edilmiş,
        Kenar Yoğunluğu (Edge Density) tabanlı kusursuz kare bulucu.
        """
        H, W = img.shape[:2]
        
        # Tahta ekran genişliğini kaplar ve tam karedir
        w_board = W
        h_board = W
        
        # Görüntüyü siyah beyaza çevir ve kenarları (çizgileri) bul
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Her yatay satırdaki kenar piksellerini topla
        row_edges = edges.sum(axis=1)
        
        best_y0 = 0
        max_edge_sum = -1
        
        # Tahtanın başlayabileceği makul Y koordinatlarını tara
        # (Ekranın en üst %10'u ile en alt %40'ı arasında bir yerde başlar)
        search_min = int(H * 0.10)
        search_max = int(H * 0.40)
        
        for y in range(search_min, search_max):
            if y + h_board > H:
                break
            
            # y'den başlayıp h_board kadar aşağı inen penceredeki toplam çizgileri say
            current_edge_sum = row_edges[y : y + h_board].sum()
            
            if current_edge_sum > max_edge_sum:
                max_edge_sum = current_edge_sum
                best_y0 = y
                
        # Tahta 15 hücreden oluşur. 1 hücrenin boyutu = h_board / 15
        cell_size = h_board / 15
        
        # Senin harika tespitin: 1 tam blok (1.0) aşağı kaydırıyoruz!
        best_y0 = int(best_y0 + (cell_size * 1.0))
        # --------------------------------------------------------------
                
        return (0, best_y0, w_board, h_board)

    # ---------- 15×15 grid ----------

    def _slice_grid(self, board: np.ndarray):
        H, W = board.shape[:2]
        cell_h, cell_w = H / self.BOARD_DIM, W / self.BOARD_DIM
        inset_y = int(cell_h * self.TILE_INSET_FRAC)
        inset_x = int(cell_w * self.TILE_INSET_FRAC)
        for r in range(self.BOARD_DIM):
            for c in range(self.BOARD_DIM):
                y0 = int(r * cell_h) + inset_y
                y1 = int((r + 1) * cell_h) - inset_y
                x0 = int(c * cell_w) + inset_x
                x1 = int((c + 1) * cell_w) - inset_x
                yield (r, c, board[y0:y1, x0:x1].copy())

    # ---------- İstaka çıkarımı ----------

    def _extract_rack(
        self, img: np.ndarray, board_bbox: tuple[int, int, int, int],
    ) -> tuple[Optional[np.ndarray], list[tuple[int, np.ndarray]]]:
        """
        Kusursuz connected components tabanlı isteka tespiti ve kesimi.
        Renk filtresiyle 7 taşı bulur ve gerçek sınırlarına göre kırpar.
        """
        H, W = img.shape[:2]
        bx, by, bw, bh = board_bbox
        below_top = by + bh
        if below_top >= H - 5:
            return None, []
        below = img[below_top:H]

        # 1. Sarı ıstaka taşlarının olduğu renk tayfını bul
        hsv_below = cv2.cvtColor(below, cv2.COLOR_BGR2HSV)
        mask = mask_in_hsv(below, HEX_COLORS["RACK_TILE_BG"], h_tol=15, s_tol=120, v_tol=120)
        
        # 2. Gürültüyü temizle (küçük tozları sil)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

        # 3. Connected Components ile 7 taşı bul
        n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(mask_cleaned, 8)
        
        # Taş adaylarını topla (fiziksel taş boyutlarına uymayanları filtrele)
        bh_ref = bh / self.BOARD_DIM # Tahtadaki ortalama taş boyutu referansı
        slot_bboxes = []
        for i in range(1, n_lab):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Fiziksel boyut filtresi: Ne çok minik ne çok devasa
            if area < (bh_ref * bh_ref) * 0.4 or w < bh_ref*0.5 or h < bh_ref*0.5:
                continue
            # En boy oranı filtresi: Tam kareye yakın olmalı
            if not (0.6 < w / max(1, h) < 1.6):
                continue
            slot_bboxes.append((x, y, w, h))

        # Soldan sağa sırala
        slot_bboxes.sort(key=lambda b: b[0])
        
        # Sadece en iyi 7 tanesini (veya daha azını) tut
        slot_bboxes = slot_bboxes[:self.RACK_SLOTS]

        out: list[tuple[int, np.ndarray]] = []
        rack_strip = None # Debug için
        
        if len(slot_bboxes) > 0:
            # 4. Tespit edilen sınırların etrafına biraz pay (padding) ekleyip kes
            for i, (x_s, y_s, w_s, h_s) in enumerate(slot_bboxes):
                # Taşın etrafına %10'luk bir emniyet payı ekle
                pad_w = int(w_s * 0.10)
                pad_h = int(h_s * 0.10)
                
                x0 = max(0, x_s - pad_w)
                y0 = max(0, y_s - pad_h)
                x1 = min(below.shape[1], x_s + w_s + pad_w)
                y1 = min(below.shape[0], y_s + h_s + pad_h)
                
                out.append((i, below[y0:y1, x0:x1].copy()))
            
            # Debug için birleştirilmiş şeridi oluştur (eski kodun devamlılığı için)
            min_x, min_y = min(b[0] for b in slot_bboxes), min(b[1] for b in slot_bboxes)
            max_x, max_y = max(b[0]+b[2] for b in slot_bboxes), max(b[1]+b[3] for b in slot_bboxes)
            rack_strip = below[max(0, min_y-10):min(below.shape[0], max_y+10), 
                              max(0, min_x-10):min(below.shape[1], max_x+10)].copy()

        return rack_strip, out

    # ---------- Debug ----------

    def _save_grid_overlay(self, warped: np.ndarray) -> None:
        if not self.debug_dir:
            return
        H, W = warped.shape[:2]
        img = warped.copy()
        for i in range(self.BOARD_DIM + 1):
            cv2.line(img, (0, int(i * H / self.BOARD_DIM)),
                     (W, int(i * H / self.BOARD_DIM)), (0, 0, 255), 1)
            cv2.line(img, (int(i * W / self.BOARD_DIM), 0),
                     (int(i * W / self.BOARD_DIM), H), (0, 0, 255), 1)
        cell = H / self.BOARD_DIM
        for r in range(self.BOARD_DIM):
            for c in range(self.BOARD_DIM):
                cv2.putText(img, f"{r},{c}", (int(c * cell + 3), int(r * cell + 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(self.debug_dir / "board_grid.png"), img)


# =========================================================================
#  Letter normalization (şablon ve hücre için ortak pipeline)
# =========================================================================

def normalize_letter(
    tile_bgr: np.ndarray,
    *,
    target_size: int = 64,
    is_rack: bool = False,  # <-- SİHİRLİ ANAHTAR BURADA
) -> np.ndarray:
    """
    Noktalı harfleri (Ö, Ü, İ) koruyan optimize edilmiş sürüm.
    İsteka ve Tahta için farklı kenar kırpma (inset) oranları kullanır.
    """
    h, w = tile_bgr.shape[:2]
    if h < 8 or w < 8:
        return np.full((target_size, target_size), 255, np.uint8)

    # SENİN STRATEJİN: İsteka ise %15 kırp (gölgeleri yoksay), tahta ise %6 kırp (noktaları koru)
    edge_inset_frac = 0.08 if is_rack else 0.06

    inset_y = int(h * edge_inset_frac)
    inset_x = int(w * edge_inset_frac)
    core = tile_bgr[inset_y: h - inset_y, inset_x: w - inset_x]
    ch, cw = core.shape[:2]

    gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border = max(2, int(min(ch, cw) * 0.06))
    ink[:border, :] = 0
    ink[-border:, :] = 0
    ink[:, :border] = 0
    ink[:, -border:] = 0

    # Sayı bölgesini sıfırla (sağ-üst köşedeki ², ¹, ⁴ vb. rakamlar)
    ink[: int(ch * 0.40), int(cw * 0.55):] = 0

    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
    if n_lab <= 1:
        return np.full((target_size, target_size), 255, np.uint8)

    H_ink, W_ink = ink.shape

    # Ana harf bileşeni: SAYI kırpıldıktan sonra en büyük alanlı (kenara
    # dokunsa bile). M gibi geniş harflerde harf kenara değer ama sayı
    # silindiği için kalan en büyük komponent yine harftir.
    valid = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n_lab)
             if stats[i, cv2.CC_STAT_AREA] >= 30]
    if not valid:
        return np.full((target_size, target_size), 255, np.uint8)
    valid.sort(key=lambda x: -x[0])
    idx_main = valid[0][1]

    main_area = stats[idx_main, cv2.CC_STAT_AREA]
    main_top = stats[idx_main, cv2.CC_STAT_TOP]
    main_h = stats[idx_main, cv2.CC_STAT_HEIGHT]
    main_cx = cents[idx_main][0]

    clean = np.zeros_like(ink)
    for i in range(1, n_lab):
        if i == idx_main:
            clean[lab == i] = 255
            continue
        a = stats[i, cv2.CC_STAT_AREA]
        if a < max(12, main_area * 0.04):
            continue

        cx, cy = cents[i]
        top_i = stats[i, cv2.CC_STAT_TOP]
        bot_i = top_i + stats[i, cv2.CC_STAT_HEIGHT]

        # SADECE ÜSTTEKİ noktaları kabul et (İ, Ö, Ü, J için).
        # Alttaki bileşenler taşın gölge çizgisi olabilir; onları içine
        # alırsak şablon-hücre bbox'ları farklı oluşur ve eşleştirme bozulur.
        is_dot_above = bot_i <= main_top + 3
        horiz_close = abs(cx - main_cx) < cw * 0.50

        if is_dot_above and horiz_close:
            clean[lab == i] = 255

    ink = clean

    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return np.full((target_size, target_size), 255, np.uint8)
        
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    cropped = ink[y0:y1 + 1, x0:x1 + 1]

    bh, bw_ = cropped.shape
    side = max(bh, bw_)
    pad = max(3, side // 8)
    canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    oy, ox = pad + (side - bh) // 2, pad + (side - bw_) // 2
    canvas[oy:oy + bh, ox:ox + bw_] = cropped
    canvas = cv2.bitwise_not(canvas)
    
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

# =========================================================================
#  TileReader
# =========================================================================

@dataclass
class TileResult:
    letter: Optional[str]
    is_joker: bool
    type: str
    confidence: float
    coords: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict:
        d = {
            "letter": self.letter,
            "is_joker": self.is_joker,
            "type": self.type,
            "confidence": round(self.confidence, 3),
        }
        if self.coords is not None:
            d["x_y_coordinates"] = self.coords
        return d
    
class TileReader:
    JOKER_DOT_PIXELS = 6
    JOKER_ROI_PX = 30
    LETTER_INK_MIN_PIX = 60
    TEMPLATE_SIZE = 64

    def __init__(
        self,
        templates_dir: Union[str, Path],
        *,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.templates_dir = Path(templates_dir)
        self.confidence_threshold = confidence_threshold
        self.templates: dict[str, np.ndarray] = self._load_templates()

    def _load_templates(self) -> dict[str, np.ndarray]:
        if not self.templates_dir.is_dir():
            raise FileNotFoundError(f"Şablon dizini yok: {self.templates_dir}")
        out: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for L in TURKISH_LETTERS:
            path = self._find_template_file(L)
            if path is None:
                missing.append(L)
                continue
            
            # cv2.imread yerine Türkçe karakter dostu okuma yöntemi:
            file_bytes = np.fromfile(str(path), dtype=np.uint8)
            tpl = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if tpl is None:
                raise IOError(f"Şablon okunamadı: {path}")
            out[L] = normalize_letter(tpl, target_size=self.TEMPLATE_SIZE, is_rack=False)
        if missing:
            warnings.warn(f"Eksik şablon harfleri: {missing}")
        return out

    def _find_template_file(self, L: str) -> Optional[Path]:
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            p = self.templates_dir / f"{L}.{ext}"
            if p.exists():
                return p
        return None

    # ---------- Public ----------

    def read_tile(
        self,
        image: np.ndarray,
        source: str = "BOARD",
        coords: Optional[tuple[int, int]] = None,
    ) -> TileResult:
        if image is None or image.size == 0:
            return TileResult(None, False, "EMPTY", 0.0,
                              coords if source == "BOARD" else None)
        source = source.upper()
        if source not in ("BOARD", "RACK"):
            raise ValueError("source 'BOARD' veya 'RACK' olmalı")

        tile_class = self._classify_tile(image, source)
        if tile_class == "EMPTY":
            return TileResult(None, False, "EMPTY", 1.0,
                              coords if source == "BOARD" else None)
        if tile_class == "Y3":
            return TileResult(None, False, "Y3", 1.0, coords)

        tile_type = "BOARD_TILE" if source == "BOARD" else "RACK_TILE"
        is_joker = self._has_joker_dot(image)

        bw = normalize_letter(image, target_size=self.TEMPLATE_SIZE, is_rack=(source == "RACK"))
        if coords:
            cv2.imwrite(f"vision_debug/debug_tile_{coords[0]}_{coords[1]}.png", bw)
        ink_pixels = int((bw == 0).sum())

        # Boş joker tespiti (rack için): normalize sonucu yatay-çubuk benzeri
        # ya da çok az ink → harf yok demek, sadece taşın gölge bandı yakalandı
        if source == "RACK":
            ys_b, xs_b = np.where(bw == 0)
            if len(xs_b) > 0:
                bw_w = xs_b.max() - xs_b.min() + 1
                bw_h = ys_b.max() - ys_b.min() + 1
                aspect = bw_w / max(1, bw_h)
                if (aspect > 3.0 and bw_h < 15) or ink_pixels < 80:
                    return TileResult("?", True, "RACK_TILE", 1.0, None)
            else:
                return TileResult("?", True, "RACK_TILE", 1.0, None)

        best_letter, conf = self._best_match(bw)
        if conf < self.confidence_threshold:
            warnings.warn(
                f"Düşük güvenli eşleşme: '{best_letter}' conf={conf:.2f} "
                f"coords={coords} source={source}"
            )

        return TileResult(
            letter=best_letter, 
            is_joker=is_joker,
            type=tile_type,
            confidence=conf,
            coords=coords if source == "BOARD" else None,
        )

    # ---------- Sınıflandırma ----------

    def _classify_tile(self, image: np.ndarray, source: str) -> str:
        h, w = image.shape[:2]
        
        # --- ÇÖZÜM BURADA: Sadece merkeze değil, tüm taş yüzeyine bakıyoruz ---
        # Kenarlardan (çizgilerden) %15 içerideki tüm alanı analiz et
        inset_y, inset_x = max(1, int(h * 0.15)), max(1, int(w * 0.15))
        roi = image[inset_y: h - inset_y, inset_x: w - inset_x]
        # ------------------------------------------------------------------------
        
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        med_h = int(np.median(roi_hsv[:, :, 0]))
        med_s = int(np.median(roi_hsv[:, :, 1]))
        med_v = int(np.median(roi_hsv[:, :, 2]))

        if source == "RACK" and 85 <= med_h <= 110 and med_s > 100 and med_v < 200:
            return "EMPTY"

        if source == "BOARD" and med_s < 25:
            return "EMPTY"

        if source == "BOARD":
            y3_bg = int((mask_in_hsv(image, HEX_COLORS["Y3_BG"],
                                     h_tol=12, s_tol=80, v_tol=60) > 0).sum())
            y3_st = int((mask_in_hsv(image, HEX_COLORS["Y3_STAR_COLOR"],
                                     h_tol=10, s_tol=80, v_tol=80) > 0).sum())
            area = h * w
            if y3_bg > area * 0.15 and y3_st > area * 0.05:
                return "Y3"

        if source == "BOARD":
            if 85 <= med_h <= 110 and med_s < 130:
                return "EMPTY"   # H2 mavi
            if 130 <= med_h <= 175:
                return "EMPTY"   # H3 mor
            if 35 <= med_h <= 85:
                return "EMPTY"   # K2 yeşil
            if 5 <= med_h <= 25 and med_s < 130 and med_v < 230:
                return "EMPTY"   # K3 kahve

        return "TILE"

    # ---------- Joker ----------

    def _has_joker_dot(self, image: np.ndarray) -> bool:
        h, w = image.shape[:2]
        roi_h = min(self.JOKER_ROI_PX, h // 3)
        roi_w = min(self.JOKER_ROI_PX, w // 3)
        roi = image[: roi_h, : roi_w]
        mask = red_mask(roi)
        return int((mask > 0).sum()) >= self.JOKER_DOT_PIXELS

    # ---------- Şablon eşleştirme ----------

    def _best_match(self, bw: np.ndarray) -> tuple[str, float]:
        if not self.templates:
            return "?", 0.0
            
        scores = []
        for L, tpl in self.templates.items():
            res = cv2.matchTemplate(bw, tpl, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            scores.append((s, L))
            
        # Skorları büyükten küçüğe sırala
        scores.sort(key=lambda x: -x[0])
        best_score, best_letter = scores[0]
        
        if len(scores) >= 2:
            second_score, second_letter = scores[1]
            top2_letters = {best_letter, second_letter}
            
            # 1. TEST: I ve İ Karışıklığı
            if top2_letters == {"I", "İ"} and (best_score - second_score) < 0.08:
                ink = (bw == 0).astype(np.uint8) * 255
                n_lab, _ = cv2.connectedComponents(ink, connectivity=8)
                if (n_lab - 1) >= 2:
                    best_letter = "İ"
                else:
                    best_letter = "I"

        # 2. TEST: T ve I Karışıklığı (KUSURSUZ GEOMETRİ TESTİ)
        if best_letter in ["I", "İ"]:
            # DİKKAT: Harfin KENDİSİNİ (siyah mürekkebi) buluyoruz (bw == 0)
            ys, xs = np.where(bw == 0)
            if len(xs) > 0:
                w = xs.max() - xs.min() + 1
                h = ys.max() - ys.min() + 1
                aspect_ratio = w / float(max(1, h))
                
                # 'I' harfi ipincedir (oran genelde 0.15 - 0.25 arasıdır).
                # Eğer oran 0.40'tan büyükse, üstte yatay bir çizgi vardır ve bu KESİN T'dir!
                if aspect_ratio > 0.40:
                    best_letter = "T"
                    
                    # Eğer T seçtiysek, güven skorunu (confidence) da T'nin skoru yapalım
                    for s, L in scores:
                        if L == "T":
                            best_score = s
                            break

        conf = (best_score + 1.0) / 2.0
        return best_letter, conf

# =========================================================================
#  Demo
# =========================================================================

def _demo(screenshot_path: str, templates_dir: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"\n► Görüntü   : {screenshot_path}")
    print(f"► Şablonlar : {templates_dir}")

    debug_dir = Path("vision_debug")
    gen = TileGenerator(debug_dir=debug_dir)
    grid = gen.process(screenshot_path)
    print(f"\n✓ Tahta bbox  : {grid.board_bbox}")
    print(f"✓ Tahta hücre : {len(grid.board_tiles)}")
    print(f"✓ İstaka slot : {len(grid.rack_tiles)}")

    reader = TileReader(templates_dir)
    print(f"✓ Yüklü şablon: {len(reader.templates)}/29")

    print("\n► Tahta okunuyor...")
    grid_chars: list[list[str]] = [["·"] * 15 for _ in range(15)]
    n_tiles = n_jokers = n_y3 = 0
    for (r, c, tile_img) in grid.board_tiles:
        res = reader.read_tile(tile_img, source="BOARD", coords=(r, c))
        if res.type == "BOARD_TILE":
            n_tiles += 1
            grid_chars[r][c] = res.letter.lower() if res.is_joker else res.letter
            if res.is_joker:
                n_jokers += 1
        elif res.type == "Y3":
            n_y3 += 1
            grid_chars[r][c] = "★"
    print(f"  → {n_tiles} taş, {n_jokers} joker, {n_y3} Y3")
    print("\n  Tahta haritası:")
    for row in grid_chars:
        print("   " + " ".join(row))

    print("\n► İstaka okunuyor...")
    rack: list[str] = []
    for (i, slot_img) in grid.rack_tiles:
        
        # --- YENİ EKLENEN DEBUG SATIRI: Istaka kesimlerini kaydet ---
        cv2.imwrite(str(debug_dir / f"debug_rack_slot_{i}.png"), slot_img)
        # -------------------------------------------------------------
        
        res = reader.read_tile(slot_img, source="RACK")
        if res.type == "EMPTY":
            rack.append("_")
        elif res.is_joker:
            rack.append("?")
        else:
            rack.append(res.letter or "?")
        print(f"  slot[{i}] → {res.to_dict()}")

def extract_game_data(image_path: Union[str, Path], templates_dir: Union[str, Path] = "templates") -> dict:
    """
    Dış modüller (main.py) tarafından çağrılacak ana API fonksiyonu.
    """
    gen = TileGenerator()
    grid = gen.process(image_path)
    reader = TileReader(templates_dir)

    board_matrix = [["." for _ in range(15)] for _ in range(15)]
    y3_pos = None
    
    for (r, c, tile_img) in grid.board_tiles:
        res = reader.read_tile(tile_img, source="BOARD", coords=(r, c))
        if res.type == "BOARD_TILE":
            board_matrix[r][c] = res.letter + "*" if res.is_joker else res.letter
        elif res.type == "Y3":
            board_matrix[r][c] = "★"
            y3_pos = (r, c)

    rack_list = []
    for (i, slot_img) in grid.rack_tiles:
        res = reader.read_tile(slot_img, source="RACK")
        if res.type != "EMPTY":
            rack_list.append("?" if res.is_joker else (res.letter or "?"))

    return {
        "board": board_matrix,
        "rack": rack_list,
        "y3_pos": y3_pos
    }

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _demo(sys.argv[1], sys.argv[2])
    else:
        print("Kullanım: python vision.py <screenshot> <templates_dir>")
