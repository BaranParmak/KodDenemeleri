import cv2
import mediapipe as mp
import numpy as np
import math
import time
import logging
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s â€” %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KonfigÃ¼rasyon
# ---------------------------------------------------------------------------

class AnalysisConfig(BaseModel):
    camera_index: int = Field(default=0, ge=0)
    frame_width: int = Field(default=640, gt=0)
    frame_height: int = Field(default=480, gt=0)

    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model_complexity: int = Field(default=1, ge=0, le=2)

    calibration_duration: float = Field(default=3.0, gt=0)
    lip_compression_ratio: float = Field(default=0.6, gt=0.0, le=1.0)

    touch_min_duration: float = Field(default=0.3, ge=0.0)
    zone_settle_duration: float = Field(default=0.12, ge=0.0)
    touch_exit_cooldown: float = Field(default=1.0, ge=0.0)

    mar_smooth_window: int = Field(default=5, ge=1)
    mar_min_consecutive_frames: int = Field(default=3, ge=1)
    mar_resume_cooldown: float = Field(default=0.5, ge=0.0)

    hand_exit_frames: int = Field(default=10, ge=1)
    face_bbox_padding: int = Field(default=20, ge=0)

    output_dir: str = Field(default="./output")


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

class InterviewState(str, Enum):
    PREP = "PREP"
    ANSWER = "ANSWER"


MOUTH_LEFT  = 61
MOUTH_RIGHT = 291
UPPER_LIP   = 13
LOWER_LIP   = 14

# ---------------------------------------------------------------------------
# ZONE REFERANS LANDMARK'LARI â€” Centroid Distance iÃ§in
# ---------------------------------------------------------------------------
# Her zone iÃ§in o bÃ¶lgenin geometrik merkezini en iyi temsil eden
# landmark'lar seÃ§ildi. SayÄ± az olabilir â€” Ã¶nemli olan ortalama
# konumun doÄŸru Ã§Ä±kmasÄ±.
#
# MediaPipe Face Mesh kaynaÄŸÄ±:
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/
#   face_geometry/data/canonical_face_model_uv_visualization.png

ZONE_REF_LANDMARKS: dict[str, list[int]] = {
    # Burun: kÃ¶prÃ¼ + kanat + uÃ§
    "nose": [1, 2, 4, 5, 19, 94, 131, 360],

    # AÄŸÄ±z: dudak kÃ¶ÅŸeleri + orta Ã¼st/alt
    "mouth": [61, 291, 13, 14, 17, 0, 269],

    # Ã‡ene: tam alt merkez
    "chin": [152, 175, 199, 200, 148, 377],

    # Sol gÃ¶z (kamera gÃ¶rÃ¼ntÃ¼sÃ¼nde sol)
    "left_eye": [33, 133, 159, 145, 160, 144, 163, 7],

    # SaÄŸ gÃ¶z (kamera gÃ¶rÃ¼ntÃ¼sÃ¼nde saÄŸ)
    "right_eye": [362, 263, 386, 374, 387, 373, 390, 249],

    # Sol yanak â€” iÃ§ yanak
    "left_cheek": [50, 101, 118, 187, 207, 206, 36, 209],

    # SaÄŸ yanak â€” iÃ§ yanak, kulaktan uzak noktalar
    "right_cheek": [280, 330, 347, 411, 427, 426, 266, 429],

    # AlÄ±n: Ã¼st merkez
    "forehead": [10, 9, 8, 107, 336, 151, 21, 251],
}

# Kucuk ve birbirine yakin bolgeleri biraz genisletmek
# (goz / burun / agiz karismasini azaltir).
ZONE_POLY_SCALE: dict[str, float] = {
    "nose": 1.45,
    "mouth": 1.35,
    "chin": 1.20,
    "left_eye": 1.45,
    "right_eye": 1.45,
    "left_cheek": 1.18,
    "right_cheek": 1.18,
    "forehead": 1.25,
}

# Ã–ncelik sadece eÅŸit uzaklÄ±k durumunda kullanÄ±lÄ±r (tiebreaker).
# KÃ¼Ã§Ã¼k/spesifik bÃ¶lgeler Ã¶nce â†’ eÅŸit puanda burun aÄŸÄ±zdan Ã¶nce gelir.
ZONE_PRIORITY = [
    "nose", "mouth", "chin",
    "left_eye", "right_eye",
    "left_cheek", "right_cheek",
    "forehead",
]

# Parmak ucu landmark indeksleri â€” yÃ¼ze temas eden bÃ¶lgeler bunlar.
# AvuÃ§ (palm) kullanmak yanlÄ±ÅŸ: el saÄŸdan alÄ±na gelince
# avuÃ§ centroid'i kulaÄŸa/yanaÄŸa daha yakÄ±n kalÄ±r.
# 4=baÅŸ parmak ucu, 8=iÅŸaret, 12=orta, 16=yÃ¼zÃ¼k, 20=serÃ§e
FINGERTIP_LANDMARKS = [4, 8, 12, 16, 20]

# Orta eklemler â€” sadece parmak ucu yoksa fallback
MID_JOINT_LANDMARKS = [3, 7, 11, 15, 19]

ZONE_LABELS_TR = {
    "nose":        "Burun",
    "mouth":       "Agiz",
    "chin":        "Cene",
    "left_eye":    "Sol Goz",
    "right_eye":   "Sag Goz",
    "left_cheek":  "Sol Yanak",
    "right_cheek": "Sag Yanak",
    "forehead":    "Alin",
    "unknown":     "Bilinmiyor",
    }

VALID_ZONES = set(ZONE_REF_LANDMARKS.keys())
VALID_ZONES_WITH_UNKNOWN = VALID_ZONES | {"unknown"}

# ---------------------------------------------------------------------------
# State Makineleri
# ---------------------------------------------------------------------------

@dataclass
class CalibrationState:
    is_calibrated: bool = False
    baseline_mar: float = 0.0
    compression_threshold: float = 0.0
    start_time: Optional[float] = None
    values: list = field(default_factory=list)

    def start(self) -> None:
        self.start_time = time.time()
        self.values = []
        logger.info("Kalibrasyon baÅŸladÄ±.")

    def update(self, mar_value: float, duration: float, compression_ratio: float) -> bool:
        self.values.append(mar_value)
        if (time.time() - self.start_time) >= duration:
            self.baseline_mar = float(np.mean(self.values))
            self.compression_threshold = self.baseline_mar * compression_ratio
            self.is_calibrated = True
            logger.info(
                "Kalibrasyon tamamlandÄ±. baseline_mar=%.4f, threshold=%.4f",
                self.baseline_mar, self.compression_threshold,
            )
            return True
        return False

    @property
    def in_progress(self) -> bool:
        return self.start_time is not None and not self.is_calibrated


@dataclass
class TouchStateMachine:
    _is_touching: bool = False
    _touch_counted: bool = False
    _start_time: float = 0.0
    _exit_until: float = 0.0
    _zone_vote_start: float = 0.0
    _zone_votes: list = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self._is_touching or time.time() < self._exit_until

    def update(
        self,
        hand_present: bool,
        current_time: float,
        cfg: AnalysisConfig,
        metrics: "SessionMetrics",
        results,
        zone_geometry: dict,
        w: int,
        h: int,
        zone_hint: Optional[str] = None,
    ) -> None:
        if hand_present:
            if not self._is_touching:
                if current_time < self._exit_until:
                    self._exit_until = 0.0
                else:
                    self._touch_counted = False
                    self._start_time = current_time
                    self._zone_vote_start = 0.0
                    self._zone_votes = []
                self._is_touching = True

            if not self._touch_counted:
                if (current_time - self._start_time) >= cfg.touch_min_duration:
                    zone = zone_hint if zone_hint else classify_zone_by_distance_px(results, zone_geometry, w, h)
                    if self._zone_vote_start == 0.0:
                        self._zone_vote_start = current_time
                    if zone in VALID_ZONES:
                        self._zone_votes.append(zone)
                    if (current_time - self._zone_vote_start) >= cfg.zone_settle_duration:
                        voted_zone = Counter(self._zone_votes).most_common(1)[0][0] if self._zone_votes else "unknown"
                        metrics.record_touch(current_time, voted_zone)
                        self._touch_counted = True
        else:
            if self._is_touching:
                self._is_touching = False
                self._exit_until = current_time + cfg.touch_exit_cooldown
                self._zone_vote_start = 0.0
                self._zone_votes = []
            elif current_time >= self._exit_until and self._touch_counted:
                self._touch_counted = False
                self._start_time = 0.0
                self._exit_until = 0.0
                self._zone_vote_start = 0.0
                self._zone_votes = []


@dataclass
class LipCompressionStateMachine:
    _consecutive_frames: int = 0
    _is_compressing: bool = False
    _mar_frozen: bool = False
    _resume_after: float = 0.0

    @property
    def is_compressing(self) -> bool:
        return self._is_compressing

    @property
    def mar_frozen(self) -> bool:
        return self._mar_frozen

    def update(
        self,
        hand_in_face: bool,
        mar_value: float,
        mar_buffer: deque,
        calibration: CalibrationState,
        current_time: float,
        cfg: AnalysisConfig,
        metrics: "SessionMetrics",
    ) -> None:
        if hand_in_face:
            self._mar_frozen = True
            self._is_compressing = False
            self._consecutive_frames = 0
            mar_buffer.clear()
            self._resume_after = current_time + cfg.mar_resume_cooldown
            return

        self._mar_frozen = False

        if current_time < self._resume_after:
            mar_buffer.clear()
            self._consecutive_frames = 0
            self._is_compressing = False
            return

        mar_buffer.append(mar_value)
        smooth_mar = float(np.mean(mar_buffer))

        if smooth_mar < calibration.compression_threshold:
            self._consecutive_frames += 1
            if self._consecutive_frames >= cfg.mar_min_consecutive_frames:
                if not self._is_compressing:
                    metrics.record_compression(current_time)
                    self._is_compressing = True
        else:
            self._consecutive_frames = 0
            self._is_compressing = False


@dataclass
class SessionMetrics:
    touch_events: int = 0
    lip_compression_events: int = 0
    touch_times: list = field(default_factory=list)
    compression_times: list = field(default_factory=list)
    zone_counts: dict = field(default_factory=lambda: {
        z: 0 for z in list(ZONE_REF_LANDMARKS.keys()) + ["unknown"]
    })
    session_start: float = field(default_factory=time.time)

    def record_touch(self, timestamp: float, zone: str = "unknown") -> None:
        self.touch_events += 1
        if zone not in self.zone_counts:
            zone = "unknown"
        self.zone_counts[zone] = self.zone_counts.get(zone, 0) + 1
        self.touch_times.append({
            "time_seconds": round(timestamp - self.session_start, 2),
            "zone": zone,
        })
        logger.debug("Touch #%d @ %.2fs â†’ %s", self.touch_events, timestamp - self.session_start, zone)

    def record_compression(self, timestamp: float) -> None:
        self.lip_compression_events += 1
        self.compression_times.append(round(timestamp - self.session_start, 2))
        logger.debug("Compression #%d @ %.2fs", self.lip_compression_events, timestamp - self.session_start)

    def summary(self) -> dict:
        elapsed = time.time() - self.session_start
        per_minute = lambda n: round(n / (elapsed / 60), 2) if elapsed > 0 else 0.0
        return {
            "session_start_timestamp": datetime.fromtimestamp(self.session_start).isoformat(),
            "total_session_seconds": round(elapsed, 1),
            "touch_events": self.touch_events,
            "lip_compression_events": self.lip_compression_events,
            "touch_per_minute": per_minute(self.touch_events),
            "lip_compression_per_minute": per_minute(self.lip_compression_events),
            "zone_counts": self.zone_counts,
            "touch_times": self.touch_times,
            "compression_times_seconds": self.compression_times,
        }


# ---------------------------------------------------------------------------
# Geometri YardÄ±mcÄ±larÄ±
# ---------------------------------------------------------------------------

def calculate_mar(face_landmarks, w: int, h: int) -> float:
    def px(idx):
        lm = face_landmarks.landmark[idx]
        return lm.x * w, lm.y * h
    hor = math.dist(px(MOUTH_LEFT), px(MOUTH_RIGHT))
    ver = math.dist(px(UPPER_LIP), px(LOWER_LIP))
    return ver / hor if hor > 0 else 0.0


def get_face_bbox(face_landmarks, w: int, h: int, padding: int) -> tuple:
    coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    return (
        max(0, int(x_min) - padding),
        max(0, int(y_min) - padding),
        min(w, int(x_max) + padding),
        min(h, int(y_max) + padding),
    )


def compute_zone_centroids(face_landmarks, w: int, h: int) -> dict[str, np.ndarray]:
    """
    Her zone iÃ§in referans landmark'lardan 2D centroid hesaplar.
    Returns: {"nose": np.array([cx, cy]), "mouth": ..., ...}
    """
    centroids = {}
    for zone, indices in ZONE_REF_LANDMARKS.items():
        coords = np.array([
            [face_landmarks.landmark[i].x * w,
             face_landmarks.landmark[i].y * h]
            for i in indices
        ])
        centroids[zone] = coords.mean(axis=0)
    return centroids


def _expand_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    return center + (points - center) * scale


def compute_zone_geometry(face_landmarks, w: int, h: int) -> dict[str, dict[str, np.ndarray | float]]:
    """
    Zone geometrisini hesaplar:
      - centroid: zone merkezi
      - poly:     genisletilmis polygon
      - radius:   centroid'e tipik uzaklik (normalize icin)
    """
    geometry: dict[str, dict[str, np.ndarray | float]] = {}
    for zone, indices in ZONE_REF_LANDMARKS.items():
        pts = np.array([
            [face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h]
            for i in indices
        ], dtype=np.float32)
        centroid = pts.mean(axis=0)
        scale = ZONE_POLY_SCALE.get(zone, 1.2)
        poly = _expand_polygon(pts, scale).astype(np.float32)
        radius = float(np.mean(np.linalg.norm(pts - centroid, axis=1))) + 1e-6
        geometry[zone] = {
            "centroid": centroid.astype(np.float32),
            "poly": poly,
            "radius": radius,
        }
    return geometry


def get_fingertip_points(hand_landmarks, w: int, h: int) -> list[np.ndarray]:
    """
    Her parmak ucu icin 2D nokta dondurur.
    Parmak ucu yoksa orta eklem fallback.
    """
    if hand_landmarks is None:
        return []

    tips = [
        np.array([hand_landmarks.landmark[i].x * w, hand_landmarks.landmark[i].y * h], dtype=np.float32)
        for i in FINGERTIP_LANDMARKS
        if i < len(hand_landmarks.landmark)
    ]
    if tips:
        return tips

    mids = [
        np.array([hand_landmarks.landmark[i].x * w, hand_landmarks.landmark[i].y * h], dtype=np.float32)
        for i in MID_JOINT_LANDMARKS
        if i < len(hand_landmarks.landmark)
    ]
    return mids


def get_fingertip_centroid(hand_landmarks, w: int, h: int) -> Optional[np.ndarray]:
    """
    Parmak uclarinin centroidini dondurur.

    Neden parmak ucu, avuc degil?
    El sandan alina gelince avuc centroidi kulak/yanaga yakin kalir
    -> yanlis zone. Parmak uclari yuze temas eden asil noktalardÄ±r.

    Parmak ucu yoksa (kapali el, asiri aci) orta eklem fallback.
    """
    if hand_landmarks is None:
        return None
    tips = np.array(get_fingertip_points(hand_landmarks, w, h), dtype=np.float32)
    if len(tips) > 0:
        return tips.mean(axis=0)
    mids = np.array([
        [hand_landmarks.landmark[i].x * w,
         hand_landmarks.landmark[i].y * h]
        for i in MID_JOINT_LANDMARKS
        if i < len(hand_landmarks.landmark)
    ], dtype=np.float32)
    return mids.mean(axis=0) if len(mids) > 0 else None


def classify_zone_by_distance_px(results, zone_geometry: dict, w: int, h: int) -> str:
    """
    Parmak ucu centroid -> zone centroid en yakin eslesme.

    Her aktif elin parmak uclari (FINGERTIP_LANDMARKS) toplanir,
    ortalamasi alinir. Bu nokta zone centroidlerine karsi olculur.
    En yakin zone doner; esit mesafede ZONE_PRIORITY tiebreaker.
    """
    tip_points: list[np.ndarray] = []
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        tip_points.extend(get_fingertip_points(hand, w, h))

    if not tip_points:
        return "unknown"

    # 1) Polygon icine dusen parmak noktalarina agirlikli oy ver.
    zone_scores = {z: 0.0 for z in ZONE_PRIORITY}
    any_inside = False

    for pt in tip_points:
        for zone in ZONE_PRIORITY:
            g = zone_geometry.get(zone)
            if g is None:
                continue
            poly = np.asarray(g["poly"], dtype=np.float32)
            inside = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0
            if not inside:
                continue
            any_inside = True
            centroid = np.asarray(g["centroid"], dtype=np.float32)
            radius = float(g["radius"])
            norm_d = float(np.linalg.norm(pt - centroid) / max(radius, 1e-6))
            score = 1.0 / (0.25 + norm_d)
            # Alin buyuk bir bolge oldugu icin dogrudan secilmesini zorlastir.
            if zone == "forehead":
                score *= 0.82
            zone_scores[zone] += score

    if any_inside:
        best_zone = max(ZONE_PRIORITY, key=lambda z: zone_scores.get(z, 0.0))
        if zone_scores.get(best_zone, 0.0) > 0:
            return best_zone

    # 2) Fallback: En yakin parmak-nokta / zone centroid eslesmesi (normalize).
    best_zone = "unknown"
    best_dist = float("inf")
    for pt in tip_points:
        for zone in ZONE_PRIORITY:
            g = zone_geometry.get(zone)
            if g is None:
                continue
            centroid = np.asarray(g["centroid"], dtype=np.float32)
            radius = float(g["radius"])
            norm_d = float(np.linalg.norm(pt - centroid) / max(radius, 1e-6))
            if zone == "forehead":
                norm_d *= 1.15
            if norm_d < best_dist:
                best_dist = norm_d
                best_zone = zone

    # Esik: Yuzden uzak noktalarin zorla bir zone'a map edilmesini engeller.
    return best_zone if best_dist <= 2.35 else "unknown"


def any_hand_in_bbox(results, bbox: tuple, w: int, h: int, min_landmarks: int = 1) -> bool:
    x_min, y_min, x_max, y_max = bbox
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand is None:
            continue
        coords = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])
        inside = (
            (coords[:, 0] > x_min) & (coords[:, 0] < x_max) &
            (coords[:, 1] > y_min) & (coords[:, 1] < y_max)
        )
        if np.sum(inside) >= min_landmarks:
            return True
    return False


class HandPresenceFilter:
    def __init__(self, exit_frames: int = 10, entry_min_lm: int = 1, stay_min_lm: int = 3) -> None:
        self._state: bool = False
        self._absent_frames: int = 0
        self._exit_frames = exit_frames
        self._entry_min_lm = entry_min_lm
        self._stay_min_lm = stay_min_lm

    def update(self, results, bbox: tuple, w: int, h: int) -> bool:
        min_lm = self._entry_min_lm if not self._state else self._stay_min_lm
        raw = any_hand_in_bbox(results, bbox, w, h, min_landmarks=min_lm)
        if raw:
            self._state = True
            self._absent_frames = 0
        else:
            if self._state:
                self._absent_frames += 1
                if self._absent_frames >= self._exit_frames:
                    self._state = False
                    self._absent_frames = 0
        return self._state


class ZoneStabilizer:
    """
    Zone tahminini frame bazli dalgalanmaya karsi stabil hale getirir.
    - Coklu frame oylama (temporal smoothing)
    - Zone degisimi icin histerezis
    """
    def __init__(self, window_size: int = 7, switch_confirm_frames: int = 3, min_votes: int = 2) -> None:
        self._window = deque(maxlen=max(3, window_size))
        self._stable_zone: str = "unknown"
        self._candidate_zone: str = "unknown"
        self._candidate_streak: int = 0
        self._switch_confirm_frames = max(1, switch_confirm_frames)
        self._min_votes = max(1, min_votes)

    def reset(self) -> None:
        self._window.clear()
        self._stable_zone = "unknown"
        self._candidate_zone = "unknown"
        self._candidate_streak = 0

    def update(self, zone: str) -> str:
        if zone not in VALID_ZONES_WITH_UNKNOWN:
            zone = "unknown"
        self._window.append(zone)
        votes = Counter(self._window)
        majority_zone, majority_count = votes.most_common(1)[0]

        if majority_count < self._min_votes:
            return self._stable_zone

        if self._stable_zone == "unknown":
            self._stable_zone = majority_zone
            self._candidate_zone = "unknown"
            self._candidate_streak = 0
            return self._stable_zone

        if majority_zone == self._stable_zone:
            self._candidate_zone = "unknown"
            self._candidate_streak = 0
            return self._stable_zone

        if majority_zone == self._candidate_zone:
            self._candidate_streak += 1
        else:
            self._candidate_zone = majority_zone
            self._candidate_streak = 1

        if self._candidate_streak >= self._switch_confirm_frames:
            self._stable_zone = majority_zone
            self._candidate_zone = "unknown"
            self._candidate_streak = 0

        return self._stable_zone


# ---------------------------------------------------------------------------
# UI Render
# ---------------------------------------------------------------------------

def draw_text_with_bg(img, text: str, pos: tuple, font_scale: float = 0.8,
                      color: tuple = (255, 255, 255), thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def render_ui(
    img,
    interview_state: InterviewState,
    calibration: CalibrationState,
    metrics: SessionMetrics,
    lip_sm: LipCompressionStateMachine,
    mar_value: float,
    current_time: float,
    cfg: AnalysisConfig,
    zone_geometry: dict,
    active_zone: Optional[str] = None,
) -> None:
    state_color = (0, 255, 0) if interview_state == InterviewState.ANSWER else (255, 255, 0)
    y = 70
    draw_text_with_bg(img, f"State: {interview_state.value}", (20, y), color=state_color)
    y += 40

    if calibration.in_progress:
        elapsed = current_time - calibration.start_time
        remaining = max(0, int(cfg.calibration_duration - elapsed))
        draw_text_with_bg(img, f"Kalibrasyon... {remaining}s", (20, 40), color=(0, 255, 255))
        return

    if not calibration.is_calibrated:
        draw_text_with_bg(img, "Kalibrasyon icin 'C' ye basin", (20, 40), color=(0, 0, 255))
        return

    draw_text_with_bg(img, f"Touches: {metrics.touch_events}", (20, y), color=(0, 165, 255))
    y += 40
    draw_text_with_bg(img, f"Lip Compressions: {metrics.lip_compression_events}", (20, y), color=(0, 0, 255))
    y += 40

    if lip_sm.mar_frozen:
        draw_text_with_bg(img, "MAR: Donduruldu", (20, y), font_scale=0.6, color=(0, 100, 255))
    else:
        draw_text_with_bg(img, f"MAR: {mar_value:.3f}", (20, y), font_scale=0.6, color=(200, 200, 200))
    y += 35

    if metrics.touch_times:
        last_zone = metrics.touch_times[-1]["zone"]
        label = ZONE_LABELS_TR.get(last_zone, last_zone)
        draw_text_with_bg(img, f"Son bolge: {label}", (20, y), font_scale=0.6, color=(180, 180, 0))

    # Debug: Zone centroid'lerini kÃ¼Ã§Ã¼k dairelerle Ã§iz
    # for zone, g in zone_geometry.items():
    #     centroid = g["centroid"]
    #     cx, cy = int(centroid[0]), int(centroid[1])
    #     cv2.circle(img, (cx, cy), 4, (0, 255, 200), -1)
    #     cv2.putText(img, zone[:3], (cx + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)


def render_alerts(img, touch_active: bool, is_compressing: bool, active_zone: Optional[str] = None) -> None:
    h = img.shape[0]
    if touch_active:
        label = ZONE_LABELS_TR.get(active_zone, "") if active_zone else ""
        text = f"Yuz Temasi! [{label}]" if label else "Yuz Temasi!"
        draw_text_with_bg(img, text, (20, h - 90), color=(0, 165, 255))
    if is_compressing:
        draw_text_with_bg(img, "Dudak Sikma!", (20, h - 50), color=(0, 0, 255))


# ---------------------------------------------------------------------------
# Klavye Handler
# ---------------------------------------------------------------------------

def handle_keypress(key: int, interview_state: InterviewState,
                    calibration: CalibrationState) -> tuple:
    if key in (ord("p"), ord("P")):
        return InterviewState.PREP, False
    if key in (ord("a"), ord("A")):
        return InterviewState.ANSWER, False
    if key in (ord("c"), ord("C")) and not calibration.is_calibrated:
        calibration.start()
    if key == 27:
        return interview_state, True
    return interview_state, False


# ---------------------------------------------------------------------------
# Ã‡Ä±ktÄ± Kaydetme
# ---------------------------------------------------------------------------

def save_results(metrics: SessionMetrics, calibration: CalibrationState,
                 cfg: AnalysisConfig) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.fromtimestamp(metrics.session_start).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"behavior_analysis_{timestamp_str}.json"
    payload = {
        "meta": {
            "module": "behavior_analysis",
            "version": "7.0",
            "config": cfg.model_dump(),
        },
        "calibration": {
            "baseline_mar": round(calibration.baseline_mar, 4),
            "compression_threshold": round(calibration.compression_threshold, 4),
        },
        "metrics": metrics.summary(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("SonuÃ§lar kaydedildi: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Ana Loop
# ---------------------------------------------------------------------------

def run(cfg: AnalysisConfig) -> tuple:
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    calibration    = CalibrationState()
    touch_sm       = TouchStateMachine()
    lip_sm         = LipCompressionStateMachine()
    metrics        = SessionMetrics()
    interview_state = InterviewState.PREP
    mar_buffer     = deque(maxlen=cfg.mar_smooth_window)
    hand_filter    = HandPresenceFilter(exit_frames=cfg.hand_exit_frames)
    zone_stabilizer = ZoneStabilizer(window_size=7, switch_confirm_frames=3, min_votes=2)
    zone_geometry: dict = {}
    active_zone: Optional[str] = None

    logger.info("Sistem hazÄ±r. Kalibrasyon iÃ§in 'C', Ã§Ä±kÄ±ÅŸ iÃ§in ESC.")

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        model_complexity=cfg.model_complexity,
    ) as holistic:

        while True:
            success, img = cap.read()
            if not success:
                logger.warning("Kamera frame okunamadÄ±.")
                break

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            current_time = time.time()

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            key = cv2.waitKey(1) & 0xFF
            interview_state, quit_flag = handle_keypress(key, interview_state, calibration)
            if quit_flag:
                break

            mar_value: float = 0.0

            if results.face_landmarks:
                face_bbox = get_face_bbox(results.face_landmarks, w, h, cfg.face_bbox_padding)
                mar_value = calculate_mar(results.face_landmarks, w, h)

                # Her frame'de zone geometrisini guncelle
                zone_geometry = compute_zone_geometry(results.face_landmarks, w, h)

                if calibration.in_progress:
                    calibration.update(mar_value, cfg.calibration_duration, cfg.lip_compression_ratio)

                elif calibration.is_calibrated:
                    hand_in_face = hand_filter.update(results, face_bbox, w, h)
                    stable_zone = "unknown"
                    zone_for_touch = "unknown"
                    prev_touch_events = metrics.touch_events

                    if hand_in_face:
                        raw_zone = classify_zone_by_distance_px(results, zone_geometry, w, h)
                        stable_zone = zone_stabilizer.update(raw_zone)
                        zone_for_touch = raw_zone
                    else:
                        zone_stabilizer.reset()

                    # Touch state machine â€” zone kaydÄ± touch_min_duration sonra yapÄ±lÄ±r
                    touch_sm.update(
                        hand_present=hand_in_face,
                        current_time=current_time,
                        cfg=cfg,
                        metrics=metrics,
                        results=results,
                        zone_geometry=zone_geometry,
                        w=w,
                        h=h,
                        zone_hint=zone_for_touch,
                    )

                    # Sadece yeni touch event oldugunda zone degissin.
                    if metrics.touch_events > prev_touch_events and metrics.touch_times:
                        active_zone = metrics.touch_times[-1]["zone"]
                    elif not touch_sm.is_active:
                        active_zone = None

                    lip_sm.update(
                        hand_in_face=touch_sm.is_active,
                        mar_value=mar_value,
                        mar_buffer=mar_buffer,
                        calibration=calibration,
                        current_time=current_time,
                        cfg=cfg,
                        metrics=metrics,
                    )

            render_ui(img, interview_state, calibration, metrics,
                      lip_sm, mar_value, current_time, cfg, zone_geometry, active_zone)
            render_alerts(img, touch_sm.is_active, lip_sm.is_compressing, active_zone)
            cv2.imshow("Behavior Analysis", img)

    cap.release()
    cv2.destroyAllWindows()
    return metrics, calibration


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = AnalysisConfig()
    metrics, calibration = run(cfg)
    output_path = save_results(metrics, calibration, cfg)
    summary = metrics.summary()
    logger.info("=" * 40)
    logger.info("SESSION SUMMARY")
    for k, v in summary.items():
        logger.info("  %-40s %s", k, v)
    logger.info("Ã‡Ä±ktÄ±: %s", output_path)
    logger.info("=" * 40)


if __name__ == "__main__":
    main()

