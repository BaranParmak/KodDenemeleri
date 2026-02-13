"""
HEAD POSE ESTIMATION - %100 STABİL VERSİYON
Yukarı/Aşağı yönleri düzeltildi, FPS darboğazı giderildi, Tuş kontrolleri iyileştirildi.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# ===================== GLOBAL DEĞİŞKENLER =====================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,  
    min_tracking_confidence=0.3    
)

# 6 Kritik Nokta
LANDMARK_IDS = [1, 152, 263, 33, 291, 61]

# 3D Yüz Modeli
model_points_3d = np.array([
    [0.0, 0.0, 0.0],            
    [0.0, -330.0, -65.0],       
    [-225.0, 170.0, -135.0],    
    [225.0, 170.0, -135.0],     
    [-150.0, -150.0, -125.0],   
    [150.0, -150.0, -125.0]     
], dtype=np.float64)

# Stabilizasyon
pose_history = deque(maxlen=5)

# Kalibrasyon (Saniye yerine sadece 50 kare toplamasını isteyeceğiz, bu sayede FPS düşük olsa da çalışır)
calibration_start = None
calibration_data = []
calibrated = False
neutral_pose = None
calibration_frames_needed = 50  # 150'den 50'ye düşürdük, çok daha hızlı ve garantili bitecek.

# Eşik Değerleri (Thresholds)
base_thresholds = {'pitch': 15.0, 'yaw': 20.0, 'roll': 15.0}
dynamic_thresholds = base_thresholds.copy()

# Kamera
camera_matrix = None
dist_coeffs = np.zeros((4, 1))

# Metrikler
warning_count = 0
total_frames = 0
valid_frames = 0


# ===================== FONKSİYONLAR =====================

def initialize_camera(w, h):
    global camera_matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

def extract_landmarks(face_landmarks, w, h):
    image_points = []
    for idx in LANDMARK_IDS:
        landmark = face_landmarks.landmark[idx]
        x = landmark.x * w
        y = landmark.y * h
        image_points.append([x, y])
    return np.array(image_points, dtype=np.float64)

def solve_pnp(image_points):
    if camera_matrix is None: return None
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success: return None
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    
    pitch = angles[0]
    yaw = -angles[1]  
    roll = angles[2]
    return pitch, yaw, roll

def stabilize_pose(pitch, yaw, roll):
    global pose_history
    pose_history.append((pitch, yaw, roll))
    if len(pose_history) > 0:
        return np.mean([p[0] for p in pose_history]), np.mean([p[1] for p in pose_history]), np.mean([p[2] for p in pose_history])
    return pitch, yaw, roll

def add_calibration_frame(pitch, yaw, roll):
    global calibration_data, calibrated, neutral_pose, dynamic_thresholds, calibration_frames_needed
    
    calibration_data.append((pitch, yaw, roll))
    
    if len(calibration_data) >= calibration_frames_needed:
        avg_pitch = np.mean([p[0] for p in calibration_data])
        avg_yaw = np.mean([p[1] for p in calibration_data])
        avg_roll = np.mean([p[2] for p in calibration_data])
        
        neutral_pose = (avg_pitch, avg_yaw, avg_roll)
        calibrated = True
        
        std_pitch = np.std([p[0] for p in calibration_data])
        std_yaw = np.std([p[1] for p in calibration_data])
        std_roll = np.std([p[2] for p in calibration_data])
        
        dynamic_thresholds = {
            'pitch': max(base_thresholds['pitch'], std_pitch * 3),
            'yaw': max(base_thresholds['yaw'], std_yaw * 3),
            'roll': max(base_thresholds['roll'], std_roll * 3)
        }
        print(f"\n✅ KALİBRASYON TAMAMLANDI!")
        return True
    return False

def get_pose_deviation(pitch, yaw, roll):
    if not calibrated or neutral_pose is None:
        return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
    return {'pitch': pitch - neutral_pose[0], 'yaw': yaw - neutral_pose[1], 'roll': roll - neutral_pose[2]}

def check_pose_limits(deviation):
    return {
        'pitch': abs(deviation['pitch']) > dynamic_thresholds['pitch'],
        'yaw': abs(deviation['yaw']) > dynamic_thresholds['yaw'],
        'roll': abs(deviation['roll']) > dynamic_thresholds['roll']
    }

def draw_text(img, text, pos, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
    x, y = pos
    cv2.rectangle(img, (x-5, y-th-5), (x+tw+5, y+5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, 0.6, color, 2)


# ===================== ANA DÖNGÜ =====================

def main():
    global camera_matrix, calibration_start, calibrated, calibration_data
    global neutral_pose, warning_count, total_frames, valid_frames, dynamic_thresholds
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    print("=" * 60)
    print("HEAD POSE ESTIMATION (STABİL VERSİYON)")
    print("=" * 60)
    print("Lütfen açılan kamera penceresinin üzerine farenizle TIKLAYIN.")
    print("Kontroller: [C] Kalibrasyon | [R] Sıfırla | [Q] Çıkış")
    print("=" * 60)
    
    while True:
        success, img = cap.read()
        if not success: break
        
        total_frames += 1
        h, w, _ = img.shape
        
        if camera_matrix is None:
            initialize_camera(w, h)
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_display = img.copy()
        
        # ========== KALİBRASYON MODU ==========
        if not calibrated and calibration_start is not None:
            progress = len(calibration_data) / calibration_frames_needed
            bar_width = int(400 * progress)
            
            cv2.rectangle(img_display, (50, 30), (450, 80), (50, 50, 50), -1)
            cv2.rectangle(img_display, (50, 30), (50 + bar_width, 80), (0, 255, 0), -1)
            draw_text(img_display, "KALIBRASYON - Sabit Durun!", (50, 110), (0, 255, 255))
            draw_text(img_display, f"Toplanan Veri: {len(calibration_data)} / {calibration_frames_needed}", (50, 140))
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                image_points = extract_landmarks(face_landmarks, w, h)
                pose_result = solve_pnp(image_points)
                
                if pose_result is not None:
                    pitch, yaw, roll = pose_result
                    pitch, yaw, roll = stabilize_pose(pitch, yaw, roll)
                    add_calibration_frame(pitch, yaw, roll)
        
        # ========== NORMAL MOD ==========
        else:
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                image_points = extract_landmarks(face_landmarks, w, h)
                
                valid_frames += 1
                pose_result = solve_pnp(image_points)
                
                if pose_result is not None:
                    pitch, yaw, roll = pose_result
                    pitch, yaw, roll = stabilize_pose(pitch, yaw, roll)
                    
                    for point in image_points:
                        cv2.circle(img_display, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
                    
                    if not calibrated:
                        draw_text(img_display, "KALIBRASYON BEKLENIYOR - 'C' basin", (10, 30), (0, 165, 255))
                    else:
                        draw_text(img_display, "KALIBRASYON OK", (10, 30), (0, 255, 0))
                    
                    y = 70
                    draw_text(img_display, f"Pitch: {pitch:.1f}", (10, y))
                    draw_text(img_display, f"Yaw: {yaw:.1f}", (10, y+30))
                    draw_text(img_display, f"Roll: {roll:.1f}", (10, y+60))
                    
                    if calibrated and neutral_pose is not None:
                        deviation = get_pose_deviation(pitch, yaw, roll)
                        limits = check_pose_limits(deviation)
                        
                        y = 200
                        draw_text(img_display, "SAPMA:", (10, y), (0, 255, 255))
                        
                        color = (0, 0, 255) if limits['pitch'] else (0, 255, 0)
                        draw_text(img_display, f"P: {deviation['pitch']:+.1f}", (10, y+30), color)
                        
                        color = (0, 0, 255) if limits['yaw'] else (0, 255, 0)
                        draw_text(img_display, f"Y: {deviation['yaw']:+.1f}", (10, y+55), color)
                        
                        color = (0, 0, 255) if limits['roll'] else (0, 255, 0)
                        draw_text(img_display, f"R: {deviation['roll']:+.1f}", (10, y+80), color)
                        
                        if any(limits.values()):
                            warning_count += 1
                            warning_messages = []
                            
                            if limits['yaw']:
                                if deviation['yaw'] > 0: warning_messages.append("SOLA Bakiyorsunuz!")
                                else: warning_messages.append("SAGA Bakiyorsunuz!")
                            
                            if limits['pitch']:
                                if deviation['pitch'] > 0: warning_messages.append("ASAGI Bakiyorsunuz!")
                                else: warning_messages.append("YUKARI Bakiyorsunuz!")
                            
                            if limits['roll']:
                                warning_messages.append("Basinizi Duz Tutun!")
                            
                            for i, msg in enumerate(warning_messages):
                                draw_text(img_display, msg, (10, h-80+(i*35)), (0, 0, 255))
                        
                        draw_text(img_display, f"Uyari: {warning_count}", (w-200, 30), (255, 255, 255))
            else:
                draw_text(img_display, "Yuz Tespit Edilemedi", (10, 30), (0, 0, 255))
        
        cv2.imshow('Head Pose Estimation', img_display)
        
        # --- KLAVYE KONTROLLERİ BÖLÜMÜ ---
        key = cv2.waitKey(1) & 0xFF
        
        # Sadece hata ayıklama için (tuş basıyorsa terminalde görünür)
        if key != 255: 
            pass # İstersen buraya print(key) yazarak klavyenin kod gönderip göndermediğini görebilirsin.
            
        if key in [ord('q'), ord('Q'), 27]:  # Q, q veya ESC tuşu
            break
        elif key in [ord('c'), ord('C')]:    # C veya c tuşu (Büyük küçük harf sorunu çözüldü)
            print("\n🔄 Kalibrasyon başlatıldı, veri toplanıyor...")
            calibration_start = time.time()
            calibration_data = []
            calibrated = False
        elif key in [ord('r'), ord('R')]:    # R veya r tuşu
            print("\n🔄 Kalibrasyon sıfırlandı!")
            calibration_start = None
            calibration_data = []
            calibrated = False
            neutral_pose = None
            dynamic_thresholds = base_thresholds.copy()
            pose_history.clear()
            
    print(f"\n📊 ÖZET: {total_frames} frame işlendi.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
