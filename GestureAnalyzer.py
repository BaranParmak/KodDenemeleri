import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# ================= CONFIG =================

CALIBRATION_TIME = 3
TOUCH_MIN_DURATION = 0.3   # saniye
MAR_TALKING_DELTA = 0.015

# Interview states
STATE_PREP = "PREP"
STATE_ANSWER = "ANSWER"

# ================= GLOBAL =================

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
UPPER_LIP = 13
LOWER_LIP = 14

# ================= METRIC STORAGE =================

touch_events = 0
lip_bite_events = 0

touch_times = []
bite_times = []

# state flags
is_touching = False
is_biting = False
touch_counted = False  

touch_start_time = 0
occlusion_cooldown_until = 0  # YENİ: Kapanma (Occlusion) bekleme süresi

# calibration
baseline_mar = None
compression_threshold = None
is_calibrated = False

# interview state
interview_state = STATE_PREP

# ================= FUNCTIONS =================

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_mar(face_landmarks, w, h):
    p_left_x = face_landmarks.landmark[MOUTH_LEFT].x * w
    p_left_y = face_landmarks.landmark[MOUTH_LEFT].y * h
    
    p_right_x = face_landmarks.landmark[MOUTH_RIGHT].x * w
    p_right_y = face_landmarks.landmark[MOUTH_RIGHT].y * h
    
    p_upper_x = face_landmarks.landmark[UPPER_LIP].x * w
    p_upper_y = face_landmarks.landmark[UPPER_LIP].y * h
    
    p_lower_x = face_landmarks.landmark[LOWER_LIP].x * w
    p_lower_y = face_landmarks.landmark[LOWER_LIP].y * h

    hor_dist = math.hypot(p_right_x - p_left_x, p_right_y - p_left_y)
    ver_dist = math.hypot(p_lower_x - p_upper_x, p_lower_y - p_upper_y)

    if hor_dist == 0:
        return 0

    return ver_dist / hor_dist

def get_face_bbox(face_landmarks, w, h, padding=20):
    coords = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
    
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)

    x_min = max(0, int(min_coords[0]) - padding)
    y_min = max(0, int(min_coords[1]) - padding)
    x_max = min(w, int(max_coords[0]) + padding)
    y_max = min(h, int(max_coords[1]) + padding)

    return x_min, y_min, x_max, y_max

def check_hand_face_intersection(hand_landmarks, face_bbox, w, h):
    if not hand_landmarks or not face_bbox:
        return False

    x_min, y_min, x_max, y_max = face_bbox

    for lm in hand_landmarks.landmark:
        hx, hy = int(lm.x * w), int(lm.y * h)
        if x_min < hx < x_max and y_min < hy < y_max:
            return True

    return False

def draw_text_with_bg(img, text, pos, font_scale=0.8, color=(255, 255, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

# ================= MAIN =================

def main():

    global baseline_mar
    global compression_threshold
    global is_calibrated
    global touch_events, lip_bite_events
    global is_touching, is_biting, touch_counted
    global touch_start_time, occlusion_cooldown_until
    global interview_state

    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    calibration_values = []
    calibration_start = None

    last_mar_values = deque(maxlen=5)
    
    session_start_time = time.time()  

    print("Sistem hazır. Kalibrasyon için 'C' tuşuna basın.")

    while True:

        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        face_bbox = None
        mar_value = 0
        current_touch = False  

        current_time = time.time()

        key = cv2.waitKey(1) & 0xFF

        if key == ord("p") or key == ord("P"):
            interview_state = STATE_PREP
        elif key == ord("a") or key == ord("A"):
            interview_state = STATE_ANSWER
        elif key == ord('c') or key == ord('C'):
            if not is_calibrated:
                print("Kalibrasyon başlıyor...")
                calibration_start = time.time()
                calibration_values = []
        elif key == 27: # ESC
            break

        # ===== FACE & HAND ANALYSIS =====
        if results.face_landmarks:

            face_bbox = get_face_bbox(results.face_landmarks, w, h)
            mar_value = calculate_mar(results.face_landmarks, w, h)

            cv2.rectangle(img, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 165, 0), 2)

            # -------- HAND ANALYSIS --------
            if is_calibrated:
                if results.left_hand_landmarks:
                    if check_hand_face_intersection(results.left_hand_landmarks, face_bbox, w, h):
                        current_touch = True

                if results.right_hand_landmarks:
                    if check_hand_face_intersection(results.right_hand_landmarks, face_bbox, w, h):
                        current_touch = True
                
                # YENİ MANTIK: El yüzdeyse sistemi 0.5 saniye sağırlaştır (Cooldown)
                if current_touch:
                    occlusion_cooldown_until = current_time + 0.5

            # -------- CALIBRATION --------
            if not is_calibrated and calibration_start is not None:

                calibration_values.append(mar_value)
                elapsed = current_time - calibration_start

                draw_text_with_bg(img, f"Calibration... {int(CALIBRATION_TIME - elapsed)}s", (20, 40), color=(0, 255, 255))

                if elapsed >= CALIBRATION_TIME:
                    baseline_mar = np.mean(calibration_values)
                    compression_threshold = baseline_mar * 0.6
                    is_calibrated = True

                    print("Kalibrasyon tamamlandı.")
                    
            elif not is_calibrated:
                draw_text_with_bg(img, "Kalibrasyon icin 'C' tusuna basin", (20, 40), color=(0, 0, 255))

            # -------- BEHAVIOR ANALYSIS --------
            elif is_calibrated:

                # YENİ MANTIK: Eğer el yüzdeyse veya el çekildikten sonraki 0.5 saniye içindeysek...
                if current_time < occlusion_cooldown_until:
                    # Bozuk verilerin ortalamayı etkilememesi için hafızayı sıfırla
                    last_mar_values.clear() 
                    is_biting = False
                else:
                    # El uzakta ve sistem stabil. MAR hesaplamasına devam et.
                    last_mar_values.append(mar_value)
                    smooth_mar = np.mean(last_mar_values)

                    # ===== LIP BITE =====
                    if smooth_mar < compression_threshold:
                        if not is_biting:
                            lip_bite_events += 1
                            bite_times.append(current_time)
                            is_biting = True
                        draw_text_with_bg(img, "Dudak Sikma Tespit Edildi!", (20, h - 50), color=(0, 0, 255))
                    else:
                        is_biting = False

                
                # -------- TOUCH TIME FILTER --------
                if current_touch:
                    if not is_touching:
                        touch_start_time = current_time
                        is_touching = True
                        touch_counted = False  
                        
                    elif not touch_counted:
                        elapsed_touch = current_time - touch_start_time
                        if elapsed_touch > TOUCH_MIN_DURATION:
                            touch_events += 1
                            touch_times.append(current_time)
                            touch_counted = True  
                            
                    if touch_counted: 
                         draw_text_with_bg(img, "Yuz Temasi!", (20, h - 90), color=(0, 165, 255))
                else:
                    is_touching = False
                    touch_counted = False
                    touch_start_time = 0

        # ===== UI =====
        state_color = (0, 255, 0) if interview_state == STATE_ANSWER else (255, 255, 0)
        
        y_pos = 70
        draw_text_with_bg(img, f"State: {interview_state}", (20, y_pos), color=state_color)
        y_pos += 40
        
        if is_calibrated:
            draw_text_with_bg(img, f"Touches: {touch_events}", (20, y_pos), color=(0, 165, 255))
            y_pos += 40
            draw_text_with_bg(img, f"Lip Bites: {lip_bite_events}", (20, y_pos), color=(0, 0, 255))
            y_pos += 40
            
            # Ekranda kullanıcıya sistemin MAR durumunu göstermek için ufak bir bilgi (Opsiyonel)
            if current_time < occlusion_cooldown_until:
                draw_text_with_bg(img, "MAR: Durduruldu (El yuzde)", (20, y_pos), font_scale=0.6, color=(100, 100, 100))
            else:
                draw_text_with_bg(img, f"MAR: {mar_value:.3f}", (20, y_pos), font_scale=0.6, color=(200, 200, 200))

        cv2.imshow("Behavior Analysis", img)

    # ===== SESSION SUMMARY =====
    print("\n====================")
    print("Session Summary")
    print("====================")
    print(f"Total Touch events: {touch_events}")
    print(f"Total Lip bite events: {lip_bite_events}")
    
    total_session_time = current_time - session_start_time
    
    if total_session_time > 0 and is_calibrated:
        print(f"Total Session Time: {total_session_time:.1f} seconds")
        print(f"Touch events per minute: {(touch_events / (total_session_time / 60)):.2f}")
        print(f"Lip bite events per minute: {(lip_bite_events / (total_session_time / 60)):.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
