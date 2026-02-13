import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import numpy as np
import time

# Webcam
cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

# Landmark ID'leri
# Sol: [Üst, Alt, Sol, Sağ]
LEFT_EYE = [159, 23, 130, 243]
# Sağ: [Üst, Alt, Sol, Sağ]
RIGHT_EYE = [386, 374, 362, 263]

ratioList = []
blinkCounter = 0
blinkFrameCounter = 0
blinkClosedFrames = 3
color = (255, 0, 255)

# --------- KALIBRASYON ----------
calibration_duration = 5
calibration_start = None
calibration_values = []
calibrated = False
blink_threshold = 30 

# --------- DAVRANIŞSAL METRİKLER ----------
interview_start_time = time.time()
blink_times = []

BURST_WINDOW = 3.0  # saniye
BURST_MIN_BLINKS = 3 # 3 saniyede en az 3 kırpma
burst_counter = 0
burst_active = False # Burst durumunu takip etmek için bayrak

print("Sistem Başlatılıyor...")

def calculate_ratio(face, eye_indices):
    """Verilen göz indeksleri için oranı hesaplar"""
    try:
        # Noktaları al
        up = face[eye_indices[0]]
        down = face[eye_indices[1]]
        left = face[eye_indices[2]]
        right = face[eye_indices[3]]
        
        # Mesafeleri ölç
        ver_dist, _ = detector.findDistance(up, down)
        hor_dist, _ = detector.findDistance(left, right)
        
        # Sıfıra bölünme hatasını önle
        if hor_dist == 0: return 100
        
        return (ver_dist / hor_dist) * 100
    except:
        return 100

while True:
    success, img = cap.read()
    if not success: break
    
    # img = cv2.flip(img, 1) # Ayna görüntüsü istersen açabilirsin
    img, faces = detector.findFaceMesh(img, draw=False)
    
    img_display = img.copy() # Ekrana basılacak temiz kopya

    if faces:
        face = faces[0]

        # 1. Ratio Hesaplama (Çift Göz)
        left_ratio = calculate_ratio(face, LEFT_EYE)
        right_ratio = calculate_ratio(face, RIGHT_EYE)
        
        # İki gözün ortalaması (Daha stabil)
        ratio = (left_ratio + right_ratio) / 2

        # Smoothing
        ratioList.append(ratio)
        if len(ratioList) > 5: ratioList.pop(0)
        ratioAvg = np.mean(ratioList)

        # 2. KALIBRASYON MODU
        if not calibrated:
            if calibration_start is None:
                calibration_start = time.time()

            elapsed = time.time() - calibration_start
            calibration_values.append(ratioAvg)

            # Progress UI
            progress = min(elapsed / calibration_duration, 1)
            bar_w = int(400 * progress)
            
            cv2.rectangle(img_display, (50, 50), (450, 90), (50, 50, 50), -1)
            cv2.rectangle(img_display, (50, 50), (50 + bar_w, 90), (0, 255, 0), -1)
            cvzone.putTextRect(img_display, "Kalibrasyon - Ekrana Bakin", (50, 120), scale=1.5)

            if elapsed >= calibration_duration:
                baseline = np.mean(calibration_values)
                std = np.std(calibration_values)
                # Baseline'dan 2 standart sapma aşağısı güvenli eşiktir
                blink_threshold = baseline - (2.5 * std) 
                calibrated = True
                print(f"Kalibrasyon Bitti. Eşik: {blink_threshold:.2f}")

        # 3. NORMAL MOD (ANALİZ)
        else:
            current_time = time.time() - interview_start_time

            # Kırpma Algılama
            if ratioAvg < blink_threshold:
                blinkFrameCounter += 1
                color = (0, 200, 0)
            else:
                if blinkFrameCounter >= blinkClosedFrames:
                    blinkCounter += 1
                    # Zaman damgasını kaydet
                    blink_times.append(current_time)
                
                blinkFrameCounter = 0
                color = (255, 0, 255)

            # --- METRİK HESAPLAMALARI ---
            
            # A) Blink Rate (Dakikadaki Hız)
            duration_minutes = max(current_time / 60, 0.01)
            blink_rate = blinkCounter / duration_minutes

            # B) Variability (Değişkenlik - Stres Göstergesi)
            if len(blink_times) > 1:
                intervals = np.diff(blink_times) # Kırpmalar arası süreler
                blink_variability = np.std(intervals) # Sürelerin düzensizliği
            else:
                blink_variability = 0

            # C) Burst Detection (DÜZELTİLMİŞ MANTIK)
            # Son X saniyedeki kırpmaları filtrele
            recent_blinks = [t for t in blink_times if current_time - t <= BURST_WINDOW]
            
            if len(recent_blinks) >= BURST_MIN_BLINKS:
                if not burst_active: # Eğer zaten burst modunda değilsek say
                    burst_counter += 1
                    burst_active = True # Burst başladı, kilitle
            else:
                burst_active = False # Kriterin altına düşünce kilidi aç

            # --- UI GÖSTERİMİ ---
            # Verileri sol tarafa alt alta yazalım
            cvzone.putTextRect(img_display, f'Count: {blinkCounter}', (20, 60), scale=1.5, thickness=2, colorR=color)
            cvzone.putTextRect(img_display, f'Rate: {blink_rate:.1f}/m', (20, 105), scale=1.5, thickness=2)
            cvzone.putTextRect(img_display, f'Var: {blink_variability:.2f}', (20, 150), scale=1.5, thickness=2)
            
            # Burst uyarısı (Kırmızı Yanıp Sönen Efekt yapılabilir)
            burst_color = (0, 0, 255) if burst_active else (50, 50, 50)
            cvzone.putTextRect(img_display, f'BURST: {burst_counter}', (20, 195), scale=1.5, thickness=2, colorR=burst_color)

            # Grafiği güncelle ve ekle
            imgPlot = plotY.update(ratioAvg, color)
            img_display = cv2.resize(img_display, (640, 360))
            imgStack = cvzone.stackImages([img_display, imgPlot], 2, 1)
            
            cv2.imshow("Advanced Blink Analysis", imgStack)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

    # Yüz yoksa veya kalibrasyon sırasındaysa
    img_display = cv2.resize(img_display, (640, 360))
    cv2.imshow("Advanced Blink Analysis", img_display)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
