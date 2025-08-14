from djitellopy import Tello
import cv2
import time
import numpy as np
import onnxruntime as ort
import datetime
import os
import shutil, subprocess, sys

# ======= Model/Session =======
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "yolov5s.onnx")
print("Lade Modell aus:", MODEL_PATH)

try:
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
except Exception as e:
    raise SystemExit(f"ONNX Runtime konnte die Datei nicht laden:\n{MODEL_PATH}\n{e}")

input_name = sess.get_inputs()[0].name

# ======= Einstellungen =======
INPUT_SIZE = 640
CONF_THRES = 0.35
IOU_THRES  = 0.45
PERSON_CLASS_ID = 0

# Distanz/Regler
TARGET_H = 350
DEAD_XY  = 50
DEAD_Z   = 25

MAX_FB = 70
MAX_LR = 0      # seitwärts aktuell deaktiviert
MAX_UD = 70
MAX_YAW = 80

Kp_yaw = 0.26
Kd_yaw = 0.12
K_fb   = 0.60
K_ud   = 0.36
Kd_ud  = 0.16

# Begrenzer / Glättung
YAW_SLEW_PER_STEP = 12
UD_SLEW_PER_STEP  = 10
FB_ALPHA = 0.35

TANH_GAIN = 2.5

FB_ALIGN_GATE = 0.35
FB_VERT_GATE  = 0.35
FB_VERT_CREEP = 0.20

RECORD_ANNOTATED = True
REC_DIR = os.path.join(HERE, "recordings")
os.makedirs(REC_DIR, exist_ok=True)
video_out = None
REC_FPS = 15

BATTERY_INTERVAL = 5.0
LOOP_HZ = 35
DT = 1.0 / LOOP_HZ

# ======= Utilities =======
def clamp(v, lo, hi): return max(lo, min(hi, v))

def letterbox(im, new_size=INPUT_SIZE, color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    pad_h, pad_w = new_size - nh, new_size - nw
    top = pad_h // 2; bottom = pad_h - top
    left = pad_w // 2; right  = pad_w - left
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, left, top

def nms(boxes, scores, iou_thres):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, iou_thres)
    if len(idxs) == 0: return []
    return [int(i) for i in np.array(idxs).reshape(-1)]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

# ======= Tello Safe-Send =======
def safe_send_rc(t, lr, fb, ud, yaw, win_name="Tello - Person Follow (YOLO + PD)"):
    try:
        t.send_rc_control(lr, fb, ud, yaw)
        return True
    except OSError as e:
        print(f"[NETZWERKFEHLER] {e}")
        try:
            cv2.displayOverlay(win_name, "NETZWERK VERLOREN – WLAN prüfen. ESC = landen", 3000)
        except Exception:
            pass
        return False
    except Exception as e:
        print(f"[SEND FEHLER] {e}")
        return False

# ======= Tello Setup =======
cv2.setUseOptimized(True)
try: cv2.setNumThreads(1)
except: pass

t = Tello()
t.connect()
try:
    t.set_video_fps(Tello.FPS_15)
    t.set_video_bitrate(Tello.BITRATE_1MBPS)
except: pass

battery = t.get_battery()
print("Akku:", battery, "%")

t.streamon(); time.sleep(0.25)
t.takeoff();  time.sleep(0.4)
try: t.move_up(40)
except: pass

t.send_rc_control(0,0,0,0)
frame_reader = t.get_frame_read()

# --- Aufnahme vorbereiten ---
print("Warte auf erstes Kamerabild für Recording-Setup...")
start_wait_ts = time.time()
first = None
while first is None:
    first = frame_reader.frame
    if time.time() - start_wait_ts > 5.0:
        print("[WARN] Kein Frame nach 5s – versuche weiter...")
        start_wait_ts = time.time()
    cv2.waitKey(1)

H, W = first.shape[:2]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(REC_DIR, f"tello_follow_{ts}.mp4")
fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
video_out = cv2.VideoWriter(out_path, fourcc, REC_FPS, (W, H))
if not video_out.isOpened():
    print("[REC] mp4v nicht verfuegbar -> Fallback auf AVI/XVID")
    out_path = os.path.join(REC_DIR, f"tello_follow_{ts}.avi")
    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    video_out = cv2.VideoWriter(out_path, fourcc, REC_FPS, (W, H))
if not video_out.isOpened():
    raise SystemExit("[REC] Konnte VideoWriter weder fuer MP4 noch AVI oeffnen.")
print(f"[REC] Aufnahme gestartet: {out_path} (fix {REC_FPS} FPS)")
written_frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX
last_batt = time.time()
start_ts = time.time()
print("Tracking aktiv. ESC: landen | SPACE: Hover | X: PANIK-STOP")

# Zustände Controller
ema_cx = ema_cy = None
EMA_ALPHA = 0.45
prev_err_x = 0
prev_yaw_cmd = 0
prev_fb_cmd = 0
prev_track_box = None   # (x1,y1,x2,y2)

try:
    while True:
        frame = frame_reader.frame
        if frame is None:
            cv2.waitKey(1)
            continue
        h, w = frame.shape[:2]

        # Rohbild aufheben, falls wir ohne Overlay speichern wollen
        raw_frame = frame.copy()

        # ======= YOLO Inferenz =======
        inp, ratio, padw, padh = letterbox(frame, INPUT_SIZE)
        blob = (inp.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        out = sess.run(None, {input_name: blob})[0]
        if out.ndim == 3:
            out = out[0]

        boxes = []; scores = []
        for det in out:
            conf = det[4]
            if conf < CONF_THRES: continue
            class_scores = det[5:]
            cls_id = int(np.argmax(class_scores))
            if cls_id != PERSON_CLASS_ID: continue
            score = float(conf * class_scores[cls_id])
            if score < CONF_THRES: continue
            cx, cy, bw, bh = det[0:4]
            x1 = (cx - bw/2 - padw) / ratio
            y1 = (cy - bh/2 - padh) / ratio
            x2 = (cx + bw/2 - padw) / ratio
            y2 = (cy + bh/2 - padh) / ratio
            x1 = clamp(int(x1), 0, w-1); y1 = clamp(int(y1), 0, h-1)
            x2 = clamp(int(x2), 0, w-1); y2 = clamp(int(y2), 0, h-1)
            if x2 <= x1 or y2 <= y1: continue
            boxes.append([x1, y1, x2-x1, y2-y1]); scores.append(score)

        # ======= Regelung =======
        lr = fb = ud = yaw = 0
        bh_pix = 0

        pick = nms(boxes, scores, IOU_THRES) if boxes else []
        best_box = None
        if pick:
            cand = [boxes[i] for i in pick]
            if prev_track_box is not None:
                ious = []
                for b in cand:
                    x1,y1,w1,h1 = b
                    bb = (x1,y1,x1+w1,y1+h1)
                    ious.append(iou(bb, prev_track_box))
                best_idx = int(np.argmax(ious))
                if ious[best_idx] < 0.2:
                    areas = [c[2]*c[3] for c in cand]
                    best_idx = int(np.argmax(areas))
            else:
                areas = [c[2]*c[3] for c in cand]
                best_idx = int(np.argmax(areas))
            x,y,bw,bbh = cand[best_idx]
            best_box = (x, y, x+bw, y+bbh)

        if best_box is not None:
            x1,y1,x2,y2 = best_box
            bw = x2-x1; bbh = y2-y1
            cx = x1 + bw//2
            cy = y1 + bbh//2
            bh_pix = bbh

            prev_track_box = (x1,y1,x2,y2)

            # Zielpunkt glätten
            if ema_cx is None:
                ema_cx, ema_cy = cx, cy
            else:
                ema_cx = int(EMA_ALPHA*cx + (1-EMA_ALPHA)*ema_cx)
                ema_cy = int(EMA_ALPHA*cy + (1-EMA_ALPHA)*ema_cy)

            # Fehler
            err_x = ema_cx - w//2
            err_y = ema_cy - h//2
            err_z = bbh - TARGET_H

            if abs(err_x) < DEAD_XY: err_x = 0
            if abs(err_y) < DEAD_XY: err_y = 0
            if abs(err_z) < DEAD_Z:  err_z = 0

            # normierte Fehler
            err_norm_h = err_x / (w/2.0)
            err_norm_v = err_y / (h/2.0)

            # Yaw PD
            p_term = Kp_yaw * MAX_YAW * np.tanh(TANH_GAIN * err_norm_h)
            try:
                prev_err_norm_h
            except NameError:
                prev_err_norm_h = 0.0
            derr_norm_h = (err_norm_h - prev_err_norm_h) / DT
            d_term = Kd_yaw * MAX_YAW * derr_norm_h
            prev_err_norm_h = err_norm_h

            yaw_cmd = int(clamp(p_term + d_term, -MAX_YAW, MAX_YAW))
            delta = yaw_cmd - prev_yaw_cmd
            if   delta >  YAW_SLEW_PER_STEP: yaw_cmd = prev_yaw_cmd + YAW_SLEW_PER_STEP
            elif delta < -YAW_SLEW_PER_STEP: yaw_cmd = prev_yaw_cmd - YAW_SLEW_PER_STEP
            yaw = yaw_cmd
            prev_yaw_cmd = yaw

            # Vor/zurück mit Align-Gates
            fb_cmd = int(clamp(-K_fb * err_z, -MAX_FB, MAX_FB))
            if abs(err_norm_h) > FB_ALIGN_GATE:
                fb_cmd = int(fb_cmd * 0.20)
            if abs(err_norm_v) > FB_VERT_GATE:
                fb_cmd = int(fb_cmd * FB_VERT_CREEP)
            if fb_cmd < 0:
                fb_cmd = int(fb_cmd * 0.8)
            if time.time() - start_ts < 1.0:
                fb_cmd = int(fb_cmd * 0.8)
            fb = int(FB_ALPHA * fb_cmd + (1 - FB_ALPHA) * prev_fb_cmd)

            # Höhe PD
            prev_ud_err = globals().get("_prev_ud_err", 0.0)
            ud_err  = -err_y
            derr_y  = (ud_err - prev_ud_err) / DT
            globals()["_prev_ud_err"] = ud_err
            ud_cmd = int(clamp(K_ud * ud_err + Kd_ud * derr_y, -MAX_UD, MAX_UD))
            prev_ud_cmd = getattr(__builtins__, "_prev_ud_cmd", 0)
            delta_ud = ud_cmd - prev_ud_cmd
            if   delta_ud >  UD_SLEW_PER_STEP: ud_cmd = prev_ud_cmd + UD_SLEW_PER_STEP
            elif delta_ud < -UD_SLEW_PER_STEP: ud_cmd = prev_ud_cmd - UD_SLEW_PER_STEP
            ud = ud_cmd
            __builtins__._prev_ud_cmd = ud

            # ======= OVERLAY (Detection + Werte) =======
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(frame, (ema_cx, ema_cy), 5, (0,255,0), -1)
            cv2.line(frame, (w//2, h//2), (ema_cx, ema_cy), (0,255,0), 1)
            # Battery + Werte oben links
            cv2.putText(frame, f"Battery: {battery}%", (10, 24), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"bh:{bh_pix}px  target:{TARGET_H}px  fb:{fb}  yaw:{yaw}",
                        (10, 48), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

            prev_err_x   = err_x
            prev_yaw_cmd = yaw
            prev_fb_cmd  = fb

        else:
            # kein Ziel -> weich ausrollen
            ema_cx = ema_cy = None
            prev_err_x = 0
            prev_yaw_cmd = int(prev_yaw_cmd * 0.7)
            prev_fb_cmd  = int(prev_fb_cmd  * 0.7)
            prev_track_box = None
            cv2.putText(frame, "Target LOST - Hovering", (10, 26), font, 0.7, (0,0,255), 2)

        # ======= Aufnahme (mit/ohne Overlay) =======
        frame_to_save = frame if RECORD_ANNOTATED else raw_frame
        if frame_to_save.shape[0] != H or frame_to_save.shape[1] != W:
            print("[REC] Framegröße änderte sich, setze Writer neu.")
            video_out.release()
            H, W = frame_to_save.shape[:2]
            video_out = cv2.VideoWriter(out_path, fourcc, REC_FPS, (W, H))
            if not video_out.isOpened():
                print("[REC] Neuer Writer konnte nicht geöffnet werden – Recording deaktiviert.")
                video_out = None
        if video_out is not None:
            video_out.write(frame_to_save)
            written_frames += 1

        # ======= Anzeige =======
        disp = cv2.resize(frame, (w//2, h//2))
        win_name = "Tello - Person Follow (YOLO + PD)"
        # Steuerhilfe unten
        cv2.putText(disp, "ESC: landen | SPACE: hover | X: panik-stop",
                    (10, disp.shape[0]-10), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imshow(win_name, disp)

        # Eingaben
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            lr = fb = ud = yaw = 0
            prev_yaw_cmd = prev_fb_cmd = 0
        elif key == ord('x'):
            lr = fb = ud = yaw = 0
            prev_yaw_cmd = prev_fb_cmd = 0
        elif key == 27:  # ESC
            print("Lande...")
            safe_send_rc(t, 0,0,0,0, win_name)
            try: t.land()
            except Exception as e: print("[Land-Fehler]", e)
            break

        # ======= Setpoints schicken =======
        ok = safe_send_rc(t, 0 if MAX_LR==0 else lr, fb, ud, yaw, win_name)  # direkt an Tello

        # Akku-Log + Aktualisierung der Overlay-Battery
        if time.time() - last_batt > BATTERY_INTERVAL:
            try:
                battery = t.get_battery()
                print("Akku:", battery, "%", "| bh:", bh_pix, "fb:", fb, "yaw:", yaw)
            except Exception as e:
                print("[WARN] Telemetrie-Timeout:", e)
            last_batt = time.time()

        time.sleep(DT)

finally:
    try: safe_send_rc(t, 0,0,0,0)
    except: pass
    try: t.streamoff()
    except: pass
    try:
        if video_out is not None:
            video_out.release()
            try:
                size_mb = os.path.getsize(out_path) / (1024*1024)
                print(f"[REC] Aufnahme gespeichert: {out_path} ({size_mb:.1f} MB, {written_frames} Frames)")
            except Exception:
                print("[REC] Aufnahme gespeichert.")
    except Exception:
        pass
    try: t.end()
    except: pass
    cv2.destroyAllWindows()

    # --- WhatsApp-kompatibler Export (optional) ---
    def transcode_to_whatsapp(in_path):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            print("[REC] FFmpeg nicht gefunden – Transcode übersprungen.")
            return None
        out_wa = os.path.splitext(in_path)[0] + "_wa.mp4"
        cmd = [
            ffmpeg, "-y",
            "-i", in_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level:v", "3.1",
            "-preset", "veryfast",
            "-r", "30",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            out_wa
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[REC] WhatsApp-Datei erstellt: {out_wa}")
            return out_wa
        except Exception as e:
            print("[REC] Transcode fehlgeschlagen:", e)
            return None

    try:
        transcode_to_whatsapp(out_path)
    except Exception as e:
        print("[REC] Konnte WhatsApp-Export nicht erstellen:", e)
