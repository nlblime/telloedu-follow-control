from djitellopy import Tello
import cv2
import time
import keyboard  # pip install keyboard
import os, datetime, shutil, subprocess

# ==== Steuer-Settings ====
SPEED_FB  = 60   # vor/zurück
SPEED_LR  = 60   # links/rechts
SPEED_UD  = 60   # hoch/runter
SPEED_YAW = 60   # drehen

BATTERY_INTERVAL = 5.0   # alle X s Akku abfragen
LOOP_HZ = 30             # Steuer-Loop-Frequenz
DT = 1.0 / LOOP_HZ

# ==== Video-Settings (Qualität/Flüssigkeit) ====
# Robust & scharf: 15 fps + 2 Mbps. Für mehr „smooth“ kannst du FPS_30 wählen.
USE_FPS_30 = False             # False=15 fps (robuster), True=30 fps (glatter, mehr Last)
INITIAL_BITRATE_MBPS = 2       # 1..4 Mbit/s (2 ist guter Sweet-Spot)

# Adaptive Bitrate passt die Bitrate automatisch an die echte Decoder-FPS an.
ADAPTIVE_BITRATE = True
ADAPT_CHECK_EVERY_S = 6.0
BITRATE_MIN = 1
BITRATE_MAX = 4
FPS_DROP_THRESHOLD = 0.65      # wenn echte FPS < threshold * Soll-FPS -> Bitrate -1
FPS_RAISE_THRESHOLD = 0.95     # wenn echte FPS > threshold * Soll-FPS -> Bitrate +1
# Hinweis: Anpassungen wirken nur, wenn Tello-Firmware die Bitraten-API akzeptiert.

# ==== Aufnahme-Settings ====
AUTO_RECORD       = True                # automatisch aufnehmen
RECORD_ANNOTATED  = False               # True: mit Overlay (Akkutext etc.), False: Rohbild
HERE              = os.path.dirname(os.path.abspath(__file__))
REC_DIR           = os.path.join(HERE, "recordings")
os.makedirs(REC_DIR, exist_ok=True)

# REC_FPS immer an das Tello-Streaming anlehnen:
REC_FPS = 30 if USE_FPS_30 else 15

# globale Recorder-Variablen
video_out = None
out_path  = None
fourcc    = None
W = H = None
written_frames = 0

# FPS/Bitrate-Monitor
_decoded_frames = 0
_fps_window_t0 = None
_measured_fps = 0.0
_current_bitrate = INITIAL_BITRATE_MBPS
_last_adapt_t = 0.0

def _open_recorder_for_frame(frame):
    """Ersten passenden VideoWriter für die Framegröße öffnen (MP4, Fallback AVI)."""
    global video_out, out_path, fourcc, W, H
    H, W = frame.shape[:2]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Primär: MP4/mp4v (sehr robust)
    out_path = os.path.join(REC_DIR, f"tello_rc_{ts}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(out_path, fourcc, REC_FPS, (W, H))
    if not video_out.isOpened():
        # Fallback: AVI/XVID
        print("[REC] mp4v nicht verfügbar -> Fallback auf AVI/XVID")
        out_path = os.path.join(REC_DIR, f"tello_rc_{ts}.avi")
        fourcc   = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter(out_path, fourcc, REC_FPS, (W, H))
    if not video_out.isOpened():
        print("[REC] Konnte VideoWriter weder für MP4 noch AVI öffnen -> Aufnahme deaktiviert.")
        video_out = None
    else:
        print(f"[REC] Aufnahme gestartet: {out_path} (fix {REC_FPS} FPS)")

def _maybe_reopen_on_resize(frame):
    """Falls Tello spontan die Auflösung wechselt: Writer neu aufsetzen."""
    global video_out, fourcc, out_path, W, H
    h, w = frame.shape[:2]
    if (W is None) or (H is None):
        _open_recorder_for_frame(frame)
        return
    if (w != W) or (h != H):
        print("[REC] Framegröße änderte sich, setze Writer neu.")
        try:
            video_out.release()
        except Exception:
            pass
        _open_recorder_for_frame(frame)

def _transcode_to_whatsapp(in_path):
    """Optional: WhatsApp-kompatibles MP4 erzeugen (falls ffmpeg installiert)."""
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

# ==== Tello Setup ====
t = Tello()
t.connect()

# --- WICHTIG: erst Video-Settings, dann streamon() ---
try:
    # Auflösung hoch (manche Tello/EDU haben nur 720p; Aufruf ist dennoch safe)
    try:
        t.set_video_resolution(Tello.RESOLUTION_720P)
    except Exception:
        # einige Firmware-Versionen bieten keine Umschaltung – einfach ignorieren
        pass

    # FPS einstellen
    if USE_FPS_30:
        t.set_video_fps(Tello.FPS_30)
        print("[Video] FPS: 30")
    else:
        t.set_video_fps(Tello.FPS_15)
        print("[Video] FPS: 15")

    # Start-Bitrate
    if INITIAL_BITRATE_MBPS == 1:
        t.set_video_bitrate(Tello.BITRATE_1MBPS)
    elif INITIAL_BITRATE_MBPS == 2:
        t.set_video_bitrate(Tello.BITRATE_2MBPS)
    elif INITIAL_BITRATE_MBPS == 3:
        t.set_video_bitrate(Tello.BITRATE_3MBPS)
    elif INITIAL_BITRATE_MBPS >= 4:
        t.set_video_bitrate(Tello.BITRATE_4MBPS)
    print(f"[Video] Start-Bitrate: {INITIAL_BITRATE_MBPS} Mbps")
except Exception as e:
    print("[Hinweis] Konnte Video-Settings nicht setzen:", e)

battery = t.get_battery()
print(f"Verbunden! Akku: {battery}%")
if battery is not None and battery < 30:
    print("[Hinweis] Akku <30%: Auto-Land möglich. Bitte laden.")

t.streamon()

# Starten und etwas Höhe gewinnen, damit Optic Flow sicher arbeitet
t.takeoff()
time.sleep(0.3)
try:
    t.move_up(50)
except Exception as e:
    print("[Warnung] Konnte nicht steigen:", e)

# RC-Modus aktivieren (mind. einmal senden)
t.send_rc_control(0, 0, 0, 0)

frame_reader = t.get_frame_read()
font = cv2.FONT_HERSHEY_SIMPLEX
last_batt = time.time()

print("Realtime-Steuerung (Taste halten): W/S vor/zurück | A/D links/rechts | R/F hoch/runter | Q/E drehen | X: Stopp | ESC: landen")

# ==== Aufnahme initialisieren (wartet auf 1. Frame) ====
if AUTO_RECORD:
    print("Warte auf erstes Kamerabild für Recording-Setup...")
    start_wait_ts = time.time()
    first = None
    while first is None:
        first = frame_reader.frame
        if time.time() - start_wait_ts > 5.0:
            print("[WARN] Kein Frame nach 5s – versuche weiter...")
            start_wait_ts = time.time()
        cv2.waitKey(1)
    _open_recorder_for_frame(first)

# fixer Zeitgeber, um nahe REC_FPS zu schreiben
last_write_ts = time.time()

# FPS-Messfenster starten
_fps_window_t0 = time.time()

def _apply_bitrate(mbps:int):
    global _current_bitrate
    mbps = max(BITRATE_MIN, min(BITRATE_MAX, int(mbps)))
    if mbps == _current_bitrate:
        return
    try:
        if mbps == 1: t.set_video_bitrate(Tello.BITRATE_1MBPS)
        elif mbps == 2: t.set_video_bitrate(Tello.BITRATE_2MBPS)
        elif mbps == 3: t.set_video_bitrate(Tello.BITRATE_3MBPS)
        else: t.set_video_bitrate(Tello.BITRATE_4MBPS)
        _current_bitrate = mbps
        print(f"[Video] Bitrate angepasst -> {mbps} Mbps")
    except Exception as e:
        print("[Video] Bitrate-Set fehlgeschlagen:", e)

try:
    while True:
        # === Eingaben abfragen (Taste halten) ===
        lr = fb = ud = yw = 0
        if keyboard.is_pressed('w'): fb =  SPEED_FB
        if keyboard.is_pressed('s'): fb = -SPEED_FB
        if keyboard.is_pressed('a'): lr = -SPEED_LR
        if keyboard.is_pressed('d'): lr =  SPEED_LR
        if keyboard.is_pressed('r'): ud =  SPEED_UD
        if keyboard.is_pressed('f'): ud = -SPEED_UD
        if keyboard.is_pressed('q'): yw = -SPEED_YAW
        if keyboard.is_pressed('e'): yw =  SPEED_YAW

        # Not-Aus der Bewegung (bleibt in der Luft)
        if keyboard.is_pressed('x'):
            lr = fb = ud = yw = 0

        # ESC -> landen & raus
        if keyboard.is_pressed('esc'):
            print("Lande...")
            t.send_rc_control(0,0,0,0)
            t.land()
            break

        # === RC kontinuierlich senden ===
        t.send_rc_control(lr, fb, ud, yw)

        # === Akku periodisch abfragen ===
        if time.time() - last_batt > BATTERY_INTERVAL:
            try:
                battery = t.get_battery()
                print(f"Akkustand: {battery}%")
            except Exception as e:
                print("Fehler beim Akku-Abfragen:", e)
            last_batt = time.time()

        # === Video anzeigen & optional aufnehmen ===
        frame = frame_reader.frame
        if frame is not None:
            # FPS-Monitor aktualisieren
            _decoded_frames += 1
            now = time.time()
            win_dt = now - _fps_window_t0
            if win_dt >= 2.0:  # alle 2s aktualisieren
                _measured_fps = _decoded_frames / win_dt
                _decoded_frames = 0
                _fps_window_t0 = now

            # Adaptive Bitrate (einfacher Regler)
            if ADAPTIVE_BITRATE and (now - _last_adapt_t) >= ADAPT_CHECK_EVERY_S:
                target_fps = 30 if USE_FPS_30 else 15
                if _measured_fps < FPS_DROP_THRESHOLD * target_fps and _current_bitrate > BITRATE_MIN:
                    _apply_bitrate(_current_bitrate - 1)
                elif _measured_fps > FPS_RAISE_THRESHOLD * target_fps and _current_bitrate < BITRATE_MAX:
                    _apply_bitrate(_current_bitrate + 1)
                _last_adapt_t = now

            # Aufnahme-Frame vorbereiten (roh oder annotiert)
            frame_to_save = frame.copy()
            if RECORD_ANNOTATED:
                cv2.putText(frame_to_save, f"Battery: {battery}%", (10, 28), font, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # Recorder ggf. neu öffnen bei Größenwechsel
            if AUTO_RECORD and video_out is not None:
                _maybe_reopen_on_resize(frame_to_save)
                # mit fixer Frequenz schreiben (nahe REC_FPS)
                if (now - last_write_ts) >= (1.0 / REC_FPS):
                    video_out.write(frame_to_save)
                    written_frames += 1
                    last_write_ts = now

            # Anzeige halb skalieren -> weniger CPU, flüssiger
            display = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            # Overlay: Battery, Bitrate, gemessene FPS
            info1 = f"Battery: {battery}%   Bitrate: {_current_bitrate} Mbps   FPS: {_measured_fps:.1f}"
            cv2.putText(display, info1, (10, 28), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(display, "W/S vor-zuruck  A/D links-rechts  R/F hoch-runter  Q/E drehen  X stop  ESC landen",
                        (10, display.shape[0]-12), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Tello (Realtime RC + Recording)", display)

            # ESC über Fenster (falls Fokus dort)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Lande...")
                t.send_rc_control(0,0,0,0)
                t.land()
                break

        time.sleep(DT)  # feste Loop-Zeit (glatte Steuerung)

finally:
    try:
        t.send_rc_control(0,0,0,0)
    except:
        pass
    try:
        t.streamoff()
    except:
        pass
    try:
        t.end()
    except:
        pass

    # Recorder sauber schließen
    try:
        if video_out is not None:
            video_out.release()
            try:
                size_mb = os.path.getsize(out_path) / (1024*1024)
                print(f"[REC] Aufnahme gespeichert: {out_path} ({size_mb:.1f} MB, {written_frames} Frames)")
            except Exception:
                print(f"[REC] Aufnahme gespeichert: {out_path}")
            # Optionaler WhatsApp-Export (kommentier aus, falls nicht gewünscht)
            try:
                _transcode_to_whatsapp(out_path)
            except Exception as e:
                print("[REC] Konnte WhatsApp-Export nicht erstellen:", e)
    except Exception:
        pass

    cv2.destroyAllWindows()
