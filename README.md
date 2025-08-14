#🛩️ Tello EDU – Control & Follow

PC-gesteuertes Python-Skript für die Ryze Tello EDU mit Echtzeit-Videoanzeige, präziser manueller Steuerung und automatischer Personenverfolgung über ein YOLOv5s ONNX-Modell.  
Enthält ein Kamera-Overlay mit Akkustand, Steuerungshinweisen und optionaler Videoaufnahme.

---

##✨ Funktionen
- **Manuelle Steuerung in Echtzeit** über Tastatur  
- **Automatische Personenverfolgung** mit YOLOv5s (ONNX)  
- **Echtzeit-Videoanzeige** mit Overlay (Akku, Steuerung, Status)  
- **Videoaufnahme** als MP4 oder AVI  
- **Optionaler Export in WhatsApp-kompatibles MP4** (mit ffmpeg)  
- **Flüssige Steuerung** dank fester Loop-Rate  

---

##🛠️ Voraussetzungen

### Hardware
- **Ryze Tello EDU** Drohne  
- PC oder Laptop mit WLAN  
- Stabile Verbindung zum Tello-WLAN während des Flugs

### Software
- **Python 3.9+** (empfohlen)  
- Python-Abhängigkeiten installieren:
  ```bash
  pip install -r requirements.txt

---

##📦 Zusätzliche Ressourcen

- **YOLOv5s ONNX-Modell** *(erforderlich für Personenverfolgung)*  
  Selbst herunterladen und in denselben Ordner wie "drone_follow.py" legen.  
  Download: [YOLOv5s ONNX von Ultralytics](https://github.com/ultralytics/yolov5/releases)

- **ffmpeg** *(optional, nur für WhatsApp-kompatiblen Videoexport nötig)*  
  Download & Installation: [ffmpeg.org](https://ffmpeg.org/download.html)

