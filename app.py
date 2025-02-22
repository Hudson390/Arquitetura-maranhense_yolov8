from flask import Flask, Response, request, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import ssl

app = Flask(__name__)
model = YOLO("runs/detect/train/weights/best.pt")

def process_frame(frame):
    results = model(frame)
    for result in results:
        frame = result.plot()
    return frame

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed", methods=["POST"])
def video_feed():
    file = request.files["frame"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    processed_frame = process_frame(frame)
    _, buffer = cv2.imencode(".jpg", processed_frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")

if __name__ == "__main__":
    # Criar certificados autoassinados via Python
    cert_file = "cert.pem"
    key_file = "key.pem"
    
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.check_hostname = False
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)
    
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=context)
