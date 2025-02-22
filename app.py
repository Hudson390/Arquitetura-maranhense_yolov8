from flask import Flask, Response, request, render_template, jsonify
import cv2
import numpy as np
import base64
import ssl
from ultralytics import YOLO

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo YOLO treinado
model = YOLO("runs/detect/train/weights/best.pt")

def process_frame(frame):
    """
    Processa o frame com YOLO e retorna a imagem anotada e os rótulos detectados.
    """
    results = model(frame)
    labels = []
    
    for result in results:
        frame = result.plot()  # Plota as detecções na imagem
        labels.extend(result.names[i] for i in result.boxes.cls.int().tolist())  # Extrai os rótulos

    return frame, labels

@app.route("/")
def index():
    """Renderiza a página principal."""
    return render_template("index.html")

@app.route("/video_feed", methods=["POST"])
def video_feed():
    """
    Recebe um frame, processa com YOLO e retorna a imagem processada + rótulos detectados.
    """
    file = request.files["frame"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_frame, labels = process_frame(frame)

    _, buffer = cv2.imencode(".jpg", processed_frame)
    image_base64 = base64.b64encode(buffer).decode()

    return jsonify({"image_url": f"data:image/jpeg;base64,{image_base64}", "labels": labels})

if __name__ == "__main__":
    # Configuração HTTPS
    cert_file = "cert.pem"
    key_file = "key.pem"

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # Inicia o servidor Flask com HTTPS
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=context)
