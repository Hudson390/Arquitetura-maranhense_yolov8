from flask import Flask, Response, request, render_template, jsonify
import cv2
import numpy as np
import base64
import ssl
from ultralytics import YOLO

app = Flask(__name__)

# Carrega o modelo YOLO treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Dicionário com descrições dos objetos detectados
descriptions = {
    "poste colonial": "Objeto típico das ruas de São Luís, o poste colonial é um elemento de iluminação que preserva o estilo europeu, com estrutura em ferro fundido e detalhes ornamentais, representando a influência arquitetônica portuguesa.",
    "grade ornamental colonial": "Presente nas sacadas e janelas dos casarões coloniais, a grade ornamental colonial é feita geralmente em ferro, com desenhos curvilíneos e arabescos, conferindo elegância e proteção às fachadas.",
    "rua de pedra colonial": "As ruas de pedra, também chamadas de calçamento pé de moleque, são características do centro histórico de São Luís. Construídas com pedras irregulares, essas vias preservam a atmosfera colonial da cidade.",
    "porta arquitetura colonial": "As portas dos casarões coloniais maranhenses possuem dimensões amplas, com madeira maciça e detalhes entalhados, muitas vezes acompanhadas por bandeiras de vidro, refletindo a estética portuguesa.",
    "igreja da Se": "Principal templo católico de São Luís, a Igreja da Sé combina elementos barrocos e neoclássicos. Construída no século XVII, destaca-se pelo altar ricamente decorado e pela importância histórica na cidade.",
    "janela colonial": "Com estrutura em madeira e vidro, as janelas coloniais dos casarões maranhenses possuem venezianas para ventilação e bandeiras de vidro, garantindo luminosidade e estilo à arquitetura.",
    "escultura leao heraldica": "Localizada no Palácio dos Leões, a escultura do Leão Heráldico simboliza força e poder, representando a herança portuguesa na arquitetura maranhense."
}

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

@app.route("/detalhes")
def detalhes():
    """
    Exibe a página com descrições detalhadas dos objetos detectados.
    """
    objetos = request.args.getlist("objetos")
    textos = {obj: descriptions.get(obj, "Descrição não encontrada") for obj in objetos}
    return render_template("detalhes.html", textos=textos)

@app.route("/mapa")
def mapa():
    return render_template("mapa.html")

if __name__ == "__main__":
    # Configuração HTTPS
    cert_file = "cert.pem"
    key_file = "key.pem"

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # Inicia o servidor Flask com HTTPS
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context=context)
