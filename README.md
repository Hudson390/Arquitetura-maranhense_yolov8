<h1 align="center">🏛️ Arquitetura Maranhense com YOLOv8</h1>

<p align="center">
  Projeto de detecção de elementos da arquitetura maranhense usando o modelo <strong>YOLOv8</strong>.
</p>

<hr>

<h2>📌 Sobre o Projeto</h2>
<p>
  Este projeto utiliza a rede neural <strong>YOLOv8</strong> para identificar elementos arquitetônicos típicos do estado do Maranhão.
  Com ele, é possível treinar, detectar e visualizar os resultados em tempo real a partir de diferentes fontes de entrada, como webcam, celular ou captura de tela.
</p>

<h2>📁 Estrutura do Projeto</h2>
<ul>
  <li><strong>app.py</strong> – Interface web da aplicação.</li>
  <li><strong>train_v8.py</strong> – Treinamento do modelo YOLOv8.</li>
  <li><strong>converter_yolo.py</strong> – Conversão de datasets para o formato YOLO.</li>
  <li><strong>detectar_capturando_celular.py</strong> – Detecção em vídeo do celular.</li>
  <li><strong>detectar_capturando_tela.py</strong> – Detecção via captura de tela.</li>
  <li><strong>detectar_usando_webcam.py</strong> – Detecção com webcam.</li>
  <li><strong>requirements.txt</strong> – Dependências do projeto.</li>
  <li><strong>datasets/</strong> – Conjunto de dados utilizado.</li>
  <li><strong>templates/</strong> e <strong>static/</strong> – Interface web.</li>
</ul>

<h2>🚀 Como Executar</h2>

<ol>
  <li>Clone o repositório:<br>
    <code>git clone https://github.com/Hudson390/Arquitetura-maranhense_yolov8.git</code>
  </li>
  <li>Instale as dependências:<br>
    <code>pip install -r requirements.txt</code>
  </li>
  <li>Execute a interface ou os scripts desejados:<br>
    <code>python app.py</code><br>
    ou<br>
    <code>python detectar_usando_webcam.py</code>
  </li>
</ol>

<h2>📷 Exemplo de Detecção</h2>
<p>Inclua aqui imagens ou GIFs mostrando os resultados.</p>

<h2>🔧 Tecnologias Utilizadas</h2>
<ul>
  <li>Python</li>
  <li>Ultralytics YOLOv8</li>
  <li>OpenCV</li>
  <li>Flask (interface web)</li>
</ul>

<h2>📚 Referências</h2>
<ul>
  <li><a href="https://github.com/ultralytics/ultralytics">YOLOv8 - Ultralytics</a></li>
</ul>

<h2>📄 Licença</h2>
<p>Este projeto está sob a licença MIT.</p>
