# classifier_api
API für Zugriff auf Fashion-MNIST-Classifier 

## Installation
1. Conda-Umgebung erzeugen und aktivieren (per Anaconda Prompt)
   ```
   conda create -n dev_api python==3.10.9
   conda activate dev_api
   ```
2. Repository Clonen (per Anaconda Prompt)
   ```
   git clone https://github.com/MareenDev/classifier_api.git
   ```
3. Repository in Editor/VSCode öffnen
   ```
   cd classifier_api
   code
   ```

4. Bibliotheken aus requirements.txt in Conda-Umgebung laden (VSCode CMD oder Anaconda Promt)
   ```
   pip install -r requirements.txt
   ```

## Starten des Webservices (VSCode/Entwicklungsumgebung)
   Run Skript .\wsgi.py

## Aufruf des Webservices
   Aufruf der URL: http://127.0.0.1:5000/
   #TBD

## Beenden des Webservices
   ```
   CTRL+C in Python-Prompt
   ```