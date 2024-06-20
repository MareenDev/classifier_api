# classifier_api
##Beschreibung
Bereitstellung verschiedener Modelle (neuronale Netze) per Webservice.
Jedes Modell wird unter einer eigenen URL bereitgestellt und kann mittels POST-Request im Datenformat JSON angefragt werden.
Einem Modell sind im Post-Request Bilddaten, Bildgröße und Anzahl der Farbkanäle mit zu geben.
Aufrufpfade mit dem postfix /decision geben den Namen der identifizierten Klasse zurück. Aufrufe mit postfix /score geben die Ausgabe (Scorewerte) des Modells zurück. 
### Requestbeispiel:
![image](https://github.com/MareenDev/classifier_api/assets/115465960/cb763097-52b2-41a9-a300-fd3a20643180)

### Response zu Request:
![image](https://github.com/MareenDev/classifier_api/assets/115465960/6089f110-a694-40f3-9273-d2487964f9a7)

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

## Aufbau
