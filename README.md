# classifier_api
## Beschreibung

Bereitstellung verschiedener Modelle (neuronale Netze) per Webservice für Thesis.
Es werden 'vanilla-Modelle' und korrespondierende Modelle mit Verteidigungsmaßnahmen verwendet. 
Übersicht der in der die Thesis vorgesehene Modelle:
| ID        | Architektur           | Verteidigung  |
| ------------- |:-------------:| -----:|
| 0| CNN | - |
| 0_1| CNN|adversariales PGD-Retraining Proaktiv (Robustes Training)|
| 0_2| CNN|Data Augmentation Proaktiv (Robustes Training)|
| 0_3| CNN|Random Resize& Padding (Preprozessor)|
| 0_4| CNN|Gaussian Noise (Preprozessor)|
| 2| ResNet18 | - |
| 2_1| ResNet18|adversariales PGD-Retraining Proaktiv (Robustes Training)|
| 2_2| ResNet18|Data Augmentation Proaktiv (Robustes Training)|
| 2_3| ResNet18|Random Resize& Padding (Preprozessor)|
| 2_4| ResNet18|Gaussian Noise (Preprozessor)|
| 5| ViT | - |
| 5_1| ViT|adversariales PGD-Retraining Proaktiv (Robustes Training)|
| 5_2| ViT|Data Augmentation Proaktiv (Robustes Training)|
| 5_3| ViT|Random Resize& Padding (Preprozessor)|
| 5_4| ViT|Gaussian Noise (Preprozessor)|

Jedes Modell wird unter einer eigenen URL bereitgestellt und kann mittels POST-Request im Datenformat JSON angefragt werden.
Einem Modell sind im Post-Request Bilddaten, Bildgröße und Anzahl der Farbkanäle mit zu geben.
Aufrufpfade mit dem postfix **/decision** geben den Namen der identifizierten Klasse zurück. Aufrufe mit postfix **/score** geben die Ausgabe (Scorewerte) des Modells zurück. 

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
   Run Skript **scr/wsgi.py**
   
   **Hinweis** Da die vorliegende Modellbereitstellung ausschließlich zum Zwecke einer Thesis verwendet wird, wird von Absicherungen des Servers abgesehen. 
   Der Server wird als Entwicklungsserver verwendet. Diese Art der Nutzung ist fur eine Produktivnutzung, bzw. für andere Sicherheitsuntersuchungen nicht zu empfehlen, kann jedoch für den vorgesehenen Experimentaufbau genutzt werden.

## Beenden des Webservices
   ```
   CTRL+C in Python-Prompt
   ```

## Aufbau
Ordner **model** umfasst trainierte Modelle in Form von Pickle-Dateien.
Ordner **src** führt Pythonskripte zur Modellbereitstellung.
* Datei **src/api.py** definiert die verfügbaren Pfade/ Webservices . (Siehe Annotation @app.route) 
* Dateien mit dem präfix **src/defense** dienen der Bereitstellung von Modellen mit Verteidigungsmaßnahmen.
   - Datei **src/defenseByAdversarialTraining.py** führt ein adversariales PGD-Retraining zu einem bestehenden Modell durch.
   - Datei **src/defenseByAugmentation.py** führt ein Retraining mit Gauss-Blurr-verzerten Trainingsdaten zu einem bestehenden Modell durch.
   - Datei **src/defenseByInputTransformation.py** stellt Klassen zur Vorverarbeitung eines Dateneingangs bereit.
* Datei **src/helpers.py** bündelt Hilfsfunktionen
* Datei **src/model1.py** umfasst Klassendefinitionen zur Modellarchitektur.
* Datei **src/testTorch.py** dient dem Testing von Modellen, welche in Ordner **model** gespeichert sind. Verwendete Performancefunktion: Akkuranz. 
* Datei **src/trainTorch.py** dient dem Training von neuen Modellen. Nach abgeschlossenem Training werden die Modelle in Ordner **model** abgespeichert.
* Datei **src/wsgi.py** wird für das Starten des Webservers benötigt.


**Hinweis** Aufgrund von Größenbeschränkgung von Dateien können nicht alle Modelle, die in der Thesis verwendet werden in Ordner model bereitgestellt werden.
Diese können jedoch mit dem übrigen verfügbaren Code lokal trainiert werden.
