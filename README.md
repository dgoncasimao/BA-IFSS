# BA-IFSS: Bachelor Thesis Baseline Experiments (IFSS-Net, VGG, TransUNet)

## ⚠️ Wichtiger Hinweis: Finaler SegFormer-Workflow
Dieses Repository (\texttt{BA-IFSS}) dient der Dokumentation von **Baseline-Experimenten** mit älteren Architekturen (\texttt{IFSS-Net}, \texttt{VGG}, \texttt{TransUNet}) für die Segmentierung des M. vastus lateralis.

**Der finale, in der Thesis als überlegen ausgewiesene Workflow, der auf der SegFormer-Architektur basiert, befindet sich im separaten Repository:**
➡️ **[Goncalves2025SegformerRepo]** (Link zu deinem SegFormer-Repo)

## Overview
BA-IFSS (Bachelor Thesis on Interactive Few-Shot Siamese Network) ist ein Framework, das die Implementierung und den Vergleich von drei Segmentierungs-Architekturen (\texttt{IFSS-Net} \parencite{Chanti2021IFSSNet}, \texttt{VGG} \parencite{Simonyan2015VGGarXiv} und \texttt{TransUNet} \parencite{Chen2021TransUNet}) ermöglicht.

Das Ziel war die Segmentierung von Volumina des **M. vastus lateralis** \parencite{Ritsche2025_3DUS_MuscleVolume}. Dieses Repository enthält die Skripte und Konfigurationen für:
* **Daten-Preprocessing:** Konvertierung von NRRD-Volumina in trainierbare 2D TIFF-Slices.
* **Modell-Training:** Durchführung der Trainingsläufe, deren Ergebnisse in den W&B-Daten (Loss- und IoU-Kurven) der Thesis dokumentiert sind.
* **Volumen-Rekonstruktion:** Zusammenfügen der 2D-Masken zu 3D-Segmentierungen für die Visualisierung in 3D Slicer \parencite{Fedorov2012}.

## 🚀 Getting Started

### Prerequisites
Stellen Sie sicher, dass folgende Abhängigkeiten installiert sind:
- Python 3.8+
- pip oder conda package manager

### Setup
1.  Klonen Sie das Repository:
    ```bash
    git clone [https://github.com/dgoncasimao/BA-IFSS.git](https://github.com/dgoncasimao/BA-IFSS.git)
    cd BA-IFSS
    ```

2.  Installieren Sie die Abhängigkeiten:
    ```bash
    pip install -r requirements.txt
    ```
    Alternativ mit Conda:
    ```bash
    conda env create -f environment.yml
    conda activate ba-ifss
    ```

## ⚙️ Usage (Data Processing and Training)

1.  **Datenstruktur:** Platzieren Sie Ihre 3D NRRD-Dateien (\texttt{*.nrrd}) in einem spezifischen Eingabeverzeichnis.
2.  **Konfiguration:** Passen Sie die Hyperparameter (HP) in der Hauptkonfigurationsdatei \texttt{config.py} an (z.B. Pfade, Batch-Größe).
3.  **Preprocessing:** Führen Sie das Preprocessing-Skript aus, um die 2D-Slices zu generieren.
    ```bash
    python preprocessing/convert_nrrd_to_slices.py
    ```
4.  **Training:** Starten Sie den gewünschten Modell-Trainingslauf (z.B. IFSS-Net).
    ```bash
    python train_ifss.py
    ```

## Kontakt
Für Fragen oder Feedback kontaktieren Sie bitte:
- **Autor:** Diego Gonçalves Simão
- **GitHub:** [dgoncasimao](https://github.com/dgoncasimao)
