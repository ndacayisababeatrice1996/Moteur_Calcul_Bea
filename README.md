# Moteur de Calcul des Tables de Mortalité et d’Espérance de Vie  
**Application Web pour l’Assurance-Vie**  

*NDACAYISABA Béatrice – Matricule 17/05026 *  
*Université du Burundi – Institut de Statistique Appliquée*  


---

## Aperçu du projet

Ce dépôt contient le **code source complet** d’un **moteur de calcul automatisé des tables de mortalité et de l’espérance de vie**.  
L’application permet de générer des tables actuarielles à partir de **cinq modèles reconnus** (Lee-Carter, Gompertz, Makeham, Heligman-Lorenz, Coale-Demeny), d’afficher les résultats sous forme de tableaux et graphiques interactifs, d’exporter en Excel, et de gérer un historique utilisateur avec pagination, filtres et tri.

> **Objectif pédagogique** : Concevoir, développer, tester et documenter un logiciel actuariel robuste, reproductible et évolutif.

---

## Fonctionnalités principales

| Fonctionnalité | Description |
|----------------|-----------|
| **Calcul actuariel** | Génération de \( l_x \), \( d_x \), \( q_x \), \( p_x \), \( e_x \) pour tout âge |
| **5 modèles intégrés** | Lee-Carter, Gompertz, Makeham, Heligman-Lorenz, Coale-Demeny |
| **Visualisation interactive** | Graphiques dynamiques avec Chart.js (courbes \( q_x \), \( p_x \)) |
| **Export Excel** | Téléchargement des tables via `openpyxl` |
| **Historique utilisateur** | Stockage SQLite avec pagination (5 entrées/page), tri (date, modèle, âge), filtres |
| **Interface responsive** | Design moderne avec Tailwind CSS |
| **Architecture MVC** | Flask (backend), HTML/JS (frontend) |

---


---

## Technologies utilisées

| Couche | Technologie |
|-------|-------------|
| **Backend** | Python 3.12, Flask 2.3.2 |
| **Calculs** | NumPy 1.26.0 |
| **Base de données** | SQLite (léger, embarqué) |
| **Export** | openpyxl 3.1.2 |
| **Frontend** | HTML5, Tailwind CSS 3.4, Chart.js 4.4.0 |
| **Déploiement local** | `python app.py` |

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/ndacayisababeactrice1996/Moteur_Calcul_Bea.git
cd Moteur_Calcul_Bea

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l’application
python app.py
