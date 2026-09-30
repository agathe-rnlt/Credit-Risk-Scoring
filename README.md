# Credit-Risk-Scoring

## 📌 Présentation du Projet
Ce projet met en œuvre la chaîne de construction d'un **modèle de scoring d'octroi de crédit bancaire**, aussi conforme que possible aux exigences Bâloises (Bâle II / Bâle III). 

L'objectif est d'évaluer le risque de défaillance des emprunteurs (`loan_status`), de linéariser les prédicteurs via la transformation **Weight of Evidence (WoE)**, d'estimer une **Régression Logistique** et de calibrer une **grille de score opérationnelle** pour guider la décision d'octroi en banque de détail.

---

## 📂 Structure du Dépôt

```text
Credit-Risk-Scoring/
├── data/
│   ├── raw/
│   │   └── loan_default_prediction.csv   # Base brute (1 000 emprunteurs, 13 variables)
│   └── sets/
│       ├── train_sample.csv               # Cohorte d'apprentissage (70%, 700 obs)
│       ├── test_sample.csv                # Cohorte de validation (30%, 300 obs)
│   ├── woe/
│       ├── train_woe.csv                  # Cohorte Train transformée en WoE
│       └── test_woe.csv                   # Cohorte Test transformée en WoE
│   ├── score/
│       ├── test_scored.csv
├── notebooks/
│   ├── 01_eda.ipynb                       # 01. Analyse Exploratoire & Échantillonnage
│   ├── 02_binning_et_woe.ipynb            # 02. Discrétisation (Binning) & Transformation WoE
│   └── 03_modelisation_et_scorecard.ipynb # 03. Régression Logistique & Grille de Score
├── rrequirements.txt
└── README.md                              # Document de présentation


# Jeu de Données (`loan_default_prediction.csv`)

Le portefeuille se compose de 1 000 emprunteurs et 13 variables :

* **Variable dépendante (target) :** `loan_status` (0 = Emprunteur Sain / 42,3 %, 1 = En Défaut / 57,7 %)
* **Variables financières et démographiques :** `credit_score`, `debt_to_income`, `income`, `age`, `employment_years`, `loan_amount`, `loan_term_months`, `interest_rate`
* **Variables qualitatives et historiques :** `home_ownership`, `loan_purpose`, `previous_defaults`

---

## Pipeline de Modélisation

### 1. Notebook 01 — Analyse Exploratoire des Données (EDA)
* **Audit de qualité :** Validation d'un taux de complétude de 100 % (0 valeur manquante) et d'unicité (0 doublon sur `loan_id`).
* **Profil de risque :** Mise en évidence des 3 facteurs de risque majeurs :
  * `previous_defaults` : Taux de défaut passant de 48,15 % (0 incident) à 74,24 % (2 incidents).
  * `credit_score` : Score moyen inférieur de 10,65 % chez les défaillants (548,9 vs 614,4 points).
  * `debt_to_income` : Ratio d'endettement moyen supérieur de 20,24 % chez les défaillants (0,47 vs 0,39).
* **Découpage étanche (Train / Test Split) :** Séparation stratifiée 70 % / 30 % (700 Train / 300 Test). Le test de Kolmogorov-Smirnov sur le `credit_score` ($p\text{-value} = 0,2054 > 0,05$) confirme l'homogénéité parfaite des cohortes.

### 2. Notebook 02 — Discrétisation (Binning) & Transformation WoE
* **Validation de l'homogénéité :** Vérification systématique par tests statistiques (KS pour le numérique, Chi² pour le catégoriel).
* **Optimal Binning :** Découpage sous contraintes prudentielles (taille minimale par classe $\ge 5\%$, maximum 5 tranches).
* **Standardisation WoE :** Conversion des variables brutes en valeurs *Weight of Evidence* (WoE) pour linéariser la relation avec le log-odds de défaut.
* **Prévention du Data Leakage :** Ajustement (*fit*) des grilles de découpage strictement sur le Train, puis projection (*transform*) sur le Test.
* **Hiérarchisation par l'Information Value (IV) :** Identification des prédicteurs majeurs (`previous_defaults` $IV \approx 0,20$, `credit_score` $IV \approx 0,17$, `debt_to_income` $IV \approx 0,15$).

### 3. Notebook 03 — Modélisation, Scorecard & Diagnostic
* **Contrôle de colinéarité :** Validation d'un *Variance Inflation Factor* ($VIF < 5$) sur toutes les variables WoE.
* **Régression Logistique :** Estimation multivariée de la probabilité de défaut $P(Y=1)$.
* **Évaluation des performances Bâloises :**
  * **Train :** $\text{AUC} = 0,7775$, $\text{Gini} = 55,49\%$.
  * **Test :** $\text{AUC} = 0,6216$, $\text{Gini} = 24,32\%$, $\text{Statistique KS} = 27,02\%$.
* **Calibration de la Grille de Score (Scaling) :** Conversion des probabilités en score linéaire (300 à 850 points) avec :
  
  $$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$
  
  *(Paramètres de réglage : Target Score = 600 points pour des Odds de 50:1, PDO = 20 points).*

---


## 🛠️ Installation & Utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/agathe-rnlt/Credit-Risk-Scoring.git
cd Credit-Risk-Scoring
```

### 2. Configurer l'environnement virtuel (Python 3.11 / 3.12)
```bash
python -m venv .venv
source .venv/bin/activate        # On Mac/Linux
.venv\Scripts\activate           # On Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Exécuter les notebooks
Lancer Jupyter Lab / VS Code et exécuter séquentiellement :
* `notebooks/01_eda.ipynb`
* `notebooks/02_binning_et_woe.ipynb`
* `notebooks/03_modelisation_et_scorecard.ipynb`

---

## 📜 Licence

Ce projet est mis à disposition sous licence **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Vous êtes libre de partager et d'adapter le contenu sous réserve de créditer l'auteure originale (`agathe-rnlt`).

