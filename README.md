# Obsolescence des détecteurs de deepfakes GAN-trained face aux modèles de diffusion

> **Mémoire de fin d'études** — *Obsolescence des détecteurs de deepfakes GAN-trained face aux modèles de diffusion : gradient systématique de détectabilité selon la distance architecturale et récupération pragmatique de robustesse par stratégies d'ensemble à poids gelés*
> Auteur : Maxime Ducarme · Année : 2025–2026 · Université Catholique de Louvain

---

## Table des matières

1. [Contexte et question de recherche](#1-contexte-et-question-de-recherche)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Ressources externes](#3-ressources-externes)
   - [Dataset DF40](#31-dataset-df40)
   - [Poids pré-entraînés des détecteurs](#32-poids-pré-entraînés-des-détecteurs)
4. [Environnement et dépendances](#4-environnement-et-dépendances)
5. [Lancer le code — Deux modes](#5-lancer-le-code--deux-modes)
   - [Mode A — Pipeline complet (GPU requis)](#mode-a--pipeline-complet-gpu-requis)
   - [Mode B — Reproduire les analyses uniquement (sans GPU)](#mode-b--reproduire-les-analyses-uniquement-sans-gpu)
6. [Description des notebooks](#6-description-des-notebooks)
7. [Fichiers de prédictions — Justification du dépôt](#7-fichiers-de-prédictions--justification-du-dépôt)
8. [Garanties méthodologiques](#8-garanties-méthodologiques)
9. [Benchmark de référence](#9-benchmark-de-référence)
10. [Citation](#10-citation)

---

## 1. Contexte et question de recherche

Ce dépôt contient le pipeline expérimental complet d'un mémoire de fin d'études traitant le sujet : Obsolescence des détecteurs de deepfakes GAN-trained face aux modèles de diffusion : gradient systématique de détectabilité selon la distance architecturale et récupération pragmatique de robustesse par stratégies d'ensemble à poids gelés

Le pipeline évalue quatre détecteurs de l'ère GAN selon trois scénarios de complexité croissante — du détecteur individuel à un méta-learner par stacking — sur un sous-ensemble personnalisé du benchmark **DF40**, constitué d'images générées par quatre méthodes de diffusion distinctes.

---

## 2. Structure du dépôt

```
deepfake-detector-obsolescence/
│
├── README.md
│
├── benchmark_src/                        ← Fichiers extraits et adaptés de DeepfakeBench (voir §9)
│   ├── metrics/                          ← Classes de métriques d'évaluation (AUC, EER, etc.)
│   │   ├── __init__.py
│   │   ├── base_metrics_class.py
│   │   ├── registry.py
│   │   └── utils.py
│   └── training/
│       ├── config/
│       │   └── detector/                 ← Fichiers YAML de configuration des 4 détecteurs retenus
│       │       ├── f3net.yaml
│       │       ├── meso4.yaml
│       │       ├── ucf.yaml
│       │       └── xception.yaml
│       ├── detectors/                    ← Implémentations des classes de détecteurs
│       │   ├── base_detector.py
│       │   ├── f3net_detector.py
│       │   ├── meso4_detector.py
│       │   ├── ucf_detector.py
│       │   └── xception_detector.py
│       ├── loss/                         ← Fonctions de perte
│       └── networks/                     ← Architectures des backbones
│
├── code/                                 ← Notebooks du pipeline expérimental (ordre séquentiel)
│   ├── 01_extract_DF40.ipynb            ← Extraction du dataset + nommage vidéo-level
│   ├── 02_data_audit.ipynb              ← Audit qualité (critères D1–D6)
│   ├── 02b_correct_dataset.ipynb        ← Corrections post-audit
│   ├── 03_split_data.ipynb              ← Split Train/Val/Test au niveau vidéo
│   ├── 04_inference.ipynb               ← Inférence des détecteurs → scores P(FAKE)
│   ├── 05_ensemble.ipynb                ← Entraînement du méta-learner (stacking)
│   ├── 06_evaluation_finale.ipynb       ← Évaluation finale : AUC, EER, F1, IC bootstrap
│   ├── 07_analyses_additionnelles.ipynb ← Analyses par méthode et corrélation d'erreurs
│   └── preprocessing/
│       └── dataset_json/                ← Fichiers JSON de configuration au format DeepfakeBench
│           ├── DiT_cdf.json
│           ├── DiT_ff.json
│           ├── MidJourney.json
│           ├── SiT_cdf.json
│           ├── SiT_ff.json
│           ├── ddim_cdf.json
│           └── ddim_ff.json
│
├── data/
│   ├── audit/                            ← Rapports d'audit générés par le notebook 02
│   ├── raw/
│   │   ├── DF40_temp/
│   │   │   ├── fake/                     ← ⬅ Placer les .zip FAKE ici (voir §3.1)
│   │   │   └── real/                     ← ⬅ Placer les .zip REAL ici (voir §3.1)
│   │   ├── DF40_fake/                    ← Généré automatiquement par le notebook 01
│   │   └── DF40_real/                    ← Généré automatiquement par le notebook 01
│   ├── splits/                           ← Généré automatiquement par le notebook 03
│   └── results/                          ← ⬅ Fichiers de prédictions déposés (voir §7)
│       ├── test_probs.csv               ← P(FAKE) par image, Test Set, tous détecteurs
│       ├── val_probs.csv                ← P(FAKE) par image, Validation Set, tous détecteurs
│       ├── ensemble_scores_test.csv     ← Scores du méta-learner, Test Set
│       ├── ensemble_scenario_probs_val.csv ← Probabilités par scénario, Validation Set
│       ├── analysis_A_per_method.csv    ← Performance par méthode de diffusion (§3.2)
│       ├── analysis_B_error_correlation.csv ← Corrélation d'erreurs inter-détecteurs (§3.3)
│       ├── bootstrap_auc_val.csv        ← Intervalles de confiance AUC bootstrap (§3.4)
│       └── diffusion_methods_summary.csv ← Synthèse agrégée par méthode de génération
│
└── weights/
    └── pretrained/                       ← ⬅ Placer les poids téléchargés ici (voir §3.2)
```

> **Note sur les fichiers `.gitkeep`** : les dossiers vides (`data/raw/`, `data/splits/`, `weights/pretrained/`, etc.) sont maintenus dans le dépôt via des fichiers `.gitkeep`. Ils seront peuplés soit en exécutant le pipeline, soit en y plaçant manuellement les fichiers requis tels que décrits ci-dessous.

---

## 3. Ressources externes

### 3.1 Dataset DF40

Le dataset utilisé dans ce projet est un **sous-ensemble personnalisé de DF40**, constitué et validé dans le cadre de ce mémoire. DF40 est un benchmark à grande échelle couvrant 40 techniques de génération de deepfakes distinctes.

**Accéder au dataset :** [https://github.com/YZY-stack/DF40](https://github.com/YZY-stack/DF40)

L'accès est accordé sur demande via le dépôt officiel DF40. Suivre les instructions fournies sur ce dépôt.

Une fois les fichiers téléchargés, les placer aux emplacements suivants **exactement** (les noms de fichiers sont codés en dur dans le notebook 01) :

**Sources FAKE** → `data/raw/DF40_temp/fake/`

| Fichier | Images extraites dans le cadre de ce projet |
|---|---|
| `MidJourney.zip` | 1 600 |
| `ddim.zip` | 1 300 |
| `DiT.zip` | 358 |
| `SiT.zip` | 258 |

**Sources REAL** → `data/raw/DF40_temp/real/`

| Fichier | Vidéos | Frames extraites dans ce projet |
|---|---|---|
| `FaceForensics++_real_data_for_DF40.zip` | 999 | 1 998 (2 frames/vidéo) |
| `Celeb-DF-v2_real_data_for_DF40.zip` | 888 | 1 776 (2 frames/vidéo) |

**Composition du dataset retenu pour ce projet :** ~7 290 images au total — ~3 774 REAL + ~3 516 FAKE — ratio approximativement équilibré.

**Méthodes exclues** (présentes dans DF40 mais écartées de cette étude pour qualité visuelle insuffisante, décision mars 2026) : `sd2.1`, `pixart`, `CollabDiff`.

> ⚠️ Les images du dataset ne sont **pas** incluses dans ce dépôt. Les droits appartiennent aux fournisseurs originaux de DF40.

---

### 3.2 Poids pré-entraînés des détecteurs

Les quatre détecteurs retenus sont évalués en mode **zéro-shot** : ils sont chargés avec leurs poids pré-entraînés originaux issus du benchmark DeepfakeBench, sans aucun réentraînement sur les données DF40. C'est un choix méthodologique délibéré — réentraîner les détecteurs confondrait la capacité de généralisation avec une adaptation spécifique au dataset.

**Télécharger les poids :** [DeepfakeBench Releases — v1.0.1](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)

Ce lien correspond à la **section 5 (Evaluation)** du [README de DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).

Après téléchargement, placer les fichiers de poids dans `weights/pretrained/`. Les fichiers YAML dans `benchmark_src/training/config/detector/` référencent ces chemins et peuvent nécessiter des ajustements selon les noms de fichiers locaux.

**Détecteurs retenus pour ce projet :**

| Détecteur | Architecture | Fichier de config |
|---|---|---|
| Xception | Xception | `xception.yaml` |
| MesoNet (Meso4) | CNN léger | `meso4.yaml` |
| F3Net | Domaine fréquentiel | `f3net.yaml` |
| UCF | Incertitude-aware | `ucf.yaml` |

> ⚠️ Les fichiers de poids ne sont **pas** inclus dans ce dépôt en raison de leur taille. Ils doivent être téléchargés depuis la page de release DeepfakeBench indiquée ci-dessus.

---

## 4. Environnement et dépendances

**Environnement principal : Google Colab**

L'intégralité des notebooks a été développée et testée exclusivement sur Google Colab (Python 3.10+, runtime GPU). Le chemin de montage Google Drive `/content/drive/MyDrive/Memoire_Deepfakes` est supposé tout au long du pipeline. Si vous souhaitez exécuter ce pipeline dans un environnement différent (machine locale, cluster HPC), vous devrez ajuster toutes les variables de chemin dans la **cellule 1 du notebook 01** (`PROJECT_ROOT`, `DATA_DIR`, etc.) et vous assurer que les dépendances système nécessaires sont disponibles. **Cette adaptation est laissée à la charge du lecteur.**

**Dépendances principales** (installées en ligne dans les notebooks via `pip`) :

```
torch
torchvision
numpy
pandas
scikit-learn
pillow
tqdm
matplotlib
pyyaml
```

> Un fichier `requirements.txt` avec les versions épinglées correspondant au runtime Colab utilisé lors de l'expérimentation est fourni à la racine du dépôt à titre de référence.

**Seed de reproductibilité :** `RANDOM_SEED = 42` — défini globalement dans le notebook 01 via `random.seed(42)` et `np.random.seed(42)`, et propagé à l'ensemble des notebooks suivants.

---

## 5. Lancer le code — Deux modes

Ce dépôt supporte deux modes d'utilisation distincts selon vos objectifs et les ressources disponibles.

---

### Mode A — Pipeline complet (GPU requis)

**Utilisez ce mode** pour reproduire l'intégralité de l'expérience depuis les données DF40 brutes jusqu'aux résultats finaux. Cela nécessite un runtime GPU (Google Colab GPU recommandé), l'accès au dataset DF40 (voir §3.1) et les poids pré-entraînés (voir §3.2).

Exécuter les notebooks **dans l'ordre séquentiel strict** :

```
01_extract_DF40.ipynb           → Extraction et renommage des images depuis les .zip
02_data_audit.ipynb             → Audit qualité du dataset extrait
02b_correct_dataset.ipynb       → Corrections identifiées lors de l'audit
03_split_data.ipynb             → Split Train / Validation / Test au niveau vidéo
04_inference.ipynb              → Inférence des 4 détecteurs → génération des scores P(FAKE)
05_ensemble.ipynb               → Entraînement du méta-learner par stacking
06_evaluation_finale.ipynb      → Calcul AUC, EER, F1, IC bootstrap (sections 3.1–3.4)
07_analyses_additionnelles.ipynb → Analyses par méthode et corrélation d'erreurs
```

**Avant de commencer**, vérifier que :
1. Les fichiers `.zip` DF40 sont placés dans `data/raw/DF40_temp/` (voir §3.1)
2. Les fichiers de poids sont placés dans `weights/pretrained/` (voir §3.2)
3. Le Google Drive est monté et `PROJECT_ROOT` dans le notebook 01 correspond à votre chemin Drive réel
4. Un runtime GPU est actif dans Colab (`Exécution → Modifier le type d'exécution → GPU T4`)

**Durée estimée :** les notebooks 01–03 sont limités par le CPU (1–2 heures) ; les notebooks 04–05 sont limités par le GPU et constituent la partie la plus coûteuse du pipeline.

---

### Mode B — Reproduire les analyses uniquement (sans GPU)

**Utilisez ce mode** pour reproduire les analyses statistiques et les résultats des **sections 3.2 à 3.4** du mémoire sans relancer le pipeline d'inférence complet. Tous les fichiers de prédictions intermédiaires sont déjà déposés dans `data/results/` (voir §7 pour la justification).

Exécuter uniquement :

```
06_evaluation_finale.ipynb       → Toutes les métriques et figures depuis les fichiers de prédictions
07_analyses_additionnelles.ipynb → Analyses par méthode et corrélation d'erreurs
```

**Aucun GPU, aucun téléchargement DF40, aucun fichier de poids requis.** Un runtime CPU standard sur Colab est suffisant.

---

## 6. Description des notebooks

### `01_extract_DF40.ipynb` — Extraction du dataset (v4)

Extrait les images depuis les archives `.zip` DF40 brutes et applique une **convention de nommage vidéo-level** conçue pour prévenir le data leakage lors de l'étape de split. Chaque image REAL est renommée `{prefix}_vid{video_id:04d}_f{frame_idx:02d}.jpg` (ex. `FF_vid0042_f00.jpg`), encodant l'identité de la vidéo source directement dans le nom de fichier. Deux frames par vidéo sont extraites avec un espacement temporel uniforme. Un `real_video_manifest.csv` est généré et requis par le notebook 03.

### `02_data_audit.ipynb` — Audit qualité

Applique un ensemble structuré de critères qualité (D1–D6) aux images extraites : vérification de résolution, taux de détection de visages, détection de corruptions, vérification de l'équilibre des classes et détection de doublons. Génère des rapports d'audit dans `data/audit/`.

### `02b_correct_dataset.ipynb` — Corrections post-audit

Applique les corrections ciblées aux anomalies identifiées dans le notebook 02.

### `03_split_data.ipynb` — Split vidéo-level

Divise le dataset en ensembles Train (70%) / Validation (15%) / Test (15%) **au niveau de la vidéo**, en utilisant `real_video_manifest.csv` pour garantir que toutes les frames d'une même vidéo source se retrouvent dans le même split. Cela élimine le data leakage basé sur l'identité. Le **Validation Set est isolé à partir de ce point** et n'est jamais utilisé pour la sélection de seuils ou le réglage de modèles — uniquement pour l'évaluation non biaisée et comme signal d'entraînement du méta-learner. Les CSV de split sont écrits dans `data/splits/`.

### `04_inference.ipynb` — Inférence des détecteurs

Exécute chacun des 4 détecteurs en mode zéro-shot sur les trois ensembles. Produit les scores de probabilité `P(FAKE)` par image. Les résultats sont sauvegardés dans `data/results/test_probs.csv` et `data/results/val_probs.csv`.

### `05_ensemble.ipynb` — Méta-learner (ensemble par stacking)

Entraîne un méta-learner de régression logistique par stacking en utilisant les sorties des détecteurs sur le Validation Set comme features. Évalue trois scénarios d'ensemble (Scénario A : détecteurs individuels ; Scénario B : moyennage simple ; Scénario C : stacking entraîné à poids gelés). Les poids du méta-learner et les scores d'ensemble sont sauvegardés dans `data/results/`.

### `06_evaluation_finale.ipynb` — Évaluation finale

Calcule l'ensemble des métriques rapportées sur le Test Set : AUC, EER, F1-score, accuracy, et intervalles de confiance bootstrap (1 000 rééchantillonnages, `seed=42`) pour l'AUC. Produit toutes les figures et tableaux apparaissant dans les sections 3.1 à 3.4 du mémoire.

### `07_analyses_additionnelles.ipynb` — Analyses complémentaires

Réalise la désagrégation de la performance par méthode de diffusion (section 3.2) et l'analyse de corrélation d'erreurs inter-détecteurs (section 3.3). Lit depuis les fichiers CSV déposés et produit les figures supplémentaires.

---

## 7. Fichiers de prédictions — Justification du dépôt

Le dossier `data/results/` contient les fichiers intermédiaires pré-calculés suivants :

| Fichier | Contenu | Utilisé dans |
|---|---|---|
| `test_probs.csv` | P(FAKE) par image par détecteur, Test Set | §3.1, §3.4 |
| `val_probs.csv` | P(FAKE) par image par détecteur, Validation Set | §3.2, §3.3, §3.4 |
| `ensemble_scores_test.csv` | Scores du méta-learner, Test Set | §3.4 |
| `ensemble_scenario_probs_val.csv` | Probabilités par scénario, Validation Set | §3.4 |
| `analysis_A_per_method.csv` | Désagrégation AUC par méthode de diffusion | §3.2 |
| `analysis_B_error_correlation.csv` | Matrice de corrélation d'erreurs inter-détecteurs | §3.3 |
| `bootstrap_auc_val.csv` | Résultats bootstrap CI pour l'AUC | §3.4 |
| `diffusion_methods_summary.csv` | Synthèse agrégée par méthode de génération | §3.2 |

**Pourquoi ces fichiers sont déposés :** l'étape d'inférence (notebook 04) est la partie la plus coûteuse en ressources du pipeline, nécessitant un GPU et plusieurs heures d'exécution. Le dépôt de ces scores de prédictions permet à tout lecteur de reproduire **l'ensemble des analyses statistiques, métriques et figures des sections 3.2 à 3.4** (notebooks 06–07) sans accès à un GPU, au dataset DF40, ni aux poids des détecteurs. Il s'agit d'une mesure de reproductibilité délibérée — cohérente avec les engagements méthodologiques explicites de ce travail (seed fixé, protocole anti-HARKing, Validation Set isolé) — et recommandée par le promoteur de ce mémoire.

Ces fichiers ne se substituent pas au pipeline : ce sont des sorties intermédiaires, pas des résultats finaux. Tout lecteur souhaitant les vérifier indépendamment peut les régénérer en exécutant le Mode A dans son intégralité.

---

## 8. Garanties méthodologiques

Ce pipeline a été conçu avec les engagements explicites de rigueur et de reproductibilité suivants :

**Seed fixé.** `RANDOM_SEED = 42` est défini globalement dans le notebook 01 et propagé à l'ensemble des notebooks suivants. Toutes les opérations d'échantillonnage (sélection de frames, split train/val/test, rééchantillonnage bootstrap) utilisent ce seed.

**Split vidéo-level.** Le split Train/Val/Test est réalisé au niveau de la vidéo (et non au niveau de la frame). Cela élimine le data leakage basé sur l'identité qui surviendrait si des frames de la même vidéo source apparaissaient dans plusieurs ensembles — un biais connu dans les benchmarks de détection de deepfakes.

**Isolation du Validation Set.** Le Validation Set n'est jamais utilisé pour l'optimisation de seuils ou le réglage d'hyperparamètres. Ses seuls rôles sont : (1) fournir le signal d'entraînement pour le méta-learner par stacking (notebook 05), et (2) calculer les intervalles de confiance bootstrap non biaisés (notebook 06). Les résultats du Test Set sont calculés une seule fois, à l'étape d'évaluation finale.

**Protocole anti-HARKing.** Toutes les hypothèses et plans d'analyse ont été fixés avant d'observer les résultats du Test Set. Aucune re-spécification post-hoc de métriques ou d'analyses de sous-groupes n'a été réalisée après inspection du Test Set.

**Évaluation zéro-shot.** Les détecteurs sont utilisés avec leurs poids pré-entraînés originaux, sans aucun réentraînement sur les données DF40. Cela garantit que la performance mesurée reflète la capacité de généralisation, et non une adaptation spécifique au dataset.

**Ensemble à poids gelés.** Le Scénario C (méta-learner par stacking) entraîne uniquement la couche de combinaison logistique sur les sorties des détecteurs — les poids des détecteurs eux-mêmes restent intégralement gelés. Cela reflète une contrainte pragmatique réaliste : aucun accès aux données d'entraînement originales des détecteurs n'est supposé.

---

## 9. Benchmark de référence

Ce travail s'appuie directement sur le framework **DeepfakeBench**. Le dossier `benchmark_src/` de ce dépôt contient un **sous-ensemble de fichiers sources extraits et adaptés de DeepfakeBench**, limité aux composants strictement nécessaires à cette expérimentation (métriques, configurations YAML des 4 détecteurs retenus, implémentations des détecteurs, fonctions de perte et architectures associées). Toutes les modifications apportées par rapport au code original sont documentées dans les fichiers concernés. Le copyright du code original appartient à ses auteurs ; cette version adaptée est utilisée exclusivement dans le cadre d'une recherche académique non commerciale.

> Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu.
> *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection.*
> NeurIPS 2023 Datasets & Benchmarks Track.
> [[Article]](https://arxiv.org/abs/2307.01426) · [[Dépôt]](https://github.com/SCLBD/DeepfakeBench) · [[Poids pré-entraînés]](https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1)

Le dataset utilisé dans cette étude est un sous-ensemble personnalisé de **DF40** :

> Zhiyuan Yan et al.
> *DF40: Toward Next-Generation Deepfake Detection.*
> [[Dépôt]](https://github.com/YZY-stack/DF40)

---

## 10. Citation

Si vous référencez ou construisez sur ce travail, merci de citer :

```bibtex
@mastersthesis{ducarme2026deepfake,
  author  = {Ducarme, Maxime},
  title   = {Obsolescence des détecteurs de deepfakes :
             Évaluation de la performance des modèles GANs
             face à la génération par diffusion},
  school  = {[Université Catholique de Louvain]},
  year    = {2026},
  url     = {https://github.com/ducamax07/deepfake-detector-obsolescence}
}
```

---
Ce repository ne sera pas maintenu. 
