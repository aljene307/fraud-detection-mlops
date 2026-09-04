# Notes de projet — Fraud Detection MLOps

Double usage : mémoire technique du projet, et fiche de révision pour les
entretiens. La partie factuelle est remplie ; la partie « à compléter » est
volontairement vide — l'exercice consiste à l'écrire avec mes propres mots.

État au commit `c6648a5` · Phase 0 et Phase 1 terminées · 140 tests.

---

# 1. Ce qui a été construit, step par step

| Step | Commit | Ce qui a été fait |
|---|---|---|
| **0.A** | `1b5b46b` | Init du dépôt sur `main`, `.gitignore` (données, `mlflow.db`, token Kaggle), `.gitattributes` en LF. Renommage `SPEC.md.md` → `SPEC.md`, nettoyage du wrapper heredoc collé dans `CLAUDE.md`. |
| **0.B** | `279ffec` | `pyproject.toml` avec versions figées, venv, squelette `src/`. `src/config.py` centralise chemins et identifiants MLflow. Sonde PyPI : aucune dépendance sans wheel Python 3.13. |
| **0.C** | `255d906` | `src/data/download.py` : API Kaggle avec repli manuel, puis validation dure en 4 couches (existence → parsing → colonnes → contenu). Code de retour non nul pour la CI. |
| **0.C + 0.D** | `4015e44` | SHA-256 figé dans `config.py`. Pipeline de variables `src/features/` : `log_amount`, `hour_of_day`, `Time` retiré, split stratifié 60/20/20 (`--strategy=time` en option), `split_manifest.json`. |
| **0.E** | `2fb6440` | 42 tests sur validation et découpage, sur données synthétiques. Correction d'un bug : le manifeste était écrit en dur dans `data/processed/` alors que les parquets suivaient `--out-dir`. |
| **1.A** | `5cf652f` | `src/training/metrics.py` : PR-AUC via `average_precision_score`, `recall_at_precision`, matrice de confusion. 34 tests, dont la démonstration ROC-AUC 0,9774 vs PR-AUC 0,0417. |
| **1.B** | `f45a319` | `src/training/train.py` : trois stratégies de déséquilibre. Seuil choisi sur la validation, figé pour le test. 25 tests dont l'espion anti-fuite SMOTE. |
| **1.C** | `c3b4ef6` | `src/training/tracking.py` : runs MLflow, artefacts, registre, alias `@production`. Correction de `config.py` (SQLite au lieu du store fichier, refusé par MLflow 3). Promotion manuelle via `--promote`. |
| **1.E** | `c6648a5` | `config/gate.yaml` (plancher 0,80) + `src/training/gate.py`, lecteur exécutable à code de retour 0/1. 23 tests. |

*Il n'y a pas de step 1.D : les tests de métriques ont été écrits avec le module
lui-même en 1.A, et le figeage du SHA-256 (0.C) a été livré avec le découpage
(0.D) dans un seul commit.*

---

# 2. Architecture

```
   creditcard.csv (Kaggle, 284 807 lignes, 492 fraudes)
          │
          ▼
   VALIDATION DURE                      src/data/download.py
   4 couches : existe → parse → colonnes → contenu
   8 contrôles, dont SHA-256 comparé à la référence figée
          │
          ▼
   VARIABLES + DÉCOUPAGE                src/features/
   log_amount, hour_of_day, Time retiré  → 30 variables
   stratifié 60/20/20, graine 42         → split_manifest.json
          │
          ▼
   ENTRAÎNEMENT                          src/training/train.py
   logreg  ·  xgb+scale_pos_weight  ·  xgb+SMOTE
   seuil choisi sur VAL, figé pour TEST
          │
          ├──────────────► MLflow          src/training/tracking.py
          │                params / métriques val_ test_ / tags
          │                artefacts : courbe PR, matrice de confusion
          │                tags : commit git, git_dirty, SHA du dataset
          │                        │
          │                        ▼
          │                REGISTRE  fraud-detector
          │                versions 1..n, alias @production (manuel)
          │                        │
          │                        ▼
          │                models:/fraud-detector@production
          │                (ce que chargera le serving en Phase 2)
          ▼
   GATE QUALITÉ                          config/gate.yaml + gate.py
   plancher test_pr_auc >= 0.80  → code de retour 0 ou 1
```

**Répartition du découpage** (graine 42, stratifié) :

| split | lignes | fraudes | taux |
|---|---|---|---|
| train | 170 884 | 295 | 0,1726 % |
| val | 56 961 | 98 | 0,1720 % |
| test | 56 962 | 99 | 0,1738 % |

---

# 3. Mes résultats

PR-AUC du hasard (taux de positifs) : **0,001738**

| modèle | PR-AUC test | × hasard | ROC-AUC | recall@p90 | TP | FP | FN | précision au seuil |
|---|---|---|---|---|---|---|---|---|
| **xgb** (`scale_pos_weight`) | **0,8727** | 502× | 0,9793 | 0,8384 | 80 | 6 | 19 | 93,0 % |
| xgb_smote | 0,8685 | 500× | 0,9851 | 0,8384 | 82 | 2 | 17 | 97,6 % |
| logreg | 0,7602 | 437× | 0,9689 | 0,0000 | — | — | — | seuil inatteignable |

**PR-AUC de validation** : xgb 0,8234 · xgb_smote 0,8172 · logreg 0,6850

Seuil figé de `xgb` : **0,80801547** (choisi sur la validation à ≥ 90 % de précision).

**Trois observations à savoir commenter :**

- `logreg` plafonne à **87 % de précision** : il n'atteint jamais la cible de
  90 %, donc `recall@p90 = 0` et son seuil vaut `None`. Ce n'est pas un bug,
  c'est un modèle incapable d'opérer au point de fonctionnement demandé.
- `xgb_smote` a une **meilleure précision** (97,6 % contre 93,0 %) mais une
  PR-AUC légèrement inférieure : il est plus prudent à son seuil, moins bon en
  classement global.
- Le test score **plus haut** que la validation (0,8727 contre 0,8234). Écart de
  0,049 dû au seul hasard du découpage — chaque fold n'a que ~98 fraudes.

---

# 4. Décisions techniques et leurs raisons

| Décision | Raison |
|---|---|
| **PR-AUC en métrique principale** | La courbe PR ne contient aucun vrai négatif. Le ROC divise les faux positifs par 284 315 vrais négatifs, ce qui les rend invisibles ; la précision les divise par le nombre d'alertes émises. Démontré : même modèle, ROC-AUC 0,9774 et PR-AUC 0,0417. |
| **`average_precision_score`, pas `auc(recall, precision)`** | L'interpolation linéaire est invalide dans l'espace PR (Davis & Goadrich, 2006). Écart mesuré sur courbe clairsemée : 0,5111 contre 0,4794. |
| **Toujours logguer la PR-AUC du hasard** | 0,85 est excellent en PR-AUC et médiocre en ROC-AUC. Sans la référence (0,001738), le nombre n'a pas d'échelle. |
| **Découpage stratifié** | Avec 492 fraudes au total, un tirage non stratifié ferait varier le nombre de positifs par fold et les métriques bougeraient sans rapport avec le modèle. |
| **3 folds et non 2** | La validation sert à choisir le seuil ; le test n'est lu qu'une fois. Choisir le seuil sur le test puis publier une métrique sur ce même test gonfle le résultat. |
| **`Time` retiré** | Décalage d'horloge absolu sur une fenêtre de 48 h : le garder laisserait le modèle mémoriser *quand* les données ont été collectées. Remplacé par `hour_of_day`. |
| **Scaler dans le `Pipeline`, pas en fichier séparé** | Un `scaler.joblib` séparé oblige à penser à l'appliquer à l'inférence ; l'oublier ne lève aucune erreur et produit des scores faux. Dans le pipeline, l'oubli est impossible par construction. |
| **`scale_pos_weight` plutôt que SMOTE par défaut** | Repondère le gradient sans dupliquer ni inventer de données. SMOTE fabrique ~170 000 points depuis 295 fraudes réelles. |
| **SMOTE sur le train uniquement, après le découpage** | Appliqué avant, un point interpolé depuis une fraude du test se retrouve dans le train : quasi-doublons des deux côtés, scores spectaculaires et sans signification. |
| **`eval_metric="aucpr"` pour l'arrêt anticipé** | Surveiller `logloss` ferait arrêter au meilleur moment pour un critère qui n'est pas le nôtre. |
| **Alias `@production`, pas les stages** | Les stages sont dépréciés depuis MLflow 2.9 et seront supprimés. Un alias est un pointeur mutable : revenir en arrière consiste à le repointer, sans redéploiement ni changement de code. Plusieurs alias peuvent coexister (`@champion`, `@challenger`), sans machine à états imposée. |
| **Backend SQLite, pas `./mlruns`** | MLflow 3 refuse le store fichier (« maintenance mode »), et le registre n'a jamais fonctionné avec lui — or l'alias en dépend. |
| **Pas de `mlflow.autolog()`** | Il journalise `training_accuracy_score` sans rien demander : il violerait la règle du projet en silence. Logging explicite + garde-fou qui lève sur toute clé contenant `accuracy`. |
| **Tag `git_dirty`** | Un run lancé depuis un arbre de travail modifié n'est pas rejouable à partir de son SHA. Le taguer, c'est refuser de laisser croire le contraire. |
| **Promotion manuelle (`--promote`)** | Un run expérimental ne doit pas devenir la production parce qu'il a fini de tourner. En Phase 4, la CI devient l'autorité de promotion. |
| **Gate à 0,80** | Encadré par deux bornes mesurées : sous le pire fold d'un bon modèle (val 0,8234) pour ne pas rejeter `xgb`, au-dessus du linéaire (0,7602) pour bloquer un retour à `logreg`. Fenêtre [0,78–0,81]. Soit ~1,5× le bruit de fold mesuré (0,049). |
| **Un seul critère bloquant** | `recall@p90` dépend du seuil, donc bouge davantage. Deux critères doubleraient les échecs à tort — et un gate auquel on ne fait pas confiance finit désactivé. |
| **Tests sur données synthétiques** | 138 des 140 tests tournent sans le CSV de 144 Mo. La CI de la Phase 4 sera verte sur une machine vierge. |

---

# 5. Dettes techniques pour la Phase 2

**1. Les probabilités ne sont pas calibrées.**
`scale_pos_weight` repondère la fonction de coût : le modèle répond
« probabilité si la fraude représentait 50 % du trafic ». Un score de 0,81 ne
signifie **pas** « 81 % de chances que ce soit une fraude ». Ça ne gêne pas la
PR-AUC (qui ne juge que le classement) ni le seuil (choisi empiriquement), mais
l'API ne devra pas présenter ces nombres comme des probabilités réelles. Si une
vraie probabilité devient nécessaire, il faudra une étape de calibration
(Platt / isotonic) ajustée sur la validation.

**2. `fastapi` et `uvicorn` arrivent transitivement par MLflow.**
Ils sont dans `requirements.lock.txt` sans être déclarés dans `pyproject.toml`.
Le jour où MLflow change de framework web, le serving casse sans qu'on ait
touché à quoi que ce soit. À déclarer explicitement dès le premier fichier de
serving — même raisonnement que `pyyaml` au step 1.E.

**Point d'attention pour le serving** : le modèle doit être chargé **au
démarrage**, pas à chaque requête (coût de plusieurs centaines de millisecondes).
Ce qui impose de gérer le cas « l'alias `@production` n'existe pas encore ».

---

# 6. À compléter par moi

> Ces réponses sont volontairement vides. L'exercice est de les écrire avec mes
> propres mots, sans relire la partie 4 — c'est la vérification que j'ai compris.

## Pourquoi l'accuracy est trompeuse ici ?

*(à compléter)*

<br><br><br>

---

## Comment j'ai choisi mon seuil ?

*(à compléter)*

<br><br><br>

---

## Pourquoi un registre plutôt qu'un pickle ?

*(à compléter)*

<br><br><br>

---

## Ce que le gate empêche ?

*(à compléter)*

<br><br><br>

---

## Pourquoi SMOTE n'a pas gagné ?

*(à compléter)*

<br><br><br>
