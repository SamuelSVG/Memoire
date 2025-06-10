# Comparaison empirique des caractéristiques des dépôts logiciels entre plusieurs plateformes

Ce projet permet d’analyser des dépôts de code hébergés sur différentes plateformes (comme GitHub ou GitLab) en sélectionnant des dépôts pertinnents et en extrayant des données techniques et sociales comme le nombre de commits, la taille du dépôt, la licence, le nombre de stars, etc. via une méthodologie qui combine un clone et les API des plateformes.

---

## Prérequis

Avant de commencer, assurez-vous que les éléments suivants sont installés et configurés sur votre machine :

- Un environnement **Python** (version **3.12** ou supérieure recommandée)
- **Docker** installé et opérationnel
- Des **tokens d’authentification personnels** pour chaque plateforme à analyser

---

## Étapes d’utilisation

### 1. Installation des dépendances

Un fichier `requirements.txt` est fourni pour installer toutes les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

---

### 2. Configuration des accès aux plateformes

Créez un fichier `.env` à la racine du projet contenant vos tokens d’authentification pour chaque plateforme :

```env
GITHUB_TOKEN=your_github_token
GITLAB_TOKEN=your_gitlab_token
...
```

> **Attention** : Les tokens doivent disposer des **autorisations nécessaires** pour accéder en lecture aux API des plateformes concernées.

---

### 3. Initialisation de l’environnement

Une cellule nommée **Setup Cell** est fournie au début du **Jupyter notebook** pour :

- Importer les différents packages
- Initialiser l’environnement d’exécution

---

### 4. Collecte et analyse des données

Des **cellules de test** sont disponibles dans le notebook pour :

- Exécuter des analyses sur une faible quantité de données
- Explorer les fonctionnalités principales

Une **documentation détaillée** est incluse pour vous aider à créer de nouvelles analyses personnalisées.

---

### 5. Note importante : Docker obligatoire

Pour récupérer automatiquement :

- La **licence** d’un dépôt
- La **répartition des langages** utilisés

Il est **indispensable** que **Docker** soit installé et fonctionnel.

Les outils `github-linguist` et `licensee` sont utilisés dans des **conteneurs Docker** pour garantir la **portabilité** et la **reproductibilité** des résultats.

---

### Note importante : Toutes les fonctions développées dans le cadre de la recherche lors de ce travail ont été conservées dans le code. Seules les fonctions utilisées dans le notebook sont utilisées dans le travail.

---
