# 🤖 Assistant Support Technique Multi-Produits (RAG)

Ce système utilise la technologie **RAG** (Retrieval-Augmented Generation) pour répondre à des questions techniques en se basant exclusivement sur vos propres documents PDF.

## 🌟 Fonctionnalités
- **Lecture de PDF** : Charge et analyse automatiquement tous les PDF dans le dossier `data/`.
- **Réponses Précises** : Utilise l'IA de **Groq** (Llama 3.3) pour générer des réponses basées sur le contexte.
- **Citations des Sources** : Indique toujours de quel document provient l'information.
- **Interface Web** : Une interface simple et interactive construite avec **Streamlit**.

---

## 📁 Structure du Projet

- **`app.py`** : L'interface utilisateur. C'est le fichier que vous lancez pour voir l'application.
- **`rag_engine.py`** : Le "cerveau" du projet. Il gère la découpe des textes et la recherche dans les documents.
- **`data/`** : Le dossier où vous devez déposer vos manuels, guides et FAQs au format PDF.
- **`.env`** : Contient votre clé API secrète pour Groq.
- **`requirements.txt`** : La liste des bibliothèques Python nécessaires.
- **`venv/`** : Votre environnement virtuel (isole les installations du projet).

---

## 🚀 Installation et Lancement

### 1. Préparation de l'environnement
Si ce n'est pas déjà fait, créez et activez votre environnement virtuel :
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Installation des dépendances
Installez tous les outils nécessaires :
```bash
pip install -r requirements.txt
```

### 3. Configuration de la clé API
Ouvrez le fichier `.env` et remplacez la valeur par votre clé Groq :
`GROQ_API_KEY=gsk_xxxxxxxxxxxx...`

### 4. Lancement de l'application
```bash
streamlit run app.py
```

---

## 🧠 Comment ça marche ? (Le concept RAG)

1. **Chargement** : Le système lit vos PDF dans `data/`.
2. **Découpage (Chunking)** : Les longs documents sont découpés en petits morceaux de 1000 caractères.
3. **Embeddings** : Chaque morceau est transformé en une liste de nombres (vecteurs) qui représentent son "sens".
4. **Recherche (Retrieval)** : Quand vous posez une question, le système cherche les morceaux de PDF qui y ressemblent le plus.
5. **Génération** : L'IA de Groq reçoit votre question + les morceaux trouvés et rédige une réponse en français.

---

## 🛠️ Dépannage
- **Erreur de module manquant** : Vérifiez que vous avez bien activé le `venv` et fait `pip install -r requirements.txt`.
- **Modèle obsolète** : Le projet utilise `llama-3.3-70b-versatile`, qui est le modèle le plus stable actuellement.
- **Aucun document trouvé** : Assurez-vous que vos fichiers dans `data/` finissent bien par `.pdf`.

---
*Développé avec ❤️ pour un support technique intelligent.*
