# 📚 Assistant Académique Intelligent — RAG + Agents

> Projet universitaire 2026 — Thème : **Assistant académique sur les notes de cours**

Un assistant IA complet combinant **RAG** (Retrieval-Augmented Generation) et des **agents autonomes** pour répondre aux questions des étudiants en ML/Deep Learning, effectuer des calculs scientifiques, gérer une liste de tâches, obtenir la météo, et effectuer des recherches web.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Interface Streamlit                      │
│                       app.py                             │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│              Agent Intelligent (agent.py)              │
│                                                          │
│  1. Keyword pre-filter  (O(1), zero LLM cost)            │
│  2. LLM classifier      (gpt-4o-mini, fallback only)     │
│                                                          │
│  Mémoire: ConversationBufferWindowMemory (LangChain)     │
└──────┬────────┬────────┬────────┬──────────┬────────────┘
       │        │        │        │          │
    RAG     Calc.    To-do   Weather   Web search   Chat
       │
┌──────▼──────────────────────────────────┐
│           Pipeline RAG (rag_pipeline.py) │
│                                          │
│  FAISS vector store  ←  OpenAI Embeddings│
│  PyPDF / Docx2txt loaders                │
│  RecursiveCharacterTextSplitter          │
│  GPT-4o-mini  →  answer + citations      │
│  ConversationBufferWindowMemory (k=6)    │
└──────────────────────────────────────────┘
```

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/fatima-299/accademic_assistant.git
cd academic-assistant
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

```bash
cp .env.example .env
# Éditez .env et remplacez sk-your-openai-api-key-here par votre vraie clé
```

### 5. Construire la base vectorielle FAISS

```bash
python build_db.py
```

> À refaire uniquement si vous modifiez les fichiers dans `data/`.

### 6. Lancer l'application

```bash
streamlit run app.py
```

L'application sera disponible sur `http://localhost:8501`

---

## 📁 Structure du projet

```
academic_assistant/
├── app.py                    # Interface Streamlit (Partie 4)
├── build_db.py               # Script de construction de l'index FAISS
├── requirements.txt          # Dépendances Python
├── .env.example              # Template de configuration
├── .env                      # Clé API (ne pas committer)
├── .gitignore
├── README.md
├── todo_list.json            # Persistance des tâches (auto-créé)
├── data/                     # Documents académiques (PDF / DOCX)
│   ├── machineLearning.pdf
│   ├── ANNs.pdf
│   ├── MLBOOK.pdf
│   └── Neural Networks and Deep Learning.pdf
├── db/
│   └── faiss_index/          # Index vectoriel FAISS (auto-généré)
└── src/
    ├── document_loader.py    # Chargement PDF / DOCX
    ├── vector_store.py       # Construction et chargement FAISS
    ├── rag_pipeline.py       # Pipeline RAG + citations + mémoire LangChain
    ├── agent.py             # Agent (keyword filter + LLM classifier)
    └── tools.py              # Calculatrice, To-do, Météo, Recherche web
```

---

## 🎯 Fonctionnalités par partie

### Partie 1 — Pipeline RAG

| Composant | Détail |
|-----------|--------|
| Ingestion | PDF (`PyPDFLoader`), DOCX (`Docx2txtLoader`) |
| Chunking | `RecursiveCharacterTextSplitter` — chunk=800, overlap=150 |
| Vectorisation | `OpenAIEmbeddings` (`text-embedding-ada-002`) |
| Stockage | FAISS (persisté dans `db/faiss_index/`) |
| Retrieval | Top-3 chunks par similarité cosinus |
| Génération | GPT-4o-mini avec citations fichier + numéro de page |

**Exemple de citation dans la réponse :**
```
Sources used:
  [1] machineLearning.pdf — page 12
  [2] ANNs.pdf — page 5
```

### Partie 2 — Agents et Outils

| Outil | Description | Exemple |
|-------|-------------|---------|
| 🧮 Calculatrice | Évaluateur AST sécurisé, fonctions scientifiques | `calculate sqrt(144) + log(100,10)` |
| 🌤️ Météo | API Open-Meteo (gratuite, temps réel) | `weather in Paris` |
| 🌐 Recherche web | Tavily, avec titre + URL + extrait | `ChatGPT updates` |
| 📝 To-do list | Persistante (JSON), add/show/delete/clear | `add task: revise chapter 3` |

### Partie 3 — Intégration & Routage intelligent

Le routeur fonctionne en deux étapes :

1. **Keyword pre-filter** — expressions régulières qui capturent les cas évidents (`add task:`, `calculate`, `weather in`, `hello`, mots-clés académiques). Coût : zéro appel LLM.
2. **LLM classifier** — `gpt-4o-mini` utilisé uniquement pour les requêtes ambiguës. Retourne un JSON `{"action": "...", "input": "..."}`.

| Route | Déclenché par | Exemple |
|-------|---------------|---------|
| `rag` | Mots-clés ML / neural / deep learning | `what is backpropagation?` |
| `calculator` | Expressions mathématiques | `calculate sin(pi/2)` |
| `todo` | `add task`, `show tasks`, etc. | `add task: revise chapter 2` |
| `weather` | `weather`, `temperature`, `forecast` | `weather in London` |
| `web` | Actualité, recherches en ligne | `latest LLM news` |
| `chat` | Salutations, conversation libre | `hello!` |

### Partie 4 — Mémoire & Interface

- **Mémoire conversationnelle** : `ConversationBufferWindowMemory` (LangChain, k=6 tours) — une instance pour le RAG, une pour le chat libre.
- **Interface** : Streamlit avec chat interactif, bouton "Clear conversation", spinner de chargement.
- **To-do persistante** : sauvegardée dans `todo_list.json` (survit aux redémarrages).
- **Outil de recherche web** :Tavily intégré, fournissant des résultats structurés avec titre, URL et résumé.
---

## 💬 Exemples d'utilisation

```
📚 Questions RAG :
  → "what is supervised learning?"
  → "explain it more simply"           (suivi de contexte)
  → "give me 3 key points"             (suivi de contexte)
  → "what is the difference between both"

🧮 Calculs :
  → "calculate sqrt(36) + sin(pi/2)"
  → "calculate 12*5+6"
  → "factorial(7)"

📝 To-do list :
  → "add task: revise neural networks"
  → "show tasks"
  → "delete task: 1"
  → "clear tasks"

🌤️ Météo :
  → "what is the weather in Paris?"
  → "weather in Tokyo"

🌐 Recherche web :
  → "ChatGPT updates"
  → "latest machine learning papers 2025"
```

---

## 🔧 Dépendances principales

| Package | Rôle |
|---------|------|
| `langchain` + `langchain-openai` | Framework RAG, agents, mémoire |
| `langchain-community` | Loaders PDF/DOCX, FAISS wrapper |
| `openai` | LLM et embeddings |
| `faiss-cpu` | Base vectorielle locale |
| `streamlit` | Interface web |
| `duckduckgo-search` | Recherche web (gratuite) |
| `requests` | Appels API météo |
| `pypdf` + `docx2txt` | Lecture de documents |

---

## ⚠️ Notes importantes

- Le fichier `.env` (clé API) ne doit **jamais** être commis sur GitHub. Il est dans `.gitignore`.
- Le dossier `venv/` ne doit **pas** être inclus dans le dépôt.
- La base FAISS (`db/`) peut être régénérée avec `python build_db.py` — inutile de la committer.
- Les tâches (`todo_list.json`) sont stockées localement et survivent aux redémarrages de l'app.

---

## 🚀 Déploiement (optionnel)

### Streamlit Cloud

1. Poussez le code sur GitHub (sans `.env`, `venv/`, `db/`)
2. Connectez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Déployez et ajoutez `OPENAI_API_KEY` dans **Settings → Secrets**
4. Lancez `python build_db.py` une première fois via le terminal cloud

---

*Projet réalisé avec LangChain, FAISS, OpenAI et Streamlit — 2026*