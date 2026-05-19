import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# CONFIGURATION DU SYSTÈME RAG
# ==========================================
DATA_DIR = Path("data")             # Dossier où sont stockés les PDF
CHROMA_PATH = Path("chroma_db")      # Dossier de la base de données vectorielle
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Modèle qui transforme le texte en chiffres
GROQ_MODEL = "llama-3.3-70b-versatile" # Le cerveau de l'IA (LLM)
CHUNK_SIZE = 1000                   # Réduit légèrement pour avoir plus de morceaux différents
CHUNK_OVERLAP = 200                 # Chevauchement entre les morceaux
TOP_K = 10                          # Augmenté à 10 pour capturer plus d'informations différentes

def load_pdf_documents(data_dir: Path):
    """
    Étape 1 : Chargement des fichiers PDF.
    Lit tous les PDF dans le dossier 'data' et extrait leur contenu textuel.
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        return []
        
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        return []
    
    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                # Nettoyage : enlève les caractères bizarres souvent trouvés dans les PDF
                page.page_content = page.page_content.replace('\x00', '')
                # On garde le nom du fichier dans les métadonnées pour savoir d'où vient l'info
                page.metadata["source"] = pdf_path.name
            documents.extend(pages)
        except Exception as e:
            print(f"Erreur lors du chargement de {pdf_path.name}: {e}")
    return documents

def split_documents(documents):
    """
    Étape 2 : Découpage (Chunking).
    Découpe les longs textes en petits morceaux plus faciles à analyser pour l'IA.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # On essaie de couper en priorité sur les doubles sauts de ligne, puis les points, etc.
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)

import streamlit as st

@st.cache_resource
def get_embedding_model():
    """
    Charge le modèle d'embeddings et le garde en cache pour éviter de le recharger.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def get_vectorstore(chunks=None, force_reload=False):
    """
    Étape 3 : Base de données Vectorielle.
    Transforme les morceaux de texte en vecteurs (embeddings) et les stocke dans ChromaDB.
    """
    embeddings = get_embedding_model()
    
    # Si on veut tout recommencer à zéro (Reset)
    if force_reload and CHROMA_PATH.exists():
        import shutil
        import time
        for i in range(3):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except Exception:
                time.sleep(1)
        
    if chunks:
        # Création d'une nouvelle base à partir des morceaux de texte
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PATH)
        )
    else:
        # Chargement de la base existante sur le disque
        return Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )

def get_indexed_files(vectorstore):
    """
    Utile pour l'indexation incrémentale : 
    Récupère la liste des noms de fichiers PDF qui sont déjà dans la base.
    """
    if not vectorstore:
        return set()
    try:
        results = vectorstore.get()
        if not results or "metadatas" not in results:
            return set()
        return {m["source"] for m in results["metadatas"] if "source" in m}
    except Exception:
        return set()

def build_prompt():
    """
    Étape 4 : Le Prompt (Les instructions).
    Définit comment l'IA doit se comporter et utiliser les documents.
    """
    template = """Vous êtes l'Assistant Technique Expert. 
Votre mission est d'extraire et de synthétiser TOUTES les informations utiles trouvées dans les documents fournis.

RÈGLES DE RÉPONSE :
1. EXHAUSTIVITÉ : Ne vous contentez pas d'une réponse courte. Listez toutes les caractéristiques, spécifications techniques, composants et instructions trouvés dans le contexte.
2. STRUCTURE : Utilisez des titres, des listes à puces et des tableaux si nécessaire pour organiser les informations.
3. FIDÉLITÉ : Basez-vous uniquement sur les extraits fournis. Si une information spécifique manque, mentionnez ce qui est disponible.
4. ABSENCE TOTALE D'INFO : Si vraiment aucun extrait ne mentionne le sujet, dites : "Désolé, je ne trouve pas d'informations sur ce sujet dans les documents indexés."
5. STYLE : Professionnel et détaillé.

CONTEXTE TECHNIQUE :
{context}

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE DÉTAILLÉE DE L'EXPERT :
"""
    return PromptTemplate.from_template(template)

def format_context(docs):
    """
    Formate les morceaux de texte trouvés pour qu'ils soient lisibles par l'IA dans le prompt.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Document inconnu")
        page = doc.metadata.get("page", "?")
        if isinstance(page, int): page += 1
        content = doc.page_content.strip()
        # On ajoute des balises claires pour que l'IA sache d'où vient chaque info
        parts.append(f"--- EXTRAIT {i} | SOURCE: {source} | PAGE: {page} ---\n{content}")
    return "\n\n".join(parts)

def get_answer(question, vectorstore):
    """
    Étape 5 : La recherche et la génération de réponse.
    C'est ici que tout se rejoint.
    """
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La clé API Groq est manquante dans le fichier .env")

    # Configuration de l'IA Groq
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1, # 0.1 = réponse très factuelle et peu créative (mieux pour la technique)
        api_key=api_key,
        max_retries=2
    )
    
    # On demande à la base de données de trouver les morceaux les plus "similaires" à la question
    # Utilisation de search_type="mmr" pour plus de diversité dans les extraits récupérés
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": 20, # Analyse 20 documents pour en choisir 10 diversifiés
            "lambda_mult": 0.5 # Équilibre entre pertinence et diversité
        }
    )
    
    relevant_docs = retriever.invoke(question)
    
    if not relevant_docs:
        return "Désolé, je n'ai trouvé aucun document pertinent pour répondre à votre question.", []

    # On prépare le contexte et le prompt final
    context = format_context(relevant_docs)
    prompt_template = build_prompt()
    final_prompt = prompt_template.format(context=context, question=question)
    
    # On envoie tout à l'IA et elle génère la réponse
    response = llm.invoke(final_prompt)
    
    # On extrait les sources pour les afficher dans l'interface
    sources_set = set()
    for doc in relevant_docs:
        s = doc.metadata.get("source", "Inconnue")
        p = doc.metadata.get("page", "?")
        if isinstance(p, int): p += 1
        sources_set.add(f"{s} (p.{p})")
    
    return response.content, sorted(list(sources_set))
