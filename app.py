import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
import rag_engine as engine

# ==========================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Support IA Multi-Produits",
    page_icon="🤖",
    layout="wide" # Utilise toute la largeur de l'écran
)

# Chargement des variables secrètes (comme la clé API Groq)
load_dotenv()

# CSS personnalisé pour rendre l'interface plus "Premium"
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .source-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        background-color: #f0f2f6;
        color: #1f77b4;
        font-size: 0.85rem;
        margin: 4px;
        border: 1px solid #d1d5da;
        font-weight: 500;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de l'historique de chat s'il n'existe pas encore
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_vectorstore():
    """Charge la base de données une seule fois et la garde en mémoire."""
    if engine.CHROMA_PATH.exists():
        with st.spinner("Chargement de la base de données..."):
            try:
                return engine.get_vectorstore()
            except Exception:
                return None
    return None

# Chargement de la base de données au démarrage de l'app
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

# ==========================================
# BARRE LATÉRALE (SIDEBAR)
# ==========================================
with st.sidebar:
    # Petit logo ou icône
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.title("⚙️ Configuration")
    
    # Indicateur de connexion à l'API
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("API Groq : Connectée ✅")
    else:
        st.error("API Groq : Non trouvée ❌")
        st.info("Ajoutez votre GROQ_API_KEY dans le fichier .env")

    st.markdown("---")
    
    # SECTION : Gestion des documents
    st.subheader("📚 Documents PDF")
    
    # Widget pour uploader des nouveaux fichiers PDF
    uploaded_files = st.file_uploader("Ajouter des manuels (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if not engine.DATA_DIR.exists():
            engine.DATA_DIR.mkdir(parents=True)
        
        saved_count = 0
        for uploaded_file in uploaded_files:
            file_path = engine.DATA_DIR / uploaded_file.name
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
        
        if saved_count > 0:
            st.info(f"{saved_count} nouveau(x) fichier(s) ajouté(s).")

    # BOUTON : Mise à jour intelligente (Indexation incrémentale)
    if st.button("🔄 Mise à jour intelligente", help="Ajoute uniquement les nouveaux fichiers à la base"):
        with st.spinner("Vérification des documents..."):
            try:
                if not st.session_state.vectorstore:
                    st.session_state.vectorstore = engine.get_vectorstore()
                
                # On regarde ce qui est déjà indexé pour ne pas le refaire
                indexed_files = engine.get_indexed_files(st.session_state.vectorstore)
                all_pdf_paths = list(engine.DATA_DIR.glob("*.pdf"))
                new_pdf_paths = [p for p in all_pdf_paths if p.name not in indexed_files]
                
                if not new_pdf_paths:
                    st.success("Tout est déjà à jour ! ✅")
                else:
                    st.info(f"Analyse de {len(new_pdf_paths)} nouveau(x) fichier(s)...")
                    new_docs = []
                    for pdf_path in new_pdf_paths:
                        from langchain_community.document_loaders import PyPDFLoader
                        loader = PyPDFLoader(str(pdf_path))
                        pages = loader.load()
                        for page in pages:
                            page.metadata["source"] = pdf_path.name
                        new_docs.extend(pages)
                    
                    if new_docs:
                        chunks = engine.split_documents(new_docs)
                        st.session_state.vectorstore.add_documents(chunks)
                        st.success("Indexation terminée !")
                        st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    # BOUTON : Réinitialisation totale
    if st.button("⚠️ Tout réinitialiser", help="Efface tout et recommence l'indexation"):
        with st.spinner("Réinitialisation en cours..."):
            try:
                documents = engine.load_pdf_documents(engine.DATA_DIR)
                if documents:
                    chunks = engine.split_documents(documents)
                    st.session_state.vectorstore = engine.get_vectorstore(chunks=chunks, force_reload=True)
                    st.success("Base réinitialisée !")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")

    st.markdown("---")
    
    # Affichage de la liste des fichiers PDF présents
    st.subheader("📄 Fichiers indexés")
    if engine.DATA_DIR.exists():
        files = list(engine.DATA_DIR.glob("*.pdf"))
        if files:
            for f in files:
                st.caption(f"• {f.name}")
        else:
            st.caption("Aucun document.")
    
    st.markdown("---")
    # Bouton pour vider l'historique des messages
    if st.button("🗑️ Effacer le chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# ZONE DE CHAT PRINCIPALE
# ==========================================
st.title("🤖 Assistant Support Technique")
st.markdown("Posez vos questions techniques. Je cherche la réponse dans les manuels officiels.")

# Affichage de tous les messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si le message a des sources, on les affiche joliment
        if "sources" in message and message["sources"]:
            st.markdown("**Sources :**")
            source_html = "".join([f'<span class="source-tag">📍 {s}</span>' for s in message["sources"]])
            st.markdown(source_html, unsafe_allow_html=True)

# Barre de saisie pour l'utilisateur
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    if not st.session_state.vectorstore:
        st.warning("⚠️ La base de données est vide. Ajoutez des PDF et cliquez sur 'Mise à jour intelligente'.")
    else:
        # 1. On affiche le message de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. On génère la réponse de l'IA
        with st.chat_message("assistant"):
            with st.spinner("Analyse des manuels en cours..."):
                try:
                    # Appel au moteur RAG
                    answer, sources = engine.get_answer(prompt, st.session_state.vectorstore)
                    
                    # Affichage de la réponse texte
                    st.markdown(answer)
                    
                    # Affichage des sources trouvées
                    if sources:
                        st.markdown("**Sources utilisées :**")
                        source_html = "".join([f'<span class="source-tag">📍 {s}</span>' for s in sources])
                        st.markdown(source_html, unsafe_allow_html=True)
                    
                    # On sauvegarde la réponse dans l'historique
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Oups ! Une erreur est survenue : {e}")
