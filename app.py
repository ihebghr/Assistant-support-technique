import streamlit as st
import os
from rag_engine import RAGEngine
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Assistant Support Multi-Produits", layout="wide")

# Initialize RAG Engine in session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar for configuration and file status
with st.sidebar:
    st.title("Configuration")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY manquante dans .env")
    else:
        st.success("Groq API connectée")

    if st.button("Actualiser la base de documents"):
        with st.spinner("Indexation des documents en cours..."):
            status = st.session_state.rag_engine.load_and_process_pdfs()
            st.info(status)

    st.markdown("---")
    st.markdown("### Documents supportés")
    data_dir = "data/"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
        if files:
            for f in files:
                st.write(f"- {f}")
        else:
            st.write("Aucun PDF dans /data")
    else:
        st.write("Dossier /data introuvable")

# Main UI
st.title("🤖 Assistant Support Technique")
st.markdown("Posez vos questions sur les produits supportés.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la documentation..."):
            response = st.session_state.rag_engine.get_response(prompt)
            answer = response["answer"]
            sources = response["sources"]
            
            full_response = answer
            if sources:
                full_response += "\n\n**Sources :**\n" + "\n".join([f"- {os.path.basename(s)}" for s in sources])
            
            st.markdown(full_response)
            
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
