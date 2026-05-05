import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, data_path="data/", model_name="llama-3.3-70b-versatile"):
        self.data_path = data_path
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm = ChatGroq(
            temperature=0,
            model_name=self.model_name,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def load_and_process_pdfs(self):
        """Load PDFs from data folder and create vector store."""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return "Dossier 'data/' créé. Veuillez y ajouter des fichiers PDF."

        loader = DirectoryLoader(self.data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            return "Aucun document PDF trouvé dans le dossier 'data/'."

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        return f"{len(documents)} documents chargés et indexés."

    def get_response(self, query):
        """Query the RAG system."""
        if not self.vector_store:
            status = self.load_and_process_pdfs()
            if not self.vector_store:
                return {"answer": status, "sources": []}

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "Vous êtes un assistant de support technique expert. "
            "Utilisez les éléments de contexte suivants pour répondre à la question. "
            "Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. "
            "Répondez en français de manière claire et professionnelle. "
            "Cite les sources à la fin de ta réponse."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        
        # Extract unique sources
        sources = list(set([doc.metadata.get("source", "Inconnu") for doc in response["context"]]))
        
        return {
            "answer": response["answer"],
            "sources": sources
        }

if __name__ == "__main__":
    # Test if Groq key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Erreur : GROQ_API_KEY non trouvée dans le fichier .env")
    else:
        engine = RAGEngine()
        print(engine.load_and_process_pdfs())
        # result = engine.get_response("Comment réinitialiser le produit ?")
        # print(result["answer"])
