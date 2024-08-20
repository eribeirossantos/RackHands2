import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def extrair_texto_pdf(pdf_docs):
    texto = ""
    for pdf in pdf_docs:
        leitor_pdf = PdfReader(pdf)
        for pagina in leitor_pdf.pages:
            texto += pagina.extract_text()
    return texto

def dividir_texto_em_partes(texto):
    divisor_texto = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    partes = divisor_texto.split_text(texto)
    return partes


def criar_repositorio_vetores(partes_texto):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    repositorio_vetores = FAISS.from_texts(partes_texto, embedding=embeddings)
    repositorio_vetores.save_local("faiss_index")


def criar_cadeia_conversacional():

    template_prompt = """
    Responda à pergunta da maneira mais detalhada possível usando o contexto fornecido, certificando-se de fornecer todos os detalhes. Se a resposta não estiver no
    contexto fornecido, diga "resposta não disponível no contexto", não forneça a resposta errada\n\n
    Contexto:\n {context}?\n
    Pergunta: \n{question}\n

    Resposta:
    """

    modelo = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template=template_prompt, input_variables=["context", "question"])
    cadeia = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)

    return cadeia



def entrada_usuario(pergunta_usuario):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index/index.faiss"  # Specify the full path to the index file

    if not os.path.exists(index_path):
        st.error(f"Index file not found at: {index_path}. Please create the index first.")
        return    
    novo_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = novo_db.similarity_search(pergunta_usuario)

    cadeia = criar_cadeia_conversacional()

    
    resposta = cadeia(
        {"input_documents": docs, "question": pergunta_usuario}
        , return_only_outputs=True)

    print(resposta)
    st.write("Resposta: ", resposta["output_text"])




def main():
    st.set_page_config("Bate-papo PDF")
    st.header("Converse com PDF usando Gemini 💁")

    pergunta_usuario = st.text_input("Faça uma pergunta sobre os arquivos PDF")

    if pergunta_usuario:
        entrada_usuario(pergunta_usuario)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Envie seus arquivos PDF e clique no botão Enviar & Processar", accept_multiple_files=True)
        if st.button("Enviar & Processar"):
            with st.spinner("Processando..."):
                texto_bruto = extrair_texto_pdf(pdf_docs)
                partes_texto = dividir_texto_em_partes(texto_bruto)
                criar_repositorio_vetores(partes_texto)
                st.success("Concluído")



if __name__ == "__main__":
    main()
