import streamlit as st
import os
import tempfile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from gtts import gTTS
import yake
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize models and tools
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Function to handle document embedding
def vector_embedding(docs):
    embeddings = NVIDIAEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_documents = text_splitter.split_documents(docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors, final_documents

# Function to convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Function to extract keywords
def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]


# Streamlit UI setup
st.title("Study Easy")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")



uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Embed Documents") and uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        # Load the document from the temporary location
        loader = PyPDFLoader(file_path=temp_file_path)
        docs.extend(loader.load())
        
        # Delete the temporary file after use
        os.remove(temp_file_path)
    
    st.session_state.vectors, st.session_state.final_documents = vector_embedding(docs)
    st.write("Vector Store DB Is Ready")

if prompt1 and 'vectors' in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt1})
    
    # Extracting relevant information
    st.header("Relevant Information")
    st.write("Response:", response['answer'])

    # Converting response to audio
    st.header("Audio")
    speech_file = text_to_speech(response['answer'])
    audio_file = open(speech_file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

   
    for i, doc in enumerate(response["context"]):
        st.write(f"Document {i+1}:")

       
        # Extract and display key terms
        keywords = extract_keywords(doc.page_content)
        st.write("Keywords:", ", ".join(keywords))

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write("--------------------------------")
