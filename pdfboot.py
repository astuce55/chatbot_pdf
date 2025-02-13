# Import necessary modules and define env variables

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer  # Utiliser SentenceTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Utiliser FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from langchain_groq import ChatGroq
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

os.environ["GROQ_API_KEY"] = "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"

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Classe personnalisée pour utiliser SentenceTransformers
class CustomEmbeddings(Embeddings):  # Hériter de la classe Embeddings
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        """Encode une liste de textes."""
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        """Encode une seule requête."""
        return self.model.encode([text], convert_to_numpy=True)[0]
# text_splitter and system template

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
give the answer according to the language.for exemple if the question is in french, give the answer in french; if the question is in english, give the answer in english

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

@cl.on_chat_start
async def on_chat_start():

    await cl.Message(content="Hello there, Welcome to Askmesomething related to data").send()
    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    with open(file.path, "rb") as f:
        pdf_stream = BytesIO(f.read())

    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a FAISS vector store
    # Encodage des textes
    embeddings = CustomEmbeddings()
    encoded_texts = embeddings.embed_documents(texts)  # Produit un tableau numpy

    # Création du vectorstore FAISS
    docsearch = FAISS.from_texts(
        texts,
        embedding=embeddings,  # Passer les vecteurs déjà encodés
        metadatas=metadatas
    )


    # Créer une instance du LLM
    llm = ChatGroq(temperature=0)  # Configurez ChatGroq

    # Create a chain that uses the FAISS vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs=chain_type_kwargs,
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    if not chain:
        await cl.Message(content="The chain is not initialized. Please restart the chat!").send()
        return

    logging.debug(f"Received message: {message}")
    logging.debug(f"Chain: {chain}")
    
    try:
        # Utilisation de `ainvoke` au lieu de `acall`
        res = await chain.ainvoke({"question": message.content})

        #print(message.content)
        # Vérifiez la structure de `res` pour ajuster l'accès
        answer = res.get("answer", "No answer available.")
        sources = res.get("sources", "").strip()
        source_elements = []

        # Get the metadata and texts from the user session
        metadatas = cl.user_session.get("metadatas")
        all_sources = [m["source"] for m in metadatas]


        texts = cl.user_session.get("texts")

        if sources:
            found_sources = []

            # Add the sources to the message
            for source in sources.split(","):
                source_name = source.strip().replace(".", "").replace("-pl", "").strip()
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        #print("All sources in metadata:", all_sources)
        #print("Sources returned by the model:", sources)

        await cl.Message(content=answer, elements=source_elements).send()

    except Exception as e:
        logging.error(f"Error during processing: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
