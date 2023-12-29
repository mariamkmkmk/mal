import os

import openai
openai.api_key = "..."
os.environ["OPENAI_API_KEY"] = "..."
openai.api_type = "..."
os.environ["OPENAI_API_TYPE"] = "..."
openai.api_base = "..."
os.environ["OPENAI_API_BASE"] = "..."
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from tqdm import tqdm
import time


def create_db(data_folder, persist_directory):
    for filename in os.listdir(data_folder):
         if filename.endswith('.pdf'):
            print(f"reading {filename}...")
            # Construct the full path to the PDF file
            file_path = os.path.join(data_folder, filename)

            # Load the PDF document
            loader = PyPDFLoader(file_path)
            raw_documents = loader.load()

            # Split the text from the document into chunks
            text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
            split_documents = text_splitter.split_documents(raw_documents)


            db = Chroma.from_documents(split_documents, OpenAIEmbeddings(deployment="<your-model>", chunk_size=3, timeout=60, show_progress_bar=True, retry_min_seconds=15), persist_directory=persist_directory)
            print("Sleeping...")
            time.sleep(60)
    return db

data_folder = "files/"
persist_directory = 'db/'
db = create_db(data_folder, persist_directory)

print("Saving your db as pickle file...")
db.persist()
print("Saved!")