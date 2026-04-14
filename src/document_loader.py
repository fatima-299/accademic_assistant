import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_documents(data_path="data"):
    documents = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data folder '{data_path}' does not exist.")

    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        raise ValueError("No PDF or DOCX documents were found in the data folder.")

    return documents