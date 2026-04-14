from dotenv import load_dotenv
from src.vector_store import build_vector_store

load_dotenv()

if __name__ == "__main__":
    build_vector_store()
    print("Vector database built successfully.")