import os

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    
def load_document(file):
    name, extension = os.path.splitext(file)
    Loader = ''
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        Loader = PyPDFLoader
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        Loader = Docx2txtLoader
    else:
        print('Document format not supported!')
        return
    print(f'{file}')
    loader = Loader(file)
    data = loader.load()
    return data

def create_embeddings_chroma(chunks, persistent_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_directory)
    return vector_store

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def generate_vector_store(file_path, vector_db_emebedding_prccessor=create_embeddings_chroma):
    data = load_document(file_path)
    chunks = chunk_data(data, chunk_size=256)
    vector_store = vector_db_emebedding_prccessor(chunks)
    return vector_store