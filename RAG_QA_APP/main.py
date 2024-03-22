import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    cost = total_tokens / 1000 * 0.0004
    return total_tokens, cost
    
def load_document(file):
    name, extension = os.path.splitext(file)
    loader = ''
    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format not supported!')
        return
    print(f'{file}')
    return loader.load()

def create_embeddings_chroma(chunks, persistent_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persistent_directory)
    return vector_store

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def generate_vector_store(file_path, vector_db_emebedding_prccessor=create_embeddings_chroma):
    data = load_document(file_path)
    chunks = chunk_data(data, chunk_size=256)
    vector_store = vector_db_emebedding_prccessor(chunks)
    return vector_store

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = chroma.Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, question, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(question)
    return answer

def clear_history():
    if 'history' in  st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
        
    st.image('img.png')
    st.subheader('LLM Question-Answering Application')
    with st.sidebar:
        api_key = st.text_input('OpenAI API key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        
        uploaded_file = st.file_uploader('Upload a file', type=['pdf', 'docx', 'txt'], on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=512, on_change=clear_history)
        add_data = st.button('Submit', on_click=clear_history)
        
        
        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('../files', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, chunks: {len(chunks)}')
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                
                vector_store = create_embeddings(chunks)
                
                st.session_state.vs = vector_store
                st.success('File Uploaded, chunked and embedded successfully.')
        
    question = st.text_input('Ask a question about the content of your file: ')
    answer = ''
    if question:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, question, k)
            st.text_area('LLM Answer: ', value=answer)
            
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            
            value = f'Q: {question} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)