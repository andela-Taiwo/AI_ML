from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(), override=True)

def ask_gemini(text, img, model='gemini-pro-vision'):
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': img}
        ]
    )
    response = llm.invoke([message])

    return response

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')