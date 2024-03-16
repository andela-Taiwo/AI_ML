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