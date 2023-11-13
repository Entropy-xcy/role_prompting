from langchain.chat_models import ChatOpenAI
from role_format import format_role_prompting
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]

system_message = format_role_prompting({
    "occupation": "lawyer",
    "education": "Law Degree",
    "gender": "male",
    "age": "mid-aged",
    "nationality": "American"
})
message = system_message + [HumanMessage(content="What are the general procedures of a lawsuit?")]

chat = ChatOpenAI()

response = chat(message)
print(response)
