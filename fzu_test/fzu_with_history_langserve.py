import os
import re
os.environ["QIANFAN_AK"] = ""
os.environ["QIANFAN_SK"] = ""

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
# db = FAISS.from_documents(documents,QianfanEmbeddingsEndpoint())
db=FAISS.load_local("faiss/xxgk/xxjj",QianfanEmbeddingsEndpoint(),allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})
from langchain_community.chat_models import QianfanChatEndpoint
model=QianfanChatEndpoint(streaming=True,model="ERNIE-3.5-8K",temperature=0.1)
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)
system_prompt = (

    """
    你的名字是福大灵犀，你知道福州大学的各种信息，请在必要的时候在回答的开头表明你的身份，Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    *Always say 请问还有需要查询的信息吗 at the end of the answer*.

    {context}
    
    """
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt, output_parser=parser)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}
def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


from fastapi import FastAPI, HTTPException
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
add_routes(app, conversational_rag_chain, path="/fzu-chat")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
