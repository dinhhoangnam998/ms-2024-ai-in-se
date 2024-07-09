from fastapi import FastAPI

from rag_chain import conversational_rag_chain, invoke_and_save

app = FastAPI()


@app.get("/chat")
async def chat(question: str, session_id: str):
    # return conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": session_id}}, )
    return invoke_and_save(session_id, question)