from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage

import asyncio

from typing_extensions import List, AsyncGenerator

from agent import build_agent
from embed_data import load_uploaded_pdfs, embed_docs, clear_all_pgvector_data
from pydantic import BaseModel

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"]
)

class RequestModel(BaseModel):
    question:str
    temperature:float
    max_retrievals:int

class ResponseModel(BaseModel):
    response:str

agent = build_agent()

@app.post("/upload")
async def upload(files:List[UploadFile] = File(...)):
    docs = await load_uploaded_pdfs(files)
    embed_docs(docs)
    return {"message" : "Document uploaded succesfully."}

@app.post("/query")
async def stream_query(req: RequestModel):
    current_state = {
        "messages": [HumanMessage(content=req.question)],
        "temperature": req.temperature,
        "num_results": req.max_retrievals
    }

    async def token_stream() -> AsyncGenerator[str, None]:
        async for step in agent.astream(
            current_state,
            config={
                "configurable": {
                    "thread_id": "user123",
                    "temperature": req.temperature,
                    "num_results": req.max_retrievals
                }
            },
            stream_mode="values"
        ):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                yield msg.content

    return StreamingResponse(token_stream(), media_type="text/plain")
    

    


