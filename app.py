import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


class BlogRequest(BaseModel):
    topic: str
    language: str = ""

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic must not be blank")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"", "english", "hindi", "french"}:
            raise ValueError("language must be english, hindi, or french")
        return value


@app.post("/blogs")
async def create_blogs(request: BlogRequest):
    topic = request.topic
    language = request.language

    groqllm = GroqLLM()
    llm = groqllm.get_llm()

    graph_builder = GraphBuilder(llm)

    if language:
        graph = graph_builder.setup_graph(usecase="language")
        state = graph.invoke({"topic": topic, "current_language": language})

    else:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})

    return {"data": state}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
