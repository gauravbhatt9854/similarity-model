from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Skill Task Similarity API")

# Allow all origins (testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/")
def root():
    return {"status": "ok", "message": "API running"}


@app.get("/ping")
def ping():
    return {"ping": "pong"}


class SimilarityRequest(BaseModel):
    user_skills: list[str]
    task_title: str
    task_description: str


@app.post("/similarity")
def similarity(data: SimilarityRequest):
    skills_text = " ".join(data.user_skills)
    task_text = f"{data.task_title}. {data.task_description}"

    u_emb = model.encode([skills_text])
    t_emb = model.encode([task_text])

    score = float(cosine_similarity(u_emb, t_emb)[0][0])
    return {"score": round(score * 100, 2)}

