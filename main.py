from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import numpy as np

app = FastAPI(title="Task Recommendation AI Model")

# 🔥 Research-grade SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ===================== INPUT SCHEMA =====================

class SimilarityRequest(BaseModel):
    user_skills: list[str]
    task_title: str
    task_description: str
    priority: str
    urgency: bool
    deadline: str   # ISO string


# ===================== WEIGHT FUNCTIONS =====================

def priority_boost(priority: str) -> float:
    if priority == "High":
        return 0.10
    if priority == "Medium":
        return 0.05
    return 0.02


def urgency_boost(urgency: bool) -> float:
    return 0.15 if urgency else 0.0


def deadline_boost(deadline: str) -> float:
    try:
        deadline_dt = datetime.fromisoformat(deadline.replace("Z", ""))
        hours_left = (deadline_dt - datetime.utcnow()).total_seconds() / 3600

        if hours_left <= 24:
            return 0.15
        if hours_left <= 72:
            return 0.08
        return 0.03
    except:
        return 0.0


# ===================== CORE AI LOGIC =====================

@app.post("/similarity")
def calculate_similarity(req: SimilarityRequest):
    """
    Research-grade semantic similarity using SBERT + contextual boosts
    """

    # 1️⃣ Combine user skills into one semantic sentence
    user_text = " ".join(req.user_skills)

    # 2️⃣ Combine task context
    task_text = f"{req.task_title}. {req.task_description}"

    # 3️⃣ Generate embeddings
    user_embedding = model.encode([user_text])
    task_embedding = model.encode([task_text])

    # 4️⃣ Cosine similarity (semantic match)
    semantic_score = cosine_similarity(user_embedding, task_embedding)[0][0]

    # 5️⃣ Context-aware boosting
    score = float(semantic_score)
    score += priority_boost(req.priority)
    score += urgency_boost(req.urgency)
    score += deadline_boost(req.deadline)

    # 6️⃣ Normalize (important for ranking)
    score = min(score, 1.0)

    return {
        "score": round(score, 4),
        "semantic_similarity": round(float(semantic_score), 4),
        "model": "Sentence-BERT (all-MiniLM-L6-v2)"
    }
