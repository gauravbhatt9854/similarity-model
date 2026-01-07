from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import math

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI(title="Task Relevance ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD NLP MODEL
# -----------------------------
# Sentence Embedding Model (Pretrained)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# DATA MODELS
# -----------------------------
class Location(BaseModel):
    latitude: float
    longitude: float


class SimilarityRequest(BaseModel):
    # USER
    user_skills: List[str]
    user_location: Optional[Location]

    # TASK
    task_title: str
    task_description: str
    task_location: Location
    task_priority: str
    task_deadline: datetime


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geo-points (km)
    """
    R = 6371  # Earth radius (km)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def priority_weight(priority: str):
    """
    Convert priority to numeric weight
    """
    return {
        "Low": 0.2,
        "Medium": 0.5,
        "High": 1.0,
    }.get(priority, 0.5)


def deadline_urgency(deadline: datetime):
    """
    Calculate urgency based on deadline
    """
    days_left = (deadline - datetime.utcnow()).days

    if days_left <= 1:
        return 1.0
    elif days_left <= 3:
        return 0.8
    elif days_left <= 7:
        return 0.5
    else:
        return 0.2


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "ML API running"}


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/similarity")
def similarity(data: SimilarityRequest):
    """
    Final relevance score using:
    - Skill similarity (ML)
    - Location distance
    - Task priority
    - Deadline urgency
    """

    # -----------------------------
    # 1️⃣ SKILL SIMILARITY (ML)
    # -----------------------------
    skills_text = " ".join(data.user_skills)
    task_text = f"{data.task_title}. {data.task_description}"

    user_embedding = model.encode([skills_text])
    task_embedding = model.encode([task_text])

    skill_score = cosine_similarity(
        user_embedding, task_embedding
    )[0][0]  # 0 → 1

    # -----------------------------
    # 2️⃣ LOCATION SCORE
    # -----------------------------
    if data.user_location:
        distance_km = haversine(
            data.user_location.latitude,
            data.user_location.longitude,
            data.task_location.latitude,
            data.task_location.longitude,
        )
        # Tasks within 50km preferred
        distance_score = max(0.0, 1 - (distance_km / 50))
    else:
        distance_score = 0.3  # default if no location

    # -----------------------------
    # 3️⃣ PRIORITY SCORE
    # -----------------------------
    priority_score = priority_weight(data.task_priority)

    # -----------------------------
    # 4️⃣ DEADLINE URGENCY SCORE
    # -----------------------------
    deadline_score = deadline_urgency(data.task_deadline)

    # -----------------------------
    # 🎯 FINAL WEIGHTED SCORE
    # -----------------------------
    final_score = (
        skill_score * 0.50 +
        distance_score * 0.20 +
        priority_score * 0.15 +
        deadline_score * 0.15
    )

    return {
        "score": round(final_score * 100, 2),
        "details": {
            "skill_similarity": round(skill_score * 100, 2),
            "distance_score": round(distance_score * 100, 2),
            "priority_score": round(priority_score * 100, 2),
            "deadline_score": round(deadline_score * 100, 2),
        },
    }
