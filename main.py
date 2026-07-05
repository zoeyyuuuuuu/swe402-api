from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, json

app = FastAPI(
    title="AI Job Threat Predictor",
    description="Predicts AI automation threat level for a given job role and provides actionable recommendations.",
    version="1.0",
    docs_url="/docs",
    swagger_ui_parameters={"syntaxHighlight": False}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model artifacts once at startup (not on every request)
model     = joblib.load("model_artifacts/rf_model.joblib")
scaler    = joblib.load("model_artifacts/scaler.joblib")
le_domain = joblib.load("model_artifacts/le_domain.joblib")
le_threat = joblib.load("model_artifacts/le_threat.joblib")

with open("model_artifacts/domain_classes.json") as f:
    domain_classes = json.load(f)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class JobInput(BaseModel):
    tasks: int
    ai_models: int
    ai_workload_ratio: float
    domain: str

class BatchInput(BaseModel):
    jobs: list[JobInput]

# ── Helper: build prediction result for one job ───────────────────────────────

def run_prediction(job: JobInput) -> dict:
    # Input validation with meaningful error messages
    if job.tasks <= 0:
        raise HTTPException(status_code=400, detail="Tasks must be greater than 0")
    if job.ai_models <= 0:
        raise HTTPException(status_code=400, detail="AI Models must be greater than 0")
    if not (0 < job.ai_workload_ratio < 1):
        raise HTTPException(status_code=400, detail="AI Workload Ratio must be between 0 and 1")

    # Encode domain
    try:
        domain_enc = le_domain.transform([job.domain])[0]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{job.domain}'. Call /domains to see valid values."
        )

    # Scale features and run prediction
    features        = np.array([[job.tasks, job.ai_models, job.ai_workload_ratio, domain_enc]])
    features_scaled = scaler.transform(features)
    pred_enc        = model.predict(features_scaled)[0]
    pred_proba      = model.predict_proba(features_scaled)[0]
    threat          = le_threat.inverse_transform([pred_enc])[0]

    proba_dict = {
        le_threat.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(pred_proba)
    }

    # Detailed recommendations per threat level
    recommendations = {
        "High": {
            "summary": (
                "This job faces HIGH risk of AI automation. "
                "Immediate action is strongly recommended."
            ),
            "actions": [
                "Enrol in reskilling programs focused on AI-complementary skills such as "
                "critical thinking, creativity, and emotional intelligence.",
                "Transition toward roles that require human judgment, leadership, and "
                "interpersonal skills which AI cannot replicate.",
                "Learn to work alongside AI tools (e.g. prompt engineering, AI supervision) "
                "to remain relevant in an AI-augmented workplace.",
                "Consider upskilling in data literacy, digital skills, or domain expertise "
                "to move into higher-value roles.",
                "HR departments should prioritise this job category for workforce "
                "transformation planning immediately."
            ],
            "urgency": "Immediate — act within 6 months"
        },
        "Medium": {
            "summary": (
                "This job faces MODERATE risk of AI automation. "
                "Proactive upskilling is recommended."
            ),
            "actions": [
                "Identify which specific tasks within this role are most automatable and "
                "focus on tasks that require human judgement.",
                "Upskill in areas such as problem-solving, communication, and stakeholder "
                "management to differentiate from automated alternatives.",
                "Explore AI tools relevant to this domain to improve personal productivity "
                "and demonstrate adaptability to employers.",
                "Consider cross-training into adjacent roles that carry lower automation risk.",
                "Monitor developments in this domain and reassess the risk level annually."
            ],
            "urgency": "Moderate — plan within 12 months"
        },
        "Low": {
            "summary": (
                "This job faces LOW risk of AI automation. "
                "Continue developing domain expertise."
            ),
            "actions": [
                "Continue building deep domain expertise as human specialisation remains "
                "a key differentiator against automation.",
                "Stay informed about AI advancements in your field and leverage them as "
                "productivity tools rather than viewing them as threats.",
                "Focus on developing soft skills such as leadership, creativity, and "
                "complex decision-making that AI currently cannot perform.",
                "Mentor others in your domain to strengthen professional value and "
                "organisational visibility.",
                "Periodically reassess as AI capabilities continue to evolve rapidly."
            ],
            "urgency": "Low — monitor annually"
        }
    }

    return {
        "threat_level":   threat,
        "confidence":     round(float(pred_proba.max()), 4),
        "probabilities":  proba_dict,
        "recommendation": recommendations.get(threat, {}),
        "input":          job.dict()
    }

# ── Endpoints ─────────────────────────────────────────────────────────────────

# GET / — health check
@app.get("/")
def root():
    return {
        "message": "AI Job Threat Predictor API is running",
        "version": "1.0",
        "endpoints": {
            "health":        "GET /",
            "domains":       "GET /domains",
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch"
        }
    }

# GET /domains — list all valid domain values
@app.get("/domains")
def get_domains():
    return {
        "domains": domain_classes,
        "count":   len(domain_classes)
    }

# POST /predict — predict threat level for a single job
@app.post("/predict")
def predict(job: JobInput):
    return run_prediction(job)

# POST /predict/batch — predict threat level for multiple jobs at once
@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    results = [run_prediction(job) for job in batch.jobs]
    summary = {
        "High":   sum(1 for r in results if r["threat_level"] == "High"),
        "Medium": sum(1 for r in results if r["threat_level"] == "Medium"),
        "Low":    sum(1 for r in results if r["threat_level"] == "Low"),
    }
    return {
        "total":   len(results),
        "summary": summary,
        "results": results
    }