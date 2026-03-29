from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime
import uuid
import logging

from pydantic import BaseModel, Field

# 1. Import your models from schemas.py
from schemas import ClaimInput, ExtractedEntities, MappedCode, AdjudicationResult, AuditLog

# 2. Import your agent functions
from rag_agent import get_mapped_codes
from policy_agent import policy_adjudication_agent

# Initialize App and Logging
app = FastAPI(title="ClinGuard AI: Autonomous Healthcare Claims Agent")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. PYDANTIC MODELS (Strict Data Guardrails)
# ==========================================

# class ClaimInput(BaseModel):
#     patient_id: str
#     encounter_text: str = Field(..., description="Raw clinical note or OCR extracted text")
#     payer_id: str

# class ExtractedEntities(BaseModel):
#     diagnoses: List[str]
#     procedures: List[str]

# class MappedCode(BaseModel):
#     entity: str
#     code: str
#     code_type: str # "ICD-10" or "CPT"
#     confidence_score: float
#     source_document: str

# class AdjudicationResult(BaseModel):
#     status: str # "APPROVED", "DENIED", or "HUMAN_REVIEW"
#     reasoning: str
#     policy_clause_cited: Optional[str] = None

# class AuditLog(BaseModel):
#     transaction_id: str
#     timestamp: datetime
#     input_data: ClaimInput
#     extracted_entities: ExtractedEntities
#     mapped_codes: List[MappedCode]
#     final_decision: AdjudicationResult

# ==========================================
# 2. MOCK AGENT FUNCTIONS (To be wired with LangChain/HF)
# ==========================================

def extraction_agent(text: str) -> ExtractedEntities:
    """Agent 1: Extracts medical entities from unstructured text."""
    # TODO: Replace with LangChain Extraction Chain
    logger.info("Extraction Agent processing text...")
    return ExtractedEntities(
        diagnoses=["Type 2 Diabetes Mellitus", "Essential Hypertension"],
        procedures=["Comprehensive Metabolic Panel"]
    )

def coding_guardrail_agent(entities: ExtractedEntities) -> List[MappedCode]:
    """Agent 2: Strict RAG mapping to ICD-10/CPT vector database."""
    logger.info("Coding Agent mapping entities via FAISS similarity search...")
    return get_mapped_codes(entities)

# def coding_guardrail_agent(entities: ExtractedEntities) -> List[MappedCode]:
#     """Agent 2: Strict RAG mapping to ICD-10/CPT vector database."""
#     # TODO: Replace with HuggingFace embeddings search against Vector DB
#     logger.info("Coding Agent mapping entities to official codes...")
#     return [
#         MappedCode(
#             entity="Type 2 Diabetes Mellitus", 
#             code="E11.9", 
#             code_type="ICD-10", 
#             confidence_score=0.98,
#             source_document="icd10_cm_2026.pdf_page_42"
#         ),
#         MappedCode(
#             entity="Comprehensive Metabolic Panel", 
#             code="80053", 
#             code_type="CPT", 
#             confidence_score=0.95,
#             source_document="cpt_2026.pdf_page_112"
#         )
#     ]


# def policy_adjudication_agent(codes: List[MappedCode], payer_id: str) -> AdjudicationResult:
#     """Agent 3: RAG against payer policies to approve/deny/escalate."""
#     logger.info(f"Policy Agent cross-referencing codes for payer {payer_id}...")
#     return policy_adjudication_agent(codes, payer_id)

# def policy_adjudication_agent(codes: List[MappedCode], payer_id: str) -> AdjudicationResult:
#     """Agent 3: RAG against payer policies to approve/deny/escalate."""
#     # TODO: Replace with LangChain QA Chain against Payer Policy PDFs
#     logger.info(f"Policy Agent cross-referencing codes for payer {payer_id}...")
    
#     # Edge-Case Guardrail: Check confidence scores
#     for code in codes:
#         if code.confidence_score < 0.85:
#             return AdjudicationResult(
#                 status="HUMAN_REVIEW",
#                 reasoning=f"Low confidence ({code.confidence_score}) mapping for entity: {code.entity}.",
#             )

#     return AdjudicationResult(
#         status="APPROVED",
#         reasoning="All procedures are medically necessary for the primary diagnosis under the current policy.",
#         policy_clause_cited="Aetna_Policy_2026_Section_4.2"
#     )

def log_audit_trail(audit_data: AuditLog):
    """Audit Engine: Saves the immutable JSON trail."""
    # In a real app, this writes to a database.
    logger.info(f"AUDIT TRAIL SAVED FOR TXN: {audit_data.transaction_id}")
    # print(audit_data.model_dump_json(indent=2))

# ==========================================
# 3. FASTAPI ROUTER / ENDPOINTS
# ==========================================

@app.post("/api/v1/process-claim", response_model=AuditLog)
async def process_claim(claim: ClaimInput, background_tasks: BackgroundTasks):
    """
    Main orchestration endpoint. 
    Passes data through the multi-agent pipeline and generates an auditable JSON trail.
    """
    transaction_id = f"txn_{uuid.uuid4().hex[:8]}"
    logger.info(f"Starting pipeline for transaction {transaction_id}")

    try:
        # Step 1: Extract
        entities = extraction_agent(claim.encounter_text)
        
        # Step 2: Map to Codes (Compliance Guardrail 1)
        mapped_codes = coding_guardrail_agent(entities)
        
        # Step 3: Adjudicate against Policy (Compliance Guardrail 2)
        decision = policy_adjudication_agent(mapped_codes, claim.payer_id)
        
        # Step 4: Construct Audit Payload
        audit_payload = AuditLog(
            transaction_id=transaction_id,
            timestamp=datetime.utcnow(),
            input_data=claim,
            extracted_entities=entities,
            mapped_codes=mapped_codes,
            final_decision=decision
        )

        # Step 5: Save Audit Log asynchronously 
        background_tasks.add_task(log_audit_trail, audit_payload)

        return audit_payload

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Agent Processing Error")

@app.get("/health")
def health_check():
    return {"status": "operational", "agents": ["extraction", "coding", "policy"]}