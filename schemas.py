# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ClaimInput(BaseModel):
    patient_id: str
    encounter_text: str = Field(..., description="Raw clinical note or OCR extracted text")
    payer_id: str

class ExtractedEntities(BaseModel):
    diagnoses: List[str]
    procedures: List[str]

class MappedCode(BaseModel):
    entity: str
    code: str
    code_type: str # "ICD-10" or "CPT"
    confidence_score: float
    source_document: str

class AdjudicationResult(BaseModel):
    status: str # "APPROVED", "DENIED", or "HUMAN_REVIEW"
    reasoning: str
    policy_clause_cited: Optional[str] = None

class AuditLog(BaseModel):
    transaction_id: str
    timestamp: datetime
    input_data: ClaimInput
    extracted_entities: ExtractedEntities
    mapped_codes: List[MappedCode]
    final_decision: AdjudicationResult