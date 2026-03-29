import os
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# You can import the Pydantic models from your main file
from schemas import ExtractedEntities, MappedCode 

logger = logging.getLogger(__name__)

# Initialize the local Hugging Face embedding model (fast and lightweight)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

DB_PATH = "./faiss_medical_db"

def initialize_vector_db():
    """
    Creates a dummy local FAISS database with some ICD-10 and CPT codes if it doesn't exist.
    In a real scenario, you'd ingest a massive CSV of official medical codes here.
    """
    if os.path.exists(DB_PATH):
        logger.info("Loading existing FAISS medical vector database...")
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    logger.info("Initializing new FAISS medical vector database with dummy codes...")
    
    # Dummy medical code dataset
    documents = [
        Document(page_content="Type 2 Diabetes Mellitus without complications", metadata={"code": "E11.9", "type": "ICD-10", "source": "icd10_cm_2026.pdf_page_42"}),
        Document(page_content="Essential (primary) hypertension", metadata={"code": "I10", "type": "ICD-10", "source": "icd10_cm_2026.pdf_page_89"}),
        Document(page_content="Comprehensive metabolic panel (CMP) blood test", metadata={"code": "80053", "type": "CPT", "source": "cpt_2026.pdf_page_112"}),
        Document(page_content="Magnetic resonance imaging (MRI) of the brain", metadata={"code": "70551", "type": "CPT", "source": "cpt_2026.pdf_page_205"})
    ]
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(DB_PATH)
    return vectorstore

# Load the DB globally when the module starts
vectorstore = initialize_vector_db()

def get_mapped_codes(entities: ExtractedEntities) -> List[MappedCode]:
    """
    Agent 2 Logic: Takes extracted entities and queries the local vector DB for the closest medical code.
    Enforces Guardrail #1: Strict mapping based on semantic similarity.
    """
    results = []
    
    # Process Diagnoses (ICD-10)
    for diagnosis in entities.diagnoses:
        # Perform similarity search; FAISS returns (Document, L2 distance score)
        # Lower L2 distance means higher similarity. We convert it to a pseudo-confidence score.
        docs_and_scores = vectorstore.similarity_search_with_score(diagnosis, k=1)
        
        if docs_and_scores:
            doc, distance = docs_and_scores[0]
            # Convert L2 distance to a 0-1 confidence scale (approximate heuristic for this embedding model)
            confidence = max(0.0, 1.0 - (distance / 2.0)) 
            
            results.append(MappedCode(
                entity=diagnosis,
                code=doc.metadata.get("code", "UNKNOWN"),
                code_type=doc.metadata.get("type", "UNKNOWN"),
                confidence_score=round(confidence, 2),
                source_document=doc.metadata.get("source", "UNKNOWN")
            ))

    # Process Procedures (CPT)
    for procedure in entities.procedures:
        docs_and_scores = vectorstore.similarity_search_with_score(procedure, k=1)
        
        if docs_and_scores:
            doc, distance = docs_and_scores[0]
            confidence = max(0.0, 1.0 - (distance / 2.0))
            
            results.append(MappedCode(
                entity=procedure,
                code=doc.metadata.get("code", "UNKNOWN"),
                code_type=doc.metadata.get("type", "UNKNOWN"),
                confidence_score=round(confidence, 2),
                source_document=doc.metadata.get("source", "UNKNOWN")
            ))

    return results