import os
import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import Pydantic models from your main file
from schemas import MappedCode, AdjudicationResult

logger = logging.getLogger(__name__)

# Ensure you set your API key in your terminal: export OPENAI_API_KEY="your-key"
# For a hackathon, a cloud LLM is safest for generation, while embeddings remain local.
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini") 

# Re-use the local embeddings for policy search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 1. DUMMY POLICY VECTOR STORE
# ==========================================
def initialize_policy_db():
    """
    Creates a temporary in-memory FAISS database containing our dummy insurance policies.
    """
    dummy_policies = [
        Document(
            page_content="Section 4.1: Comprehensive Metabolic Panels (CPT 80053) are fully covered for preventative care and routine screening without prior authorization.",
            metadata={"payer_id": "AETNA_123", "clause": "Aetna_Policy_2026_Section_4.1"}
        ),
        Document(
            page_content="Section 4.2: Magnetic Resonance Imaging (MRI) of the brain (CPT 70551) requires prior authorization. If no prior authorization is present, the claim must be DENIED.",
            metadata={"payer_id": "AETNA_123", "clause": "Aetna_Policy_2026_Section_4.2"}
        ),
        Document(
            page_content="Section 5.1: Treatments for Essential Hypertension (ICD-10 I10) and Type 2 Diabetes (ICD-10 E11.9) are approved standard care.",
            metadata={"payer_id": "AETNA_123", "clause": "Aetna_Policy_2026_Section_5.1"}
        )
    ]
    return FAISS.from_documents(dummy_policies, embeddings)

policy_vectorstore = initialize_policy_db()

# ==========================================
# 2. THE LANGCHAIN ADJUDICATION PIPELINE
# ==========================================
def policy_adjudication_agent(codes: List[MappedCode], payer_id: str) -> AdjudicationResult:
    """
    Agent 3: RAG against payer policies to adjudicate the claim.
    """
    logger.info(f"Policy Agent evaluating codes for payer {payer_id}...")

    # Guardrail #1: Check confidence scores from the previous agent
    for code in codes:
        if code.confidence_score < 0.85:
            logger.warning(f"Low confidence detected ({code.confidence_score}). Escalating to human.")
            return AdjudicationResult(
                status="HUMAN_REVIEW",
                reasoning=f"Escalated due to low confidence ({code.confidence_score}) when mapping entity: '{code.entity}'.",
                policy_clause_cited="N/A - Internal Guardrail Triggered"
            )

    # Step 1: Retrieve relevant policy clauses for these specific codes
    search_query = " ".join([f"{c.code_type} {c.code}" for c in codes])
    retrieved_docs = policy_vectorstore.similarity_search(search_query, k=2)
    
    policy_context = "\n".join([
        f"[{doc.metadata['clause']}]: {doc.page_content}" 
        for doc in retrieved_docs
    ])

    # Step 2: Set up the strict Pydantic Output Parser
    parser = PydanticOutputParser(pydantic_object=AdjudicationResult)

    # Step 3: Build the Prompt
    prompt = PromptTemplate(
        template="""
        You are an expert Medical Claims Adjudicator AI. 
        Your job is to review the submitted medical codes against the insurance policy context provided and make a final decision.

        Policy Context:
        {context}

        Submitted Codes:
        {codes_list}

        Rules:
        1. If all codes are covered under the policy, status is "APPROVED".
        2. If ANY code violates the policy (e.g., requires missing prior authorization), status is "DENIED".
        3. You MUST cite the exact policy clause used to make your decision.

        {format_instructions}
        """,
        input_variables=["context", "codes_list"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Step 4: Execute the Chain
    formatted_codes = ", ".join([f"{c.code_type} {c.code} ({c.entity})" for c in codes])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": policy_context,
            "codes_list": formatted_codes
        })
        return result
    except Exception as e:
        logger.error(f"LLM Adjudication Failed: {str(e)}")
        # Ultimate Fallback Guardrail
        return AdjudicationResult(
            status="HUMAN_REVIEW",
            reasoning="LLM processing error or hallucination detected during output parsing.",
            policy_clause_cited="System Error Guardrail"
        )