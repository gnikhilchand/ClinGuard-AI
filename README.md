# ClinGuard AI 🏥🛡️
**Autonomous Healthcare Claims Agent with Strict Compliance Guardrails**

*Built for the ET Gen AI Hackathon 2026 - Problem Statement 5 (Domain-Specialized AI Agents with Compliance Guardrails)*

## 📖 Overview
ClinGuard AI is a multi-agent system designed to automate the highly regulated process of healthcare claims adjudication. By utilizing a Retrieval-Augmented Generation (RAG) architecture and strict deterministic guardrails, the system processes unstructured clinical notes, maps them to official medical codes (ICD-10/CPT), and adjudicates them against payer policies with 100% auditable reasoning.

### The Problem It Solves
Human medical coders take an average of 15 minutes per claim, with a 15% error rate leading to delayed revenue and compliance fines. ClinGuard AI reduces processing time to under 10 seconds per claim while strictly enforcing payer policies to prevent upcoding, downcoding, and hallucinations.

## 🏗️ Multi-Agent Architecture
This system utilizes a localized orchestration layer powered by FastAPI, coordinating three specialized agents:

1. **Extraction Agent:** Parses unstructured clinical encounter text to identify diagnoses and procedures.
2. **Coding Guardrail Agent (The Mapper):** Uses local Hugging Face embeddings (`all-MiniLM-L6-v2`) and FAISS vector search to semantically map entities to official ICD-10 and CPT codes. 
    * *Guardrail:* Extracts L2 distance scores. If the mapping confidence is below 85%, it triggers a Human-in-the-Loop escalation to prevent hallucinated codes.
3. **Policy Adjudication Agent (The Judge):** Uses a LangChain pipeline and `gpt-4o-mini` to cross-reference the mapped codes against a vector database of payer policies. 
    * *Guardrail:* Uses `PydanticOutputParser` to force the LLM into a strict JSON schema, requiring it to cite the exact policy clause used for its decision.

## ⚙️ Tech Stack
* **Orchestration:** FastAPI, Python 3.11
* **AI/LLM Framework:** LangChain, OpenAI (`gpt-4o-mini`)
* **Vector Database:** FAISS (Local, In-Memory)
* **Embeddings:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Data Validation:** Pydantic

## 🚀 Quick Setup & Installation

## 1. Clone the repository**
```bash
git clone https://github.com/gnikhilchand/ClinGuard-AI.git
cd clinguard-ai
```
## 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install fastapi uvicorn pydantic langchain langchain-openai langchain-community langchain-huggingface sentence-transformers faiss-cpu python-dotenv
```
## 3. Set your Environment Variables
Create a .env file in the root directory and add your OpenAI key:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```
## 4. Run the Application
```bash
uvicorn main:app --reload
```
## 🧪 Testing the API
Once the server is running, navigate to the Swagger UI at http://127.0.0.1:8000/docs.

Endpoint: POST /api/v1/process-claim

Sample Request Payload:
```bash
{
  "patient_id": "P-98765",
  "encounter_text": "Patient presents with elevated blood sugar. Diagnosed with Type 2 Diabetes Mellitus without complications. Ordered a Comprehensive metabolic panel (CMP) blood test.",
  "payer_id": "AETNA_123"
}
```
The system will return a fully auditable JSON trail detailing the extracted entities, the vector-mapped medical codes, the confidence scores, the final decision (APPROVED/DENIED/HUMAN_REVIEW), and the exact policy clause cited.
