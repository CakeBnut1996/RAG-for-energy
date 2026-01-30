from pydantic import BaseModel, Field
from generation_utils.llm_client import LLMClient

# === DATA MODELS ===

class FaithfulnessResult(BaseModel):
    score: float = Field(..., description="0.0 to 1.0 (0=Hallucination, 1=Grounded)")
    reasoning: str = Field(..., description="Evidence of hallucination or grounding")

class AbstentionResult(BaseModel):
    is_correct_refusal: bool = Field(..., description="True if refusal was appropriate")
    reasoning: str = Field(..., description="Why the refusal was correct/incorrect")

# === THE UNIVERSAL JUDGE ===

class UniversalJudge:
    def __init__(self, provider: str, model_name: str):
        """
        The Judge uses the LLMClient for structured output.
        """
        self.llm = LLMClient(provider, model_name)


    def evaluate_faithfulness(self, query: str, context: str, answer: str) -> FaithfulnessResult:
        prompt = f"""
        JUDGE TASK: Faithfulness
        Context: "{context[:10000]}"
        Generated Answer: "{answer}"

        Does the Answer contain info NOT in Context? (1.0 = Fully Grounded, 0.0 = Hallucinated).
        """
        return self.llm.generate_structured(prompt, FaithfulnessResult)

    def evaluate_abstention(self, query: str, context: str, answer: str) -> AbstentionResult:
        prompt = f"""
        JUDGE TASK: Abstention
        User Question: {query}
        Context (Irrelevant): "{context[:5000]}"
        Generated Answer: "{answer}"

        Did the model correctly refuse to answer? (True/False).
        """
        return self.llm.generate_structured(prompt, AbstentionResult)