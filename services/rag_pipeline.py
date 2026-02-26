import re

from config import LLM_MAX_NEW_TOKENS, LLM_MODEL_NAME, LLM_TEMPERATURE
from services.image_analyzer import analyze_image_symptoms
from services.prompt_builder import build_rag_prompt
from services.safety_guardrails import get_safety_notes
from services.vector_store import SearchResult, load_vector_store

_generator = None
_generator_task = None


def _extract_entry_parts(text: str) -> dict[str, str]:
    disease_match = re.search(r"^\s*([^\n:]+):", text, flags=re.MULTILINE)
    cause_match = re.search(r"Cause:\s*([^\n]+)", text, flags=re.IGNORECASE)
    symptoms_match = re.search(r"Symptoms:\s*([^\n]+)", text, flags=re.IGNORECASE)
    treatment_match = re.search(r"Treatment:\s*([^\n]+)", text, flags=re.IGNORECASE)
    prevention_match = re.search(r"Prevention:\s*([^\n]+)", text, flags=re.IGNORECASE)
    return {
        "disease": disease_match.group(1).strip() if disease_match else "",
        "cause": cause_match.group(1).strip() if cause_match else "",
        "symptoms": symptoms_match.group(1).strip() if symptoms_match else "",
        "treatment": treatment_match.group(1).strip() if treatment_match else "",
        "prevention": prevention_match.group(1).strip() if prevention_match else "",
    }


def _format_farmer_answer_from_docs(
    search_results: list[SearchResult], plant_part: str = "Any"
) -> str:
    for result in search_results:
        parts = _extract_entry_parts(result.document.page_content)
        if parts["disease"] and (parts["treatment"] or parts["prevention"]):
            treatment = parts["treatment"] or "I do not have enough information in the knowledge base."
            prevention = parts["prevention"] or "I do not have enough information in the knowledge base."
            why_match_parts = []
            if parts["symptoms"]:
                why_match_parts.append(f"The observed signs match: {parts['symptoms']}")
            if parts["cause"]:
                why_match_parts.append(f"Likely reason: {parts['cause']}")
            why_match_text = " ".join(why_match_parts).strip() or "This matches the visible symptoms."

            inspection_step = "- Check the affected part again after 2-3 days."
            dryness_step = "- Keep the affected area dry and clean."
            if "fruit" in plant_part.lower():
                inspection_step = "- Check more fruits for similar internal or surface damage."
                dryness_step = "- Keep harvested fruits dry and handle gently."
            elif "tree bark" in plant_part.lower():
                inspection_step = "- Check nearby branches and bark cracks in 2-3 days."
                dryness_step = "- Keep bark wounds clean and avoid extra injury."

            return (
                f"1. Disease Name\n{parts['disease']}\n{why_match_text}\n\n"
                f"2. What You Should Do Now\n"
                f"- Start now: {treatment}\n"
                f"- Remove badly affected plant parts.\n"
                f"{dryness_step}\n"
                f"{inspection_step}\n"
                f"- If spreading fast, ask local agri officer.\n\n"
                f"3. Precautions\n"
                f"- {prevention}\n"
                f"- Do not overwater the plant.\n"
                f"- Keep space between plants for airflow.\n"
                f"- Check the affected part every day for new damage.\n"
                f"- Clean tools after use."
            )
    return "I do not have enough information in the knowledge base."


def _get_generator():
    global _generator, _generator_task
    if _generator is None:
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc

        candidates = [
            ("text2text-generation", LLM_MODEL_NAME),
            ("text-generation", LLM_MODEL_NAME),
            ("text-generation", "distilgpt2"),
        ]
        errors = []
        for task, model_name in candidates:
            try:
                _generator = pipeline(task, model=model_name, tokenizer=model_name)
                _generator_task = task
                break
            except Exception as exc:
                errors.append(f"{task}/{model_name}: {exc}")

        if _generator is None:
            raise RuntimeError(
                "Could not initialize a local text generation pipeline.\n"
                + "\n".join(errors)
            )
    return _generator, _generator_task


def _top_matches(search_results: list[SearchResult]) -> list[dict]:
    items: list[dict] = []
    for result in search_results:
        parts = _extract_entry_parts(result.document.page_content)
        if parts["disease"]:
            items.append(
                {
                    "disease": parts["disease"],
                    "confidence": round(result.score * 100, 1),
                    "symptoms": parts["symptoms"],
                }
            )
    return items


def run_rag(
    query: str,
    image_bytes: bytes | None = None,
    category: str = "Any",
    crop: str = "Any",
    stage: str = "Any",
    region: str = "Any",
    plant_part: str = "Any",
    crop_name: str = "",
) -> dict:
    text_query = query.strip()
    image_observations = analyze_image_symptoms(image_bytes or b"")
    if not text_query and not image_observations:
        raise ValueError("Upload a plant image or enter symptoms.")

    retrieval_query = text_query
    context_hint = (
        f"Category: {category}. "
        f"Crop type: {crop_name or crop}. "
        f"Plant part: {plant_part}. "
        f"Growth stage: {stage}. "
        f"Region: {region}. "
    )
    retrieval_query = f"{context_hint}\n{retrieval_query}".strip()
    if image_observations:
        retrieval_query = (
            f"{context_hint}\n"
            f"User symptoms: {text_query or 'None provided'}\n"
            f"{image_observations}"
        )

    vectorstore = load_vector_store()
    search_results = vectorstore.similarity_search_with_score(
        retrieval_query,
        k=3,
        crop=(crop_name or crop),
        stage=stage,
        region=region,
    )
    if not search_results:
        answer = "I do not have enough information in the knowledge base."
        return {"answer": answer, "top_matches": [], "safety_notes": get_safety_notes(answer)}

    context = "\n\n".join([result.document.page_content for result in search_results])
    final_query = (
        f"{context_hint}\n"
        f"User symptoms: {text_query or 'None provided'}\n"
        f"{image_observations or ''}"
    ).strip()
    prompt = build_rag_prompt(context=context, query=final_query)

    generator, generator_task = _get_generator()
    outputs = generator(
        prompt,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        do_sample=LLM_TEMPERATURE > 0,
    )
    generated_text = outputs[0]["generated_text"].strip()
    if generator_task == "text-generation" and generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    if (not generated_text) or ("I do not have enough information in the knowledge base." in generated_text):
        generated_text = _format_farmer_answer_from_docs(search_results, plant_part=plant_part)

    return {
        "answer": generated_text,
        "top_matches": _top_matches(search_results),
        "safety_notes": get_safety_notes(generated_text),
        "image_observations": image_observations,
    }
