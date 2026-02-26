from services.rag_pipeline import _extract_entry_parts
from services.safety_guardrails import get_safety_notes


def test_extract_entry_parts():
    sample = (
        "Tomato Early Blight:\n"
        "Cause: Fungus Alternaria solani.\n"
        "Symptoms: Brown spots with ring pattern.\n"
        "Treatment: Remove infected leaves.\n"
        "Prevention: Keep leaves dry.\n"
    )
    parsed = _extract_entry_parts(sample)
    assert parsed["disease"] == "Tomato Early Blight"
    assert "Brown spots" in parsed["symptoms"]
    assert "Remove infected leaves." == parsed["treatment"]


def test_safety_notes_mentions_spray_and_fertilizer():
    text = "Use fungicide spray and nitrogen fertilizer."
    notes = get_safety_notes(text)
    assert any("label" in note.lower() for note in notes)
    assert any("split doses" in note.lower() for note in notes)
