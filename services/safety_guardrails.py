def get_safety_notes(answer_text: str) -> list[str]:
    text = answer_text.lower()
    notes: list[str] = []

    if "fungicide" in text or "bactericide" in text or "spray" in text:
        notes.append("Read the product label fully before spraying.")
        notes.append("Wear gloves, mask, and eye protection while spraying.")
        notes.append("Do not spray during strong wind or hot midday sun.")
        notes.append("Keep children, animals, and food items away during spraying.")

    if "fertilizer" in text or "nitrogen" in text or "potash" in text or "phosphorus" in text:
        notes.append("Use small split doses. Do not add too much fertilizer at once.")
        notes.append("Water the soil after fertilizer use unless product label says otherwise.")
        notes.append("Do a soil test when possible before repeating fertilizer.")

    notes.append("If plants get worse after 3 to 5 days, contact a local agriculture expert.")
    return notes
