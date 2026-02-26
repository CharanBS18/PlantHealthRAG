_translator_cache = {}


def _fallback_translate(text: str, language: str) -> str:
    replacements = {
        "Hindi": {
            "Disease Name": "रोग का नाम",
            "What You Should Do Now": "अभी क्या करना चाहिए",
            "Precautions": "सावधानियां",
            "Top Matches": "शीर्ष मिलान",
            "Safety Notes": "सुरक्षा नोट",
            "Symptom clue": "लक्षण संकेत",
            "confidence": "विश्वास",
        },
        "Telugu": {
            "Disease Name": "వ్యాధి పేరు",
            "What You Should Do Now": "ఇప్పుడే చేయాల్సినవి",
            "Precautions": "జాగ్రత్తలు",
            "Top Matches": "అత్యుత్తమ సరిపోలికలు",
            "Safety Notes": "భద్రత సూచనలు",
            "Symptom clue": "లక్షణ సూచన",
            "confidence": "నమ్మకం",
        },
    }
    mapping = replacements.get(language, {})
    out = text
    for src, dst in mapping.items():
        out = out.replace(src, dst)
    return out


def _get_translator(language: str):
    if language in _translator_cache:
        return _translator_cache[language]

    model_map = {
        "Hindi": "Helsinki-NLP/opus-mt-en-hi",
        "Telugu": "Helsinki-NLP/opus-mt-en-te",
    }
    model_name = model_map.get(language)
    if not model_name:
        _translator_cache[language] = None
        return None

    try:
        from transformers import pipeline
    except ModuleNotFoundError:
        _translator_cache[language] = None
        return None

    task_candidates = ["translation_en_to_hi", "translation_en_to_te", "translation"]
    translator = None
    for task in task_candidates:
        try:
            translator = pipeline(task, model=model_name, tokenizer=model_name)
            break
        except Exception:
            continue

    _translator_cache[language] = translator
    return translator


def translate_output(text: str, language: str) -> str:
    if language == "English":
        return text

    translator = _get_translator(language)
    if translator is None:
        return _fallback_translate(text, language)

    try:
        # Line-wise translation keeps structure and improves model stability.
        translated_lines = []
        for line in text.splitlines():
            if not line.strip():
                translated_lines.append("")
                continue
            translated = translator(line, max_new_tokens=256)
            translated_text = translated[0].get("translation_text", "").strip()
            translated_lines.append(translated_text or line)
        return "\n".join(translated_lines)
    except Exception:
        return _fallback_translate(text, language)
