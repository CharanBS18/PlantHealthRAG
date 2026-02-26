from io import BytesIO

from config import VISION_MODEL_NAME

_vision_pipeline = None
_vision_task = None
_zero_shot_pipeline = None


def _build_pipeline(task: str, model_name: str):
    try:
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return pipeline(task, model=model_name)


def _get_vision_pipeline():
    global _vision_pipeline, _vision_task
    if _vision_pipeline is None:
        candidates = [
            ("image-to-text", VISION_MODEL_NAME),
            ("image-text-to-text", VISION_MODEL_NAME),
            ("visual-question-answering", "dandelin/vilt-b32-finetuned-vqa"),
        ]
        errors = []
        for task, model_name in candidates:
            try:
                _vision_pipeline = _build_pipeline(task, model_name)
                _vision_task = task
                break
            except Exception as exc:  # pragma: no cover - depends on local env/models
                errors.append(f"{task}: {exc}")

        if _vision_pipeline is None:
            raise RuntimeError(
                "Could not initialize a local vision pipeline. "
                "Tried image captioning and VQA tasks.\n"
                + "\n".join(errors)
            )

    return _vision_pipeline, _vision_task


def _run_image_text_to_text(model, image):
    prompt = "Describe visible plant symptoms from this image in one short sentence."
    attempts = [
        lambda: model(image, text=prompt),
        lambda: model({"image": image, "text": prompt}),
        lambda: model({"images": image, "text": prompt}),
    ]
    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            continue
    raise RuntimeError("image-text-to-text pipeline could not process image input.")


def _run_vqa(image) -> str:
    model = _build_pipeline("visual-question-answering", "dandelin/vilt-b32-finetuned-vqa")
    questions = [
        "What visible disease symptoms appear on this plant, fruit, or tree part?",
        "Are there yellow spots, brown lesions, rotting tissue, mold, wilting, or curling?",
        "If the fruit is cut open, what internal problem is visible?",
        "What is the likely visible condition in this image?",
    ]
    answers = []
    for question in questions:
        output = model(image=image, question=question)
        if isinstance(output, list) and output:
            answer = (output[0].get("answer") or "").strip()
            if answer:
                answers.append(answer)
    return f"Image observation: {'; '.join(answers)}" if answers else ""


def _get_zero_shot_image_classifier():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        _zero_shot_pipeline = _build_pipeline(
            "zero-shot-image-classification", "openai/clip-vit-base-patch32"
        )
    return _zero_shot_pipeline


def _run_zero_shot_classifier(image) -> str:
    labels = [
        "early blight disease",
        "late blight disease",
        "leaf curl disease",
        "fruit rot disease",
        "anthracnose fruit spot",
        "citrus canker disease",
        "apple scab disease",
        "mango stem end rot",
        "banana anthracnose",
        "tree bark canker",
        "powdery mildew disease",
        "downy mildew disease",
        "nitrogen deficiency",
        "potassium deficiency",
        "phosphorus deficiency",
        "healthy plant",
    ]
    try:
        classifier = _get_zero_shot_image_classifier()
        results = classifier(image, candidate_labels=labels)
    except Exception:
        return ""
    if not results:
        return ""
    top = results[0]
    label = str(top.get("label", "")).strip()
    score = float(top.get("score", 0.0))
    if not label:
        return ""
    return f"Image likely shows: {label} ({round(score * 100, 1)}% confidence)."


def analyze_image_symptoms(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is not installed. Run: pip install -r requirements.txt"
        ) from exc

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    model, task = _get_vision_pipeline()

    zero_shot_hint = _run_zero_shot_classifier(image)

    if task == "image-to-text":
        results = model(image)
        if not results:
            return zero_shot_hint
        caption = (results[0].get("generated_text") or "").strip()
        if not caption or "describe visible leaf symptoms" in caption.lower():
            caption = _run_vqa(image)
        observation = f"Image observation: {caption}" if caption else ""
        return f"{observation} {zero_shot_hint}".strip()

    if task == "image-text-to-text":
        try:
            results = _run_image_text_to_text(model, image)
            if not results:
                return zero_shot_hint
            caption = (results[0].get("generated_text") or "").strip()
            if not caption or "describe visible leaf symptoms" in caption.lower():
                caption = _run_vqa(image)
            observation = f"Image observation: {caption}" if caption else ""
            return f"{observation} {zero_shot_hint}".strip()
        except Exception:
            vqa_text = _run_vqa(image)
            return f"{vqa_text} {zero_shot_hint}".strip()

    if task == "visual-question-answering":
        vqa_text = _run_vqa(image)
        return f"{vqa_text} {zero_shot_hint}".strip()

    return ""
