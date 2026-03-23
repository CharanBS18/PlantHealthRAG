import os

import streamlit as st

from config import DEFAULT_LANGUAGE
from services.history_store import load_history, save_history_entry
from services.language_support import translate_output
from services.rag_pipeline import run_rag
from services.vector_store import build_vector_store
from services.voice_output import synthesize_speech

st.set_page_config(page_title="Plant Health RAG", page_icon="🌱", layout="wide")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "last_audio_format" not in st.session_state:
    st.session_state.last_audio_format = "audio/wav"

st.title("🌿 Plant Health RAG Assistant (Offline)")
st.caption("Upload plant/leaf image and get simple action guidance for farmers.")

with st.sidebar:
    st.subheader("Diagnosis Filters")
    category = st.selectbox(
        "Plant Category",
        ["Any", "Field Crop", "Vegetable", "Fruit Crop", "Tree"],
        index=0,
    )
    crop = st.selectbox(
        "Crop",
        [
            "Any",
            "Tomato",
            "Rice",
            "Chili",
            "Mango",
            "Banana",
            "Citrus",
            "Guava",
            "Apple",
            "Grape",
            "Pomegranate",
            "Coconut",
            "Arecanut",
            "Many crops",
        ],
        index=0,
    )
    crop_name = st.text_input("Exact crop/fruit/tree name (optional)", value="")
    plant_part = st.selectbox(
        "Affected Part",
        ["Any", "Leaf", "Fruit surface", "Cut fruit (inside)", "Stem", "Root", "Tree bark", "Whole plant/tree"],
        index=0,
    )
    stage = st.selectbox(
        "Growth Stage",
        ["Any", "Early", "Vegetative", "Flowering", "Fruiting", "Maturity"],
        index=0,
    )
    region = st.selectbox(
        "Region/Weather",
        ["Any", "Warm humid", "Cool wet", "Tropical", "Dry"],
        index=0,
    )
    language = st.selectbox("Output Language", ["English", "Hindi", "Telugu"], index=["English", "Hindi", "Telugu"].index(DEFAULT_LANGUAGE) if DEFAULT_LANGUAGE in ["English", "Hindi", "Telugu"] else 0)
    st.caption("Language option currently translates headings and key labels offline.")

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("Input")
    if st.button("Build Knowledge Base Index"):
        try:
            build_vector_store()
            st.success("Vector index created!")
            st.rerun()  # Refresh the page to update the UI
        except Exception as exc:
            st.error(f"Failed to build vector index: {exc}")
            st.info("If this persists, check that data/plant_knowledge.txt exists and is readable.")

    uploaded_file = st.file_uploader(
        "Upload plant/leaf image",
        type=["png", "jpg", "jpeg", "webp"],
    )
    query = st.text_area(
        "Describe symptoms (optional):",
        placeholder="Example: brown circular spots on older tomato leaves",
    )

    if uploaded_file:
        st.image(uploaded_file.getvalue(), caption="Uploaded Image", use_container_width=True)

    analyze_clicked = st.button("Analyze Plant")

with col_right:
    st.subheader("Recent History")
    history = load_history(limit=8)
    if not history:
        st.caption("No saved analyses yet.")
    else:
        for item in history:
            stamp = item.get("timestamp_utc", "unknown-time")
            short_disease = item.get("top_disease", "Unknown")
            st.markdown(f"- `{stamp}`: **{short_disease}**")

if analyze_clicked:
    from services.vector_store import INDEX_PATH, DOCS_PATH
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        st.error("Vector index not found. Please click 'Build Knowledge Base Index' first.")
        st.info("The knowledge base needs to be built before you can analyze plants.")
    elif not query.strip() and not uploaded_file:
        st.warning("Upload an image or enter symptoms.")
    else:
        image_bytes = uploaded_file.getvalue() if uploaded_file else None
        with st.spinner("Analyzing using local RAG..."):
            try:
                result = run_rag(
                    query=query,
                    image_bytes=image_bytes,
                    category=category,
                    crop=crop,
                    stage=stage,
                    region=region,
                    plant_part=plant_part,
                    crop_name=crop_name.strip(),
                )
                answer = result["answer"]
                translated_answer = translate_output(answer, language)
                st.session_state.last_answer = translated_answer

                st.success("Diagnosis Complete")
                st.markdown(translated_answer)

                top_matches = result.get("top_matches", [])
                if top_matches:
                    st.markdown(f"### {translate_output('Top Matches', language)}")
                    for idx, item in enumerate(top_matches, start=1):
                        symptoms_text = item.get("symptoms") or "No symptom detail found."
                        line = (
                            f"{idx}. **{item['disease']}** - `{item['confidence']}%` confidence\n"
                            f"   - Symptom clue: {symptoms_text}"
                        )
                        st.markdown(
                            translate_output(line, language)
                        )

                notes = result.get("safety_notes", [])
                if notes:
                    st.markdown(f"### {translate_output('Safety Notes', language)}")
                    for note in notes:
                        st.markdown(f"- {translate_output(note, language)}")

                top_disease = top_matches[0]["disease"] if top_matches else "Unknown"
                save_history_entry(
                    {
                        "query": query,
                        "crop": crop,
                        "crop_name": crop_name.strip(),
                        "plant_part": plant_part,
                        "category": category,
                        "stage": stage,
                        "region": region,
                        "top_disease": top_disease,
                        "answer": answer,
                    }
                )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

if st.session_state.last_answer:
    st.markdown("### Voice Output")
    if st.button("Generate Audio Advice"):
        st.session_state.last_audio_bytes = None
        audio_payload = synthesize_speech(st.session_state.last_answer)
        if audio_payload:
            audio_bytes, audio_format = audio_payload
            st.session_state.last_audio_bytes = audio_bytes
            st.session_state.last_audio_format = audio_format
        else:
            st.warning("Text-to-speech could not generate valid audio on this system.")
    if st.session_state.last_audio_bytes:
        st.audio(st.session_state.last_audio_bytes, format=st.session_state.last_audio_format)
