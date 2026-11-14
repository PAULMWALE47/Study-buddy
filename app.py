import io
import html
import json
import math
import os
import re
import struct
import time
import wave
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load a local, Git-ignored .env file for convenient development.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Change this one value if the selected Gemini model is unavailable.
MODEL_NAME = "gemini-3.5-flash-lite"


# Read the key from the local environment, Streamlit Cloud secrets, or the local session input.
def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key and not api_key.lower().startswith(("your-", "your_", "replace-", "paste-")):
        return api_key
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
    except StreamlitSecretNotFoundError:
        api_key = ""
    if api_key and not api_key.lower().startswith(("your-", "your_", "replace-", "paste-")):
        return api_key
    api_key = st.session_state.get("gemini_api_key", "").strip()
    if api_key and not api_key.lower().startswith(("your-", "your_", "replace-", "paste-")):
        return api_key
    return ""


# Extract text from one supported uploaded file.
def extract_text(file):
    name = file.name
    extension = os.path.splitext(name)[1].lower()
    file_data = io.BytesIO(file.getvalue())

    try:
        if extension == ".pdf":
            from pypdf import PdfReader

            return "\n\n".join(page.extract_text() or "" for page in PdfReader(file_data).pages)
        if extension == ".docx":
            from docx import Document

            document = Document(file_data)
            return "\n\n".join(paragraph.text for paragraph in document.paragraphs)
        if extension == ".pptx":
            from pptx import Presentation

            presentation = Presentation(file_data)
            slides = []
            for slide in presentation.slides:
                slides.append("\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text")))
            return "\n\n".join(slides)
        if extension in (".txt", ".md"):
            return file_data.read().decode("utf-8", errors="replace")
        if extension == ".csv":
            return pd.read_csv(file_data).to_string(index=False)
        if extension == ".xlsx":
            sheets = pd.read_excel(file_data, sheet_name=None)
            return "\n\n".join(f"{sheet_name}\n{sheet.to_string(index=False)}" for sheet_name, sheet in sheets.items())
        if extension in (".png", ".jpg"):
            try:
                import pytesseract
                from PIL import Image
            except ImportError:
                st.warning(f"OCR is not available for {name}. Install pytesseract and Pillow to read images.")
                return ""
            try:
                return pytesseract.image_to_string(Image.open(file_data))
            except pytesseract.TesseractNotFoundError:
                st.warning(f"OCR is not available for {name}. Install the Tesseract application to read images.")
                return ""

        st.warning(f"Unsupported file type: {name}")
        return ""
    except Exception as error:
        st.warning(f"Could not read {name}: {error}")
        return ""


# Combine paragraphs into roughly 180-word chunks and keep their source names.
def chunk_text(text, source, max_words=180):
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []
    current_words = []

    for paragraph in paragraphs:
        words = paragraph.split()
        while words:
            room = max_words - len(current_words)
            current_words.extend(words[:room])
            words = words[room:]
            if len(current_words) >= max_words:
                chunks.append({"text": " ".join(current_words), "source": source})
                current_words = []

    if current_words:
        chunks.append({"text": " ".join(current_words), "source": source})
    return chunks


# Build a TF-IDF index for all extracted chunks.
def build_index(chunks):
    if not chunks:
        return None, None
    try:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
        return vectorizer, matrix
    except ValueError as error:
        st.warning(f"No searchable terms were found in the uploaded text: {error}")
        return None, None


# Find the chunks whose word patterns best match a query.
def retrieve(query, vectorizer, matrix, chunks, top_k=4):
    if not query.strip() or vectorizer is None or matrix is None or not chunks:
        return []
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    best_indexes = scores.argsort()[::-1][: min(top_k, len(chunks))]
    return [
        {"text": chunks[index]["text"], "source": chunks[index]["source"], "score": float(scores[index])}
        for index in best_indexes
        if scores[index] > 0
    ]


# Keep all Gemini requests in this function, including JSON requests.
def ask_model(prompt, want_json=False):
    st.session_state["last_model_error"] = ""
    api_key = get_api_key()
    if not api_key:
        message = (
            "Gemini API key is missing. Set GEMINI_API_KEY before starting Streamlit, "
            "or enter a key in the sidebar."
        )
        st.session_state["last_model_error"] = message
        st.error(message)
        return ""

    for attempt in range(2):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            if want_json:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
            else:
                response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            answer = response.text
            if not answer:
                message = (
                    "Gemini returned no text. The request may have been blocked, "
                    "or the model returned an empty response."
                )
                st.session_state["last_model_error"] = message
                st.error(message)
                return ""
            return answer
        except Exception as error:
            details = f"{getattr(error, 'code', '')} {getattr(error, 'status_code', '')} {error}".lower()
            is_quota_error = any(word in details for word in ("429", "quota", "resource_exhausted", "resource exhausted"))
            if is_quota_error and attempt == 0:
                st.warning("The API limit was reached. Waiting about 20 seconds before one retry.")
                time.sleep(20)
                continue
            if is_quota_error:
                message = "API limit reached, please wait a minute and try again."
            elif any(word in details for word in ("401", "unauthenticated", "api key not valid", "invalid api key")):
                message = "Gemini rejected the API key. Use a valid Gemini API key in the sidebar or environment."
            elif any(word in details for word in ("404", "not found", "model not found")):
                message = f"Gemini could not find model {MODEL_NAME}. Check the MODEL_NAME setting at the top of app.py."
            else:
                safe_error = str(error).replace(api_key, "[hidden key]")
                message = f"Gemini request failed: {safe_error}"
            st.session_state["last_model_error"] = message
            st.error(message)
            return ""
    return ""


# Remove optional Markdown fences and decode a model-generated JSON value.
def parse_json_response(response):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


# Generate 50 flashcards from retrieved document context.
def generate_flashcards(matches, topic=""):
    context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in matches)
    topic_line = f"Focus on this topic: {topic}." if topic else "Cover the most important concepts."
    prompt = (
        "Use only the context below. Create exactly 50 distinct, concise study flashcards. "
        "Avoid repeating the same fact. "
        'Return JSON only as an array of objects: [{"front": "...", "back": "..."}].\n'
        f"{topic_line}\n\nContext:\n{context}"
    )
    response = ask_model(prompt, want_json=True)
    if not response:
        return []
    try:
        cards = parse_json_response(response)
        if not isinstance(cards, list) or len(cards) != 50 or not all(
            isinstance(card, dict)
            and isinstance(card.get("front"), str)
            and isinstance(card.get("back"), str)
            and card["front"].strip()
            and card["back"].strip()
            for card in cards
        ):
            raise ValueError("The response did not contain exactly 50 valid flashcards.")
        return cards
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read the generated flashcards: {error}")
        return []


# Generate plain, open-ended practice questions from retrieved document context.
def generate_practice_questions(matches, topic, number_of_questions):
    context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in matches)
    topic_line = f"Focus on this topic: {topic}." if topic else "Cover the key ideas in the notes."
    prompt = (
        f"Use only the context below. Create exactly {number_of_questions} short-answer practice questions. "
        "These must be plain open-ended questions, not multiple-choice. Do not include answer options. "
        'Return JSON only as an array of objects with "question", "answer", "explanation", "topic", '
        'and "source" string fields.\n'
        f"{topic_line}\n\nContext:\n{context}"
    )
    response = ask_model(prompt, want_json=True)
    if not response:
        return []
    try:
        questions = parse_json_response(response)
        required_fields = ("question", "answer", "explanation", "topic", "source")
        if not isinstance(questions, list) or len(questions) != number_of_questions or not all(
            isinstance(question, dict)
            and all(isinstance(question.get(field), str) and question[field].strip() for field in required_fields)
            for question in questions
        ):
            raise ValueError("The response did not match the open-ended question format.")
        return questions
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read the generated practice questions: {error}")
        return []


# Generate and validate all quiz questions in one model request, retrying bad JSON once.
def generate_quiz(matches, topic, number_of_questions):
    context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in matches)
    topic_line = f"Topic: {topic}." if topic else "Topic: whole documents."
    prompt = (
        f"Use only the context below. Generate exactly {number_of_questions} multiple-choice questions. "
        "Each question must have four options and one correct answer. "
        'Return JSON only as an array with this shape: [{"question": "...", '
        '"options": ["A...", "B...", "C...", "D..."], "answer_index": 0, '
        '"topic": "short topic label", "explanation": "why it is correct", '
        '"source": "file name"}]. The answer_index must be 0, 1, 2, or 3.\n'
        f"{topic_line}\n\nContext:\n{context}"
    )

    for attempt in range(2):
        response = ask_model(prompt, want_json=True)
        if not response:
            return []
        try:
            questions = parse_json_response(response)
            valid_questions = (
                isinstance(questions, list)
                and len(questions) == number_of_questions
                and all(
                    isinstance(question, dict)
                    and isinstance(question.get("question"), str)
                    and isinstance(question.get("options"), list)
                    and len(question["options"]) == 4
                    and all(isinstance(option, str) for option in question["options"])
                    and isinstance(question.get("answer_index"), int)
                    and not isinstance(question.get("answer_index"), bool)
                    and 0 <= question["answer_index"] < 4
                    and isinstance(question.get("topic"), str)
                    and isinstance(question.get("explanation"), str)
                    and isinstance(question.get("source"), str)
                    for question in questions
                )
            )
            if valid_questions:
                return questions[:number_of_questions]
            raise ValueError("The response did not match the quiz format.")
        except (json.JSONDecodeError, ValueError) as error:
            if attempt == 0:
                prompt += "\nReturn valid JSON only, following the requested schema exactly."
            else:
                st.error(f"Could not read the generated quiz after one retry: {error}")
    return []


# Compare the selected answers with the answer keys in Python.
def mark_quiz(questions, selected_answers):
    marked_questions = []
    correct_count = 0
    for index, question in enumerate(questions):
        selected = selected_answers.get(index)
        is_correct = selected == question["answer_index"]
        correct_count += int(is_correct)
        marked_questions.append(
            {
                **question,
                "selected_index": selected,
                "is_correct": is_correct,
            }
        )
    total = len(questions)
    percent = round(correct_count * 100 / total) if total else 0
    return {
        "questions": marked_questions,
        "score": correct_count,
        "total": total,
        "percent": percent,
        "completed_at": datetime.now().astimezone().isoformat(timespec="minutes"),
    }


# Explain each selected wrong answer in one model call.
def explain_wrong_answers(marked_questions, chunks, vectorizer, matrix, use_notes):
    wrong_answers = [
        (index, item)
        for index, item in enumerate(marked_questions)
        if not item["is_correct"] and item["selected_index"] is not None
    ]
    if not wrong_answers:
        return {}

    records = []
    for index, item in wrong_answers:
        record = {
            "index": index,
            "question": item["question"],
            "student_answer": item["options"][item["selected_index"]],
            "correct_answer": item["options"][item["answer_index"]],
        }
        if use_notes:
            query = f"{item['question']} {record['student_answer']} {record['correct_answer']}"
            matches = retrieve(query, vectorizer, matrix, chunks, top_k=2)
            record["document_context"] = [
                {"source": match["source"], "text": match["text"]} for match in matches
            ]
        records.append(record)

    instruction = (
        "Use only the document context in each item. Explain briefly why the student's answer is "
        "incorrect and why the correct answer fits. If the context is not enough, say so."
        if use_notes
        else "Use general knowledge only; do not refer to uploaded notes. Explain briefly why the "
        "student's answer is incorrect and why the correct answer fits."
    )
    prompt = (
        f"{instruction} Return JSON only as an array of objects with integer 'index' and string "
        "'explanation' fields.\n\nItems:\n"
        + json.dumps(records, ensure_ascii=False)
    )
    response = ask_model(prompt, want_json=True)
    if not response:
        st.session_state["wrong_explanation_error"] = (
            st.session_state.get("last_model_error") or "Gemini could not explain the missed answers."
        )
        return {}

    try:
        explanations = parse_json_response(response)
        if not isinstance(explanations, list):
            raise ValueError("The response was not a list.")
        parsed_explanations = {
            item["index"]: item["explanation"]
            for item in explanations
            if isinstance(item, dict)
            and isinstance(item.get("index"), int)
            and not isinstance(item.get("index"), bool)
            and 0 <= item["index"] < len(marked_questions)
            and isinstance(item.get("explanation"), str)
        }
        expected_indexes = {index for index, _ in wrong_answers}
        if not expected_indexes.issubset(parsed_explanations):
            raise ValueError("The response did not explain every incorrect answer.")
        return parsed_explanations
    except (json.JSONDecodeError, ValueError) as error:
        st.session_state["wrong_explanation_error"] = f"Could not read the explanations: {error}"
        return {}


# Create a brief, low-volume two-note chime for positive quiz feedback.
def make_success_sound():
    sample_rate = 22050
    audio_frames = bytearray()
    for frequency, duration in ((660, 0.12), (880, 0.16)):
        sample_count = int(sample_rate * duration)
        for sample_index in range(sample_count):
            fade = min(1, sample_index / (sample_rate * 0.01), (sample_count - sample_index) / (sample_rate * 0.04))
            amplitude = 0.045 * max(0, fade)
            value = int(32767 * amplitude * math.sin(2 * math.pi * frequency * sample_index / sample_rate))
            audio_frames.extend(struct.pack("<h", value))

    audio_file = io.BytesIO()
    with wave.open(audio_file, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(audio_frames)
    return audio_file.getvalue()


# Let the student hear a question and its choices with the browser's built-in speech.
def render_question_reader(question, options, question_number):
    spoken_text = html.escape(
        f"Question {question_number}. {question}. "
        + " ".join(f"Option {index + 1}: {option}." for index, option in enumerate(options))
    )
    button_id = f"read-question-{question_number}"
    text_id = f"spoken-text-{question_number}"
    components.html(
        f"""
        <style>
            body {{ margin: 0; font-family: sans-serif; }}
            button {{
                border: 1px solid #8b5e3c;
                border-radius: 9px;
                padding: 0.45rem 0.8rem;
                color: #fffcf5;
                background: #8b5e3c;
                font-weight: 600;
                cursor: pointer;
            }}
            button:focus-visible {{ outline: 3px solid #c59b72; outline-offset: 2px; }}
        </style>
        <button id="{button_id}" type="button" aria-label="Read question and answer choices aloud">
            Read question and choices aloud
        </button>
        <span id="{text_id}" hidden>{spoken_text}</span>
        <script>
            const readButton = document.getElementById("{button_id}");
            readButton.addEventListener("click", () => {{
                if (!("speechSynthesis" in window)) {{
                    readButton.textContent = "Speech is not supported in this browser";
                    return;
                }}
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(
                    document.getElementById("{text_id}").textContent
                );
                utterance.rate = 0.92;
                window.speechSynthesis.speak(utterance);
            }});
        </script>
        """,
        height=48,
        scrolling=False,
    )


# Read saved quiz attempts so they remain available after restarting the app.
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quiz_history.json")


def load_quiz_history():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
            history = json.load(history_file)
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as error:
        st.warning(f"Could not load saved quiz history: {error}")
        return []

    if not isinstance(history, list) or not all(
        isinstance(attempt, dict) and isinstance(attempt.get("questions"), list)
        for attempt in history
    ):
        st.warning("The saved quiz history file has an invalid format. Starting with an empty history.")
        return []
    return history


# Save quiz attempts locally; this file is excluded from version control.
def save_quiz_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
        json.dump(history, history_file, ensure_ascii=False, indent=2)


# Summarize quiz history and explain all weak topics in one model request.
def build_study_report(history, chunks, vectorizer, matrix):
    topic_scores = {}
    for attempt in history:
        for question in attempt["questions"]:
            topic = question.get("topic") or "General"
            score = topic_scores.setdefault(topic, {"correct": 0, "total": 0, "wrong": []})
            score["total"] += 1
            if question["is_correct"]:
                score["correct"] += 1
            else:
                score["wrong"].append(question)

    weak_topics = []
    for topic, score in topic_scores.items():
        score["percent"] = score["correct"] * 100 / score["total"]
        if score["percent"] < 60:
            weak_topics.append(topic)
    weak_topics.sort(key=lambda topic: (topic_scores[topic]["percent"], topic.lower()))

    explanations = {}
    source_names = {}
    if weak_topics:
        context_parts = []
        for topic in weak_topics:
            matches = retrieve(topic, vectorizer, matrix, chunks, top_k=2)
            source_names[topic] = list(dict.fromkeys(item["source"] for item in matches))
            context_parts.append(
                f"Topic: {topic}\n"
                f"Wrong answers: "
                f"{'; '.join(question['question'] for question in topic_scores[topic]['wrong'])}\n"
                + "\n".join(f"Source: {item['source']}\n{item['text']}" for item in matches)
            )
        if any(source_names.values()):
            prompt = (
                "Use only the context below. For every listed topic, write a short, clear explanation "
                "that helps the student revise. Return JSON only as an array of objects with "
                '"topic" and "explanation" string fields.\n\nContext:\n'
                + "\n\n".join(context_parts)
            )
            response = ask_model(prompt, want_json=True)
            if response:
                try:
                    parsed = parse_json_response(response)
                    if isinstance(parsed, list):
                        explanations = {
                            item["topic"]: item["explanation"]
                            for item in parsed
                            if isinstance(item, dict)
                            and isinstance(item.get("topic"), str)
                            and isinstance(item.get("explanation"), str)
                        }
                except (json.JSONDecodeError, TypeError):
                    st.warning("The study explanations could not be read from the model response.")

    # Format one downloadable Markdown report.
    lines = ["# Study Report", "", "## Score per topic", ""]
    for topic in sorted(topic_scores, key=str.lower):
        score = topic_scores[topic]
        lines.append(f"- **{topic}:** {score['correct']}/{score['total']}")
    lines.extend(["", "## Areas to study", ""])
    if not weak_topics:
        lines.append("No topics are currently below 60%.")
    else:
        for topic in weak_topics:
            score = topic_scores[topic]
            sources = source_names.get(topic, [])
            lines.append(f"### {topic} ({score['correct']}/{score['total']}, {score['percent']:.0f}%)")
            lines.append(explanations.get(topic, "Revision explanation unavailable."))
            lines.append(f"Sources: {', '.join(sources) if sources else 'No matching document chunks found.'}")
            lines.append("")
    return "\n".join(lines)


# Initialize the values that persist while this browser session is open.
st.set_page_config(page_title="Study Buddy RAG", page_icon="📚", layout="wide")
for key, default in {
    "chunks": [],
    "vectorizer": None,
    "matrix": None,
    "file_statuses": [],
    "document_version": 0,
    "chat_messages": [],
    "last_model_error": "",
    "flashcards_cache": {},
    "active_flashcards_key": None,
    "flashcard_index": 0,
    "show_flashcard_answer": False,
    "practice_cache": {},
    "active_practice_key": None,
    "practice_index": 0,
    "show_practice_answer": False,
    "quiz_cache": {},
    "current_quiz_key": None,
    "quiz_submitted": False,
    "quiz_result": None,
    "wrong_explanation_error": "",
    "success_sound_played": False,
    "quiz_history": [],
    "history_version": 0,
    "history_loaded": False,
    "history_storage_error": "",
    "report_cache": {},
}.items():
    st.session_state.setdefault(key, default)

# Load saved attempts once for this browser session.
if not st.session_state.history_loaded:
    st.session_state.quiz_history = load_quiz_history()
    st.session_state.history_version = len(st.session_state.quiz_history)
    st.session_state.history_loaded = True

# Style the app with a calm study-focused palette and clearer content surfaces.
st.markdown(
    """
    <style>
    :root {
        --ink: #3e2a20;
        --muted: #806a5b;
        --accent: #8b5e3c;
        --line: #e6d7c6;
        --surface: #fffcf5;
    }

    [data-testid="stAppViewContainer"] {
        background: #f8f3e9;
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #efe3d3;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted) !important;
    }
    .block-container {
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .study-hero {
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e5d2bd;
        border-radius: 22px;
        background: linear-gradient(120deg, #fffdf8 15%, #f1e4d2 100%);
        box-shadow: 0 12px 32px rgba(62, 42, 32, 0.07);
    }
    .study-hero .eyebrow {
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .study-hero h1 {
        margin: 0.35rem 0 0.4rem;
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 2.7rem);
        font-weight: 750;
        letter-spacing: -0.045em;
    }
    .study-hero p {
        max-width: 650px;
        margin: 0;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
    }
    div[data-testid="stTabs"] > div[role="tablist"] {
        gap: 0.4rem;
        border-bottom: 1px solid var(--line);
    }
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 3rem;
        padding: 0.65rem 1rem;
        border-radius: 12px 12px 0 0;
        color: var(--muted) !important;
        opacity: 1 !important;
        font-weight: 650;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: inherit !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        background: #f1e5d4;
    }
    h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.7rem;
        padding: 0.55rem 1.05rem;
        border: 1px solid var(--accent);
        border-radius: 10px;
        color: #fffcf5;
        background: var(--accent);
        font-weight: 650;
        box-shadow: 0 3px 8px rgba(62, 42, 32, 0.12);
        transition: border-color 120ms ease, background 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: #70472d;
        color: #fffcf5;
        background: #70472d;
        transform: translateY(-1px);
        box-shadow: 0 5px 12px rgba(62, 42, 32, 0.18);
    }
    div[data-testid="stButton"] > button:disabled,
    div[data-testid="stDownloadButton"] > button:disabled {
        border-color: #d7c7b4;
        color: #806a5b;
        background: #e8ddce;
        box-shadow: none;
        transform: none;
    }
    div[data-testid="stFileUploader"],
    div[data-testid="stExpander"],
    div[data-testid="stChatMessage"],
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #c5aa8b;
        border-radius: 12px;
        background: var(--surface);
    }
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: var(--muted) !important;
    }
    div[data-testid="stFileUploader"] button {
        border: 1px solid var(--accent);
        border-radius: 9px;
        color: #fffcf5 !important;
        background: var(--accent) !important;
    }
    div[data-testid="stChatMessage"] {
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.65rem;
    }
    div[data-testid="stExpander"] {
        overflow: hidden;
    }
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        border-color: #d8c8b5;
        border-radius: 10px;
        background: var(--surface) !important;
    }
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: var(--ink) !important;
        background: transparent !important;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    .flashcard-stage {
        margin: 1.4rem 0;
        perspective: 1200px;
        animation: flashcard-arrive 420ms ease-out both;
    }
    .study-flashcard {
        position: relative;
        min-height: 310px;
        transform-style: preserve-3d;
        transition: transform 760ms cubic-bezier(0.2, 0.75, 0.25, 1);
    }
    .study-flashcard.is-flipped {
        transform: rotateY(180deg);
    }
    .flashcard-face {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 2rem 2.2rem;
        overflow: hidden;
        border: 1px solid #e2cfb7;
        border-radius: 22px;
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
        box-shadow: 0 18px 38px rgba(62, 42, 32, 0.13);
    }
    .flashcard-front {
        background: linear-gradient(135deg, #fffef9 10%, #f2e7d8 100%);
    }
    .flashcard-back {
        transform: rotateY(180deg);
        border-color: #d4b99b;
        background: linear-gradient(135deg, #f5e8d8 10%, #e8d2ba 100%);
    }
    .flashcard-label {
        margin-bottom: 1rem;
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
    }
    .flashcard-back .flashcard-label {
        color: #765033;
    }
    .flashcard-text {
        max-height: 190px;
        overflow-y: auto;
        color: var(--ink);
        font-size: clamp(1.25rem, 2.6vw, 1.8rem);
        font-weight: 650;
        line-height: 1.5;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .flashcard-hint {
        margin-top: 1.4rem;
        color: var(--muted);
        font-size: 0.9rem;
    }
    @keyframes flashcard-arrive {
        from { opacity: 0; transform: translateY(12px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @media (prefers-reduced-motion: reduce) {
        .flashcard-stage { animation: none; }
        .study-flashcard { transition: none; }
    }
    @media (max-width: 700px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        .study-hero {
            padding: 1.4rem;
            border-radius: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Collect the user's study materials and build the retrieval index.
st.markdown(
    """
    <section class="study-hero">
        <div class="eyebrow">Your personal study space</div>
        <h1>Study Buddy</h1>
        <p>Turn your notes into clear answers, focused flashcards, and quizzes that help you learn with confidence.</p>
    </section>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("Settings")
st.sidebar.caption("AI setup and demo options")
if get_api_key():
    st.sidebar.caption("Gemini is configured by the app owner.")
else:
    entered_api_key = st.sidebar.text_input(
        "Gemini API key (optional)",
        type="password",
        help="Used only for this browser session. Prefer configuring a server secret.",
    )
    if entered_api_key:
        st.session_state["gemini_api_key"] = entered_api_key
    if not get_api_key():
        st.sidebar.caption("Gemini features need an API key. Retrieval-only mode works without one.")
answer_mode = st.sidebar.radio(
    "Answer mode",
    ["Use uploaded notes", "Use Gemini directly", "Retrieval-only Q&A"],
    help=(
        "Gemini directly ignores notes for Q&A and wrong-answer explanations. "
        "Retrieval-only skips Gemini for Q&A. Quiz questions still use your notes."
    ),
)

upload_tab, ask_tab, cards_tab, practice_tab, quiz_tab, report_tab = st.tabs(
    ["Upload", "Questions", "Flashcards", "Practice", "Quiz", "Report"]
)

with upload_tab:
    uploaded_files = st.file_uploader(
        "Choose study files",
        accept_multiple_files=True,
    )
    st.caption("Supported: PDF, Word, PowerPoint (.pptx), text, Markdown, CSV, Excel, and PNG/JPG images.")
    if st.button("Process uploaded files", disabled=not uploaded_files):
        all_chunks = []
        statuses = []
        for uploaded_file in uploaded_files:
            text = extract_text(uploaded_file)
            statuses.append({"name": uploaded_file.name, "has_text": bool(text.strip())})
            if text.strip():
                all_chunks.extend(chunk_text(text, uploaded_file.name))
            else:
                st.warning(f"No extractable text was found in {uploaded_file.name}.")
        st.session_state.chunks = all_chunks
        st.session_state.vectorizer, st.session_state.matrix = build_index(all_chunks)
        st.session_state.file_statuses = statuses
        st.session_state.document_version += 1
        st.session_state.chat_messages = []
        st.success(f"Processed {len(uploaded_files)} file(s) and created {len(all_chunks)} text chunk(s).")

    if st.session_state.file_statuses:
        st.subheader("Files loaded")
        for status in st.session_state.file_statuses:
            label = "text extracted" if status["has_text"] else "no text extracted"
            st.write(f"- {status['name']} — {label}")
        st.write(f"**Chunks created:** {len(st.session_state.chunks)}")

# Answer from retrieved context, or show the matches without using Gemini.
with ask_tab:
    st.subheader("Ask Questions")
    for index, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("matches"):
                st.write("Sources: " + ", ".join(message["sources"]))
                with st.expander("Retrieved chunks and similarity scores"):
                    for match in message["matches"]:
                        st.markdown(f"**{match['source']} — similarity {match['score']:.3f}**")
                        st.write(match["text"])
    question = st.chat_input("Ask about your uploaded files")
    if question:
        matches = []
        if answer_mode == "Use Gemini directly":
            prompt = (
                "Answer using your general knowledge only. Do not refer to uploaded documents or "
                "claim that your answer came from a file. Explain the answer clearly and mention "
                "uncertainty when appropriate.\n\n"
                f"Question: {question}"
            )
            answer = ask_model(prompt)
            if not answer:
                answer = st.session_state.get("last_model_error") or "Gemini returned no answer."
        elif not st.session_state.chunks:
            answer = "Upload and process study files first, or select Gemini directly in the sidebar."
        else:
            matches = retrieve(question, st.session_state.vectorizer, st.session_state.matrix, st.session_state.chunks)
            sources = list(dict.fromkeys(match["source"] for match in matches))
            if answer_mode == "Retrieval-only Q&A":
                answer = (
                    "Here are the most relevant chunks from your uploaded files."
                    if matches
                    else "No matching document text was found for that question."
                )
            else:
                if matches:
                    context = "\n\n".join(
                        f"Source: {match['source']}\n{match['text']}" for match in matches
                    )
                    prompt = (
                        "Use only the context below. If the answer is not in the context, "
                        "say you couldn't find it. Mention the source file names when relevant.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}"
                    )
                    answer = ask_model(prompt)
                    if not answer:
                        answer = st.session_state.get("last_model_error") or "Gemini returned no answer."
                else:
                    answer = "I couldn't find matching information in the uploaded files."
        sources = list(dict.fromkeys(match["source"] for match in matches))
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer, "matches": matches, "sources": sources}
        )
        st.rerun()

# Cache generated flashcards so repeated clicks do not spend another API request.
with cards_tab:
    st.subheader("Flashcards")
    card_topic = st.text_input("Optional topic for flashcards", key="card_topic")
    if st.button("Generate 50 flashcards"):
        if not st.session_state.chunks:
            st.warning("Upload and process study files before generating flashcards.")
        else:
            normalized_topic = card_topic.strip().lower()
            cache_key = f"{st.session_state.document_version}:50:{normalized_topic}"
            if cache_key not in st.session_state.flashcards_cache:
                matches = retrieve(
                    card_topic or "important concepts terms definitions and facts",
                    st.session_state.vectorizer,
                    st.session_state.matrix,
                    st.session_state.chunks,
                    top_k=5,
                )
                if matches:
                    cards = generate_flashcards(matches, card_topic.strip())
                    if cards:
                        st.session_state.flashcards_cache[cache_key] = cards
                else:
                    st.warning("No matching document text was found for that topic.")
            if cache_key in st.session_state.flashcards_cache:
                st.session_state.active_flashcards_key = cache_key
                st.session_state.flashcard_index = 0
                st.session_state.show_flashcard_answer = False

    active_key = st.session_state.active_flashcards_key
    if (
        active_key in st.session_state.flashcards_cache
        and len(st.session_state.flashcards_cache[active_key]) == 50
    ):
        cards = st.session_state.flashcards_cache[active_key]
        card_index = st.session_state.flashcard_index
        card = cards[card_index]
        is_flipped = st.session_state.show_flashcard_answer
        question = html.escape(str(card["front"]))
        answer = html.escape(str(card["back"]))
        flip_class = " is-flipped" if is_flipped else ""
        st.caption(f"Card {card_index + 1} of {len(cards)}")
        st.markdown(
            f"""
            <div class="flashcard-stage">
                <div class="study-flashcard{flip_class}">
                    <div class="flashcard-face flashcard-front">
                        <div class="flashcard-label">QUESTION</div>
                        <div class="flashcard-text">{question}</div>
                        <div class="flashcard-hint">Think of your answer, then flip the card.</div>
                    </div>
                    <div class="flashcard-face flashcard-back">
                        <div class="flashcard-label">ANSWER</div>
                        <div class="flashcard-text">{answer}</div>
                        <div class="flashcard-hint">Flip back when you’re ready to review.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Flip to question" if is_flipped else "Flip to answer",
            key=f"flip_flashcard_{active_key}_{card_index}",
        ):
            st.session_state.show_flashcard_answer = not is_flipped
            st.rerun()
        previous, next_card = st.columns(2)
        with previous:
            if st.button("Previous", disabled=st.session_state.flashcard_index == 0):
                st.session_state.flashcard_index -= 1
                st.session_state.show_flashcard_answer = False
                st.rerun()
        with next_card:
            if st.button("Next", disabled=st.session_state.flashcard_index >= len(cards) - 1):
                st.session_state.flashcard_index += 1
                st.session_state.show_flashcard_answer = False
                st.rerun()

# Generate plain short-answer questions that students can answer before revealing guidance.
with practice_tab:
    st.subheader("Open-ended practice")
    st.caption("Practice with plain questions from your notes—there are no multiple-choice options.")
    practice_topic = st.text_input("Optional topic for practice", key="practice_topic")
    practice_count = st.slider(
        "Number of practice questions",
        min_value=5,
        max_value=20,
        value=10,
        key="practice_question_count",
    )
    if st.button("Generate practice questions"):
        if not st.session_state.chunks:
            st.warning("Upload and process study files before generating practice questions.")
        else:
            normalized_topic = practice_topic.strip()
            cache_key = (
                f"{st.session_state.document_version}:{practice_count}:{normalized_topic.lower()}"
            )
            if cache_key not in st.session_state.practice_cache:
                matches = retrieve(
                    normalized_topic or "important concepts and facts",
                    st.session_state.vectorizer,
                    st.session_state.matrix,
                    st.session_state.chunks,
                    top_k=5,
                )
                if matches:
                    questions = generate_practice_questions(matches, normalized_topic, practice_count)
                    if questions:
                        st.session_state.practice_cache[cache_key] = questions
                else:
                    st.warning("No matching document text was found for that topic.")
            if cache_key in st.session_state.practice_cache:
                st.session_state.active_practice_key = cache_key
                st.session_state.practice_index = 0
                st.session_state.show_practice_answer = False

    active_practice_key = st.session_state.active_practice_key
    if active_practice_key in st.session_state.practice_cache:
        practice_questions = st.session_state.practice_cache[active_practice_key]
        practice_index = st.session_state.practice_index
        practice_item = practice_questions[practice_index]
        st.caption(f"Question {practice_index + 1} of {len(practice_questions)}")
        st.markdown(f"### {practice_item['question']}")
        st.text_area(
            "Your answer (optional)",
            key=f"practice_answer_{active_practice_key}_{practice_index}",
            disabled=st.session_state.show_practice_answer,
        )
        if st.button(
            "Show suggested answer",
            key=f"reveal_practice_{active_practice_key}_{practice_index}",
            disabled=st.session_state.show_practice_answer,
        ):
            st.session_state.show_practice_answer = True
            st.rerun()
        if st.session_state.show_practice_answer:
            st.markdown(f"**Suggested answer:** {practice_item['answer']}")
            st.write(practice_item["explanation"])
            st.caption(f"Topic: {practice_item['topic']} · Source: {practice_item['source']}")

        previous, next_question = st.columns(2)
        with previous:
            if st.button(
                "Previous",
                key=f"previous_practice_{active_practice_key}_{practice_index}",
                disabled=practice_index == 0,
            ):
                st.session_state.practice_index -= 1
                st.session_state.show_practice_answer = False
                st.rerun()
        with next_question:
            if st.button(
                "Next",
                key=f"next_practice_{active_practice_key}_{practice_index}",
                disabled=practice_index >= len(practice_questions) - 1,
            ):
                st.session_state.practice_index += 1
                st.session_state.show_practice_answer = False
                st.rerun()

# Load a cached quiz or generate one, then score it locally on submission.
with quiz_tab:
    st.subheader("Quiz")
    quiz_topic = st.text_input("Topic (leave blank for whole documents)", key="quiz_topic")
    question_count = st.slider("Number of questions", min_value=5, max_value=60, value=5)
    if st.button("Generate quiz"):
        if not st.session_state.chunks:
            st.warning("Upload and process study files before generating a quiz.")
        else:
            normalized_topic = quiz_topic.strip()
            cache_key = (
                f"{st.session_state.document_version}:{normalized_topic.lower()}:{question_count}"
            )
            if cache_key not in st.session_state.quiz_cache:
                matches = retrieve(
                    normalized_topic or "important concepts and facts",
                    st.session_state.vectorizer,
                    st.session_state.matrix,
                    st.session_state.chunks,
                    top_k=5,
                )
                if matches:
                    questions = generate_quiz(matches, normalized_topic, question_count)
                    if questions:
                        st.session_state.quiz_cache[cache_key] = questions
                else:
                    st.warning("No matching document text was found for that topic.")
            if cache_key in st.session_state.quiz_cache:
                st.session_state.current_quiz_key = cache_key
                st.session_state.quiz_submitted = False
                st.session_state.quiz_result = None
                st.session_state.success_sound_played = False
                st.session_state.wrong_explanation_error = ""

    current_key = st.session_state.current_quiz_key
    if current_key in st.session_state.quiz_cache:
        questions = st.session_state.quiz_cache[current_key]
        selected_answers = {}
        for index, question_item in enumerate(questions):
            st.markdown(f"**{index + 1}. {question_item['question']}**")
            render_question_reader(question_item["question"], question_item["options"], index + 1)
            option_key = f"quiz_answer_{current_key}_{index}"
            selected = st.radio(
                "Choose one answer",
                question_item["options"],
                index=None,
                key=option_key,
                disabled=st.session_state.quiz_submitted,
                label_visibility="collapsed",
            )
            if selected is not None:
                selected_answers[index] = question_item["options"].index(selected)

        if not st.session_state.quiz_submitted:
            if st.button("Submit quiz"):
                result = mark_quiz(questions, selected_answers)
                st.session_state.wrong_explanation_error = ""
                if any(
                    not item["is_correct"] and item["selected_index"] is not None
                    for item in result["questions"]
                ):
                    with st.spinner("Explaining missed answers..."):
                        explanations = explain_wrong_answers(
                            result["questions"],
                            st.session_state.chunks,
                            st.session_state.vectorizer,
                            st.session_state.matrix,
                            use_notes=answer_mode != "Use Gemini directly",
                        )
                    for question_index, explanation in explanations.items():
                        result["questions"][question_index]["wrong_explanation"] = explanation
                    result["wrong_explanation_error"] = st.session_state.wrong_explanation_error
                st.session_state.quiz_result = result
                st.session_state.quiz_submitted = True
                st.session_state.success_sound_played = False
                st.session_state.quiz_history.append(result)
                st.session_state.history_version += 1
                try:
                    save_quiz_history(st.session_state.quiz_history)
                    st.session_state.history_storage_error = ""
                except OSError as error:
                    st.session_state.history_storage_error = f"Could not save quiz history: {error}"
                st.rerun()

        if st.session_state.quiz_result:
            result = st.session_state.quiz_result
            st.markdown(f"## Score: {result['score']} / {result['total']} ({result['percent']}%)")
            if result["score"] > 0 and not st.session_state.success_sound_played:
                st.audio(make_success_sound(), format="audio/wav", autoplay=True)
                st.caption("A soft chime plays when your submitted quiz includes a correct answer.")
                st.session_state.success_sound_played = True
            if result.get("wrong_explanation_error"):
                st.info(result["wrong_explanation_error"])
            for index, item in enumerate(result["questions"]):
                label = "Correct" if item["is_correct"] else (
                    "Unanswered" if item["selected_index"] is None else "Wrong"
                )
                st.markdown(f"**{index + 1}. {label} — {item['topic']}**")
                if not item["is_correct"]:
                    correct_answer = item["options"][item["answer_index"]]
                    if item["selected_index"] is not None:
                        st.write(f"Your answer: {item['options'][item['selected_index']]}")
                    st.write(f"Correct answer: {correct_answer}")
                    st.markdown(f"**Why the correct answer is right:** {item['explanation']}")
                    if item.get("wrong_explanation"):
                        st.markdown(f"**Why your answer was wrong:** {item['wrong_explanation']}")
                else:
                    st.write(item["explanation"])
                st.caption(f"Source: {item['source']}")

# Build one cached revision report from all quiz attempts so far.
with report_tab:
    st.subheader("Study Report")
    st.caption("Quiz attempts are saved locally in quiz_history.json.")
    if st.session_state.history_storage_error:
        st.error(st.session_state.history_storage_error)
    if not st.session_state.quiz_history:
        st.info("Submit a quiz to create a study report.")
    else:
        with st.expander(f"Quiz history ({len(st.session_state.quiz_history)} attempts)", expanded=True):
            for attempt_number, attempt in enumerate(reversed(st.session_state.quiz_history), start=1):
                completed_at = attempt.get("completed_at", "Earlier attempt")
                st.write(
                    f"**Attempt {len(st.session_state.quiz_history) - attempt_number + 1}** · "
                    f"{completed_at} — {attempt['score']} / {attempt['total']} ({attempt['percent']}%)"
                )

        report_key = f"{st.session_state.document_version}:{st.session_state.history_version}"
        if st.button("Generate study report") and report_key not in st.session_state.report_cache:
            st.session_state.report_cache[report_key] = build_study_report(
                st.session_state.quiz_history,
                st.session_state.chunks,
                st.session_state.vectorizer,
                st.session_state.matrix,
            )
        if report_key in st.session_state.report_cache:
            report = st.session_state.report_cache[report_key]
            st.markdown(report)
            st.download_button(
                "Download report as Markdown",
                data=report,
                file_name="study_report.md",
                mime="text/markdown",
            )
