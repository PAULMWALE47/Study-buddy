import io
import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st
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


# Generate a small set of flashcards from retrieved document context.
def generate_flashcards(matches, topic=""):
    context = "\n\n".join(f"Source: {item['source']}\n{item['text']}" for item in matches)
    topic_line = f"Focus on this topic: {topic}." if topic else "Cover the most important concepts."
    prompt = (
        "Use only the context below. Create about 10 useful study flashcards. "
        'Return JSON only as an array of objects: [{"front": "...", "back": "..."}].\n'
        f"{topic_line}\n\nContext:\n{context}"
    )
    response = ask_model(prompt, want_json=True)
    if not response:
        return []
    try:
        cards = parse_json_response(response)
        if not isinstance(cards, list) or not all(
            isinstance(card, dict) and card.get("front") and card.get("back") for card in cards
        ):
            raise ValueError("The response did not contain valid flashcards.")
        return cards
    except (json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read the generated flashcards: {error}")
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
    "quiz_cache": {},
    "current_quiz_key": None,
    "quiz_submitted": False,
    "quiz_result": None,
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
        --ink: #17263c;
        --muted: #64748b;
        --accent: #148a78;
        --line: #e2e8f0;
        --surface: #ffffff;
    }

    [data-testid="stAppViewContainer"] {
        background: #f5f7fb;
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #edf2f7;
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
        border: 1px solid #dce8e7;
        border-radius: 22px;
        background: linear-gradient(120deg, #ffffff 15%, #e8f5f2 100%);
        box-shadow: 0 12px 32px rgba(23, 38, 60, 0.06);
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
        background: #e8f5f2;
    }
    h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.7rem;
        padding: 0.55rem 1.05rem;
        border: 1px solid #cbd9df;
        border-radius: 10px;
        color: var(--ink);
        background: var(--surface);
        font-weight: 650;
        transition: border-color 120ms ease, background 120ms ease, transform 120ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
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
        border: 1px dashed #b9c9d2;
        border-radius: 12px;
        background: var(--surface);
    }
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {
        color: var(--muted) !important;
    }
    div[data-testid="stFileUploader"] button {
        border: 1px solid #cbd9df;
        border-radius: 9px;
        color: var(--ink) !important;
        background: #ffffff !important;
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
        border-color: #d5dee8;
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
retrieval_only = st.sidebar.toggle("Retrieval-only mode (no AI)", value=False)

upload_tab, ask_tab, cards_tab, quiz_tab, report_tab = st.tabs(
    ["Upload", "Questions", "Flashcards", "Quiz", "Report"]
)

with upload_tab:
    uploaded_files = st.file_uploader(
        "Choose study files",
        accept_multiple_files=True,
    )
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
        if not st.session_state.chunks:
            st.warning("Upload and process study files before asking a question.")
        else:
            matches = retrieve(question, st.session_state.vectorizer, st.session_state.matrix, st.session_state.chunks)
            sources = list(dict.fromkeys(match["source"] for match in matches))
            st.session_state.chat_messages.append({"role": "user", "content": question})
            if retrieval_only:
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
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer, "matches": matches, "sources": sources}
            )
            st.rerun()

# Cache generated flashcards so repeated clicks do not spend another API request.
with cards_tab:
    st.subheader("Flashcards")
    card_topic = st.text_input("Optional topic for flashcards", key="card_topic")
    if st.button("Generate flashcards"):
        if not st.session_state.chunks:
            st.warning("Upload and process study files before generating flashcards.")
        else:
            cache_key = f"{st.session_state.document_version}:{card_topic.strip().lower()}"
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
    if active_key in st.session_state.flashcards_cache:
        cards = st.session_state.flashcards_cache[active_key]
        card = cards[st.session_state.flashcard_index]
        st.markdown(f"**Card {st.session_state.flashcard_index + 1} of {len(cards)}**")
        st.markdown(f"### {card['front']}")
        if st.button("Show answer", disabled=st.session_state.show_flashcard_answer):
            st.session_state.show_flashcard_answer = True
            st.rerun()
        if st.session_state.show_flashcard_answer:
            st.info(card["back"])
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

    current_key = st.session_state.current_quiz_key
    if current_key in st.session_state.quiz_cache:
        questions = st.session_state.quiz_cache[current_key]
        selected_answers = {}
        for index, question_item in enumerate(questions):
            st.markdown(f"**{index + 1}. {question_item['question']}**")
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
                st.session_state.quiz_result = result
                st.session_state.quiz_submitted = True
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
            for index, item in enumerate(result["questions"]):
                label = "Correct" if item["is_correct"] else "Wrong"
                st.markdown(f"**{index + 1}. {label} — {item['topic']}**")
                if not item["is_correct"]:
                    correct_answer = item["options"][item["answer_index"]]
                    st.write(f"Correct answer: {correct_answer}")
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
