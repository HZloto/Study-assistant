# -*- coding: utf-8 -*-
import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai # New SDK import
from google.genai import types # New SDK import for types
# from PIL import Image # No longer needed unless images are added later
import ast # For safely evaluating the dictionary string from Gemini
import random # To shuffle flashcards if needed (optional)
import traceback # For printing full tracebacks
import json # For parsing JSON response directly

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        st.error("Error: GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment variables.")
        st.stop()
    else:
        st.warning("Using GEMINI_API_KEY. Consider switching to GOOGLE_API_KEY.")

# --- Client Initialization (Corrected as per latest SDK docs) ---
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error creating Google GenAI Client: {e}")
    traceback.print_exc()
    st.stop()

# Recommended model (Using 1.5 Flash, good for structured output and general tasks)
MODEL_NAME = "gemini-2.0-flash"
# If you specifically need 2.0-flash capabilities, change it back:
# MODEL_NAME = "gemini-2.0-flash"

# --- Helper Functions (extract_text_from_folder, get_available_courses remain same) ---

def extract_text_from_folder(folder_path):
    # (Function remains the same)
    all_text = ""
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' not found."
    try:
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        txt_files.sort()
        for file_name in txt_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                all_text += file.read() + "\n\n"
        return all_text.strip()
    except Exception as e:
        st.error(f"Error reading files from {folder_path}: {e}")
        return ""

def get_available_courses():
    # (Function remains the same)
    courses_dir = "Courses"
    if not os.path.exists(courses_dir):
        return []
    try:
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        return sorted([folder for folder in os.listdir(courses_dir)
                if os.path.isdir(os.path.join(courses_dir, folder))],
                key=natural_sort_key)
    except Exception as e:
        st.error(f"Error accessing Courses directory: {e}")
        return []


# --- Text Extraction Helper (Unified for both generation functions) ---
def safe_extract_response_text(response):
    """Safely extracts text from Gemini response object, handling potential errors."""
    # (Function remains the same as previous correct version)
    try:
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            first_candidate = response.candidates[0]
            if hasattr(first_candidate, 'content') and first_candidate.content.parts:
                 text_parts = [part.text for part in first_candidate.content.parts if hasattr(part, 'text')]
                 if text_parts:
                     return "".join(text_parts)

        elif hasattr(response, 'parts') and response.parts:
             return "".join(part.text for part in response.parts if hasattr(part, 'text'))

        print("DEBUG: Full Gemini Response Object in safe_extract_response_text:", response)
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        block_reason_prompt = None
        if prompt_feedback:
            block_reason_prompt = getattr(prompt_feedback, 'block_reason', None)

        block_reason_candidate = None
        finish_reason_candidate = "UNKNOWN"
        if hasattr(response, 'candidates') and response.candidates:
            try:
                first_candidate = response.candidates[0]
                finish_reason_candidate = getattr(first_candidate, 'finish_reason', 'UNKNOWN')
                if hasattr(first_candidate, 'safety_ratings'):
                    for rating in first_candidate.safety_ratings:
                         if hasattr(rating, 'blocked') and rating.blocked:
                             block_reason_candidate = f"BLOCKED (Category: {getattr(rating, 'category', 'UNKNOWN')})"
                             break
            except Exception as e:
                print(f"DEBUG: Error accessing candidate details: {e}")

        error_message = f"Received response, but couldn't extract text content."
        details = []
        if block_reason_prompt: details.append(f"Prompt Feedback: BLOCKED ({block_reason_prompt})")
        if block_reason_candidate: details.append(f"Candidate Info: {block_reason_candidate}")
        elif finish_reason_candidate not in ("STOP", "UNKNOWN", None):
             finish_reason_str = str(finish_reason_candidate)
             if finish_reason_str not in ('FinishReason.STOP', 'FinishReason.UNKNOWN', 'None'):
                  details.append(f"Candidate Finish Reason: {finish_reason_str}")

        if details: error_message += " " + ". ".join(details) + "."
        st.warning(error_message)
        return None

    except Exception as e:
        st.error(f"Critical error during response text extraction: {e}")
        print(f"Response Text Extraction Error: {e}")
        traceback.print_exc()
        return None


# --- generate_response Function (CORRECTED to use client.models.generate_content) ---
def generate_response(prompt, lecture_notes_text, exam_mode=False, chat_history=None):
    """
    Generates a response using the Gemini model with context and chat history,
    aligned with the latest google-genai SDK practices using client.models.

    Args:
        prompt (str): The user's latest input (question or answer).
        lecture_notes_text (str): The loaded course material.
        exam_mode (bool): Flag indicating if exam mode is active.
        chat_history (list): List of previous messages [{'role': 'user'/'model', 'content': str}]

    Returns:
        str: The generated text response from the model, or an error message.
    """
    global client # Use the global client initialized correctly

    try:
        # --- System Prompt Setup --- (Keep the detailed prompts)
        if exam_mode:
            system_instruction_text = """You are an expert tutor for the MIT course 14.73 (Challenges of Global Poverty).
You are in **EXAM MODE**. Your goal is to test the user's understanding with various question types based *strictly* on the provided lecture notes context (which starts with "COURSE CONTEXT:").

**HERE ARE EXAMPLES OF HOW THE EXAM SHOULD WORK:**

--- EXAMPLE 1 (Select All) ---
Tutor:
**Question 3:** For which types of borrower might a “flexible” repayment plan (e.g., a grace period or a repayment holiday) be most beneficial? Select all that apply.
A. One whose income is highly variable
B. One who needs to stick to a routine
C. One who plans to make a large investment that won’t pay off right away
D. One who just needs to pay for one large, unexpected expense (e.g., a medical bill)


Tutor:
✅ **Partially Correct!** (A and C are good choices)
**Explanation:**
*   Flexibility helps borrowers with volatile income smooth consumption.
*   Grace periods are useful for investments with delayed returns.
*   Routine-focused borrowers might prefer fixed schedules.
*   Single large expenses might be better handled by emergency funds or specific loan types, though flexibility *could* help repayment.
---
**Question 4:** [Next Question...]
--- EXAMPLE 1 END ---

--- EXAMPLE 2 (Fill-in-the-blank) ---
Tutor:
**Question 8:** Fill in the blank. Overall, the results from Barboni and Agarwal (2022) suggest that offering entrepreneurs a choice between a traditional loan and a flexible loan with a higher interest rate is ______ for the client and ______ for the bank.


Tutor:
✅ **Correct!**
**Explanation:**
*   Clients benefited from the choice (higher sales, flexibility).
*   Banks saw similar default rates but potentially slightly lower profits due to complexity or delayed repayment, hence neutral/ambiguous impact overall for the bank in that study.
---
**Question 9:** [Next Question...]
--- EXAMPLE 2 END ---

--- EXAMPLE 3 (True/False) ---
Tutor:
**Question 10:** True or false? Considering the evidence presented here and in lecture, increasing access to credit is not a path to increasing the wellbeing of a meaningful number of the world’s poor.
A. True
B. False


Tutor:
✅ **Correct!**
**Explanation:**
*   While average impacts might be small, evidence shows significant benefits for *some* borrowers.
*   Indirect effects (wages, competition) can be positive.
*   Innovations in microfinance continue to explore ways to improve impact. Access to credit *can* be a path, though not a universal solution.
---
**Question 11:** [Next Question...]
--- EXAMPLE 3 END ---

**YOUR TASK NOW:**

1.  Ask challenging questions aimed at covering the whole topic based *only* on the provided COURSE CONTEXT. Use a variety of formats:
    *   Standard Multiple Choice (A, B, C, D) - ask for a single letter answer.
    *   Select All That Apply (A, B, C, D, E, F...) - ask user to list letters separated by commas (e.g., "A, C").
    *   Fill-in-the-Blank - indicate blanks with `______` and ask user to provide the missing word(s), separated by commas if multiple blanks.
    *   True/False - present as A/B options or ask for "True" or "False".
2.  Clearly format each question starting with "**Question X:**".
3.  Wait for the user's answer. The user might provide single letters, multiple letters (e.g., "A, C", "A,C"), words ("good, neutral"), or "True"/"False".
4.  When you receive the user's answer, evaluate it based on the *immediately preceding* question you asked.
5.  Start your response *immediately* with "✅ **Correct!**", "❌ **Incorrect.**", or "✅ **Partially Correct!**".
6.  Provide a concise **Explanation:** using 2-4 bullet points derived *only* from the lecture notes, explaining why the answer is right, wrong, or partially right.
7.  Use a markdown separator (`---`).
8.  Immediately after the separator, provide the *next* question, formatted correctly.
9.  Do NOT add conversational filler. Just provide evaluation, explanation, separator, and the next question.
10. If the user's input doesn't seem like a valid answer format *for the question you just asked* (e.g., they type a full sentence), gently remind them of the expected format for *that question type* or tell them they can type 'exit exam mode'.
11. If the user asks to exit, acknowledge and stop providing questions.
"""
        else: # Standard Tutor Mode
            system_instruction_text = """You are a helpful and concise tutor for the MIT course 14.73 (Challenges of Global Poverty).
Use the provided lecture notes context (which starts with "COURSE CONTEXT:") as your primary knowledge source to answer student questions clearly.
If a question cannot be answered from the notes, say so.
Be conversational but stick to the course material."""

        # --- Construct `contents` list for the API (Using types.Content) ---
        api_contents = []
        context_string = f"COURSE CONTEXT:\n{lecture_notes_text}"
        # Prepend context and role confirmation as the first turns
        api_contents.append(types.Content(role="user", parts=[types.Part(text=context_string)]))
        api_contents.append(types.Content(role="model", parts=[types.Part(text="Okay, I have loaded the course context and understand my role.")]))

        if chat_history:
            for message in chat_history:
                # Filtering logic (remains the same)
                if message["role"] == "model" and (
                    message["content"].startswith("Welcome!") or
                    "loaded. Ask me questions" in message["content"] or
                    "exiting exam mode" in message["content"] or
                    "exiting flashcard mode" in message["content"] or
                    "already in exam mode" in message["content"] or
                    message["content"].startswith("Okay, I have loaded") or
                    "Please answer with a single letter" in message["content"]
                    ):
                    continue
                if message["role"] == "user" and (
                     message["content"].startswith("COURSE CONTEXT:") or
                     message["content"].strip().lower() == "exit exam mode" or
                     message["content"].strip().lower() == "exit flashcard mode" or
                     message["content"].strip().lower() in ["start exam mode", "final exam mode", "exam", "start exam", "start flashcard mode", "flashcards"]
                     ):
                     continue

                # Append valid history messages using types.Content
                if isinstance(message["content"], str):
                    try:
                        api_contents.append(types.Content(role=message["role"], parts=[types.Part(text=message["content"])]))
                    except ValueError as ve: # Handle potential invalid roles
                         print(f"Skipping message due to invalid role '{message['role']}': {ve}")
                         st.warning(f"Skipping message with role {message['role']}")
                         continue
                else:
                     st.warning(f"Skipping message with unexpected content type: {type(message['content'])}")

        # Add the latest user prompt as the final Content object
        api_contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        # --- Prepare Generation Config (Using types.GenerateContentConfig) ---
        generation_config_object = types.GenerateContentConfig(
            temperature=0.5 if exam_mode else 0.7,
            max_output_tokens=3072,
            # Include system instruction here as per latest SDK docs
            system_instruction=system_instruction_text
        )

        # --- Call the Gemini API using client.models.generate_content ---
        response = client.models.generate_content(
            model=MODEL_NAME, # Use the configured model name
            contents=api_contents,
            config=generation_config_object
        )

        # --- Safely Extract Text using the unified helper ---
        response_text = safe_extract_response_text(response)

        if response_text is None:
             # Error/warning already handled by helper
             return "Sorry, I couldn't generate a valid response. The request might have been blocked or resulted in no usable output. Please check the logs or try rephrasing."
        else:
            return response_text

    except Exception as e:
        # Catch potential API errors or other issues
        st.error(f"Error generating response from Gemini: {e}")
        print(f"Gemini API Error during generate_response: {e}")
        traceback.print_exc()
        # Provide a user-friendly error message
        return "An critical error occurred while trying to get a response. Please check the server logs or contact support."


# --- generate_flashcards Function (Remains as updated in previous step - already correct) ---
def generate_flashcards(lecture_notes_text, course_name):
    """
    Generates flashcards using the Gemini model based on lecture notes,
    aligned with the latest google-genai SDK practices.
    """
    global client # Use the globally defined client object

    flashcard_prompt = f"""
    Based strictly on the following course material for {course_name}, generate a minimum of 20 flashcards.
    Focus on the most important key concepts, definitions, methodologies, findings, and potential ambiguities discussed that are likely to be tested on a final exam. Ensure the questions require understanding beyond simple recall where possible.

    Return the flashcards *only* as a valid JSON object (a dictionary) where keys are the questions (front of the card) and values are the concise answers (back of the card). The output must be *only* the JSON object itself, starting with {{ and ending with }}, with no surrounding text, comments, explanations, or markdown code fences like ```json ... ```.

    Example format:
    {{
        "What is the 'poverty trap' concept discussed in the context of Sachs?": "A self-reinforcing mechanism where poverty leads to conditions (like low savings, poor health, low education) that prevent escaping poverty, requiring external aid ('big push') to break the cycle.",
        "According to Banerjee & Duflo, what is a common characteristic of the businesses run by the poor?": "They are often small-scale, undifferentiated, operate in crowded markets, and face constraints like lack of access to credit or insurance, limiting growth potential.",
        "What was a key finding of the study on deworming in Kenya regarding educational outcomes?": "Deworming significantly reduced school absenteeism and improved attendance, suggesting health interventions can have substantial educational benefits.",
        "Define 'randomized controlled trial' (RCT) in the context of development economics.": "An experimental method where eligible units (individuals, villages, schools) are randomly assigned to either receive an intervention (treatment group) or not (control group) to measure the causal impact of the intervention.",
        "What is a potential limitation of using average treatment effects from RCTs to guide policy?": "Average effects might hide significant variation (heterogeneity) in impact across different subgroups, meaning the intervention might be highly effective for some but ineffective or harmful for others."
    }}

    COURSE MATERIAL:
    ---
    {lecture_notes_text}
    ---

    Generate the JSON object containing at least 20 flashcards now:
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=flashcard_prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
        )

        response_text = safe_extract_response_text(response)
        if not response_text:
            st.error("Flashcard generation failed: Received no text content from the model even when requesting JSON.")
            print("DEBUG: Full response object when text was missing (JSON requested):", response)
            return None

        try:
            flashcards_dict = json.loads(response_text)
            if not isinstance(flashcards_dict, dict):
                st.error("Flashcard generation failed: Parsed JSON result is not a dictionary.")
                print("DEBUG: Parsed JSON result type:", type(flashcards_dict))
                print("DEBUG: Raw response text:", response_text)
                return None
            if len(flashcards_dict) == 0:
                 st.error("Flashcard generation resulted in an empty dictionary.")
                 return None
            if len(flashcards_dict) < 20:
                 st.warning(f"Generated {len(flashcards_dict)} flashcards, which is less than the requested minimum of 20. Proceeding anyway.")

            flashcards_dict = {str(k).strip(): str(v).strip() for k, v in flashcards_dict.items()}
            return flashcards_dict

        except json.JSONDecodeError as e:
            st.error(f"Flashcard generation failed: Could not parse the response as valid JSON: {e}")
            st.error("The model did not return a valid JSON object despite being asked to.")
            print("DEBUG: Raw response text that failed JSON parsing:\n", response_text)
            return None

    except Exception as e:
        st.error(f"An error occurred during flashcard generation: {e}")
        print(f"Gemini API Error during generate_flashcards: {e}")
        traceback.print_exc()
        return None


# --- Streamlit App (UI Code remains the same) ---

st.set_page_config(page_title="Study Bot", page_icon="📚", layout="wide")

# --- Initialize Session State --- (Remains the same)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "content": "Welcome! Please select a course to load materials."}]
if "exam_mode" not in st.session_state:
    st.session_state.exam_mode = False
if "flashcard_mode" not in st.session_state:
    st.session_state.flashcard_mode = False
if "current_lecture_text" not in st.session_state:
    st.session_state.current_lecture_text = ""
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0
if "show_flashcard_answer" not in st.session_state:
    st.session_state.show_flashcard_answer = False

# --- Sidebar --- (Remains the same)
with st.sidebar:
    st.title("📚 Study Bot")
    st.markdown(f"**Model:** `{MODEL_NAME}`")

    st.subheader("Course Selection")
    available_courses = get_available_courses()

    if not available_courses:
        st.error("No courses found in the 'Courses' directory!")
        st.stop()

    selected_course_index = available_courses.index(st.session_state.selected_course) if st.session_state.selected_course in available_courses else 0

    def on_course_change():
        st.session_state.exam_mode = False
        st.session_state.flashcard_mode = False
        st.session_state.flashcards = None
        st.session_state.flashcard_index = 0
        st.session_state.show_flashcard_answer = False
        st.session_state.messages = [{"role": "model", "content": "Select 'Load Course Materials' below."}]
        st.session_state.current_lecture_text = ""

    new_selection = st.selectbox(
        "Select a course", available_courses, index=selected_course_index, key="course_selector", on_change=on_course_change
    )
    if new_selection != st.session_state.selected_course:
       st.session_state.selected_course = new_selection

    if st.button("Load Course Materials", key="load_course_button", disabled=(st.session_state.selected_course is None)):
        if st.session_state.selected_course:
            course_path = os.path.join("Courses", st.session_state.selected_course)
            with st.spinner(f"Loading materials for {st.session_state.selected_course}..."):
                lecture_text = extract_text_from_folder(course_path)
                if lecture_text and not lecture_text.startswith("Error:"):
                    st.session_state.current_lecture_text = lecture_text
                    st.session_state.messages = [
                        {"role": "model", "content": f"Course '{st.session_state.selected_course}' loaded. Ask questions, start exam, or generate flashcards."}
                    ]
                    st.session_state.exam_mode = False
                    st.session_state.flashcard_mode = False
                    st.session_state.flashcards = None
                    st.session_state.flashcard_index = 0
                    st.session_state.show_flashcard_answer = False
                    st.success(f"Loaded '{st.session_state.selected_course}' successfully!")
                    st.rerun()
                else:
                    error_msg = lecture_text if lecture_text and lecture_text.startswith("Error:") else f"Failed to load lecture text for {st.session_state.selected_course}."
                    st.error(error_msg)
                    st.session_state.current_lecture_text = ""
                    st.session_state.messages = [{"role": "model", "content": "Failed to load course. Please select another."}]
                    st.session_state.exam_mode = False
                    st.session_state.flashcard_mode = False

    if st.session_state.current_lecture_text:
        st.divider()
        st.subheader("Study Mode")

        mode_active = st.session_state.exam_mode or st.session_state.flashcard_mode

        if st.session_state.exam_mode:
            if st.button("❌ Exit Exam Mode", type="secondary", key="exit_exam", use_container_width=True):
                st.session_state.exam_mode = False
                st.session_state.messages.append({"role": "model", "content": "Okay, exiting exam mode. How can I help you?"})
                st.rerun()
        elif st.session_state.flashcard_mode:
             if st.button("❌ Exit Flashcard Mode", type="secondary", key="exit_flashcards", use_container_width=True):
                st.session_state.flashcard_mode = False
                st.session_state.messages.append({"role": "model", "content": "Exited flashcard mode. Ask me questions or start exam mode."})
                st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Start Exam", type="primary", key="start_exam", use_container_width=True):
                    st.session_state.exam_mode = True
                    st.session_state.flashcard_mode = False
                    initial_exam_prompt = "Please start the final exam by providing the first question based on the course material. Use varied question types."
                    history_for_first_question = [msg for msg in st.session_state.messages if not (msg["role"] == "model" and (msg["content"].startswith("Welcome!") or msg["content"].startswith("Select 'Load")))]
                    with st.spinner("Starting exam and preparing the first question..."):
                        response = generate_response( # NOW calls the corrected function
                            initial_exam_prompt,
                            st.session_state.current_lecture_text,
                            exam_mode=True,
                            chat_history=history_for_first_question
                        )
                    st.session_state.messages.append({"role": "model", "content": response})
                    st.rerun()
            with col2:
                if st.button("⚡ Generate Flashcards", type="primary", key="start_flashcards", use_container_width=True):
                    st.session_state.flashcard_mode = True
                    st.session_state.exam_mode = False
                    st.session_state.flashcards = None
                    st.session_state.flashcard_index = 0
                    st.session_state.show_flashcard_answer = False
                    with st.spinner(f"Generating flashcards for {st.session_state.selected_course}... (This may take a moment)"):
                        flashcards_dict = generate_flashcards(st.session_state.current_lecture_text, st.session_state.selected_course) # Uses the correct SDK pattern
                    if flashcards_dict:
                        st.session_state.flashcards = list(flashcards_dict.items())
                        if not st.session_state.flashcards:
                             st.error("Failed to generate flashcards (empty result). Please try again.")
                             st.session_state.flashcard_mode = False
                        else:
                            st.success(f"Generated {len(st.session_state.flashcards)} flashcards!")
                    else:
                        st.error("Failed to generate flashcards. Please try again or check the logs.")
                        st.session_state.flashcard_mode = False
                    st.rerun()

        st.divider()
        st.subheader("Course Info")
        word_count = len(st.session_state.current_lecture_text.split()) if st.session_state.current_lecture_text else 0
        st.write(f"**Course:** {st.session_state.selected_course or 'None Selected'}")
        st.write(f"**Word Count:** {word_count:,}")

        status_text = "💬 QA Mode"
        if st.session_state.exam_mode:
            status_text = "📝 Exam Mode Active"
        elif st.session_state.flashcard_mode:
             card_count = len(st.session_state.flashcards) if st.session_state.flashcards else 0
             status_text = f"⚡ Flashcard Mode Active ({card_count} cards)"
        st.markdown(f"**Status:** {status_text}")

    st.divider()
    st.subheader("Tips")
    st.markdown("- **QA Mode:** Ask questions about the loaded text.")
    st.markdown("- **Exam Mode:** Answer varied questions. Check format hints if needed.")
    st.markdown("- **Flashcards:** Generate cards on key concepts. Use buttons to navigate.")

# --- Main Interface --- (Remains the same)
title_text = f"📚 {st.session_state.selected_course}" if st.session_state.selected_course else "📚 Study Bot"
st.title(title_text)

# --- Flashcard Mode Display (Clean UI) --- (Remains the same)
if st.session_state.flashcard_mode:
    if st.session_state.flashcards and len(st.session_state.flashcards) > 0:
        total_cards = len(st.session_state.flashcards)
        current_index = st.session_state.flashcard_index
        if current_index >= total_cards:
             st.session_state.flashcard_index = 0
             current_index = 0
             st.session_state.show_flashcard_answer = False

        question, answer = st.session_state.flashcards[current_index]

        st.caption(f"Card {current_index + 1} of {total_cards}")

        with st.container(border=True):
            st.subheader("❓ Question")
            st.markdown(f"> {question}")

            if st.session_state.show_flashcard_answer:
                st.divider()
                st.subheader("💡 Answer")
                st.markdown(answer)
            else:
                st.write("\n" * 2)
                st.markdown("*(Click 'Show Answer' below)*")

        st.write("")

        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col1:
            if st.button("⬅️ Previous", key="prev_card", disabled=(current_index <= 0), use_container_width=True):
                st.session_state.flashcard_index -= 1
                st.session_state.show_flashcard_answer = False
                st.rerun()
        with col2:
            flip_text = "Show Answer" if not st.session_state.show_flashcard_answer else "Show Question"
            button_type = "primary" if not st.session_state.show_flashcard_answer else "secondary"
            if st.button(f"🔄 {flip_text}", key="flip_card", type=button_type, use_container_width=True):
                st.session_state.show_flashcard_answer = not st.session_state.show_flashcard_answer
                st.rerun()
        with col3:
            if st.button("Next ➡️", key="next_card", disabled=(current_index >= total_cards - 1), use_container_width=True):
                st.session_state.flashcard_index += 1
                st.session_state.show_flashcard_answer = False
                st.rerun()

    elif not st.session_state.flashcards and st.session_state.current_lecture_text:
         st.warning("Flashcards haven't been generated yet or generation failed. Click 'Generate Flashcards' in the sidebar.")
    else:
         if not st.session_state.selected_course:
              st.info("Please select and load a course from the sidebar to begin.")
         elif not st.session_state.flashcards:
              st.info("Use the 'Generate Flashcards' button in the sidebar to create flashcards for the loaded course.")

# --- QA and Exam Mode Display (Chat Interface - now uses corrected generate_response) ---
else:
    # Message display loop (remains same)
    for message in st.session_state.messages:
        if message["role"] == "user" and message["content"].startswith("COURSE CONTEXT:"):
            continue
        if message["role"] == "model" and (
            message["content"].startswith("Okay, I have loaded") or
            message["content"].startswith("Exited flashcard mode.") or
            message["content"].startswith("Welcome!") or
            message["content"].startswith("Select 'Load")
            ):
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input logic (remains same)
    prompt_disabled = not st.session_state.current_lecture_text
    prompt_placeholder = "Ask a question or enter your answer..."
    if not st.session_state.current_lecture_text:
        prompt_placeholder = "Please load course materials first"

    if prompt := st.chat_input(prompt_placeholder, disabled=prompt_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        history_for_api = st.session_state.messages[:-1]

        is_exit_command = prompt.strip().lower() == "exit exam mode"
        is_start_exam_command = prompt.strip().lower() in ["start exam mode", "final exam mode", "exam", "start exam"]

        # Command processing and calling generate_response (now corrected version)
        if is_exit_command:
            if st.session_state.exam_mode:
                st.session_state.exam_mode = False
                response = "Okay, exiting exam mode. How can I help you with the course material?"
                st.session_state.messages.append({"role": "model", "content": response})
            else:
                 response = "You are not currently in exam mode. How can I help?"
                 st.session_state.messages.append({"role": "model", "content": response})
            st.rerun()
        elif is_start_exam_command:
            if not st.session_state.exam_mode:
                st.session_state.exam_mode = True
                st.session_state.flashcard_mode = False
                initial_exam_prompt = "Please start the final exam by providing the first question based on the course material. Use varied question types."
                with st.spinner("Starting exam and preparing the first question..."):
                    response = generate_response( # Calls corrected function
                        initial_exam_prompt,
                        st.session_state.current_lecture_text,
                        exam_mode=True,
                        chat_history=history_for_api
                    )
                st.session_state.messages.append({"role": "model", "content": response})
            else:
                response = "You are already in exam mode. Please answer the question or type 'exit exam mode'."
                st.session_state.messages.append({"role": "model", "content": response})
            st.rerun()
        elif st.session_state.exam_mode:
             with st.spinner("Evaluating your answer and preparing the next question..."):
                response = generate_response( # Calls corrected function
                    prompt,
                    st.session_state.current_lecture_text,
                    exam_mode=True,
                    chat_history=history_for_api
                )
             st.session_state.messages.append({"role": "model", "content": response})
             st.rerun()
        else: # QA mode
            with st.spinner("Thinking..."):
                response = generate_response( # Calls corrected function
                    prompt,
                    st.session_state.current_lecture_text,
                    exam_mode=False,
                    chat_history=history_for_api
                )
            st.session_state.messages.append({"role": "model", "content": response})
            st.rerun()