# -*- coding: utf-8 -*-
import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai # New SDK import
from google.genai import types # New SDK import for types
from PIL import Image # Needed for potential image parts if used later

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

# Configure the Gemini client using the NEW SDK syntax
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error creating Google GenAI Client: {e}")
    st.stop()

# Recommended model
MODEL_NAME = "gemini-2.0-flash"

# --- Helper Functions ---

def extract_text_from_folder(folder_path):
    """Extracts text from all .txt files in a folder"""
    all_text = ""
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' not found."
    try:
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        txt_files.sort() # Ensure consistent order
        for file_name in txt_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                all_text += file.read() + "\n\n"
        return all_text.strip()
    except Exception as e:
        st.error(f"Error reading files from {folder_path}: {e}")
        return ""

def get_available_courses():
    """Get available courses from the Courses directory"""
    courses_dir = "Courses"
    if not os.path.exists(courses_dir):
        return []
    try:
        # Natural sort implementation to handle numerical prefixes correctly
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            
        return sorted([folder for folder in os.listdir(courses_dir)
                if os.path.isdir(os.path.join(courses_dir, folder))], 
                key=natural_sort_key)
    except Exception as e:
        st.error(f"Error accessing Courses directory: {e}")
        return []

# We no longer need a strict is_mcq_answer function for the main logic flow
# def is_mcq_answer(text):
#     """Check if the text is a simple MCQ answer (A, B, C, or D), case-insensitive."""
#     if not text:
#         return False
#     return re.fullmatch(r"[A-Da-d]\.?\s*", text.strip()) is not None


def generate_response(prompt, lecture_notes_text, exam_mode=False, chat_history=None):
    """
    Generates a response using the Gemini model with context and chat history,
    updated for google-genai SDK v1.0+, direct Part instantiation, and few-shot prompting
    for diverse question types.

    Args:
        prompt (str): The user's latest input (question or answer).
        lecture_notes_text (str): The loaded course material.
        exam_mode (bool): Flag indicating if exam mode is active.
        chat_history (list): List of previous messages [{'role': 'user'/'model', 'content': str}]

    Returns:
        str: The generated text response from the model, or an error message.
    """
    global client

    try:
        # --- System Prompt Setup ---
        if exam_mode:
            # Enhanced system instruction with few-shot examples
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

User: A, C

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

User: good, neutral/ambiguous

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

User: False

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

        # --- Construct `contents` list for the API ---
        api_contents = []
        context_string = f"COURSE CONTEXT:\n{lecture_notes_text}"
        api_contents.append(types.Content(role="user", parts=[types.Part(text=context_string)])) # WORKAROUND
        api_contents.append(types.Content(role="model", parts=[types.Part(text="Okay, I have loaded the course context and understand my role.")])) # WORKAROUND

        if chat_history:
            for message in chat_history:
                # Filter out internal/status messages before adding to API history
                if message["role"] == "model" and (
                    message["content"].startswith("Welcome!") or
                    "loaded. Ask me questions" in message["content"] or
                    "exiting exam mode" in message["content"] or
                    "already in exam mode" in message["content"] or
                    message["content"].startswith("Okay, I have loaded") or
                    "Please answer with a single letter" in message["content"] # Keep filtering old reminder
                    ):
                    continue
                if message["role"] == "user" and (
                     message["content"].startswith("COURSE CONTEXT:") or
                     message["content"].strip().lower() == "exit exam mode" or # Filter commands
                     message["content"].strip().lower() in ["start exam mode", "final exam mode", "exam", "start exam"]
                     ):
                     continue

                if isinstance(message["content"], str):
                    try:
                        # WORKAROUND: Direct instantiation
                        api_contents.append(types.Content(role=message["role"], parts=[types.Part(text=message["content"])]))
                    except ValueError as ve:
                         print(f"Skipping message due to invalid role '{message['role']}': {ve}")
                         st.warning(f"Skipping message with role {message['role']}")
                         continue
                else:
                     st.warning(f"Skipping message with unexpected content type: {type(message['content'])}")

        # Add the latest user prompt (which could be an answer)
        api_contents.append(types.Content(role="user", parts=[types.Part(text=prompt)])) # WORKAROUND

        # --- Prepare Generation Config Object ---
        generation_config_object = types.GenerateContentConfig(
            temperature=0.5 if exam_mode else 0.7, # Slightly higher temp for more varied questions?
            max_output_tokens=3072, # Increased slightly for potentially longer explanations/questions
            system_instruction=system_instruction_text
        )

        # --- Call the Gemini API ---
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=api_contents,
            config=generation_config_object
        )

        # --- Safely Extract Text from Response ---
        if hasattr(response, 'text') and response.text:
            return response.text
        elif response.parts:
             try:
                 return "".join(part.text for part in response.parts if hasattr(part, 'text'))
             except Exception as e:
                 print(f"DEBUG: Error joining response parts: {e}, Parts: {response.parts}")
        elif response.candidates:
             try:
                 first_candidate = response.candidates[0]
                 if first_candidate.content and first_candidate.content.parts:
                     return "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
             except Exception as e:
                  print(f"DEBUG: Error extracting text from candidates: {e}, Candidates: {response.candidates}")

        # Handle blocked or empty responses
        print("DEBUG: Full Gemini Response:", response)
        safety_feedback = getattr(response, 'prompt_feedback', None)
        finish_reason_prompt = "UNKNOWN"
        block_reason_prompt = None
        if safety_feedback:
            block_reason_prompt = getattr(safety_feedback, 'block_reason', None)
            if block_reason_prompt:
                 finish_reason_prompt = f"PROMPT_BLOCKED ({block_reason_prompt})"

        finish_reason_candidate = "UNKNOWN"
        block_reason_candidate = None
        if response.candidates:
            try:
                first_candidate = response.candidates[0]
                finish_reason_candidate = getattr(first_candidate, 'finish_reason', 'UNKNOWN')
                if hasattr(first_candidate, 'safety_ratings'):
                    for rating in first_candidate.safety_ratings:
                        if hasattr(rating, 'blocked') and rating.blocked:
                            block_reason_candidate = f"RESPONSE_BLOCKED ({getattr(rating, 'category', 'UNKNOWN')})"
                            break
            except Exception as e:
                print(f"DEBUG: Error accessing candidate details: {e}")

        error_message = f"Received response, but couldn't extract text. Prompt Feedback: {finish_reason_prompt}. Candidate Finish Reason: {finish_reason_candidate}."
        if block_reason_candidate:
             error_message += f" Candidate Block Reason: {block_reason_candidate}."
        st.warning(error_message)
        return "Sorry, I couldn't generate a valid response. The request might have been blocked due to safety settings or resulted in no usable output. Please check the console logs for details or try rephrasing."

    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        print(f"Gemini API Error during generate_response: {e}")
        import traceback
        traceback.print_exc()
        return "An critical error occurred while trying to get a response. Please check the server logs or contact support."

# --- Streamlit App ---

st.set_page_config(page_title="Study Bot", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "content": "Welcome! Please select a course to load materials."}]
if "exam_mode" not in st.session_state:
    st.session_state.exam_mode = False
if "current_lecture_text" not in st.session_state:
    st.session_state.current_lecture_text = ""
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Study Bot")
    st.markdown(f"model used: {str(MODEL_NAME)}")
    st.subheader("Course Selection")
    available_courses = get_available_courses()

    if not available_courses:
        st.error("No courses found in the 'Courses' directory!")
        st.stop()

    selected_course_index = available_courses.index(st.session_state.selected_course) if st.session_state.selected_course in available_courses else 0
    st.session_state.selected_course = st.selectbox(
        "Select a course", available_courses, index=selected_course_index, key="course_selector"
    )

    if st.button("Load Course Materials", key="load_course_button"):
        if st.session_state.selected_course:
            course_path = os.path.join("Courses", st.session_state.selected_course)
            with st.spinner(f"Loading materials for {st.session_state.selected_course}..."):
                lecture_text = extract_text_from_folder(course_path)
                if lecture_text and not lecture_text.startswith("Error:"):
                    st.session_state.current_lecture_text = lecture_text
                    st.session_state.messages = [
                        {"role": "model", "content": f"Course '{st.session_state.selected_course}' loaded. Ask me questions or start exam mode."}
                    ]
                    st.session_state.exam_mode = False
                    st.success(f"Loaded '{st.session_state.selected_course}' successfully!")
                    st.rerun()
                else:
                    error_msg = lecture_text if lecture_text and lecture_text.startswith("Error:") else f"Failed to load lecture text for {st.session_state.selected_course}."
                    st.error(error_msg)
                    st.session_state.current_lecture_text = ""
                    st.session_state.messages = [{"role": "model", "content": "Failed to load course. Please select another."}]
        else:
             st.warning("Please select a course first.")

    if st.session_state.current_lecture_text:
        st.divider()
        st.subheader("Study Mode")
        if st.session_state.exam_mode:
            if st.button("❌ Exit Exam Mode", type="secondary"):
                st.session_state.exam_mode = False
                # Add a message indicating exit, don't treat "exit" as a prompt to the model
                st.session_state.messages.append({"role": "model", "content": "Okay, exiting exam mode. How can I help you with the course material?"})
                st.rerun()
        else:
            if st.button("📝 Start Exam Mode", type="primary"):
                st.session_state.exam_mode = True
                # Prompt to start the exam process
                initial_exam_prompt = "Please start the final exam by providing the first question based on the course material. Use varied question types."
                history_for_first_question = [msg for msg in st.session_state.messages if not (msg["role"] == "model" and msg["content"].startswith("Welcome!"))]
                with st.spinner("Starting exam and preparing the first question..."):
                    response = generate_response(
                        initial_exam_prompt,
                        st.session_state.current_lecture_text,
                        exam_mode=True,
                        chat_history=history_for_first_question
                    )
                st.session_state.messages.append({"role": "model", "content": response})
                st.rerun()

        st.divider()
        st.subheader("Course Info")
        word_count = len(st.session_state.current_lecture_text.split())
        st.write(f"**Course:** {st.session_state.selected_course}")
        st.write(f"**Word Count:** {word_count:,}")
        st.markdown(f"**Status:** {'📝 Exam Mode Active' if st.session_state.exam_mode else '💬 Conversation Mode'}")

    st.divider()
    st.subheader("Tips for Exam Mode")
    st.markdown("- Answer MCQs with letter(s) (e.g., `A` or `A, C`).")
    st.markdown("- Answer Fill-in-blanks with word(s) (e.g., `good` or `good, neutral`).")
    st.markdown("- Answer True/False with `True`, `False`, or the letter.")
    st.markdown("- Type `exit exam mode` to leave.")

# --- Main Chat Interface ---
st.title(f"💬 {st.session_state.selected_course or 'Study Bot'}")

if st.session_state.exam_mode:
    st.info("📝 **EXAM MODE**: Answer the question or type 'exit exam mode'. Check tips in sidebar for answer formats.")

# Display chat messages from history
for message in st.session_state.messages:
    # Filter internal context messages before display
    if message["role"] == "user" and message["content"].startswith("COURSE CONTEXT:"):
        continue
    if message["role"] == "model" and message["content"].startswith("Okay, I have loaded"):
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle chat input
if prompt := st.chat_input("Ask a question or enter your answer..."):
    if not st.session_state.current_lecture_text:
        st.warning("Please load course materials using the sidebar first.")
        st.stop()

    # Append user message immediately to state for display
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare history for API call (all messages *before* the current user prompt)
    history_for_api = st.session_state.messages[:-1]

    # --- Process User Input ---
    is_exit_command = prompt.strip().lower() == "exit exam mode"
    is_start_command = prompt.strip().lower() in ["start exam mode", "final exam mode", "exam", "start exam"]

    if is_exit_command:
        if st.session_state.exam_mode:
            st.session_state.exam_mode = False
            response = "Okay, exiting exam mode. How can I help you with the course material?"
            st.session_state.messages.append({"role": "model", "content": response})
        else:
             response = "You are not currently in exam mode. How can I help?"
             st.session_state.messages.append({"role": "model", "content": response})
        st.rerun()

    elif is_start_command:
        if not st.session_state.exam_mode:
            st.session_state.exam_mode = True
            initial_exam_prompt = "Please start the final exam by providing the first question based on the course material. Use varied question types."
            with st.spinner("Starting exam and preparing the first question..."):
                response = generate_response(
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

    # If in exam mode and input is not a command, assume it's an answer
    elif st.session_state.exam_mode:
         with st.spinner("Evaluating your answer and preparing the next question..."):
            # Pass the user's answer ('prompt') as the latest content
            # The updated system instruction guides the model on evaluating various answer formats
            response = generate_response(
                prompt, # The user's answer IS the prompt for this turn
                st.session_state.current_lecture_text,
                exam_mode=True,
                chat_history=history_for_api # History *before* this answer
            )
         st.session_state.messages.append({"role": "model", "content": response})
         st.rerun()

    # Handle General Conversation (Not in Exam Mode)
    else:
        with st.spinner("Thinking..."):
            response = generate_response(
                prompt,
                st.session_state.current_lecture_text,
                exam_mode=False,
                chat_history=history_for_api
            )
        st.session_state.messages.append({"role": "model", "content": response})
        st.rerun()