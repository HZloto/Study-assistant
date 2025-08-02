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

# Recommended model (Using 2.5 Pro for best results, Flash as fallback)
MODEL_NAME = "gemini-2.5-pro"  # Changed to more stable Flash model
# Alternative: "gemini-1.5-pro" or "gemini-1.5-flash"

# --- Helper Functions ---

def extract_text_from_folder(folder_path):
    """Extract text from all .txt files in a folder."""
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
    """Return a list of available courses in the Courses directory."""
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

def get_available_lessons(course):
    """Return a list of available lessons for the selected course."""
    course_dir = os.path.join("Courses", course)
    if not os.path.exists(course_dir):
        return []
    try:
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        return sorted([folder for folder in os.listdir(course_dir)
                if os.path.isdir(os.path.join(course_dir, folder))],
                key=natural_sort_key)
    except Exception as e:
        st.error(f"Error accessing lessons directory for course {course}: {e}")
        return []

def get_course_info(course_name):
    """Return a formatted description of the course based on course_name."""
    course_info = {
        "14.73": "MIT 14.73: The Challenges of Global Poverty",
        "JPAL102": "JPAL 102x: Designing and Running Randomized Evaluations",
        # Add more courses as needed
    }
    return course_info.get(course_name, course_name)

def safe_extract_response_text(response):
    """Safely extracts text from Gemini response object, handling potential errors."""
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

def generate_response(prompt, lecture_notes_text, course_name, lesson_name, exam_mode=False, chat_history=None):
    """
    Generates a response using the Gemini model with context and chat history,
    aligned with the latest google-genai SDK practices using client.models.

    Args:
        prompt (str): The user's latest input (question or answer).
        lecture_notes_text (str): The loaded course material.
        course_name (str): The name of the selected course.
        lesson_name (str): The name of the selected lesson.
        exam_mode (bool): Flag indicating if exam mode is active.
        chat_history (list): List of previous messages [{'role': 'user'/'model', 'content': str}]

    Returns:
        str: The generated text response from the model, or an error message.
    """
    global client # Use the global client initialized correctly

    try:
        # Format course name for prompts
        formatted_course_name = get_course_info(course_name)
        
        # --- System Prompt Setup ---
        if exam_mode:
            system_instruction_text = f"""You are an expert tutor for {formatted_course_name}.
You are in **EXAM MODE**. Your goal is to test the user's understanding with various question types based *strictly* on the provided lecture notes context (which starts with "COURSE CONTEXT:").
The current lesson is: "{lesson_name}".

**HERE ARE EXAMPLES OF HOW THE EXAM SHOULD WORK:**

--- EXAMPLE 1 (Select All) ---
Tutor:
**Question 3:** For which types of borrower might a "flexible" repayment plan (e.g., a grace period or a repayment holiday) be most beneficial? Select all that apply.
A. One whose income is highly variable
B. One who needs to stick to a routine
C. One who plans to make a large investment that won't pay off right away
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
**Question 10:** True or false? Considering the evidence presented here and in lecture, increasing access to credit is not a path to increasing the wellbeing of a meaningful number of the world's poor.
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

1.  Ask challenging questions aimed at covering the whole topic based *only* on the provided COURSE CONTEXT for lesson "{lesson_name}". Use a variety of formats:
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
12. Be challenging: make the questions wrong answers linked to other parts of the class or slight misinterpretations of the material.
13. Always specify if there are multiple correct answers, or just one.
14. Don't be shy to add calculatory questions requesting a result or at least finding the correct formula if it is relevant to the class.
15. When the exam is over, provide a summary of the user's performance, including: Score, description of questions missed and reminder to ensure the missed concepts are reviewed.
"""
        else: # Standard Tutor Mode
            system_instruction_text = f"""You are a helpful and concise tutor for {formatted_course_name}.
You are currently helping with the lesson: "{lesson_name}".
Use the provided lecture notes context (which starts with "COURSE CONTEXT:") as your only knowledge source to answer student questions clearly.
If a question cannot be answered from the notes, say so.
Be conversational but stick to the course material."""

        # --- Construct `contents` list for the API (Using types.Content) ---
        api_contents = []
        context_string = f"COURSE CONTEXT:\n{lecture_notes_text}"
        # Prepend context and role confirmation as the first turns
        api_contents.append(types.Content(role="user", parts=[types.Part(text=context_string)]))
        api_contents.append(types.Content(role="model", parts=[types.Part(text=f"Okay, I have loaded the course context for {formatted_course_name}, lesson: {lesson_name}, and understand my role.")]))

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
        return "A critical error occurred while trying to get a response. Please check the server logs or contact support."

def estimate_tokens(text):
    """
    Rough token estimation - Gemini models typically use ~4 characters per token
    but this can vary significantly based on content complexity.
    """
    # Conservative estimate: 3.5 characters per token on average
    return len(text) // 3.5

def try_generate_with_fallback_models(client, prompt, config):
    """
    Try generating content with multiple models as fallback.
    Returns (response, model_used) or (None, None) if all fail.
    """
    # Models to try in order (most capable to most reliable)
    models_to_try = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash", 
        "gemini-1.5-pro"
    ]
    
    for model in models_to_try:
        try:
            print(f"DEBUG: Trying model: {model}")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            print(f"DEBUG: Successfully used model: {model}")
            return response, model
        except Exception as e:
            print(f"DEBUG: Model {model} failed with error: {e}")
            if "500 INTERNAL" in str(e) and model != models_to_try[-1]:
                print(f"DEBUG: Trying next model due to 500 error...")
                continue
            else:
                # Re-raise the last error if it's not a 500 or it's the last model
                if model == models_to_try[-1]:
                    raise e
                continue
    
    return None, None

def intelligent_content_truncation(text, max_chars=45000):
    """
    Intelligently truncate content by preserving complete sections/paragraphs.
    Tries to cut at natural breakpoints like headers, sections, or paragraphs.
    """
    if len(text) <= max_chars:
        return text
    
    # Try to find good breakpoints in order of preference
    breakpoints = [
        (r'\n\n[A-Z][^\.]*\n\n', 'section'),  # Section headers
        (r'\n\n.*?[\.!?]\n\n', 'paragraph'),   # Complete paragraphs
        (r'\n\n', 'double_newline'),           # Paragraph breaks
        (r'\. ', 'sentence'),                  # Sentence endings
    ]
    
    # Start with a conservative limit to ensure we have room
    target_length = min(max_chars, int(max_chars * 0.9))
    
    for pattern, breakpoint_type in breakpoints:
        matches = list(re.finditer(pattern, text[:target_length + 2000]))  # Look a bit beyond target
        if matches:
            # Find the last good match within our target length
            best_match = None
            for match in matches:
                if match.end() <= target_length:
                    best_match = match
                else:
                    break
            
            if best_match:
                truncated = text[:best_match.end()].strip()
                if len(truncated) > max_chars * 0.6:  # Ensure we keep a reasonable amount
                    return truncated + f"\n\n[Content truncated at {breakpoint_type} boundary]"
    
    # Fallback to simple character truncation
    return text[:target_length].strip() + "\n\n[Content truncated due to length limits]"

def get_flashcard_cache_path(course_name, lesson_name):
    """Generate the cache file path for flashcards."""
    cache_dir = os.path.join("data", "flashcard_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # Clean names for safe file paths
    safe_course = re.sub(r'[^\w\-_.]', '_', course_name)
    safe_lesson = re.sub(r'[^\w\-_.]', '_', lesson_name)
    filename = f"{safe_course}_{safe_lesson}_flashcards.json"
    return os.path.join(cache_dir, filename)

def load_cached_flashcards(course_name, lesson_name):
    """Load flashcards from cache if they exist."""
    cache_path = get_flashcard_cache_path(course_name, lesson_name)
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                # Validate cache structure
                if isinstance(cached_data, dict) and len(cached_data) > 0:
                    return cached_data
                else:
                    st.warning("Invalid cached flashcard format found. Will regenerate.")
                    return None
        return None
    except Exception as e:
        st.warning(f"Error loading cached flashcards: {e}. Will regenerate.")
        print(f"Cache loading error: {e}")
        return None

def save_flashcards_to_cache(flashcards_dict, course_name, lesson_name):
    """Save flashcards to cache file."""
    cache_path = get_flashcard_cache_path(course_name, lesson_name)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(flashcards_dict, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.warning(f"Failed to save flashcards to cache: {e}")
        print(f"Cache saving error: {e}")
        return False

def clear_flashcard_cache(course_name=None, lesson_name=None):
    """Clear flashcard cache. If course/lesson specified, clear only that cache."""
    cache_dir = os.path.join("data", "flashcard_cache")
    try:
        if course_name and lesson_name:
            # Clear specific cache file
            cache_path = get_flashcard_cache_path(course_name, lesson_name)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
        else:
            # Clear entire cache directory
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                return True
        return False
    except Exception as e:
        st.error(f"Error clearing cache: {e}")
        return False

def generate_flashcards(lecture_notes_text, course_name, lesson_name, force_regenerate=False):
    """
    Generates flashcards using the Gemini model based on lecture notes.
    Uses caching to avoid regenerating the same flashcards.
    
    Args:
        lecture_notes_text (str): The course material text
        course_name (str): Name of the course
        lesson_name (str): Name of the lesson
        force_regenerate (bool): If True, skip cache and regenerate flashcards
    """
    global client
    
    # Check cache first unless force to regenerate
    if not force_regenerate:
        cached_flashcards = load_cached_flashcards(course_name, lesson_name)
        if cached_flashcards:
            st.success(f"✅ Loaded {len(cached_flashcards)} flashcards from cache!")
            st.info("💡 These flashcards were previously generated. Use 'Regenerate' to create new ones.")
            return cached_flashcards
    
    # Format course name for prompts
    formatted_course_name = get_course_info(course_name)

    # Check content length and adjust strategy - Much more aggressive limits
    word_count = len(lecture_notes_text.split())
    char_count = len(lecture_notes_text)
    estimated_tokens = estimate_tokens(lecture_notes_text)
    
    # Log content statistics for debugging
    print(f"DEBUG: Content stats - Characters: {char_count:,}, Words: {word_count:,}, Estimated tokens: {estimated_tokens:.0f}")
    
    # Very conservative content size management - aim for max 25,000 characters to be safe
    if char_count > 30000:  # Much more conservative limit
        st.warning(f"Course material is very long ({char_count:,} characters, ~{estimated_tokens:.0f} tokens). Using aggressive truncation.")
        lecture_notes_text = intelligent_content_truncation(lecture_notes_text, 25000)
        target_cards = 20
        st.info(f"Content aggressively reduced to {len(lecture_notes_text):,} characters (~{estimate_tokens(lecture_notes_text):.0f} tokens).")
    elif char_count > 20000:
        st.warning(f"Course material is quite long ({char_count:,} characters, ~{estimated_tokens:.0f} tokens). Using intelligent truncation.")
        lecture_notes_text = intelligent_content_truncation(lecture_notes_text, 18000)
        target_cards = 20
        st.info(f"Content reduced to {len(lecture_notes_text):,} characters (~{estimate_tokens(lecture_notes_text):.0f} tokens).")
    elif char_count > 15000:
        st.info(f"Course material is moderately long ({char_count:,} characters, ~{estimated_tokens:.0f} tokens). Using slight truncation.")
        lecture_notes_text = intelligent_content_truncation(lecture_notes_text, 14000)
        target_cards = 20
    elif word_count > 2000:
        target_cards = 20
    else:
        target_cards = 20

    # Ultra-concise prompt to minimize token usage - accept both formats
    flashcard_prompt = f"""Create {target_cards} Q&A flashcards from: {formatted_course_name} - {lesson_name}

Return valid JSON in one of these formats:
1. {{"Question 1": "Answer 1", "Question 2": "Answer 2", ...}}
2. [{{"Q1": "Question 1", "A1": "Answer 1"}}, {{"Q2": "Question 2", "A2": "Answer 2"}}, ...]

Keep answers brief (<50 words).

CONTENT:
{lecture_notes_text}
"""

    # Calculate final prompt token estimate
    prompt_tokens = estimate_tokens(flashcard_prompt)
    print(f"DEBUG: Final prompt stats - Characters: {len(flashcard_prompt):,}, Estimated tokens: {prompt_tokens:.0f}")
    
    if prompt_tokens > 30000:  # Very conservative limit
        st.error(f"⚠️ Even after truncation, content is too large (~{prompt_tokens:.0f} tokens). Maximum recommended: ~30,000 tokens.")
        st.info("Try selecting a smaller lesson or splitting the content into multiple files.")
        return None

    try:
        # Use more conservative parameters and fallback models
        config = types.GenerateContentConfig(
            temperature=0.2,  # Lower temperature for more focused output
            max_output_tokens=3072,  # Further reduced to prevent any overflow
            response_mime_type="application/json"
        )
        
        response, model_used = try_generate_with_fallback_models(client, flashcard_prompt, config)
        if response is None:
            st.error("All models failed to generate flashcards. Please try with smaller content.")
            return None
            
        if model_used != MODEL_NAME:
            st.info(f"🔄 Used fallback model: {model_used}")

        response_text = safe_extract_response_text(response)
        if not response_text:
            st.error("Flashcard generation failed: No response text received.")
            return None

        response_text = response_text.strip()
        
        # Enhanced debugging for malformed responses
        print(f"DEBUG: Response length: {len(response_text)} characters")
        print(f"DEBUG: Response starts with: {response_text[:100]}")
        print(f"DEBUG: Response ends with: {response_text[-100:]}")

        # Enhanced response validation - handle both object and array formats
        is_valid_json = False
        if response_text.startswith('{') and response_text.endswith('}'):
            is_valid_json = True
        elif response_text.startswith('[') and response_text.endswith(']'):
            is_valid_json = True
            
        if not is_valid_json:
            st.warning("Response appears malformed or truncated. Attempting recovery...")
            print("DEBUG: Malformed response:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Try to fix common issues for object format
            if response_text.startswith('{') and not response_text.endswith('}'):
                # Find last complete entry more carefully
                lines = response_text.split('\n')
                fixed_lines = []
                for line in lines:
                    if '": "' in line and (line.strip().endswith('",') or line.strip().endswith('"')):
                        fixed_lines.append(line)
                    elif line.strip() in ['{', '}']:
                        fixed_lines.append(line)
                
                if fixed_lines and fixed_lines[0].strip() == '{':
                    # Remove trailing comma from last entry if present
                    if len(fixed_lines) > 1 and fixed_lines[-1].strip().endswith('",'):
                        fixed_lines[-1] = fixed_lines[-1].rstrip().rstrip(',') + '"'
                    fixed_lines.append('}')
                    response_text = '\n'.join(fixed_lines)
                    st.info("Successfully recovered flashcards from malformed response.")
            
            # Try to fix common issues for array format
            elif response_text.startswith('[') and not response_text.endswith(']'):
                # Try to close the array properly
                if response_text.rstrip().endswith('}'):
                    response_text = response_text.rstrip() + '\n]'
                    st.info("Successfully recovered flashcards from incomplete array.")
                elif response_text.rstrip().endswith(','):
                    # Remove trailing comma and close array
                    response_text = response_text.rstrip().rstrip(',') + '\n]'
                    st.info("Successfully recovered flashcards from incomplete array.")

        # Handle both array and object JSON formats
        try:
            parsed_data = json.loads(response_text)
            
            # Convert array format to dictionary format
            if isinstance(parsed_data, list):
                print("DEBUG: Converting array format to dictionary format")
                flashcards_dict = {}
                for i, item in enumerate(parsed_data, 1):
                    if isinstance(item, dict):
                        # Handle objects with Q1/A1, Q2/A2 pattern
                        for key, value in item.items():
                            if key.startswith('Q') and key[1:].isdigit():
                                q_num = key[1:]
                                a_key = f"A{q_num}"
                                if a_key in item:
                                    flashcards_dict[value] = item[a_key]
                                    break
                        # Handle simple question/answer pairs
                        if len(item) == 2:
                            keys = list(item.keys())
                            values = list(item.values())
                            if any(word in keys[0].lower() for word in ['question', 'q']):
                                flashcards_dict[values[0]] = values[1]
                            else:
                                flashcards_dict[values[0]] = values[1]
            elif isinstance(parsed_data, dict):
                flashcards_dict = parsed_data
            else:
                raise ValueError(f"Expected dictionary or list, got {type(parsed_data)}")
            
            if len(flashcards_dict) == 0:
                st.error("Generated empty flashcard set.")
                return None
            
            # Clean up the flashcards
            cleaned_dict = {}
            for q, a in flashcards_dict.items():
                clean_q = str(q).strip()
                clean_a = str(a).strip()
                if clean_q and clean_a and clean_q != clean_a:
                    cleaned_dict[clean_q] = clean_a
            
            if len(cleaned_dict) < 5:
                st.error(f"Too few valid flashcards generated ({len(cleaned_dict)}). Please try again.")
                return None
            
            # Save to cache before returning
            if save_flashcards_to_cache(cleaned_dict, course_name, lesson_name):
                cache_icon = "💾"
            else:
                cache_icon = "⚠️"
            
            if len(cleaned_dict) < target_cards * 0.7:  # More lenient threshold
                st.warning(f"Generated {len(cleaned_dict)} flashcards (less than target of {target_cards}). {cache_icon}")
            else:
                st.success(f"Successfully generated {len(cleaned_dict)} flashcards! {cache_icon}")
            
            return cleaned_dict

        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"JSON parsing failed: {e}")
            print("DEBUG: Failed JSON text:", response_text)
            
            # Enhanced regex fallback
            st.info("Attempting advanced text parsing recovery...")
            try:
                # More flexible regex patterns
                patterns = [
                    r'"([^"]+)":\s*"([^"]*(?:\\.[^"]*)*)"',  # Original pattern
                    r'"([^"]+)":\s*"([^"]+)"',  # Simpler pattern
                    r'(["\'])([^"\']+)\1:\s*(["\'])([^"\']+)\3',  # Mixed quotes
                ]
                
                best_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    if pattern == patterns[2]:  # Handle mixed quotes result
                        matches = [(m[1], m[3]) for m in matches]
                    if len(matches) > len(best_matches):
                        best_matches = matches
                
                if best_matches and len(best_matches) >= 5:
                    fallback_dict = {}
                    for q, a in best_matches:
                        clean_q = q.strip()
                        clean_a = a.strip()
                        if clean_q and clean_a and len(clean_q) > 10 and len(clean_a) > 10:
                            fallback_dict[clean_q] = clean_a
                    
                    if len(fallback_dict) >= 5:
                        # Save recovered flashcards to cache
                        save_flashcards_to_cache(fallback_dict, course_name, lesson_name)
                        st.success(f"Recovered {len(fallback_dict)} flashcards using text parsing! 💾")
                        return fallback_dict
                
                st.error("All recovery attempts failed. Please try generating flashcards again.")
                return None
                
            except Exception as fallback_error:
                st.error(f"Recovery parsing failed: {fallback_error}")
                return None

    except Exception as e:
        error_msg = str(e)
        
        # Enhanced error handling with specific debugging
        print(f"DEBUG: Full error details: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        
        # Handle specific token/context length errors with better messaging
        if "500 INTERNAL" in error_msg or "INTERNAL" in error_msg:
            # Calculate and display token information
            final_char_count = len(lecture_notes_text)
            final_token_estimate = estimate_tokens(lecture_notes_text)
            
            st.error("⚠️ API Error: Content processing failed (Error 500)")
            st.info("**This could be due to:**")
            st.info("• Content complexity exceeding processing limits")
            st.info("• Temporary API service issues")
            st.info("• Token limits despite character count being reasonable")
            
            with st.expander("🔍 Debug Information"):
                st.write(f"**Content length:** {final_char_count:,} characters")
                st.write(f"**Estimated tokens:** {final_token_estimate:.0f}")
                st.write(f"**Course:** {course_name}")
                st.write(f"**Lesson:** {lesson_name}")
                st.write(f"**Model:** {MODEL_NAME}")
                
            st.warning("**Try these solutions:**")
            st.info("1. Wait a few minutes and try again (temporary API issue)")
            st.info("2. Split this lesson into smaller text files")
            st.info("3. Try switching to Gemini 1.5 Flash model")
            st.info("4. Remove some content from the lesson files")
            
        elif "input context" in error_msg.lower() and "too long" in error_msg.lower():
            st.error("⚠️ Input context is too long for the model.")
            st.info("**Suggestions:**")
            st.info("• Split content into smaller lessons")
            st.info("• Use shorter text files")
            st.info("• Remove unnecessary content from files")
            st.warning(f"Current content length: {len(lecture_notes_text):,} characters.")
        elif "DEADLINE_EXCEEDED" in error_msg:
            st.error("⚠️ Request timeout: Content took too long to process.")
            st.info("Try with smaller content or try again later.")
        elif "RESOURCE_EXHAUSTED" in error_msg:
            st.error("⚠️ Rate limit exceeded. Please wait a moment before trying again.")
        else:
            st.error(f"Flashcard generation error: {e}")
            st.info("This might be a temporary API issue. Please try again in a moment.")
            st.info("If the problem persists, try with smaller content or a different lesson.")
        
        print(f"Flashcard generation error: {e}")
        traceback.print_exc()
        return None
        return None

# --- Streamlit App ---

st.set_page_config(page_title="Study Bot", page_icon="📚", layout="wide")

# --- Initialize Session State ---
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
if "selected_lesson" not in st.session_state:
    st.session_state.selected_lesson = None
if "flashcards" not in st.session_state:
    st.session_state.flashcards = None
if "flashcard_index" not in st.session_state:
    st.session_state.flashcard_index = 0
if "show_flashcard_answer" not in st.session_state:
    st.session_state.show_flashcard_answer = False

# --- Sidebar ---
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
        st.session_state.messages = [{"role": "model", "content": "Please select a lesson and click 'Load Materials'."}]
        st.session_state.current_lecture_text = ""
        st.session_state.selected_lesson = None

    new_selection = st.selectbox(
        "Select a course", available_courses, index=selected_course_index, key="course_selector", on_change=on_course_change
    )
    if new_selection != st.session_state.selected_course:
       st.session_state.selected_course = new_selection
    
    # Add lesson selection
    if st.session_state.selected_course:
        available_lessons = get_available_lessons(st.session_state.selected_course)
        lesson_index = 0
        if st.session_state.selected_lesson in available_lessons:
            lesson_index = available_lessons.index(st.session_state.selected_lesson)
        
        def on_lesson_change():
            st.session_state.exam_mode = False
            st.session_state.flashcard_mode = False
            st.session_state.flashcards = None
            st.session_state.flashcard_index = 0
            st.session_state.show_flashcard_answer = False
            st.session_state.messages = [{"role": "model", "content": "Click 'Load Materials' to continue."}]
            st.session_state.current_lecture_text = ""
        
        selected_lesson = st.selectbox(
            "Select a lesson", 
            available_lessons, 
            index=lesson_index,
            key="lesson_selector",
            on_change=on_lesson_change
        )
        
        if selected_lesson != st.session_state.selected_lesson:
            st.session_state.selected_lesson = selected_lesson

    loading_disabled = st.session_state.selected_course is None or st.session_state.selected_lesson is None
    if st.button("Load Materials", key="load_course_button", disabled=loading_disabled):
        if st.session_state.selected_course and st.session_state.selected_lesson:
            lesson_path = os.path.join("Courses", st.session_state.selected_course, st.session_state.selected_lesson)
            with st.spinner(f"Loading materials for {st.session_state.selected_course}, {st.session_state.selected_lesson}..."):
                lecture_text = extract_text_from_folder(lesson_path)
                if lecture_text and not lecture_text.startswith("Error:"):
                    st.session_state.current_lecture_text = lecture_text
                    course_info = get_course_info(st.session_state.selected_course)
                    st.session_state.messages = [
                        {"role": "model", "content": f"{course_info}, lesson '{st.session_state.selected_lesson}' loaded. Ask questions, start an exam, or generate flashcards."}
                    ]
                    st.session_state.exam_mode = False
                    st.session_state.flashcard_mode = False
                    st.session_state.flashcards = None
                    st.session_state.flashcard_index = 0
                    st.session_state.show_flashcard_answer = False
                    st.success(f"Loaded '{st.session_state.selected_lesson}' successfully!")
                    st.rerun()
                else:
                    error_msg = lecture_text if lecture_text and lecture_text.startswith("Error:") else f"Failed to load lecture text for {st.session_state.selected_lesson}."
                    st.error(error_msg)
                    st.session_state.current_lecture_text = ""
                    st.session_state.messages = [{"role": "model", "content": "Failed to load materials. Please select another lesson."}]
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
                        response = generate_response(
                            initial_exam_prompt,
                            st.session_state.current_lecture_text,
                            st.session_state.selected_course,
                            st.session_state.selected_lesson,
                            exam_mode=True,
                            chat_history=history_for_first_question
                        )
                    st.session_state.messages.append({"role": "model", "content": response})
                    st.rerun()
            with col2:
                # Check if cached flashcards exist
                cached_exists = load_cached_flashcards(st.session_state.selected_course, st.session_state.selected_lesson) is not None
                
                if cached_exists:
                    # Show two buttons vertically stacked instead of side by side to avoid nested columns
                    if st.button("⚡ Load Flashcards", type="primary", key="load_flashcards", use_container_width=True):
                        st.session_state.flashcard_mode = True
                        st.session_state.exam_mode = False
                        st.session_state.flashcards = None
                        st.session_state.flashcard_index = 0
                        st.session_state.show_flashcard_answer = False
                        
                        flashcards_dict = load_cached_flashcards(st.session_state.selected_course, st.session_state.selected_lesson)
                        if flashcards_dict:
                            st.session_state.flashcards = list(flashcards_dict.items())
                            st.success(f"Loaded {len(st.session_state.flashcards)} cached flashcards!")
                        else:
                            st.error("Failed to load cached flashcards.")
                            st.session_state.flashcard_mode = False
                        st.rerun()
                    
                    if st.button("🔄 Regenerate", type="secondary", key="regen_flashcards", use_container_width=True):
                        st.session_state.flashcard_mode = True
                        st.session_state.exam_mode = False
                        st.session_state.flashcards = None
                        st.session_state.flashcard_index = 0
                        st.session_state.show_flashcard_answer = False
                        with st.spinner(f"Regenerating flashcards for {st.session_state.selected_lesson}..."):
                            flashcards_dict = generate_flashcards(
                                st.session_state.current_lecture_text, 
                                st.session_state.selected_course,
                                st.session_state.selected_lesson,
                                force_regenerate=True
                            )
                        if flashcards_dict:
                            st.session_state.flashcards = list(flashcards_dict.items())
                            if not st.session_state.flashcards:
                                 st.error("Failed to generate flashcards (empty result). Please try again.")
                                 st.session_state.flashcard_mode = False
                        else:
                            st.error("Failed to generate flashcards. Please try again or check the logs.")
                            st.session_state.flashcard_mode = False
                        st.rerun()
                else:
                    # No cache exists, show single generate button
                    if st.button("⚡ Generate Flashcards", type="primary", key="start_flashcards", use_container_width=True):
                        st.session_state.flashcard_mode = True
                        st.session_state.exam_mode = False
                        st.session_state.flashcards = None
                        st.session_state.flashcard_index = 0
                        st.session_state.show_flashcard_answer = False
                        with st.spinner(f"Generating flashcards for {st.session_state.selected_lesson}... (This may take a moment)"):
                            flashcards_dict = generate_flashcards(
                                st.session_state.current_lecture_text, 
                                st.session_state.selected_course,
                                st.session_state.selected_lesson
                            )
                        if flashcards_dict:
                            st.session_state.flashcards = list(flashcards_dict.items())
                            if not st.session_state.flashcards:
                                 st.error("Failed to generate flashcards (empty result). Please try again.")
                                 st.session_state.flashcard_mode = False
                        else:
                            st.error("Failed to generate flashcards. Please try again or check the logs.")
                            st.session_state.flashcard_mode = False
                        st.rerun()

        st.divider()
        st.subheader("Current Selection")
        formatted_course = get_course_info(st.session_state.selected_course)
        word_count = len(st.session_state.current_lecture_text.split()) if st.session_state.current_lecture_text else 0
        char_count = len(st.session_state.current_lecture_text) if st.session_state.current_lecture_text else 0
        st.write(f"**Course:** {formatted_course}")
        st.write(f"**Lesson:** {st.session_state.selected_lesson}")
        st.write(f"**Word Count:** {word_count:,}")
        st.write(f"**Character Count:** {char_count:,}")
        
        # Add content size warning for flashcard generation
        if char_count > 60000:
            st.warning("⚠️ Large content - flashcards will be truncated")
        elif char_count > 40000:
            st.info("📊 Moderate content - may generate fewer flashcards")

        status_text = "💬 QA Mode"
        if st.session_state.exam_mode:
            status_text = "📝 Exam Mode Active"
        elif st.session_state.flashcard_mode:
             card_count = len(st.session_state.flashcards) if st.session_state.flashcards else 0
             status_text = f"⚡ Flashcard Mode Active ({card_count} cards)"
        st.markdown(f"**Status:** {status_text}")

        # Add cache status info
        cached_exists = load_cached_flashcards(st.session_state.selected_course, st.session_state.selected_lesson) is not None
        cache_status = "💾 Cached" if cached_exists else "🔄 Not Cached"
        st.markdown(f"**Flashcards:** {cache_status}")

    st.divider()
    st.subheader("Tips")
    st.markdown("- **QA Mode:** Ask questions about the loaded text.")
    st.markdown("- **Exam Mode:** Answer varied questions. Check format hints if needed.")
    st.markdown("- **Flashcards:** Generate cards on key concepts. Use buttons to navigate.")

# --- Main Interface ---
title_text = f"📚 {get_course_info(st.session_state.selected_course)}" if st.session_state.selected_course else "📚 Study Bot"
subtitle_text = f"{st.session_state.selected_lesson}" if st.session_state.selected_lesson else ""

st.title(title_text)
if subtitle_text:
    st.subheader(subtitle_text)

# --- Flashcard Mode Display ---
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
              st.info("Please select a course and lesson from the sidebar to begin.")
         elif not st.session_state.selected_lesson:
              st.info("Please select a lesson from the sidebar and click 'Load Materials'.")
         elif not st.session_state.flashcards:
              st.info("Use the 'Generate Flashcards' button in the sidebar to create flashcards for the loaded materials.")

# --- QA and Exam Mode Display ---
else:
    # Message display loop
    for message in st.session_state.messages:
        if message["role"] == "user" and message["content"].startswith("COURSE CONTEXT:"):
            continue
        if message["role"] == "model" and (
            message["content"].startswith("Okay, I have loaded") or
            message["content"].startswith("Exited flashcard mode.") or
            message["content"].startswith("Welcome!") or
            message["content"].startswith("Select 'Load") or
            message["content"].startswith("Please select")
            ):
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input logic
    prompt_disabled = not st.session_state.current_lecture_text
    prompt_placeholder = "Ask a question or enter your answer..."
    if not st.session_state.current_lecture_text:
        prompt_placeholder = "Please load course materials first"

    if prompt := st.chat_input(prompt_placeholder, disabled=prompt_disabled):
        st.session_state.messages.append({"role": "user", "content": prompt})
        history_for_api = st.session_state.messages[:-1]

        is_exit_command = prompt.strip().lower() == "exit exam mode"
        is_start_exam_command = prompt.strip().lower() in ["start exam mode", "final exam mode", "exam", "start exam"]

        # Command processing
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
                    response = generate_response(
                        initial_exam_prompt,
                        st.session_state.current_lecture_text,
                        st.session_state.selected_course,
                        st.session_state.selected_lesson,
                        exam_mode=True,
                        chat_history=history_for_api
                    )
                st.session_state.messages.append({"role": "model", "content": response})
            else:
                response = "You are already in exam mode. Please answer the question or type 'exit exam mode'."
                st.session_state.messages.append({"role": "model", "content": response})
            st.rerun()
        elif st.session_state.exam_mode:
             with st.spinner("Thinking..."):
                response = generate_response(
                    prompt,
                    st.session_state.current_lecture_text,
                    st.session_state.selected_course,
                    st.session_state.selected_lesson,
                    exam_mode=True,
                    chat_history=history_for_api
                )
             st.session_state.messages.append({"role": "model", "content": response})
             st.rerun()
        else: # QA mode
            with st.spinner("Thinking..."):
                response = generate_response(
                    prompt,
                    st.session_state.current_lecture_text,
                    st.session_state.selected_course,
                    st.session_state.selected_lesson,
                    exam_mode=False,
                    chat_history=history_for_api
                )
            st.session_state.messages.append({"role": "model", "content": response})
            st.rerun()