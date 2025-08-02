# 📚 Study Assistant

<div align="center">
  <img src="Icon.png" alt="Study Assistant Icon" width="512">
</div>

A lightweight AI-powered study support tool that helps students learn more effectively from their class transcripts and notes. Built with Streamlit and powered by Google's Gemini AI.

## ✨ Features

- **📖 Interactive Q&A**: Ask questions about your course materials and get AI-powered answers
- **📝 Exam Mode**: Take practice exams with varied question types (multiple choice, fill-in-the-blank, true/false, select all)
- **⚡ Smart Flashcards**: Auto-generate flashcards from course content with intelligent caching
- **💾 Caching System**: Saves generated flashcards to avoid redundant API calls
- **📁 Multi-Course Support**: Organize materials by courses and lessons
- **🎯 Context-Aware**: AI responses are strictly based on your provided course materials

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd Study-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   
   Create a `.env` file in the root directory:
   ```bash
   touch .env
   ```
   
   Add your Gemini API key to the `.env` file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```
   
   > ⚠️ **Important**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

4. **Add your course materials**
   
   Follow the structure in `Courses/Example-Course/`:
   ```
   Courses/
   ├── Your-Course-Name/
   │   ├── Lesson-1/
   │   │   ├── lecture_notes.txt
   │   │   └── readings.txt
   │   ├── Lesson-2/
   │   │   └── content.txt
   │   └── ...
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Start studying!**
   - Select your course and lesson from the sidebar
   - Click "Load Materials"
   - Choose your study mode: Q&A, Exam, or Flashcards

## 📁 Course Material Setup

### File Structure
```
Courses/
├── README.md                 # Guidelines for organizing courses
├── Example-Course/           # Sample course (included)
│   └── Lesson-1/
│       └── sample_content.txt
└── Your-Course/              # Your course materials
    ├── Week-1/
    │   ├── lecture_transcript.txt
    │   └── readings.txt
    └── Week-2/
        └── notes.txt
```

### Guidelines
- **Format**: Use `.txt` files only (UTF-8 encoded)
- **Organization**: Create folders for each course, then subfolders for lessons/weeks
- **Content**: Include lecture transcripts, notes, readings, or any study materials
- **Naming**: Use consistent naming conventions (e.g., "Week-1", "Chapter-2", "Lesson-3")

## 🎓 Study Modes

### 💬 Q&A Mode (Default)
- Ask questions about your course materials
- Get contextual answers based only on your content
- Perfect for clarifying concepts and reviewing material

### 📝 Exam Mode
- Take AI-generated practice exams
- Multiple question types: multiple choice, true/false, fill-in-the-blank, select all
- Immediate feedback with explanations
- Questions are based strictly on your course content

### ⚡ Flashcard Mode
- Auto-generate flashcards from course materials
- Navigate with Previous/Next buttons
- Show/hide answers for active recall practice
- Cached for fast loading on repeat visits

## 🔧 Configuration

### API Key Setup
The app supports both environment variable names:
- `GOOGLE_API_KEY` (recommended)
- `GEMINI_API_KEY` (legacy support)

### Model Configuration
- Default model: `gemini-2.5-pro`
- Optimized for educational content and question generation
- Configured for reliable JSON output for flashcards

## 💾 Data Management

### Caching
- Flashcards are automatically cached in `data/flashcard_cache/`
- Reduces API costs and improves performance
- Cache files are course and lesson-specific

### Privacy
- All course materials remain local
- Only processed content is sent to Gemini API for generating responses
- No personal data is stored or transmitted beyond what's necessary for AI responses

## 🛠️ Technical Details

### Built With
- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[Google Gemini](https://ai.google.dev/)**: AI model for content understanding and generation
- **Python 3.8+**: Core programming language

### Key Features
- Robust error handling for API calls
- Smart content parsing and recovery
- Session state management for seamless user experience
- Natural sorting for course and lesson organization

## 🤝 Contributing

### Adding Course Support
To add support for additional file formats or course structures:
1. Modify `extract_text_from_folder()` function
2. Update course info mapping in `get_course_info()`
3. Test with your specific course format

### Improving AI Responses
- Adjust temperature settings in `generate_response()` for different response styles
- Modify system prompts for different teaching approaches
- Customize question types in exam mode

## 📝 License

This project is intended for educational use. Please ensure compliance with your institution's academic integrity policies when using AI assistance for studying.

## 🐛 Troubleshooting

### Common Issues

**"No courses found"**
- Ensure you have created course folders in the `Courses/` directory
- Check that your course folders contain lesson subfolders with `.txt` files

**"API key not found"**
- Verify your `.env` file exists in the root directory
- Check that your API key is correctly formatted in the `.env` file
- Ensure the environment variable name matches `GOOGLE_API_KEY`

**"Failed to generate flashcards"**
- Your course content might be too long; try shorter lessons
- Check your API quota and billing status
- Look for special characters that might break JSON parsing

**App crashes or doesn't start**
- Install all requirements: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Verify all file paths and permissions

### Getting Help
1. Check the console output for detailed error messages
2. Verify your course material format matches the example
3. Test with the included Example-Course first
4. Check the Gemini API status and your quota limits

---

**Happy studying! 🎓✨**
