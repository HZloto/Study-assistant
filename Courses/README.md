# Course Materials Directory

This directory contains course materials for the Study Assistant application.

## Structure

Each course should follow this structure:

```
Courses/
├── Course-Name/
│   ├── Lesson-1/
│   │   ├── lecture_notes.txt
│   │   ├── additional_reading.txt
│   │   └── summary.txt
│   ├── Lesson-2/
│   │   └── [lesson files]
│   └── ...
```

## Guidelines

1. **Course Names**: Use descriptive names (e.g., "14.73", "JPAL102", "Introduction-to-Economics")
2. **Lesson Names**: Use consistent naming (e.g., "Lesson-1", "Week-1", "Chapter-1")
3. **File Format**: All course materials should be in `.txt` format
4. **Encoding**: Use UTF-8 encoding for all text files
5. **Content**: Include lecture notes, readings, summaries, or any relevant course material
6. **Size Limits**: Keep lesson content under 50,000 characters (~25-30 pages) for optimal flashcard generation

## Content Size Recommendations

- **Optimal**: 10,000-30,000 characters per lesson
- **Good**: 30,000-50,000 characters per lesson  
- **Too Large**: >100,000 characters (may cause API errors)

### For Large Content:
- Split long transcripts into multiple lessons
- Create separate lessons for different topics/weeks
- Remove repetitive or non-essential content

## Example

See the `Example-Course` folder for a sample structure with properly formatted course materials.

## Adding Your Courses

1. Create a new folder with your course name
2. Create subfolders for each lesson/topic
3. Add `.txt` files with your course materials
4. The app will automatically detect and load your courses

## Notes

- Only `.txt` files are processed by the application
- Files are loaded in alphabetical order within each lesson
- Large files may cause performance issues or API errors
- The application caches generated flashcards in the `data/` directory
- Content is automatically truncated if too long, but splitting manually is recommended

## Troubleshooting Content Size Issues

If you encounter "input context too long" errors:

1. **Check file sizes**: Use `wc -c filename.txt` to check character count
2. **Split large files**: Break content into logical chunks (by week, topic, etc.)
3. **Remove redundancy**: Delete repeated content or lengthy examples
4. **Use summaries**: Create condensed versions of lengthy materials
