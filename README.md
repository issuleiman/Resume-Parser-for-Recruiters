# Resume Parser for Recruiters

A modern resume parsing system that allows recruiters to extract and analyze key information from candidate resumes using non-transformer-based NLP techniques.

## Features

- **File Upload Support**: Accept resumes in PDF, DOC, and DOCX formats
- **Intelligent Resume Parsing**: Extract key information using rule-based NLP techniques
- **Clean, Professional UI**: Modern interface inspired by platforms like Greenhouse and Lever
- **Visual Parsing Status Indicators**: Clear success/error indicators (✅ Green #22C55E, ❌ Red #EF4444)
- **Comprehensive Data Extraction**:
  - Basic Info: Name, Email, Phone, Address
  - Skills
  - Work Experience Timeline
  - Education
  - Certifications
  - Projects
  - Awards / Achievements
- **SQL Database Storage**: Persistent storage of parsed resume data

## Tech Stack

### Parsing Technologies
- **pyresparser**: For extracting name, email, skills, education, experience
- **docx2txt / pdfminer.six**: For extracting text from resume files
- **spaCy (en_core_web_sm)**: For POS tagging, NER, and pattern-based extraction (no transformers)
- **nltk**: For tokenization, POS tagging, basic rule-based NER

### Backend
- **Python Flask**: Web framework
- **SQLAlchemy**: ORM for database operations
- **SQLite**: Default database (configurable)

### Frontend
- **HTML/CSS**: Clean, responsive UI
- **Tailwind CSS**: Modern styling
- **Custom components**: Cards, timelines, status indicators with the specified color scheme

## Setup Instructions

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required NLTK data and spaCy model:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
   python -m spacy download en_core_web_sm
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. On the main dashboard, use the "Upload Resume" card to select and upload a resume file (.pdf, .docx, or .doc).
2. Click the "Parse Resume" button to upload and parse the file.
3. The system will extract information and display a success or error status.
4. For successfully parsed resumes, you'll be redirected to a detailed view showing all extracted information.
5. The main dashboard lists all previously uploaded resumes with status indicators.

## Database Schema

- **candidate_id**: UUID for unique identification
- **name, email, phone, address**: Basic candidate information
- **skills**: Array of extracted skills
- **work_experience**: JSON array with title, company, dates, responsibilities
- **education**: JSON array with degree, institution, dates
- **certifications, projects, awards**: Additional extracted information
- **parsing_status**: Success/error indicator
- **upload_timestamp**: When the resume was processed

## Deployment (Production)

For production deployment:

1. Set environment variables:
   ```
   export SECRET_KEY="your-secure-key"
   export DATABASE_URL="postgresql://user:password@localhost/dbname"  # Optional: for PostgreSQL
   ```

2. Use a production WSGI server:
   ```
   gunicorn app:app
   ```