

import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from app.parsers.resume_parser import ResumeParser
from app.database.models import db, Candidate
import uuid

app = Flask(__name__, 
            static_folder='app/static',
            template_folder='app/templates')

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///resume_parser.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx'}

# Initialize the database
db.init_app(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    candidates = Candidate.query.order_by(Candidate.upload_timestamp.desc()).all()
    return render_template('index.html', candidates=candidates)

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
        
    file = request.files['resume']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        # Generate a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        extension = original_filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4()}.{extension}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Parse resume
        try:
            parser = ResumeParser(file_path, extension)
            parsed_data = parser.extract_data()
            
            # Save to database
            candidate = Candidate(
                candidate_id=str(uuid.uuid4()),
                file_path=file_path,
                name=parsed_data.get('name', 'Unknown'),
                email=parsed_data.get('email', ''),
                phone=parsed_data.get('phone', ''),
                address=parsed_data.get('address', ''),
                skills=parsed_data.get('skills', []),
                work_experience=parsed_data.get('work_experience', []),
                education=parsed_data.get('education', []),
                certifications=parsed_data.get('certifications', []),
                projects=parsed_data.get('projects', []),
                awards=parsed_data.get('awards', []),
                parsing_status='success'
            )
            
            db.session.add(candidate)
            db.session.commit()
            
            flash('Resume uploaded and parsed successfully!', 'success')
            return redirect(url_for('candidate_details', candidate_id=candidate.candidate_id))
            
        except Exception as e:
            # Create a record with parsing error status
            candidate = Candidate(
                candidate_id=str(uuid.uuid4()),
                file_path=file_path,
                name='Parsing Error',
                parsing_status='error',
                parsing_error=str(e)
            )
            
            db.session.add(candidate)
            db.session.commit()
            
            flash(f'Error parsing resume: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload PDF, DOC, or DOCX files.', 'error')
        return redirect(request.url)

@app.route('/candidate/<candidate_id>')
def candidate_details(candidate_id):
    candidate = Candidate.query.filter_by(candidate_id=candidate_id).first_or_404()
    return render_template('candidate_details.html', candidate=candidate)

@app.route('/delete/<candidate_id>', methods=['POST'])
def delete_candidate(candidate_id):
    candidate = Candidate.query.filter_by(candidate_id=candidate_id).first_or_404()
    
    # Delete the file if it exists
    if candidate.file_path and os.path.exists(candidate.file_path):
        os.remove(candidate.file_path)
    
    db.session.delete(candidate)
    db.session.commit()
    
    flash('Candidate record deleted successfully', 'success')
    return redirect(url_for('index'))

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16MB', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)