from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class JsonEncodedList(db.TypeDecorator):
    """Enables JSON storage by encoding and decoding on the fly."""
    impl = db.Text
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return '[]'
        return json.dumps(value)
        
    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return json.loads(value)


class Candidate(db.Model):
    __tablename__ = 'candidates'
    
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.String(36), unique=True, nullable=False)  # UUID as string
    file_path = db.Column(db.String(255), nullable=False)
    
    # Basic Information
    name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    
    # Skills and other structured data - stored as JSON
    skills = db.Column(JsonEncodedList, default=[])
    work_experience = db.Column(JsonEncodedList, default=[])
    education = db.Column(JsonEncodedList, default=[])
    certifications = db.Column(JsonEncodedList, default=[])
    projects = db.Column(JsonEncodedList, default=[])
    awards = db.Column(JsonEncodedList, default=[])
    
    # Parsing status
    parsing_status = db.Column(db.String(10), default='success')  # success or error
    parsing_error = db.Column(db.Text, nullable=True)  # store error message if parsing fails
    
    # Timestamps
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Candidate {self.name}>'