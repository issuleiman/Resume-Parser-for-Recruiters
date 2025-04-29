import os
import re
import docx2txt
import spacy
import nltk
from datetime import datetime
from pdfminer.high_level import extract_text
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import json

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    # Download the model if not available
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

class ResumeParser:
    def __init__(self, file_path, file_extension):
        self.file_path = file_path
        self.file_extension = file_extension
        self.text = self._extract_text()
        
    def _extract_text(self):
        """Extract text from resume file based on file extension"""
        if self.file_extension == 'pdf':
            return extract_text(self.file_path)
        elif self.file_extension == 'docx':
            return docx2txt.process(self.file_path)
        elif self.file_extension == 'doc':
            # For .doc files we'd need an additional library like textract
            # For this implementation, we'll return an error message
            raise ValueError("DOC format not fully supported. Please convert to DOCX or PDF.")
        else:
            raise ValueError(f"Unsupported file format: {self.file_extension}")
    
    def extract_data(self):
        """Main function to extract all data from resume with improved section detection"""
        if not self.text:
            raise ValueError("Could not extract text from file")
        
        # Process the document with spaCy
        doc = nlp(self.text)
        
        # Detect section boundaries for more accurate extraction
        sections = self._detect_section_boundaries()
        
        # Dictionary to store all parsed data
        data = {
            'name': self._extract_name(doc),
            'email': self._extract_email(),
            'phone': self._extract_phone(),
            'address': self._extract_address(doc),
            'skills': self._extract_skills(doc),
            'education': self._extract_education(doc),
            'work_experience': self._extract_work_experience(doc),
            'certifications': self._extract_certifications(doc),
            'projects': self._extract_projects(doc),
            'awards': self._extract_awards(doc),
            'summary': self._extract_summary(doc, sections.get('summary', [])),
        }
        
        return data
    
    def _extract_name(self, doc):
        """Extract candidate name using enhanced NER and pattern recognition"""
        # Names typically appear at the top of the resume
        # First check the first few lines with priority
        lines = self.text.split('\n')
        first_lines = [line.strip() for line in lines[:5] if line.strip()]
        
        # Look for name patterns in first few lines (most likely to contain the name)
        for line in first_lines:
            # Skip lines that are too long to be just a name
            if len(line) > 40:
                continue
                
            # Skip lines that look like contact info
            if '@' in line or 'http' in line or any(char in line for char in '0123456789'):
                continue
                
            # Skip lines that look like headers
            if line.lower() in ['resume', 'curriculum vitae', 'cv', 'profile']:
                continue
                
            # Look for name patterns (capitalized words without common title markers)
            name_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$', line)
            if name_match:
                return name_match.group(1).strip()
        
        # If name not found in first lines, use NER on first 300 characters
        first_section = doc[:min(len(doc), 300)]
        person_entities = [ent.text for ent in first_section.ents if ent.label_ == 'PERSON']
        
        if person_entities:
            # Get the longest person entity as it's likely to be the full name
            return max(person_entities, key=len)
        
        # Final fallback: look for capitalized words at the beginning
        initial_tokens = [token.text for token in doc[:50] if token.is_alpha and token.is_title and not token.is_stop]
        if initial_tokens and len(initial_tokens) >= 2:
            # Only consider the first 2-3 title-case tokens as the name
            name_candidate = ' '.join(initial_tokens[:3])
            # Filter out tokens that are likely to be part of headers or titles
            name_filtered = ' '.join(token for token in name_candidate.split() 
                                    if token.lower() not in ['resume', 'curriculum', 'vitae', 'cv', 'profile', 'summary'])
            if name_filtered:
                return name_filtered
        
        return "Unknown"
    
    def _extract_email(self):
        """Extract email address using improved regex patterns and validation"""
        # More comprehensive pattern for email addresses
        email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
        
        # First look for labeled emails (more reliable)
        for line in self.text.split('\n'):
            line = line.strip().lower()
            if 'email' in line or 'e-mail' in line or 'mail' in line:
                emails = re.findall(email_pattern, line)
                if emails:
                    # Validate the email format and return the first valid one
                    for email in emails:
                        if self._validate_email(email):
                            return email
        
        # Then scan the whole text for email patterns
        emails = re.findall(email_pattern, self.text)
        
        # Filter and validate emails
        valid_emails = [email for email in emails if self._validate_email(email)]
        
        # Return the first valid email found, if any
        return valid_emails[0] if valid_emails else ""
    
    def _validate_email(self, email):
        """Validate if a string is likely to be a real email address"""
        # Basic validation to filter out false positives
        if not email or len(email) > 100:
            return False
            
        # Check for invalid characters in local part
        local_part = email.split('@')[0]
        if not local_part or not re.match(r'^[\w.+-]+$', local_part):
            return False
            
        # Check for common domain extensions
        domain = email.split('@')[1] if len(email.split('@')) > 1 else ""
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.io', '.co', '.me', 
                      '.info', '.biz', '.ca', '.uk', '.de', '.jp', '.fr', '.au', 
                      '.ru', '.ch', '.it', '.nl', '.se', '.no', '.es', '.mil', '.ai']
        
        if not domain or not any(domain.lower().endswith(tld) for tld in common_tlds):
            # Additional check for other country TLDs
            if not re.search(r'\.[a-z]{2,4}$', domain.lower()):
                return False
        
        return True
    
    def _extract_phone(self):
        """Extract phone number using enhanced regex patterns"""
        # More comprehensive patterns to match various phone formats
        patterns = [
            r'(\+\d{1,3}[-\.\s]??\d{1,4}[-\.\s]??\d{1,4}[-\.\s]??\d{1,4})',  # +1 123-456-7890
            r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4})',  # 123-456-7890
            r'(\d{3}[-\.\s]??\d{4}[-\.\s]??\d{4})',  # 123-4567-8900
            r'\((\d{3})\)[-\.\s]??\d{3}[-\.\s]??\d{4}',  # (123) 456-7890
            r'\((\d{3})\)\s*\d{3}[-\.\s]??\d{4}',  # (123)456-7890
            r'(\d{5}[-\.\s]??\d{5})',  # 12345-67890 (some international formats)
            r'(\d{4}[-\.\s]??\d{3}[-\.\s]??\d{3})',  # 1234-567-890 (some international formats)
            r'([0-9]{10})',  # 10 digit number without separators
            r'(tel|phone|mobile|cell)(?:[:]?\s*)(\+?[0-9\s\(\)\-\.]{8,})',  # Prefixed formats: "Tel: 123-456-7890"
        ]
        
        # First check common labeled formats
        for line in self.text.split('\n'):
            line = line.strip().lower()
            if any(prefix in line for prefix in ["phone", "mobile", "cell", "tel"]):
                # For lines that have a phone label, try all patterns
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        # Flatten possible tuple results from groups
                        match = matches[0]
                        if isinstance(match, tuple):
                            for item in match:
                                if len(re.sub(r'[^\d]', '', item)) >= 7:  # At least 7 digits for a valid number
                                    return item.strip()
                        else:
                            return match.strip()
        
        # Then check the entire text for phone patterns
        for pattern in patterns:
            matches = re.findall(pattern, self.text)
            if matches:
                # Flatten possible tuple results from groups
                match = matches[0]
                if isinstance(match, tuple):
                    for item in match:
                        if len(re.sub(r'[^\d]', '', item)) >= 7:  # At least 7 digits
                            return item.strip()
                else:
                    # Make sure we have enough digits for a valid phone number
                    if len(re.sub(r'[^\d]', '', match)) >= 7:
                        return match.strip()
        
        # Special case for phone numbers without separators
        # Look for sequences of 10-12 digits that might be phone numbers
        digit_sequences = re.findall(r'\b\d{10,12}\b', self.text)
        if digit_sequences:
            for seq in digit_sequences:
                # Check if this looks like a phone number (not a date or other number)
                if not any(year_pattern in seq for year_pattern in ["19", "20"]):  # Avoid dates
                    # Format it properly (simple formatting as XXX-XXX-XXXX)
                    if len(seq) == 10:
                        return f"{seq[:3]}-{seq[3:6]}-{seq[6:]}"
                    return seq
        
        return ""
    
    def _extract_address(self, doc):
        """Extract address using enhanced NER, pattern recognition and contextual analysis"""
        # Look for address section indicators
        address_indicators = ['address', 'location', 'residence', 'residing at', 'live in', 'based in']
        
        # Pattern for common address components
        address_patterns = [
            r'\d+\s+[A-Za-z0-9\s,\.]+(?:Avenue|Ave|Boulevard|Blvd|Circle|Cir|Court|Ct|Drive|Dr|Lane|Ln|Park|Parkway|Pkwy|Place|Pl|Plaza|Plz|Point|Pt|Road|Rd|Square|Sq|Street|St|Terrace|Ter|Way)\b',
            r'\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',  # City, State ZIP
            r'\b[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',  # State ZIP
        ]
        
        # Look for GPE (geopolitical entity), LOC (location), FAC (facility) entities
        address_entities = []
        location_entities = []
        
        lines = self.text.split('\n')
        
        # First look for address indicators in lines
        for line in lines:
            line_lower = line.lower()
            # Check if line contains address indicators
            if any(indicator in line_lower for indicator in address_indicators):
                # This line likely contains or introduces address information
                # Process with NER
                line_doc = nlp(line)
                # Extract entities that might be part of an address
                for ent in line_doc.ents:
                    if ent.label_ in ['GPE', 'LOC', 'FAC']:
                        address_entities.append(ent.text)
                
                # Check if line contains address patterns
                for pattern in address_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        for match in matches:
                            address_entities.append(match)
        
        # If we didn't find address from indicators, extract from all entities
        if not address_entities:
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC', 'FAC']:
                    location_entities.append(ent.text)
            
            # Look for city-state-zip patterns
            for i, line in enumerate(lines[:20]):  # Check first 20 lines (header area)
                for pattern in address_patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        for match in matches:
                            location_entities.append(match)
        
        # Add zipcode/postal code detection with international formats
        postal_patterns = [
            r'\b\d{5}(?:-\d{4})?\b',  # US format: 12345 or 12345-6789
            r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b',  # Canadian format: A1A 1A1
            r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',  # UK format: AA1 1AA or A1 1AA
            r'\b\d{4}\s*[A-Z]{2}\b',  # Dutch format: 1234 AB
            r'\b\d{5}-\d{3}\b'  # Brazilian format: 12345-678
        ]
        
        postal_codes = []
        for pattern in postal_patterns:
            matches = re.findall(pattern, self.text)
            if matches:
                postal_codes.extend(matches)
        
        # Select the most likely address components
        address_components = address_entities if address_entities else location_entities[:3]
        
        # If we have postal codes, add the first one found
        if postal_codes and postal_codes[0] not in address_components:
            address_components.append(postal_codes[0])
        
        # Combine and format appropriately
        # Remove duplicates while preserving order
        unique_components = []
        for component in address_components:
            if component not in unique_components:
                unique_components.append(component)
        
        return ", ".join(unique_components) if unique_components else ""
    
    def _extract_skills(self, doc):
        """Extract skills using enhanced pattern matching and NLP techniques"""
        # Common skill section headers with various formatting possibilities
        skill_headers = [
            'skills', 'technical skills', 'core competencies', 'technologies', 'technical expertise',
            'proficiencies', 'areas of expertise', 'skill set', 'professional skills', 'competencies',
            'key skills', 'qualifications', 'technical proficiencies', 'expertise', 'strengths'
        ]
        
        # Comprehensive technical skill keywords list with categories
        programming_languages = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'ruby', 'go', 'golang',
            'php', 'swift', 'kotlin', 'r', 'matlab', 'perl', 'scala', 'bash', 'powershell', 'rust',
            'dart', 'objective-c', 'vba', 'cobol', 'fortran', 'lisp', 'haskell', 'erlang', 'clojure'
        ]
        
        web_tech = [
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind', 'jquery', 'react', 'angular', 
            'vue', 'svelte', 'nextjs', 'gatsby', 'redux', 'nodejs', 'express', 'django', 'flask',
            'spring', 'laravel', 'asp.net', 'webassembly', 'graphql', 'restful api', 'soap', 'ajax'
        ]
        
        data_tech = [
            'sql', 'nosql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'cassandra', 'redis', 'elasticsearch',
            'dynamodb', 'mariadb', 'firebase', 'neo4j', 'couchdb', 'sqlite', 'dax', 'power query',
            'data modeling', 'er diagrams', 'olap', 'etl', 'data warehousing', 'data lakes'
        ]
        
        cloud_devops = [
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'github actions',
            'gitlab ci', 'terraform', 'ansible', 'puppet', 'chef', 'prometheus', 'grafana', 'nagios',
            'cloudformation', 'azure devops', 'nginx', 'apache', 'cloudflare', 'heroku', 'netlify', 'vercel'
        ]
        
        ai_ml = [
            'machine learning', 'deep learning', 'artificial intelligence', 'nlp', 'natural language processing',
            'computer vision', 'neural networks', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
            'numpy', 'scipy', 'matplotlib', 'seaborn', 'data visualization', 'big data', 'hadoop', 'spark',
            'reinforcement learning', 'regression', 'classification', 'clustering', 'dimensionality reduction'
        ]
        
        tools_software = [
            'git', 'svn', 'jira', 'confluence', 'trello', 'asana', 'notion', 'slack', 'microsoft teams', 'linux',
            'unix', 'windows', 'macos', 'office', 'excel', 'word', 'powerpoint', 'outlook', 'visio', 'adobe',
            'photoshop', 'illustrator', 'indesign', 'figma', 'sketch', 'tableau', 'power bi', 'qlik', 'looker'
        ]
        
        soft_skills = [
            'leadership', 'communication', 'problem solving', 'teamwork', 'management', 'critical thinking',
            'time management', 'project management', 'agile', 'scrum', 'kanban', 'stakeholder management',
            'negotiation', 'conflict resolution', 'presentation', 'public speaking', 'mentoring', 'coaching'
        ]
        
        # Combine all skill categories
        all_skills = (
            programming_languages + web_tech + data_tech + cloud_devops + 
            ai_ml + tools_software + soft_skills
        )
        
        # First look for skill sections in the resume with improved section recognition
        lines = self.text.split('\n')
        skill_section = False
        skills_found = set()
        skill_section_text = ""
        
        # First pass: Identify skill sections and collect their text
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is a skill section header
            if any(header in line_lower for header in skill_headers):
                skill_section = True
                continue
            
            # If we're in a skill section, collect the text
            if skill_section:
                # Check if we've hit another section header
                if (not line or 
                    line_lower.endswith(':') or 
                    (i < len(lines) - 1 and lines[i+1].strip().endswith(':')) or
                    any(header in line_lower for header in 
                        ['experience', 'education', 'projects', 'certifications'])):
                    skill_section = False
                else:
                    skill_section_text += line + " "
        
        # Process collected skill section text with specialized parsing
        if skill_section_text:
            # Clean up the text
            skill_section_text = re.sub(r'\s+', ' ', skill_section_text)
            
            # Extract skills from bulleted lists or comma-separated values
            skill_list = []
            
            # First try to extract skills separated by bullets, commas, or other separators
            potential_skills = re.split(r'[,•|\t/;]', skill_section_text)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) > 2 and len(skill) < 40:
                    skill_list.append(skill)
            
            # Process the list to standardize and filter skills
            for skill in skill_list:
                # Handle multi-word skills and normalize them
                if any(tech in skill.lower() for tech in all_skills):
                    skills_found.add(skill.strip())
                else:
                    # For unrecognized terms, check if they might be technologies not in our list
                    doc_skill = nlp(skill)
                    # Add if it has proper noun characteristics or is in title case
                    if any(token.pos_ in ['PROPN', 'NOUN'] for token in doc_skill) or skill.istitle():
                        skills_found.add(skill.strip())
        
        # Second pass: Look for all predefined skills throughout the document
        text_lower = self.text.lower()
        
        # Function to check if a skill exists as a whole word in text
        def has_skill(skill, text):
            # Create a pattern that ensures the skill is a whole word
            pattern = r'\b' + re.escape(skill) + r'\b'
            return re.search(pattern, text, re.IGNORECASE) is not None
        
        # Check for whole-word matches of all predefined skills
        for skill in all_skills:
            if has_skill(skill, text_lower) and len(skill) > 2:
                # Capitalize multi-word skills properly
                formatted_skill = ' '.join(word.capitalize() if word not in ['of', 'and', 'the', 'in', 'on', 'with', 'for'] 
                                         else word for word in skill.split())
                skills_found.add(formatted_skill)
        
        # Look for skill phrases with consecutive capitalized words
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b', self.text):
            phrase = match.group(1)
            if 2 < len(phrase) < 40 and not any(word.lower() in ['resume', 'curriculum', 'vitae'] for word in phrase.split()):
                skills_found.add(phrase)
        
        # Sort and return the final list of skills
        return sorted(list(skills_found))
    
    def _extract_education(self, doc):
        """Extract education information using improved pattern matching and context analysis"""
        # Comprehensive education keywords
        education_headers = [
            'education', 'academic background', 'qualification', 'academic history',
            'academic qualification', 'educational qualification'
        ]
        
        degree_keywords = [
            # Bachelor degrees
            'bachelor', 'bachelors', 'b.s', 'b.a', 'b.eng', 'b.e', 'b.tech', 'b.sc', 'bs', 'ba', 'bsc', 'btech', 
            'beng', 'undergraduate', 'bachelor of science', 'bachelor of arts', 'bachelor of engineering',
            'bachelor of technology', 'b.com', 'bcom',
            
            # Master degrees
            'master', 'masters', 'm.s', 'm.a', 'm.eng', 'm.e', 'm.tech', 'm.sc', 'ms', 'ma', 'msc', 'mtech', 
            'meng', 'graduate', 'master of science', 'master of arts', 'master of engineering',
            'master of technology', 'mba', 'master of business administration',
            
            # Doctorate degrees
            'phd', 'ph.d', 'doctorate', 'doctor of philosophy', 'd.phil', 'dphil',
            
            # Associate degrees
            'associate', 'a.a', 'a.s', 'a.a.s', 'associate of arts', 'associate of science',
            
            # Other educational qualifications
            'diploma', 'certificate', 'post-graduate', 'postgraduate'
        ]
        
        institution_keywords = [
            'university', 'college', 'institute', 'school', 'academy',
            'polytechnic', 'community college', 'technical', 'campus'
        ]
        
        major_keywords = [
            'major in', 'major', 'specialization', 'specialized in', 'concentration in',
            'focus on', 'focused on', 'field of study', 'field'
        ]
        
        gpa_keywords = [
            'gpa', 'grade point average', 'cgpa', 'cumulative gpa'
        ]
        
        # Extract education section from resume
        lines = self.text.split('\n')
        education_section = False
        education_text = []
        education_entries = []
        
        # First pass: Find and extract education section
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is an education section header
            if any(header in line_lower for header in education_headers):
                education_section = True
                continue
            
            # If we're in the education section, collect the text
            if education_section:
                # Check if we've reached another section
                if (not line or line_lower.endswith(':') or 
                    any(header in line_lower for header in ['experience', 'skills', 'projects', 'certifications'])):
                    if line_lower not in ['', 'education:']:
                        education_section = False
                        continue
                
                # Add non-empty lines to our education text collection
                if line:
                    education_text.append(line)
        
        # If no clear education section found, try to find education-related information in the whole document
        if not education_text:
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in degree_keywords) and any(inst in line.lower() for inst in institution_keywords):
                    education_text.append(line)
        
        # Process collected education text to extract structured information
        current_entry = {}
        skip_line = False
        
        for i, line in enumerate(education_text):
            if skip_line:
                skip_line = False
                continue
                
            line_lower = line.lower()
            
            # Check for degree information
            degree_match = False
            for degree in degree_keywords:
                pattern = r'\b' + re.escape(degree) + r'\b'
                if re.search(pattern, line_lower, re.IGNORECASE):
                    degree_match = True
                    
                    # If we already have an entry in progress, save it before starting a new one
                    if current_entry and 'degree' in current_entry:
                        education_entries.append(current_entry)
                        current_entry = {}
                    
                    current_entry['degree'] = line
                    
                    # Look for dates in the same line
                    date_range_match = re.search(r'(19|20)\d{2}\s*(-|to|–|until|present|current|now)\s*(19|20)\d{2}|present|current|now', line, re.IGNORECASE)
                    if date_range_match:
                        current_entry['date_range'] = date_range_match.group(0)
                    else:
                        # Look for single year which might indicate graduation year
                        year_match = re.search(r'(19|20)\d{2}', line)
                        if year_match:
                            current_entry['year'] = year_match.group(0)
                    
                    # Look for institution name in the same line
                    for inst in institution_keywords:
                        if inst in line_lower:
                            # Extract the complete institution name
                            # This is a simplified approach; in a more complex scenario,
                            # we might use NER to extract the complete institution name
                            inst_pattern = r'(?i)(?:[\w\s]+\s)?' + re.escape(inst) + r'(?:\s(?:of|for|in)(?:\s[\w\s&,]+)?)?'
                            inst_match = re.search(inst_pattern, line)
                            if inst_match:
                                current_entry['institution'] = inst_match.group(0).strip()
                            break
                    
                    # Look for major/field of study
                    for major_kw in major_keywords:
                        major_pattern = r'(?i)' + re.escape(major_kw) + r'\s+([\w\s&,]+)'
                        major_match = re.search(major_pattern, line)
                        if major_match:
                            current_entry['major'] = major_match.group(1).strip()
                            break
                    
                    # Look for GPA
                    for gpa_kw in gpa_keywords:
                        gpa_pattern = r'(?i)' + re.escape(gpa_kw) + r'[\s:]*([0-9.]+)(?:/[0-9.]+)?'
                        gpa_match = re.search(gpa_pattern, line)
                        if gpa_match:
                            current_entry['gpa'] = gpa_match.group(1).strip()
                            break
                    
                    break
            
            # If this line wasn't a degree line, check if it's related to the current entry
            if not degree_match and current_entry:
                # Check for institution name
                if not 'institution' in current_entry:
                    for inst in institution_keywords:
                        if inst in line_lower:
                            current_entry['institution'] = line.strip()
                            break
                
                # Check for date range if we don't have it yet
                if not ('date_range' in current_entry or 'year' in current_entry):
                    date_range_match = re.search(r'(19|20)\d{2}\s*(-|to|–|until)\s*(19|20)\d{2}|present|current|now', line, re.IGNORECASE)
                    if date_range_match:
                        current_entry['date_range'] = date_range_match.group(0)
                    else:
                        # Look for single year which might indicate graduation year
                        year_match = re.search(r'(19|20)\d{2}', line)
                        if year_match:
                            current_entry['year'] = year_match.group(0)
                
                # Check for major/field of study
                if not 'major' in current_entry:
                    for major_kw in major_keywords:
                        if major_kw in line_lower:
                            # Extract what comes after the major keyword
                            parts = line_lower.split(major_kw)
                            if len(parts) > 1 and parts[1].strip():
                                current_entry['major'] = parts[1].strip()
                            break
                
                # Check for GPA
                if not 'gpa' in current_entry:
                    for gpa_kw in gpa_keywords:
                        if gpa_kw.lower() in line_lower:
                            gpa_pattern = r'(?i)' + re.escape(gpa_kw) + r'[\s:]*([0-9.]+)(?:/[0-9.]+)?'
                            gpa_match = re.search(gpa_pattern, line)
                            if gpa_match:
                                current_entry['gpa'] = gpa_match.group(1).strip()
                            break
        
        # Add the last entry if it exists
        if current_entry and 'degree' in current_entry:
            education_entries.append(current_entry)
        
        # If we still don't have any education entries but found education lines,
        # create a simple entry with what we know
        if not education_entries and education_text:
            combined_text = ' '.join(education_text)
            entry = {'degree': combined_text}
            
            # Try to extract key information from combined text
            # Institution
            for inst in institution_keywords:
                inst_pattern = r'(?i)(?:[\w\s]+\s)?' + re.escape(inst) + r'(?:\s(?:of|for|in)(?:\s[\w\s&,]+)?)?'
                inst_match = re.search(inst_pattern, combined_text)
                if inst_match:
                    entry['institution'] = inst_match.group(0).strip()
                    break
            
            # Date range
            date_range_match = re.search(r'(19|20)\d{2}\s*(-|to|–|until)\s*(19|20)\d{2}|present|current|now', combined_text, re.IGNORECASE)
            if date_range_match:
                entry['date_range'] = date_range_match.group(0)
            else:
                # Look for single year which might indicate graduation year
                year_match = re.search(r'(19|20)\d{2}', combined_text)
                if year_match:
                    entry['year'] = year_match.group(0)
            
            education_entries.append(entry)
        
        # Final cleanup of education entries
        for entry in education_entries:
            # Clean up the degree field (remove extra information that might be included)
            if 'degree' in entry:
                # Remove date ranges and years from degree field
                entry['degree'] = re.sub(r'(19|20)\d{2}\s*(-|to|–|until)\s*(19|20)\d{2}|present|current|now', '', entry['degree']).strip()
                entry['degree'] = re.sub(r'(19|20)\d{2}', '', entry['degree']).strip()
                
                # Remove institution name from degree if it's duplicated
                if 'institution' in entry:
                    entry['degree'] = entry['degree'].replace(entry['institution'], '').strip()
                    # Remove any trailing punctuation
                    entry['degree'] = re.sub(r'[,;-]\s*$', '', entry['degree']).strip()
        
        return education_entries
    
    def _extract_work_experience(self, doc):
        """Extract work experience information using improved pattern matching and context analysis"""
        # Common work experience section headers with variations
        work_headers = [
            'experience', 'work experience', 'professional experience', 'employment',
            'employment history', 'work history', 'career', 'professional background',
            'professional history', 'work background', 'professional profile', 'career history'
        ]
        
        # Comprehensive list of job titles for better recognition
        job_titles = [
            # Engineering roles
            'engineer', 'developer', 'programmer', 'architect', 'administrator', 'technician',
            'devops', 'sysadmin', 'webmaster', 'reliability', 'technical', 'qa', 'tester',
            
            # Management roles
            'manager', 'director', 'chief', 'head', 'lead', 'principal', 'senior', 'junior',
            'supervisor', 'team lead', 'executive', 'president', 'ceo', 'cto', 'cio', 'coo', 
            'cfo', 'vp', 'vice president', 'founder', 'co-founder',
            
            # Analysis roles
            'analyst', 'consultant', 'specialist', 'strategist', 'researcher', 'scientist',
            
            # Other common roles
            'coordinator', 'assistant', 'associate', 'representative', 'agent', 'advisor',
            'officer', 'clerk', 'secretary', 'receptionist', 'intern', 'trainee', 'apprentice',
            'designer', 'writer', 'editor', 'marketing', 'sales', 'support', 'customer',
            'product', 'project', 'program', 'account', 'business', 'finance', 'hr', 'human resources'
        ]
        
        # Common organizational keywords that might indicate company names
        company_keywords = [
            'inc', 'llc', 'ltd', 'limited', 'corp', 'corporation', 'company', 'co', 'group',
            'plc', 'associates', 'partners', 'agency', 'consulting', 'solutions', 'services',
            'technologies', 'systems', 'software', 'international', 'global', 'worldwide', 'national',
            'enterprises', 'industries', 'gmbh', 'sa', 'ag'
        ]
        
        # Month expressions for better date parsing
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
            'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
        ]
        
        # Date patterns for more comprehensive date extraction
        date_patterns = [
            # Full date formats: Month Year - Month Year
            r'(?:(?:' + '|'.join(months) + r')\.?\s+)?(19|20)\d{2}\s*(?:-|to|–|until|through|present|\s+)*?(?:(?:' + '|'.join(months) + r')\.?\s+)?((?:19|20)\d{2}|present|current|now)',
            
            # Month Year format
            r'(?:' + '|'.join(months) + r')\.?\s+(?:19|20)\d{2}',
            
            # Year only format
            r'(?:^|\s|:)(19|20)\d{2}(?:\s*-\s*(?:present|current|now|\d{4}|\s*))?',
            
            # Present/current indicator
            r'(?:^|\s|:)present|current|now(?:\s*-|$)',
        ]
        
        lines = self.text.split('\n')
        work_section = False
        work_entries = []
        current_entry = {}
        
        # First pass: Find the work experience section and collect relevant text
        work_section_text = []
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is a work experience section header
            if any(header in line_lower for header in work_headers) and (
                line_lower.endswith(':') or any(line_lower == h for h in work_headers) or 
                i + 1 < len(lines) and not lines[i + 1].strip()
            ):
                work_section = True
                continue
            
            # If we're in the work section, collect the text
            if work_section:
                # Check if we've reached another major section header
                if (not line or 
                    (line_lower.endswith(':') and len(line) < 30 and
                     any(header in line_lower for header in ['education', 'skills', 'projects', 'certifications']))):
                    # We've reached the end of the work section
                    work_section = False
                    continue
                
                # If we're still in the work section, add the line to our collection
                if line:
                    work_section_text.append(line)
        
        # If no explicit work section was found, look for job title patterns throughout the resume
        if not work_section_text:
            potential_job_lines = []
            for line in lines:
                line = line.strip()
                if line and any(title in line.lower() for title in job_titles):
                    # Does it contain a date pattern?
                    has_date = False
                    for pattern in date_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            has_date = True
                            break
                    
                    if has_date or any(company in line.lower() for company in company_keywords):
                        potential_job_lines.append(line)
            
            # Add the next few lines after each potential job line to capture responsibilities
            if potential_job_lines:
                for job_line in potential_job_lines:
                    job_index = lines.index(job_line)
                    work_section_text.append(job_line)
                    
                    # Add up to 5 lines after the job line as potential responsibilities
                    for i in range(1, 6):
                        if job_index + i < len(lines) and lines[job_index + i].strip():
                            # Stop if we hit what looks like another job or section
                            if any(header in lines[job_index + i].lower() for header in ['education:', 'skills:', 'projects:', 'certifications']):
                                break
                            work_section_text.append(lines[job_index + i].strip())
        
        # Process the collected work experience text
        i = 0
        while i < len(work_section_text):
            line = work_section_text[i]
            line_lower = line.lower()
            
            # Check for patterns that indicate a new job entry
            new_entry = False
            
            # Pattern 1: Job title with company
            title_company_match = re.search(r'^(.*?(?:' + '|'.join(job_titles) + r').*?)(?:at|@|,|\||-|–|\bat\b)\s*(.*?)(?=\(|,|–|-|\d{4}|$)', line_lower)
            
            # Pattern 2: Line contains both job title keyword and company keyword
            has_job_and_company = any(title in line_lower for title in job_titles) and any(company in line_lower for company in company_keywords)
            
            # Pattern 3: Line starts with capitalized words (potential job title)
            capitalized_start = re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}', line) and any(title in line_lower for title in job_titles)
            
            # Pattern 4: Line contains a date pattern typically associated with job entries
            date_match = None
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    date_match = match
                    break
            
            # Check if we have a new job entry
            if title_company_match or (has_job_and_company and date_match) or (capitalized_start and date_match):
                # If we already have an entry in progress, save it
                if current_entry and 'title' in current_entry and 'company' in current_entry:
                    work_entries.append(current_entry)
                
                # Start a new entry
                current_entry = {'responsibilities': []}
                
                # Extract job title
                if title_company_match:
                    current_entry['title'] = title_company_match.group(1).strip()
                    current_entry['company'] = title_company_match.group(2).strip()
                else:
                    # Use NER to try to separate job title from company
                    doc_line = nlp(line)
                    org_entities = [ent.text for ent in doc_line.ents if ent.label_ == 'ORG']
                    
                    if org_entities:
                        # Assume first ORG entity is the company
                        company = org_entities[0]
                        # Remove company from line to get title
                        title = line.replace(company, '').strip()
                        # Clean up any remaining separators
                        title = re.sub(r'(?:at|@|,|\||-|–|\bat\b)\s*$', '', title).strip()
                        
                        current_entry['title'] = title
                        current_entry['company'] = company
                    else:
                        # Default fallback: use the whole line as title and try to find company in next lines
                        current_entry['title'] = line
                        if i + 1 < len(work_section_text) and any(company in work_section_text[i+1].lower() for company in company_keywords):
                            current_entry['company'] = work_section_text[i+1]
                            i += 1
                        else:
                            current_entry['company'] = "Unknown"
                
                # Extract date information
                if date_match:
                    current_entry['date_range'] = date_match.group(0).strip()
                else:
                    # Look for date in next line if needed
                    if i + 1 < len(work_section_text):
                        next_line = work_section_text[i+1]
                        for pattern in date_patterns:
                            next_date_match = re.search(pattern, next_line, re.IGNORECASE)
                            if next_date_match:
                                current_entry['date_range'] = next_date_match.group(0).strip()
                                if next_line.strip() == next_date_match.group(0).strip():
                                    # If next line is just the date, skip it in the next iteration
                                    i += 1
                                break
                
                new_entry = True
            
            # If not a new entry, check if it's a responsibility for current entry
            if not new_entry and current_entry and 'title' in current_entry:
                # Check if line is a bullet point or other indicator of a responsibility
                if line.lstrip().startswith(('•', '-', '✓', '*', '>', '→')):
                    # It's a bullet point, add as responsibility
                    responsibility = line.lstrip('•-✓*>→ \t')
                    if responsibility:
                        current_entry['responsibilities'].append(responsibility)
                elif not any(title in line_lower for title in job_titles) and not any(pattern in line_lower for pattern in ['education:', 'skills:', 'projects:', 'certifications']):
                    # If it doesn't look like a new job entry or section header, it's probably part of current job description
                    current_entry['responsibilities'].append(line)
            
            # Move to next line
            i += 1
        
        # Add the last entry if it exists
        if current_entry and 'title' in current_entry and 'company' in current_entry:
            work_entries.append(current_entry)
        
        # Clean up the entries
        for entry in work_entries:
            # Clean up responsibilities: remove duplicates and very short entries
            if 'responsibilities' in entry:
                unique_responsibilities = []
                for resp in entry['responsibilities']:
                    if resp and len(resp) > 5 and resp not in unique_responsibilities:
                        unique_responsibilities.append(resp)
                entry['responsibilities'] = unique_responsibilities
            
            # Clean up title and company (remove date information if present)
            for field in ['title', 'company']:
                if field in entry:
                    for pattern in date_patterns:
                        entry[field] = re.sub(pattern, '', entry[field], flags=re.IGNORECASE).strip()
                    
                    # Remove trailing separators or punctuation
                    entry[field] = re.sub(r'[,;:|—–-]\s*$', '', entry[field]).strip()
        
        return work_entries
    
    def _extract_certifications(self, doc):
        """Extract certification information"""
        # Common certification section headers
        cert_headers = ['certifications', 'certificates', 'credentials', 'licenses']
        
        lines = self.text.split('\n')
        cert_section = False
        certifications = []
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is a certification section header
            if any(header == line_lower for header in cert_headers):
                cert_section = True
                continue
            
            # If we're in certification section
            if cert_section:
                # If we hit another section header, exit cert section
                if not line or (line_lower.endswith(':') and len(line) < 30):
                    cert_section = False
                    continue
                
                # Skip empty lines
                if not line:
                    continue
                
                # If line has bullet points or dashes, clean them
                cert_line = line.lstrip('•-✓*>→ \t')
                if cert_line:
                    certifications.append(cert_line)
        
        return certifications
    
    def _extract_projects(self, doc):
        """Extract project information using improved pattern matching and context analysis"""
        # Comprehensive project section headers
        project_headers = [
            'projects', 'personal projects', 'academic projects', 'key projects', 
            'professional projects', 'side projects', 'portfolio projects',
            'research projects', 'development projects', 'project experience'
        ]
        
        # Technology keywords to identify project tech stacks
        tech_keywords = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue', 'node',
            'django', 'flask', 'spring', 'php', 'laravel', 'ruby', 'rails', 'go', 'golang',
            'c++', 'c#', '.net', 'html', 'css', 'bootstrap', 'tailwind', 'jquery', 'swift',
            'kotlin', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'mongodb', 'mysql',
            'postgresql', 'sql', 'nosql', 'firebase', 'android', 'ios', 'mobile', 'web',
            'tensorflow', 'pytorch', 'machine learning', 'ai', 'data science'
        ]
        
        # Role keywords in projects
        role_keywords = [
            'developer', 'lead', 'architect', 'designer', 'manager', 'coordinator',
            'researcher', 'contributor', 'creator', 'maintainer', 'owner', 'administrator'
        ]
        
        lines = self.text.split('\n')
        project_section = False
        projects = []
        current_project = {}
        project_section_lines = []
        
        # First pass: Find the project section and collect its text
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is a project section header
            if any(header in line_lower for header in project_headers) and (
                line_lower.endswith(':') or line_lower in project_headers or
                i + 1 < len(lines) and not lines[i + 1].strip()
            ):
                project_section = True
                continue
            
            # If we're in the project section, collect the text
            if project_section:
                # Check if we've reached another major section header
                if (not line or 
                    (line_lower.endswith(':') and len(line) < 30 and
                     any(header in line_lower for header in ['education', 'skills', 'experience', 'certifications', 'awards']))):
                    # We've reached the end of the project section
                    project_section = False
                    continue
                
                # Add non-empty lines to our collection
                if line:
                    project_section_lines.append(line)
        
        # If no explicit project section was found, look for project patterns throughout the resume
        if not project_section_lines:
            # Look for lines that might indicate projects
            for i, line in enumerate(lines):
                line = line.strip()
                if line and (
                    'project' in line.lower() or
                    any(f"developed {article}" in line.lower() for article in ['a', 'an', 'the']) or
                    any(f"created {article}" in line.lower() for article in ['a', 'an', 'the']) or
                    any(f"built {article}" in line.lower() for article in ['a', 'an', 'the'])
                ):
                    project_section_lines.append(line)
                    
                    # Include the next few lines as they might contain project details
                    for j in range(1, 5):  # Look at up to 4 lines after
                        if i + j < len(lines) and lines[i + j].strip():
                            # Stop if we hit what looks like another project or section
                            if any(header in lines[i + j].lower() for header in ['education:', 'skills:', 'experience:']):
                                break
                            project_section_lines.append(lines[i + j].strip())
        
        # Process the collected project lines
        i = 0
        while i < len(project_section_lines):
            line = project_section_lines[i]
            line_lower = line.lower()
            
            # Check if this line looks like a new project title
            is_new_project = False
            
            # Pattern 1: Line starts with a capital letter and doesn't look like a bullet point
            if (not line.lstrip().startswith(('•', '-', '✓', '*', '>', '→')) and
                re.match(r'^[A-Z]', line) and
                not line.endswith(':') and
                len(line) < 100):  # Project titles are typically not very long
                
                # Additional checks to identify project titles
                is_new_project = True
                
                # Don't consider lines that are clearly part of a responsibility
                if any(phrase in line_lower for phrase in ['responsible for', 'managed', 'developed', 'implemented', 'led']):
                    if not any(keyword in line_lower for keyword in ['project', 'application', 'website', 'system', 'platform']):
                        is_new_project = False
            
            if is_new_project:
                # Save previous project if it exists
                if current_project and 'name' in current_project:
                    projects.append(current_project)
                
                # Start a new project
                current_project = {
                    'name': line,
                    'description': []
                }
                
                # Extract date/timeframe or technologies if they're in parentheses
                parens_matches = re.findall(r'\(([^)]+)\)', line)
                for parens_content in parens_matches:
                    # Check if parentheses contain technologies
                    if any(tech in parens_content.lower() for tech in tech_keywords):
                        current_project['technologies'] = parens_content
                        # Remove the parenthetical part from the name for cleaner output
                        current_project['name'] = current_project['name'].replace(f"({parens_content})", "").strip()
                    
                    # Check if parentheses contain dates or timeframe
                    elif re.search(r'(19|20)\d{2}|month|year|week|present|current', parens_content.lower()):
                        current_project['timeframe'] = parens_content
                        # Remove the parenthetical part from the name for cleaner output
                        current_project['name'] = current_project['name'].replace(f"({parens_content})", "").strip()
                    
                    # Check if parentheses contain role information
                    elif any(role in parens_content.lower() for role in role_keywords):
                        current_project['role'] = parens_content
                        # Remove the parenthetical part from the name for cleaner output
                        current_project['name'] = current_project['name'].replace(f"({parens_content})", "").strip()
                
                # Check if ":" indicates a separation between project name and description
                colon_split = line.split(':', 1)
                if len(colon_split) > 1 and colon_split[1].strip():
                    current_project['name'] = colon_split[0].strip()
                    current_project['description'].append(colon_split[1].strip())
                
                # Check next line for technology information if not already found
                if 'technologies' not in current_project and i + 1 < len(project_section_lines):
                    next_line = project_section_lines[i + 1].lower()
                    tech_keywords_found = [tech for tech in tech_keywords if tech in next_line]
                    
                    if tech_keywords_found:
                        # Found technologies in the next line
                        if any(indicator in next_line for indicator in ['technology', 'tech stack', 'tools', 'using', 'built with']):
                            current_project['technologies'] = project_section_lines[i + 1]
                            i += 1  # Skip this line in the next iteration
        
            # If not a new project, process as part of the current project
            elif current_project and 'name' in current_project:
                # Check if line is a bullet point or other indicator of a description
                if line.lstrip().startswith(('•', '-', '✓', '*', '>', '→')):
                    # It's a bullet point, add as a description
                    description_point = line.lstrip('•-✓*>→ \t')
                    if description_point:
                        current_project['description'].append(description_point)
                
                # Check for technical details or role information
                elif not 'technologies' in current_project and any(tech in line_lower for tech in tech_keywords):
                    current_project['technologies'] = line
                
                # Everything else is part of the description
                else:
                    current_project['description'].append(line)
            
            # Move to next line
            i += 1
        
        # Add the last project if it exists
        if current_project and 'name' in current_project:
            projects.append(current_project)
        
        # Clean up the project entries
        for project in projects:
            # Make sure each project has a description list
            if 'description' not in project:
                project['description'] = []
            
            # Ensure descriptions are unique and non-empty
            project['description'] = [desc for desc in project['description'] if desc and len(desc) > 5]
            project['description'] = list(dict.fromkeys(project['description']))  # Remove duplicates while preserving order
            
            # If no technologies were explicitly identified but they're mentioned in the description,
            # try to extract them
            if 'technologies' not in project and project['description']:
                tech_found = []
                for desc in project['description']:
                    for tech in tech_keywords:
                        if re.search(r'\b' + re.escape(tech) + r'\b', desc.lower()):
                            tech_found.append(tech.capitalize())
                
                if tech_found:
                    project['technologies'] = ', '.join(tech_found)
            
            # Clean up project name (remove any trailing punctuation)
            project['name'] = re.sub(r'[,;:|—–-]\s*$', '', project['name']).strip()
        
        return projects
    
    def _extract_awards(self, doc):
        """Extract awards and achievements"""
        # Common award section headers
        award_headers = [
            'awards', 'achievements', 'honors', 'accomplishments', 
            'recognition', 'scholarships'
        ]
        
        lines = self.text.split('\n')
        award_section = False
        awards = []
        
        for line in lines:
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is an award section header
            if any(header == line_lower for header in award_headers):
                award_section = True
                continue
            
            # If we're in award section
            if award_section:
                # If we hit another section header, exit award section
                if not line or (line_lower.endswith(':') and len(line) < 30):
                    award_section = False
                    continue
                
                # Skip empty lines
                if not line:
                    continue
                
                # If line has bullet points or dashes, clean them
                award_line = line.lstrip('•-✓*>→ \t')
                if award_line:
                    awards.append(award_line)
        
        return awards
    
    def _extract_summary(self, doc, summary_sections=None):
        """Extract the professional summary or objective from the resume"""
        # Summary section headers with variations
        summary_headers = [
            'summary', 'professional summary', 'profile', 'professional profile', 
            'career objective', 'objective', 'about me', 'career summary',
            'executive summary', 'personal statement'
        ]
        
        # If we have pre-detected summary sections, use them
        if summary_sections:
            summary_text = []
            for section in summary_sections:
                summary_text.append(section['text'])
            return " ".join(summary_text).strip()
        
        # Otherwise find summary manually
        lines = self.text.split('\n')
        summary_section = False
        summary_text = []
        
        # First look for summary section indicators
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Check if this line is a summary section header
            if any(header in line_lower for header in summary_headers):
                summary_section = True
                continue
                
            # If we're in the summary section
            if summary_section:
                # Check if we've reached another section
                if (not line or line_lower.endswith(':') or 
                    any(header in line_lower for header in 
                        ['experience', 'education', 'skills', 'work', 'employment'])):
                    if line_lower not in ['', 'summary:']:  # Don't end on empty line or summary header
                        summary_section = False
                    continue
                    
                # Add non-empty lines to our summary text
                if line:
                    summary_text.append(line)
        
        # If no summary section was found but we're parsing a resume, 
        # check the first few paragraphs for profile-like content
        if not summary_text and len(lines) > 5:
            # Look at first 10 non-empty lines, ignoring likely headers
            candidate_lines = [line for line in lines[:20] if line.strip() 
                             and not line.lower().endswith(':')
                             and not any(header in line.lower() for header in summary_headers)]
            
            # Identify the first paragraph-like text (multiple sentences)
            paragraph_start = None
            for i, line in enumerate(candidate_lines):
                # Skip lines that look like contact info
                if '@' in line or any(char.isdigit() for char in line) or len(line) < 15:
                    continue
                    
                # Skip lines that are likely a name
                if i == 0 and line.isupper():
                    continue
                    
                # Found a potential paragraph start
                paragraph_start = i
                break
                
            # If we found a paragraph start, collect the paragraph
            if paragraph_start is not None:
                current_paragraph = [candidate_lines[paragraph_start]]
                # Add subsequent lines that appear to be part of the same paragraph
                for i in range(paragraph_start + 1, min(paragraph_start + 5, len(candidate_lines))):
                    # Stop if we hit a line that looks like a section header
                    if candidate_lines[i].strip().endswith(':') or len(candidate_lines[i]) < 15:
                        break
                    current_paragraph.append(candidate_lines[i])
                    
                # If paragraph has multiple lines or contains multiple sentences, it's likely a summary
                paragraph_text = ' '.join(current_paragraph)
                if len(current_paragraph) > 1 or paragraph_text.count('.') > 1:
                    summary_text = current_paragraph
        
        # Join and clean the summary text
        if summary_text:
            summary = ' '.join(summary_text)
            # Remove any bullet points or other markers
            summary = re.sub(r'^[•\-*>→]+\s*', '', summary)
            return summary.strip()
            
        return ""
    
    def _detect_section_boundaries(self):
        """Detects all section boundaries in the resume for more accurate extraction"""
        # Create a comprehensive dictionary of possible section headers
        section_headers = {
            'contact': ['contact', 'contact information', 'personal information', 'personal details'],
            'summary': ['summary', 'professional summary', 'profile', 'professional profile', 'career objective', 'objective', 'about me'],
            'skills': ['skills', 'technical skills', 'core competencies', 'technologies', 'technical expertise',
                     'proficiencies', 'areas of expertise', 'skill set', 'professional skills', 'competencies',
                     'key skills', 'qualifications', 'technical proficiencies', 'expertise', 'strengths'],
            'experience': ['experience', 'work experience', 'professional experience', 'employment',
                        'employment history', 'work history', 'career', 'professional background',
                        'professional history', 'work background', 'professional profile', 'career history'],
            'education': ['education', 'academic background', 'qualification', 'academic history',
                       'academic qualification', 'educational qualification', 'educational background'],
            'projects': ['projects', 'personal projects', 'academic projects', 'key projects', 
                      'professional projects', 'side projects', 'portfolio projects',
                      'research projects', 'development projects', 'project experience'],
            'certifications': ['certifications', 'certificates', 'credentials', 'licenses', 'professional certifications'],
            'awards': ['awards', 'achievements', 'honors', 'accomplishments', 'recognition', 'scholarships'],
            'publications': ['publications', 'papers', 'research', 'articles', 'journals'],
            'languages': ['languages', 'language proficiency', 'spoken languages'],
            'interests': ['interests', 'hobbies', 'activities', 'personal interests', 'extracurricular'],
            'references': ['references', 'referees', 'professional references']
        }
        
        # Dictionary to store detected sections with their start and end line indices
        detected_sections = {}
        
        lines = self.text.split('\n')
        current_section = None
        section_start_idx = None
        
        # First pass: identify section headers
        for i, line in enumerate(lines):
            line = line.strip()
            line_lower = line.lower()
            
            # Skip empty lines for section detection
            if not line:
                continue
                
            # Check if this line is a section header
            found_section = None
            for section, headers in section_headers.items():
                # Look for exact matches or headers ending with colon
                if any(header == line_lower or line_lower == header + ':' for header in headers):
                    found_section = section
                    break
                    
                # Look for header contained in the line (with some context awareness)
                if any(header in line_lower for header in headers):
                    # Make sure it's likely a header (short line, possibly ends with colon)
                    if len(line) < 40 or line_lower.endswith(':'):
                        found_section = section
                        break
            
            # If we found a new section header
            if found_section:
                # If we were already in a section, mark its end
                if current_section and section_start_idx is not None:
                    if current_section not in detected_sections:
                        detected_sections[current_section] = []
                    
                    detected_sections[current_section].append({
                        'start': section_start_idx,
                        'end': i - 1  # End at previous line
                    })
                
                # Start tracking the new section
                current_section = found_section
                section_start_idx = i + 1  # Content starts after header
        
        # Add the last section if we were tracking one
        if current_section and section_start_idx is not None:
            if current_section not in detected_sections:
                detected_sections[current_section] = []
                
            detected_sections[current_section].append({
                'start': section_start_idx,
                'end': len(lines) - 1
            })
        
        # Second pass: Handle nested sections and overlaps
        # Sort all boundaries by start position
        all_boundaries = []
        for section, boundaries in detected_sections.items():
            for boundary in boundaries:
                all_boundaries.append({
                    'section': section,
                    'start': boundary['start'],
                    'end': boundary['end']
                })
        
        # Sort by start position
        all_boundaries.sort(key=lambda x: x['start'])
        
        # Adjust boundaries to avoid overlaps
        for i in range(1, len(all_boundaries)):
            prev = all_boundaries[i-1]
            curr = all_boundaries[i]
            
            # If current starts before previous ends, adjust previous end
            if curr['start'] <= prev['end']:
                prev['end'] = curr['start'] - 1
        
        # Rebuild the detected sections with adjusted boundaries
        cleaned_sections = {}
        for boundary in all_boundaries:
            section = boundary['section']
            if section not in cleaned_sections:
                cleaned_sections[section] = []
                
            # Only add if boundary is valid
            if boundary['start'] <= boundary['end']:
                cleaned_sections[section].append({
                    'start': boundary['start'],
                    'end': boundary['end'],
                    'text': '\n'.join(lines[boundary['start']:boundary['end']+1])
                })
        
        return cleaned_sections