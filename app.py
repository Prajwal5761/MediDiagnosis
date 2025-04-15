# app.py - Main Flask application
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import os


# Define dataset path
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'disease_symptom_dataset.csv')

# Import the medical diagnosis system
from diagnosis_system import MedicalDiagnosisSystem

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_diagnosis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Medical Diagnosis System
# diagnosis_system = MedicalDiagnosisSystem()

# Initialize Medical Diagnosis System with dataset path
diagnosis_system = MedicalDiagnosisSystem(dataset_path)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    diagnoses = db.relationship('DiagnosisRecord', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DiagnosisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)  # Store as JSON string
    diagnosis_results = db.Column(db.Text, nullable=False)  # Store as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create database if it doesn't exist
with app.app_context():
    db.create_all()


@app.template_filter('from_json')
def from_json(value):
    return json.loads(value)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']

        # Check if username or email already exists
        user_exists = User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first()
        if user_exists:
            flash('Username or email already exists')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(username=username, email=email, name=name, age=age, gender=gender)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Login successful')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    diagnoses = DiagnosisRecord.query.filter_by(user_id=user.id).order_by(DiagnosisRecord.created_at.desc()).all()
    
    return render_template('dashboard.html', user=user, diagnoses=diagnoses)

# @app.route('/diagnosis', methods=['GET', 'POST'])
# def diagnosis():
#     if 'user_id' not in session:
#         flash('Please log in to access this feature')
#         return redirect(url_for('login'))
    
#     if request.method == 'POST':
#         description = request.form['description']
        
#         # Get diagnosis
#         diagnosis_result = diagnosis_system.get_diagnosis(description)
        
#         # Save to database
#         new_diagnosis = DiagnosisRecord(
#             user_id=session['user_id'],
#             description=description,
#             symptoms=json.dumps(diagnosis_result['extracted_symptoms']),
#             diagnosis_results=json.dumps(diagnosis_result['possible_conditions'])
#         )
        
#         db.session.add(new_diagnosis)
#         db.session.commit()
        
#         # Get user information for the patient section
#         user = db.session.get(User, session['user_id'])
#         patient = {
#             'name': user.name if hasattr(user, 'name') else "Patient",
#             # Add any other patient properties needed by the template
#         }
        
#         return render_template('diagnosis_result.html', 
#                               diagnosis=diagnosis_result, 
#                               description=description, 
#                               diagnosis_id=new_diagnosis.id,
#                               patient=patient)  # Add the patient variable
    
#     return render_template('diagnosis_form.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if 'user_id' not in session:
        flash('Please log in to access this feature')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        description = request.form['description']
        
        # Get diagnosis
        diagnosis_result = diagnosis_system.get_diagnosis(description)
        
        # Extract the symptoms and conditions to pass directly to template
        extracted_symptoms = diagnosis_result.get('extracted_symptoms', {})
        symptoms = list(extracted_symptoms.keys())  # Just the symptom names
        
        possible_conditions = diagnosis_result.get('possible_conditions', [])
        conditions = []
        
        # Format conditions for the template
        for condition in possible_conditions:
            # Convert probability string like "75.5%" to float 75.5
            prob_str = condition.get('probability', '0%')
            probability = float(prob_str.rstrip('%'))
            
            conditions.append({
                'name': condition.get('disease', 'Unknown'),
                'probability': probability,
                'description': 'Common symptoms include: ' + ', '.join(
                    list(diagnosis_system.disease_models.get(condition.get('disease', ''), {}).keys())[:3]
                )
            })
        
        # Save to database
        symptoms_json = json.dumps(extracted_symptoms)
        conditions_json = json.dumps(possible_conditions)
        
        new_diagnosis = DiagnosisRecord(
            user_id=session['user_id'],
            description=description,
            symptoms=symptoms_json,
            diagnosis_results=conditions_json
        )
        
        db.session.add(new_diagnosis)
        db.session.commit()
        
        # Get user information
        user = User.query.get(session['user_id'])
        patient = {
            'name': user.name if hasattr(user, 'name') else "Patient",
            'age': user.age if hasattr(user, 'age') else "",
            'gender': user.gender if hasattr(user, 'gender') else ""
        }
        
        # Format date
        current_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        return render_template('diagnosis_result.html', 
                              symptoms=symptoms,
                              conditions=conditions,
                              description=description,
                              diagnosis_id=new_diagnosis.id,
                              diagnosis={'date': current_date, 'id': new_diagnosis.id},
                              patient=patient)
    
    return render_template('diagnosis_form.html')


@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to access your history')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    diagnoses = DiagnosisRecord.query.filter_by(user_id=user.id).order_by(DiagnosisRecord.created_at.desc()).all()
    
    return render_template('history.html', diagnoses=diagnoses)

# @app.route('/diagnosis/<int:diagnosis_id>')
# def view_diagnosis(diagnosis_id):
#     if 'user_id' not in session:
#         flash('Please log in to access this feature')
#         return redirect(url_for('login'))
    
#     diagnosis_record = DiagnosisRecord.query.get_or_404(diagnosis_id)
    
#     # Verify that the diagnosis belongs to the current user
#     if diagnosis_record.user_id != session['user_id']:
#         flash('Unauthorized access')
#         return redirect(url_for('dashboard'))
    
#     diagnosis_result = {
#         'extracted_symptoms': json.loads(diagnosis_record.symptoms),
#         'possible_conditions': json.loads(diagnosis_record.diagnosis_results),
#         'disclaimer': "IMPORTANT: This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation."
#     }
    
#     # Get user information for the patient section
#     user = db.session.get(User, session['user_id'])
#     patient = {
#         'name': user.name if hasattr(user, 'name') else "Patient",
#         # Add any other patient properties needed by the template
#     }
    
#     return render_template('diagnosis_result.html', 
#                           diagnosis=diagnosis_result, 
#                           description=diagnosis_record.description,
#                           diagnosis_id=diagnosis_id,
#                           date=diagnosis_record.created_at,
#                           patient=patient)  # Add the patient variable

@app.route('/diagnosis/<int:diagnosis_id>')
def view_diagnosis(diagnosis_id):
    if 'user_id' not in session:
        flash('Please log in to access this feature')
        return redirect(url_for('login'))
    
    diagnosis_record = DiagnosisRecord.query.get_or_404(diagnosis_id)
    
    # Verify that the diagnosis belongs to the current user
    if diagnosis_record.user_id != session['user_id']:
        flash('Unauthorized access')
        return redirect(url_for('dashboard'))
    
    try:
        extracted_symptoms = json.loads(diagnosis_record.symptoms)
        symptoms = list(extracted_symptoms.keys())  # Just the symptom names
    except:
        symptoms = []
        
    try:
        possible_conditions = json.loads(diagnosis_record.diagnosis_results)
        conditions = []
        
        # Format conditions for the template
        for condition in possible_conditions:
            # Convert probability string like "75.5%" to float 75.5
            prob_str = condition.get('probability', '0%')
            probability = float(prob_str.rstrip('%'))
            
            conditions.append({
                'name': condition.get('disease', 'Unknown'),
                'probability': probability,
                'description': 'Common symptoms include: ' + ', '.join(
                    list(diagnosis_system.disease_models.get(condition.get('disease', ''), {}).keys())[:3]
                )
            })
    except:
        conditions = []
    
    # Get user information for the patient section
    user = User.query.get(session['user_id'])
    patient = {
        'name': user.name if hasattr(user, 'name') else "Patient",
        'age': user.age if hasattr(user, 'age') else "",
        'gender': user.gender if hasattr(user, 'gender') else ""
    }
    
    # Format date
    date_str = diagnosis_record.created_at.strftime('%Y-%m-%d')
    
    return render_template('diagnosis_result.html', 
                          symptoms=symptoms,
                          conditions=conditions,
                          description=diagnosis_record.description,
                          diagnosis_id=diagnosis_id,
                          diagnosis={'date': date_str, 'id': diagnosis_id},
                          patient=patient)

if __name__ == '__main__':
    app.run(debug=True)