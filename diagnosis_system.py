# diagnosis_system.py - Medical diagnosis system module
import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os
from dataset_parser import MedicalDatasetParser

# Download necessary NLTK resources first thing at import time
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class MedicalDiagnosisSystem:
    def __init__(self, dataset_path='data/disease_symptom_dataset.csv'):
        # Initialize lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load disease models from dataset
        self.load_disease_models(dataset_path)
        
        # Define symptom dictionary - maps various expressions to standardized symptoms
        self.symptom_dict = self.build_symptom_dictionary()
        
        # Intensity modifiers with multipliers
        self.intensity_modifiers = {
            'high_intensity': 1.5,
            'medium_intensity': 1.0,
            'low_intensity': 0.5
        }
    
    def load_disease_models(self, dataset_path):
        """Load disease models from the dataset"""
        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset file not found at {dataset_path}")
            # Fallback to empty model
            self.disease_models = {}
            return
            
        # Parse the dataset
        parser = MedicalDatasetParser(dataset_path)
        self.disease_models = parser.get_disease_models()
        
        # If parsing failed, initialize with empty dict
        if not self.disease_models:
            print("Warning: Failed to load disease models from dataset")
            self.disease_models = {}
    
    def build_symptom_dictionary(self):
        """Build a comprehensive symptom dictionary based on disease models"""
        symptom_dict = {
            # Fever related
            'fever': 'fever', 'hot': 'fever', 'temperature': 'fever', 'burning up': 'fever',
            'high temperature': 'fever', 'feverish': 'fever',
            
            # Pain related
            'headache': 'headache', 'head hurts': 'headache', 'migraine': 'headache',
            'head pain': 'headache', 'pounding head': 'headache',
            'sore throat': 'sore_throat', 'throat pain': 'sore_throat', 
            'painful throat': 'sore_throat', 'scratchy throat': 'sore_throat',
            'chest pain': 'chest_pain', 'pain in chest': 'chest_pain', 
            'chest discomfort': 'chest_pain', 'painful chest': 'chest_pain',
            'stomach pain': 'abdominal_pain', 'tummy ache': 'abdominal_pain', 
            'abdominal pain': 'abdominal_pain', 'belly pain': 'abdominal_pain',
            'joint pain': 'joint_pain', 'muscle pain': 'muscle_pain',
            
            # Respiratory related
            'cough': 'cough', 'coughing': 'cough', 'hacking': 'cough',
            'runny nose': 'runny_nose', 'stuffy nose': 'congestion', 
            'congestion': 'congestion', 'congested': 'congestion',
            'sneezing': 'sneezing', 'sneeze': 'sneezing',
            'shortness of breath': 'shortness_of_breath', 'trouble breathing': 'shortness_of_breath',
            'can\'t breathe': 'shortness_of_breath', 'difficulty breathing': 'shortness_of_breath',
            'wheezing': 'wheezing',
            
            # Digestive related
            'nausea': 'nausea', 'nauseated': 'nausea', 'feel sick': 'nausea',
            'vomiting': 'vomiting', 'throwing up': 'vomiting', 'vomit': 'vomiting',
            'diarrhea': 'diarrhea', 'loose stool': 'diarrhea',
            'constipation': 'constipation', 'constipated': 'constipation',
            'indigestion': 'indigestion', 'heartburn': 'indigestion',
            'appetite': 'appetite_changes', 'not hungry': 'appetite_changes', 
            'hungry': 'appetite_changes', 'eating less': 'appetite_changes',
            
            # General symptoms
            'fatigue': 'fatigue', 'tired': 'fatigue', 'exhausted': 'fatigue', 
            'no energy': 'fatigue', 'weakness': 'fatigue',
            'dizzy': 'dizziness', 'dizziness': 'dizziness', 'lightheaded': 'dizziness',
            'faint': 'fainting', 'passed out': 'fainting', 'fainting': 'fainting',
            'sweating': 'sweating', 'sweat': 'sweating',
            'chills': 'chills', 'shivering': 'chills',
            'weight loss': 'weight_loss', 'losing weight': 'weight_loss',
            'weight gain': 'weight_gain', 'gaining weight': 'weight_gain',
            'rash': 'rash', 'skin rash': 'rash', 'hives': 'rash',
            'swelling': 'swelling', 'swollen': 'swelling',
            
            # Duration and intensity markers
            'days': 'duration', 'weeks': 'duration', 'months': 'duration',
            'severe': 'high_intensity', 'intense': 'high_intensity', 'mild': 'low_intensity',
            'slight': 'low_intensity', 'moderate': 'medium_intensity'
        }
        
        # Add all standardized symptoms from the disease models
        for disease, symptoms in self.disease_models.items():
            for symptom in symptoms:
                # Add the symptom as its own key if not already present
                if symptom not in symptom_dict.values():
                    symptom_dict[symptom.lower().replace("_", " ")] = symptom
        
        return symptom_dict
    
    # The rest of your class remains the same...
    def preprocess_text(self, text):
        """Preprocess the text by removing punctuation, lowercasing, and tokenizing"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = re.sub(f'[{string.punctuation}]', ' ', text)
            
            # Use a simpler tokenization approach that doesn't rely on punkt_tab
            tokens = text.split()
            
            # Remove stopwords and lemmatize
            processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            return processed_tokens
        except Exception as e:
            print(f"Preprocessing error: {e}")
            # Fallback tokenization
            simple_tokens = text.lower().split()
            return [token.strip(string.punctuation) for token in simple_tokens if token.strip(string.punctuation) and token not in self.stop_words]
    
    def extract_symptoms(self, text):
        """Extract symptoms from preprocessed text"""
        try:
            tokens = self.preprocess_text(text)
            
            # Look for symptoms in the text
            extracted_symptoms = {}
            intensity_modifier = 1.0  # Default intensity
            
            for i, token in enumerate(tokens):
                if token in self.symptom_dict:
                    standardized_symptom = self.symptom_dict[token]
                    
                    # Check for intensity modifiers near the symptom
                    context_range = 3  # Look 3 words before and after
                    start_idx = max(0, i - context_range)
                    end_idx = min(len(tokens), i + context_range + 1)
                    
                    for j in range(start_idx, end_idx):
                        if j < len(tokens) and tokens[j] in self.symptom_dict:
                            nearby_term = self.symptom_dict[tokens[j]]
                            if nearby_term in self.intensity_modifiers:
                                intensity_modifier = self.intensity_modifiers[nearby_term]
                    
                    # Handle special case for duration and intensity
                    if standardized_symptom in ['duration', 'high_intensity', 'medium_intensity', 'low_intensity']:
                        continue
                    
                    # Add or update symptom confidence
                    if standardized_symptom in extracted_symptoms:
                        # Increase confidence if mentioned multiple times
                        extracted_symptoms[standardized_symptom] = min(1.0, extracted_symptoms[standardized_symptom] + 0.2)
                    else:
                        extracted_symptoms[standardized_symptom] = 0.7 * intensity_modifier  # Base confidence
            
            return extracted_symptoms
        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            return {}
    
    def calculate_disease_scores(self, symptoms):
        """Calculate disease scores using fuzzy logic"""
        if not symptoms:
            return {"No significant symptoms detected": 0}
            
        disease_scores = {}
        
        for disease, disease_model in self.disease_models.items():
            # Initialize variables for fuzzy calculation
            total_weight = 0
            weighted_match = 0
            
            # Calculate number of key symptoms present
            key_symptoms_present = 0
            key_symptoms_total = 0
            
            for symptom, weight in disease_model.items():
                total_weight += weight
                key_symptoms_total += 1 if weight >= 0.7 else 0
                
                if symptom in symptoms:
                    # Weighted match based on symptom importance and confidence
                    weighted_match += weight * symptoms[symptom]
                    if weight >= 0.7:  # Consider it a key symptom if weight is 0.7+
                        key_symptoms_present += 1
            
            if total_weight > 0:
                # Base score is weighted match divided by total possible
                base_score = weighted_match / total_weight
                
                # Factor in key symptoms coverage
                key_symptom_ratio = key_symptoms_present / key_symptoms_total if key_symptoms_total > 0 else 0
                
                # Final score with emphasis on key symptoms
                disease_scores[disease] = (base_score * 0.6) + (key_symptom_ratio * 0.4)
        
        return disease_scores
    
    def get_diagnosis(self, text, threshold=0.3):
        """Main method to get diagnosis from user input"""
        try:
            # Extract symptoms
            symptoms = self.extract_symptoms(text)
            
            # Calculate disease scores
            disease_scores = self.calculate_disease_scores(symptoms)
            
            # Sort diseases by score
            sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Filter results above threshold
            results = [{"disease": disease, "score": score, "probability": f"{score*100:.1f}%"} 
                      for disease, score in sorted_diseases if score >= threshold]
            
            # Add a disclaimer
            disclaimer = "IMPORTANT: This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation."
            
            return {
                "extracted_symptoms": symptoms,
                "possible_conditions": results,
                "disclaimer": disclaimer
            }
        except Exception as e:
            print(f"Error in diagnosis process: {e}")
            return {
                "error": f"An error occurred during diagnosis: {str(e)}",
                "disclaimer": "IMPORTANT: This is not a medical diagnosis. Please consult a healthcare professional for proper evaluation."
            }