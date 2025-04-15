import pandas as pd
import json
import os

class MedicalDatasetParser:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def parse_csv_dataset(self):
        """Parse CSV format dataset with disease-symptom-weight relationships"""
        try:
            # Read the CSV file
            df = pd.read_csv(self.dataset_path)
            
            # Create a dictionary to store the disease models
            disease_models = {}
            
            # Group by disease and create a dictionary for each
            for disease, group in df.groupby('disease'):
                disease_models[disease] = {}
                
                # For each symptom in the disease group, add it with its weight
                for _, row in group.iterrows():
                    symptom = row['symptom']
                    weight = float(row['weight'])
                    disease_models[disease][symptom] = weight
            
            return disease_models
        except Exception as e:
            print(f"Error parsing dataset: {e}")
            return {}
    
    def parse_json_dataset(self):
        """Parse JSON format dataset with disease-symptom-weight relationships"""
        try:
            with open(self.dataset_path, 'r') as file:
                disease_models = json.load(file)
            return disease_models
        except Exception as e:
            print(f"Error parsing JSON dataset: {e}")
            return {}
            
    def get_disease_models(self):
        """Determine file type and parse accordingly"""
        file_ext = os.path.splitext(self.dataset_path)[1].lower()
        
        if file_ext == '.csv':
            return self.parse_csv_dataset()
        elif file_ext == '.json':
            return self.parse_json_dataset()
        else:
            print(f"Unsupported file format: {file_ext}")
            return {}