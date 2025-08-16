#!/usr/bin/env python3
"""
Fast Accurate Trainer - Prioritizes abbreviations with fast performance
"""
import mysql.connector
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'resume_analyzer'
}

BASE_DIR = Path("ml_specialization_module")
VECTOR_FILE = BASE_DIR / "spec_vectorizer.pkl"
MODEL_FILE = BASE_DIR / "spec_nn_model.pkl"
LOOKUP_FILE = BASE_DIR / "specializations_lookup.csv"

# Priority abbreviations with exact matches
PRIORITY_ABBREVS = {
    'cs': ['computer science'],
    'extc': ['electronics telecommunication', 'electronics communication'],
    'ece': ['electronics communication'],
    'it': ['information technology'],
    'mech': ['mechanical'],
    'civil': ['civil'],
    'eee': ['electrical electronics'],
    'chem': ['chemical'],
    'bio': ['biotechnology'],
    'aero': ['aerospace'],
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_data_from_db():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ups_courses_specialization", conn)
    conn.close()
    return df

def generate_smart_variants(text):
    """Generate smart variants with priority for abbreviations"""
    variants = [text]
    text_clean = text.lower().replace(' ', '')
    
    # Add without spaces
    variants.append(text_clean)
    
    # Add key words (>2 chars)
    words = text.split()
    for word in words:
        if len(word) > 2:
            variants.append(word.lower())
    
    # Add initials
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    # Add priority abbreviations
    text_lower = text.lower()
    for abbr, expansions in PRIORITY_ABBREVS.items():
        for expansion in expansions:
            if all(word in text_lower for word in expansion.split()):
                variants.append(abbr)
                break
    
    # Add only essential prefixes
    for length in [3, 4]:
        if len(text_clean) >= length:
            variants.append(text_clean[:length])
    
    return list(set(variants))

def train_fast_accurate_model():
    print("=== Fast Accurate Training ===")
    
    source_df = load_data_from_db()
    print(f"Loaded {len(source_df)} specializations")
    
    expanded_data = []
    
    for _, row in source_df.iterrows():
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        variants = generate_smart_variants(spec)
        
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    
    print(f"Generated {len(specializations)} training entries")
    
    # Fast vectorizer settings
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(2, 3),
        max_features=4000,
        lowercase=True
    )
    X = vectorizer.fit_transform(specializations)
    
    
    # Fast neighbor search
    nn_model = NearestNeighbors(
        n_neighbors=min(30, len(specializations)), 
        metric="cosine"
    )
    nn_model.fit(X)

    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn_model, f)
    
    expanded_df.to_csv(LOOKUP_FILE, index=False)
    
    print("SUCCESS: Fast accurate training complete!")
    print(f"Entries: {len(specializations)}")

if __name__ == "__main__":
    train_fast_accurate_model()