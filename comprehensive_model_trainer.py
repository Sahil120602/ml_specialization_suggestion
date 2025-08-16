#!/usr/bin/env python3
"""
Comprehensive Model Trainer - Handles ALL possible abbreviations and short forms
Keeps original logic but adds extensive abbreviation support for every specialization
"""
import mysql.connector
import pandas as pd
import pickle
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Database configuration
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
LAST_UPDATE_FILE = BASE_DIR / "last_update.txt"

def get_db_connection():
    """Get database connection with error handling"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

def get_last_db_update():
    """Get the latest update timestamp from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(updated_on) FROM ups_courses_specialization")
    result = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return result

def load_data_from_db():
    """Load specialization data from MySQL"""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ups_courses_specialization", conn)
    conn.close()
    return df

# Removed massive word abbreviation function - using pure ML learning instead

def generate_character_learning_variants(text):
    """Generate character-level variants for pure ML learning"""
    variants = [text]
    text_lower = text.lower().replace(' ', '')
    
    # Add the text without spaces for better character matching
    variants.append(text_lower)
    
    # Add individual meaningful words
    words = text.split()
    for word in words:
        if len(word) > 2:  # Add all meaningful words
            variants.append(word.lower())
    
    # Add character prefixes of different lengths for fuzzy matching
    for length in [2, 3, 4, 5, 6]:
        if len(text_lower) >= length:
            variants.append(text_lower[:length])
    
    # Add initials for common abbreviation patterns
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    # Add character suffixes for better matching
    for length in [3, 4, 5]:
        if len(text_lower) >= length:
            variants.append(text_lower[-length:])
    
    return list(set(variants))

def train_pure_ml_model():
    """Train pure ML model with character-level learning"""
    print("=== Pure ML Character-Level Training ===")
    
    # Load data
    print("Loading data from database...")
    source_df = load_data_from_db()
    print(f"Loaded {len(source_df)} specializations")
    
    # Generate character-level training data
    print("Generating character-level variants...")
    expanded_data = []
    
    for _, row in source_df.iterrows():
        # Normalize specialization name
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        
        # Generate character-level variants
        variants = generate_character_learning_variants(spec)
        
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    
    print(f"Expanded from {len(source_df)} to {len(specializations)} character-level entries")
    
    # Train model with character n-grams for maximum fuzzy matching
    print("Training pure ML model...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(1, 4),  # Single chars to 4-grams for maximum coverage
        max_features=15000,  # More features for better pattern recognition
        lowercase=True
    )
    X = vectorizer.fit_transform(specializations)
    nn_model = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
    nn_model.fit(X)
    
    # Save everything
    print("Saving model and data...")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn_model, f)
    
    # Save lookup data for consistent indexing
    expanded_df.to_csv(LOOKUP_FILE, index=False)
    LAST_UPDATE_FILE.write_text(str(get_last_db_update()))
    
    print("SUCCESS: Pure ML model training complete!")
    print(f"SUCCESS: Original entries: {len(source_df)}")
    print(f"SUCCESS: Character-level entries: {len(specializations)}")
    print("SUCCESS: Model learns patterns naturally from data!")

def verify_database():
    """Verify the database and table structure before training"""
    try:
        print("=== Database Verification ===")
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'ups_courses_specialization'
        """, (DB_CONFIG['database'],))
        
        if cursor.fetchone()[0] == 0:
            print("❌ Table 'ups_courses_specialization' not found!")
            return False
        
        print("✅ Table 'ups_courses_specialization' found")
        
        # Get record count
        cursor.execute("SELECT COUNT(*) FROM ups_courses_specialization")
        total_count = cursor.fetchone()[0]
        print(f"✅ Total records: {total_count}")
        
        # Course distribution
        cursor.execute("""
            SELECT course_id, COUNT(*) as count 
            FROM ups_courses_specialization 
            GROUP BY course_id 
            ORDER BY count DESC 
            LIMIT 10
        """)
        course_dist = cursor.fetchall()
        print(f"📈 Top Course Distribution:")
        for course_id, count in course_dist:
            print(f"   Course {course_id}: {count} specializations")
        
        cursor.close()
        conn.close()
        
        print("✅ Database verification successful!\n")
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # Just verify database
        verify_database()
    else:
        # Verify database first, then train
        if verify_database():
            try:
                train_pure_ml_model()
                print("\nReady to test with: python specialization_suggestion_realtime.py")
            except Exception as e:
                print(f"Training error: {e}")
        else:
            print("Database verification failed. Please check your database setup.")