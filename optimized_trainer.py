#!/usr/bin/env python3
"""
Optimized ML Trainer - Fast with abbreviations and comprehensive features
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
LAST_UPDATE_FILE = BASE_DIR / "last_update.txt"

# Comprehensive abbreviations for ALL domains
KEY_ABBREVS = {
    # Engineering
    'extc': 'electronics telecommunication communication',
    'ece': 'electronics communication',
    'cs': 'computer science',
    'csaiml': 'computer science artificial intelligence machine learning',
    'aiml': 'artificial intelligence machine learning',
    'ai': 'artificial intelligence',
    'ml': 'machine learning',
    'ds': 'data science',
    'aids': 'artificial intelligence data science',
    'it': 'information technology',
    'mech': 'mechanical',
    'civil': 'civil',
    'eee': 'electrical electronics',
    'chem': 'chemical',
    'bio': 'biotechnology',
    'aero': 'aerospace',
    
    # Medical
    'mbbs': 'bachelor medicine surgery',
    'md': 'doctor medicine',
    'ms': 'master surgery',
    'bds': 'bachelor dental surgery',
    'mds': 'master dental surgery',
    'cardio': 'cardiology cardiac',
    'neuro': 'neurology neurological',
    'ortho': 'orthopedic orthopedics',
    'gynec': 'gynecology obstetrics',
    'pedia': 'pediatrics',
    'radio': 'radiology',
    'patho': 'pathology',
    'pharma': 'pharmacy pharmaceutical',
    'physio': 'physiotherapy',
    'nursing': 'nursing',
    'anesthesia': 'anesthesiology',
    
    # Management/Business
    'mba': 'master business administration',
    'bba': 'bachelor business administration',
    'pgdm': 'diploma management',
    'hr': 'human resources',
    'finance': 'finance financial',
    'marketing': 'marketing',
    'ops': 'operations',
    'scm': 'supply chain management',
    'pm': 'project management',
    'retail': 'retail management',
    'hospitality': 'hospitality management',
    'mgmt': 'management',

    # Commerce
    'bcom': 'bachelor commerce',
    'mcom': 'master commerce',
    'ca': 'chartered accountant',
    'cs': 'company secretary',
    'cma': 'cost management accountant',
    'accounts': 'accounting',
    'taxation': 'taxation',
    'banking': 'banking',
    'insurance': 'insurance',
    
    # Arts & Humanities
    'ba': 'bachelor arts',
    'ma': 'master arts',
    'english': 'english literature',
    'hindi': 'hindi literature',
    'history': 'history',
    'geography': 'geography',
    'psychology': 'psychology',
    'sociology': 'sociology',
    'philosophy': 'philosophy',
    'journalism': 'journalism mass communication',
    'media': 'media communication',
    
    # Sciences
    'bsc': 'bachelor science',
    'msc': 'master science',
    'physics': 'physics',
    'chemistry': 'chemistry',
    'maths': 'mathematics',
    'stats': 'statistics',
    'micro': 'microbiology',
    'biochem': 'biochemistry',
    'biotech': 'biotechnology',
    'environ': 'environmental science',
    'geology': 'geology',
    'botany': 'botany',
    'zoology': 'zoology',
    
    # Law
    'llb': 'bachelor law',
    'llm': 'master law',
    'law': 'law legal',
    'criminal': 'criminal law',
    'corporate': 'corporate law',
    
    # Education
    'bed': 'bachelor education',
    'med': 'master education',
    'teaching': 'teaching education',
    'montessori': 'montessori education',
    
    # Design/Fashion
    'fashion': 'fashion design',
    'interior': 'interior design',
    'graphic': 'graphic design',
    'textile': 'textile design',
    'architecture': 'architecture',

}

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
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ups_courses_specialization", conn)
    conn.close()
    return df

def generate_comprehensive_short_forms(text):
    """Generate ALL possible short forms for every specialization"""
    variants = [text]
    text_clean = text.lower().replace(' ', '').replace('&', 'and')
    text_lower = text.lower()
    words = text.split()
    
    # Add without spaces
    variants.append(text_clean)
    
    # Add all meaningful words
    for word in words:
        if len(word) > 2:
            variants.append(word.lower())
    
    # Add initials (most important for CSE, ECE, etc.)
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w and w[0].isalpha()])
        variants.append(initials)
    
    # Add character prefixes for fuzzy matching (including single chars)
    for length in [1, 2, 3, 4, 5]:
        if len(text_clean) >= length:
            variants.append(text_clean[:length])
    
    # Add single character variants for medical specializations
    if any(med_word in text_lower for med_word in ['cardiology', 'cardiac', 'cancer', 'clinical', 'critical', 'chest']):
        variants.append('c')
    if any(med_word in text_lower for med_word in ['dermatology', 'dental', 'diabetes']):
        variants.append('d')
    if any(med_word in text_lower for med_word in ['emergency', 'endocrinology', 'ent']):
        variants.append('e')
    if any(med_word in text_lower for med_word in ['gastroenterology', 'general', 'gynecology']):
        variants.append('g')
    if any(med_word in text_lower for med_word in ['hematology', 'hepatology', 'heart']):
        variants.append('h')
    if any(med_word in text_lower for med_word in ['internal', 'intensive', 'immunology']):
        variants.append('i')
    if any(med_word in text_lower for med_word in ['nephrology', 'neurology', 'nuclear']):
        variants.append('n')
    if any(med_word in text_lower for med_word in ['orthopedic', 'oncology', 'ophthalmology']):
        variants.append('o')
    if any(med_word in text_lower for med_word in ['pediatric', 'pathology', 'psychiatry', 'pulmonary']):
        variants.append('p')
    if any(med_word in text_lower for med_word in ['radiology', 'rheumatology', 'respiratory']):
        variants.append('r')
    if any(med_word in text_lower for med_word in ['surgery', 'surgical', 'spine']):
        variants.append('s')
    
    # Generate short forms for EVERY specialization type
    
    # Computer Engineering patterns - separate from CSE
    if 'computer engineering' in text_lower and 'science' not in text_lower:
        variants.extend(['ce', 'comp eng', 'computer eng', 'comp engineering'])
    # CSE patterns - Computer Science Engineering  
    elif 'computer science' in text_lower and 'engineering' in text_lower:
        variants.extend(['cse', 'cs', 'comp sci eng'])
        # Add AI/ML specific patterns for CSE
        if 'artificial intelligence' in text_lower and 'machine learning' in text_lower:
            variants.extend(['cs(aiml)', 'cs aiml', 'csaiml', 'cse(aiml)', 'cse aiml'])
        elif 'artificial intelligence' in text_lower:
            variants.extend(['cs(ai)', 'cs ai', 'cse(ai)', 'cse ai'])
        elif 'machine learning' in text_lower:
            variants.extend(['cs(ml)', 'cs ml', 'cse(ml)', 'cse ml'])
    elif 'computer science' in text_lower:
        variants.extend(['cs', 'comp sci', 'computer'])
        # Add AI/ML specific patterns for CS
        if 'artificial intelligence' in text_lower and 'machine learning' in text_lower:
            variants.extend(['cs(aiml)', 'cs aiml', 'csaiml'])
        elif 'artificial intelligence' in text_lower:
            variants.extend(['cs(ai)', 'cs ai'])
        elif 'machine learning' in text_lower:
            variants.extend(['cs(ml)', 'cs ml'])
    # General computer patterns
    elif 'computer' in text_lower and 'engineering' in text_lower:
        variants.extend(['comp eng', 'computer eng'])
    
    # All Engineering short forms
    if 'engineering' in text_lower:
        # Remove 'engineering' and create short forms
        base_words = [w for w in words if w.lower() != 'engineering']
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.append(base_initials + 'e')  # Add 'E' for engineering
            variants.append(base_initials)  # Without 'E'
    
    # System/Systems patterns
    if 'system' in text_lower or 'systems' in text_lower:
        base_words = [w for w in words if w.lower() not in ['system', 'systems', 'engineering']]
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.extend([base_initials + 's', base_initials + 'se'])
    
    # Technology patterns
    if 'technology' in text_lower:
        base_words = [w for w in words if w.lower() not in ['technology', 'engineering']]
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.extend([base_initials + 't', base_initials + 'tech'])
    
    # Science patterns
    if 'science' in text_lower:
        base_words = [w for w in words if w.lower() not in ['science', 'engineering']]
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.extend([base_initials + 's', base_initials + 'sci'])
    
    # Management patterns
    if 'management' in text_lower:
        base_words = [w for w in words if w.lower() != 'management']
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.extend([base_initials + 'm', base_initials + 'mgmt'])
    
    # Design patterns
    if 'design' in text_lower:
        base_words = [w for w in words if w.lower() != 'design']
        if base_words:
            base_initials = ''.join([w[0] for w in base_words if w and w[0].isalpha()])
            variants.extend([base_initials + 'd', base_initials + 'design'])
    
    # Add common abbreviations for specific terms
    abbreviation_map = {
        'artificial intelligence': ['ai', 'artificial', 'intelligence'],
        'machine learning': ['ml', 'machine', 'learning'],
        'data science': ['ds', 'data', 'science'],
        'computer science': ['cs', 'comp', 'computer'],
        'electronics': ['ec', 'elec', 'electronics'],
        'communication': ['comm', 'communication'],
        'telecommunication': ['telecom', 'tele', 'telecommunication'],
        'mechanical': ['mech', 'mechanical'],
        'electrical': ['ee', 'elec', 'electrical'],
        'civil': ['civil', 'ce'],
        'chemical': ['chem', 'chemical'],
        'information technology': ['it', 'info tech'],
        'software': ['sw', 'soft', 'software'],
        'hardware': ['hw', 'hard', 'hardware'],
        'network': ['net', 'networking'],
        'security': ['sec', 'security'],
        'database': ['db', 'database'],
        'web': ['web', 'internet'],
        'mobile': ['mob', 'mobile'],
        'cloud': ['cloud', 'computing'],
        # Medical abbreviations
        'cardiology': ['c', 'cardio', 'cardiac'],
        'dermatology': ['d', 'derma', 'skin'],
        'emergency': ['e', 'emerg', 'emergency'],
        'gastroenterology': ['g', 'gastro', 'gi'],
        'hematology': ['h', 'hemato', 'blood'],
        'internal': ['i', 'internal', 'medicine'],
        'nephrology': ['n', 'nephro', 'kidney'],
        'orthopedic': ['o', 'ortho', 'bone'],
        'pediatric': ['p', 'pedia', 'child'],
        'radiology': ['r', 'radio', 'imaging'],
        'surgery': ['s', 'surgical', 'operation'],
    }
    
    # Apply abbreviation mapping
    for full_term, abbrevs in abbreviation_map.items():
        if full_term in text_lower:
            variants.extend(abbrevs)
    
    # Add predefined abbreviations
    for abbr, expansion in KEY_ABBREVS.items():
        if any(word in text_lower for word in expansion.split()):
            variants.append(abbr)
            variants.extend(expansion.split())
    
    return list(set(variants))

def verify_database():
    """Verify database before training"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ups_courses_specialization")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print(f"Database verified: {count} records")
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False

def train_optimized_model():
    print("=== Optimized Comprehensive Training ===")
    
    if not verify_database():
        return
    
    source_df = load_data_from_db()
    print(f"Loaded {len(source_df)} specializations")
    
    expanded_data = []
    
    for _, row in source_df.iterrows():
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        variants = generate_comprehensive_short_forms(spec)
        
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    
    print(f"Generated {len(specializations)} training entries")
    
    # Balanced vectorizer settings
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(1, 4),  # Full range for fuzzy matching
        max_features=6000,   # Balanced
        lowercase=True
    )
    X = vectorizer.fit_transform(specializations)
    
    # Comprehensive neighbor search
    nn_model = NearestNeighbors(
        n_neighbors=len(specializations), 
        metric="cosine"
    )
    nn_model.fit(X)
    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn_model, f)
    
    expanded_df.to_csv(LOOKUP_FILE, index=False)
    LAST_UPDATE_FILE.write_text(str(get_last_db_update()))
    
    print("SUCCESS: Fast optimized training complete!")
    print(f"Entries: {len(specializations)}")
    print("Model includes: smart abbreviations, essential features only")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_database()
    else:
        try:
            train_optimized_model()
        except Exception as e:
            print(f"Training error: {e}")