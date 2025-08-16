import pickle
import pandas as pd
import msvcrt
import time
import threading
import mysql.connector
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',  # Change this
    'database': 'resume_analyzer'
}

# Hardcoded course ID - Change this to filter by specific course
HARDCODED_COURSE_ID = None  # 12 = Engineering, 37 = Medical, 46 = Management, None = All courses

BASE = Path("ml_specialization_module")
VEC_FILE = BASE / "spec_vectorizer.pkl"
MODEL_FILE = BASE / "spec_nn_model.pkl"
LAST_UPDATE_FILE = BASE / "last_update.txt"

SPEC_COL = "specilization_name"

# Global variables for models
vectorizer = None
nn = None
df = None

def get_db_connection():
    """Get database connection with error handling"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

def load_data_from_db():
    """Load specialization data from MySQL"""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ups_courses_specialization", conn)
    conn.close()
    return df

def generate_character_variants(text):
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
    
    # Add character prefixes of different lengths
    for length in [2, 3, 4, 5]:
        if len(text_lower) >= length:
            variants.append(text_lower[:length])
    
    # Add initials
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    return list(set(variants))

def train_model():
    """Train pure ML model with character-level learning"""
    global vectorizer, nn, df
    
    print("Training pure ML model from MySQL database...")
    source_df = load_data_from_db()
    print(f"Loaded {len(source_df)} specializations from database")
    
    # Generate character-level variants for better fuzzy matching
    expanded_data = []
    for _, row in source_df.iterrows():
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        variants = generate_character_variants(spec)
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    print(f"Generated {len(specializations)} character-level training variants")
    
    # Train with character n-grams for fuzzy matching
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(1, 4),  # Include single characters to 4-grams
        max_features=10000,
        lowercase=True
    )
    X = vectorizer.fit_transform(specializations)
    nn = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
    nn.fit(X)
    
    # Save artifacts
    BASE.mkdir(parents=True, exist_ok=True)
    with open(VEC_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn, f)
    
    # Save the expanded dataframe for consistent indexing
    lookup_file = BASE / "specializations_lookup.csv"
    expanded_df.to_csv(lookup_file, index=False)
    
    df = expanded_df
    
    # Save last update timestamp
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(updated_on) FROM ups_courses_specialization")
        last_update = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        LAST_UPDATE_FILE.write_text(str(last_update))
    except:
        pass
    
    print(f"SUCCESS: Pure ML model trained! ({len(specializations)} entries)")

def load_models():
    """Load existing models or train new ones"""
    global vectorizer, nn, df
    
    try:
        # Check if model files exist and are recent
        if VEC_FILE.exists() and MODEL_FILE.exists():
            vectorizer = pickle.load(open(VEC_FILE, "rb"))
            nn = pickle.load(open(MODEL_FILE, "rb"))
            
            # Try to load the lookup data that was saved during training
            lookup_file = BASE / "specializations_lookup.csv"
            if lookup_file.exists():
                df = pd.read_csv(lookup_file)
                print("Loaded existing models and lookup data")
            else:
                # Fallback: regenerate data (but this might cause index mismatch)
                print("Lookup file missing, regenerating data...")
                train_model()
        else:
            raise FileNotFoundError("Model files not found")
            
    except Exception as e:
        print(f"Loading models failed: {e}")
        print("Training new models...")
        train_model()

def auto_retrain_daemon():
    """Background daemon to check for updates every 5 minutes"""
    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(updated_on) FROM ups_courses_specialization")
            db_update = str(cursor.fetchone()[0])
            cursor.close()
            conn.close()
            
            model_update = "1970-01-01 00:00:00"
            if LAST_UPDATE_FILE.exists():
                model_update = LAST_UPDATE_FILE.read_text().strip()
            
            if db_update > model_update:
                print(f"Database updated: {db_update} > {model_update}")
                train_model()
                print("Auto-retrain completed!")
                
        except Exception as e:
            print(f"Auto-retrain error: {e}")
        
        time.sleep(300)  # 5 minutes

def start_auto_retrain():
    """Start auto-retrain in background thread"""
    daemon = threading.Thread(target=auto_retrain_daemon, daemon=True)
    daemon.start()
    print("Auto-retrain daemon started (checks every 5 minutes)")

def get_suggestions(query, course_id=None, limit=20):
    """Get suggestions using ML fuzzy matching with exact match priority"""
    if not query.strip():
        return []
    
    try:
        # Clean query but preserve common patterns
        import re
        query_clean = re.sub(r'[^a-zA-Z0-9\s\(\)/\-&]', ' ', query)
        query_lower = query_clean.lower().strip()
        
        # Handle very short queries with direct matching + ML
        if len(query_lower) == 1:
            # For single character queries, use direct string matching first
            filter_course_id = course_id if course_id is not None else HARDCODED_COURSE_ID
            course_data = df[df['course_id'] == filter_course_id] if filter_course_id else df
            
            direct_matches = []
            for _, row in course_data.iterrows():
                spec_name = row[SPEC_COL]
                variant = str(row['spec_normalized']).lower()
                
                if spec_name.lower().startswith(query_lower) or variant.startswith(query_lower):
                    score = 90 if spec_name.lower().startswith(query_lower) else 80
                    direct_matches.append((spec_name, int(row['id']), int(row['course_id']), score))
            
            if direct_matches:
                # Remove duplicates
                seen_names = set()
                unique_matches = []
                for match in direct_matches:
                    if match[0] not in seen_names:
                        seen_names.add(match[0])
                        unique_matches.append(match)
                
                unique_matches.sort(key=lambda x: -x[3])
                return unique_matches[:limit]
        
        # Use ML similarity search for longer queries
        X_query = vectorizer.transform([query_lower])
        n_neighbors = min(len(df), 100)
        distances, indices = nn.kneighbors(X_query, n_neighbors=n_neighbors)
        
        suggestions = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= len(df):
                continue
                
            row = df.iloc[idx]
            
            # Filter by course_id if specified
            filter_course_id = course_id if course_id is not None else HARDCODED_COURSE_ID
            if filter_course_id is not None and row['course_id'] != filter_course_id:
                continue
                
            spec_name = row[SPEC_COL]
            spec_id = int(row['id'])
            course_id_val = int(row['course_id'])
            base_score = max(0, int((1 - dist) * 100))
            
            # Hybrid ML + Rule-based scoring
            spec_lower = spec_name.lower()
            spec_normalized = spec_lower.replace(' ', '').replace('.', '')
            query_normalized = query_lower.replace(' ', '').replace('.', '')
            
            # Rule-based boost calculation
            rule_boost = 0
            
            # 1. EXACT MATCH - Highest priority
            if query_lower == spec_lower or query_normalized == spec_normalized:
                rule_boost = 100
            
            # 2. STARTS WITH - Second priority
            elif spec_lower.startswith(query_lower) or spec_normalized.startswith(query_normalized):
                rule_boost = 90
            
            # 3. WORD START MATCH - Third priority
            elif any(word.startswith(query_lower) for word in spec_lower.split()):
                rule_boost = 70
            
            # 4. CONTAINS - Fourth priority
            elif query_lower in spec_lower or query_normalized in spec_normalized:
                rule_boost = 50
            
            # 5. ABBREVIATION PATTERNS - Special handling
            elif 'cs(aiml)' in query_lower or 'csaiml' in query_lower:
                if 'computer science' in spec_lower and 'artificial intelligence' in spec_lower and 'machine learning' in spec_lower:
                    rule_boost = 100
                elif 'artificial intelligence' in spec_lower and 'machine learning' in spec_lower:
                    rule_boost = 95
                elif 'computer science' in spec_lower and ('artificial intelligence' in spec_lower or 'machine learning' in spec_lower):
                    rule_boost = 90
            
            elif 'cs(ds)' in query_lower or 'cs ds' in query_lower:
                if 'computer science' in spec_lower and 'data science' in spec_lower:
                    rule_boost = 100
                elif 'data science' in spec_lower or 'big data' in spec_lower or 'analytics' in spec_lower:
                    rule_boost = 95
                elif 'computer science' in spec_lower:
                    rule_boost = 80
            
            elif 'ai ds' in query_lower or 'aids' in query_lower:
                if 'artificial intelligence' in spec_lower and 'data science' in spec_lower:
                    rule_boost = 100
                elif ('artificial intelligence' in spec_lower and 'data' in spec_lower) or ('ai' in spec_lower and 'data science' in spec_lower):
                    rule_boost = 95
            
            # 6. DOMAIN-SPECIFIC ABBREVIATIONS
            elif len(query_lower) <= 5:  # Short queries likely abbreviations
                # Engineering abbreviations
                if query_lower in ['cs', 'cse'] and 'computer science' in spec_lower:
                    rule_boost = 85
                elif query_lower in ['ece', 'extc'] and ('electronics' in spec_lower or 'communication' in spec_lower):
                    rule_boost = 85
                elif query_lower == 'mech' and 'mechanical' in spec_lower:
                    rule_boost = 85
                elif query_lower == 'civil' and 'civil' in spec_lower:
                    rule_boost = 85
                elif query_lower == 'bio' and 'biotechnology' in spec_lower:
                    rule_boost = 85
                # Medical abbreviations
                elif query_lower == 'cardio' and ('cardiology' in spec_lower or 'cardiac' in spec_lower):
                    rule_boost = 85
                elif query_lower == 'neuro' and ('neurology' in spec_lower or 'neurological' in spec_lower):
                    rule_boost = 85
                elif query_lower == 'ortho' and 'orthopedic' in spec_lower:
                    rule_boost = 85
            
            # Final score: Combine ML score with rule-based boost
            if rule_boost > 0:
                base_score = max(base_score, rule_boost)  # Take higher of ML or rule score
            
            # Ensure exact matches always get 100%
            if rule_boost == 100:
                base_score = 100
            
            suggestions.append((spec_name, spec_id, course_id_val, base_score))
        
        # Remove duplicates
        seen_names = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion[0] not in seen_names:
                seen_names.add(suggestion[0])
                unique_suggestions.append(suggestion)
        
        # Sort with exact match priority
        def sort_key(item):
            name, _, _, score = item
            name_lower = name.lower()
            # Exact match gets top priority
            if query_lower == name_lower:
                return (0, -score)  # 0 = highest priority
            # Starts with query gets second priority  
            elif name_lower.startswith(query_lower):
                return (1, -score)
            # Contains query gets third priority
            elif query_lower in name_lower:
                return (2, -score)
            # Everything else by score
            else:
                return (3, -score)
        
        unique_suggestions.sort(key=sort_key)
        return unique_suggestions[:limit]
        
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return []

def realtime_mode():
    """Real-time suggestion mode"""
    course_filter = f"Course ID: {HARDCODED_COURSE_ID}" if HARDCODED_COURSE_ID else "All Courses"
    title = f"MySQL Real-time Suggestions ({course_filter})"
    print(title)
    print("Start typing (ESC to exit)\n")
    current_input = ""

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                break
            elif key == b'\r':
                print("\n")
                continue
            elif key == b'\x08':
                if current_input:
                    current_input = current_input[:-1]
            elif 32 <= ord(key) <= 126:
                current_input += key.decode('utf-8')

            print("\033[2J\033[H", end="")
            print(title)
            print("Start typing (ESC to exit)\n")
            print(f"Query: {current_input}")

            if current_input:
                suggestions = get_suggestions(current_input, limit=8)
                if suggestions:
                    print(f"\nSuggestions ({len(suggestions)}):")
                    for name, spec_id, course_id, score in suggestions:
                        print(f"  - {name} ({score}%) [ID: {spec_id}, Course: {course_id}]")
                else:
                    print("\nNo matches found")
            print("\n" + "-" * 50)

def set_course_filter(course_id):
    """Update the HARDCODED_COURSE_ID in this file"""
    try:
        # Read the current file
        with open(__file__, 'r') as f:
            content = f.read()
        
        # Find and replace the HARDCODED_COURSE_ID line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('HARDCODED_COURSE_ID = '):
                if course_id.lower() == 'none' or course_id == '0':
                    lines[i] = 'HARDCODED_COURSE_ID = None  # All courses'
                else:
                    lines[i] = f'HARDCODED_COURSE_ID = {course_id}  # Course filter'
                break
        
        # Write back the modified content
        with open(__file__, 'w') as f:
            f.write('\n'.join(lines))
        
        if course_id.lower() == 'none' or course_id == '0':
            print("✅ Course filter set to: ALL COURSES")
        else:
            print(f"✅ Course filter set to: {course_id}")
        
        print("Restart the program to apply changes.")
        
    except Exception as e:
        print(f"❌ Error updating course filter: {e}")

def show_course_options():
    """Show available course filter options"""
    print("=== Course Filter Options ===")
    print("Common Course IDs:")
    print("  12 = Engineering")
    print("  37 = Medical")
    print("  46 = Management")
    print("  1  = Commerce/Business")
    print("  2  = Arts & Humanities")
    print("  7  = Sciences")
    print("  None = All courses")

if __name__ == "__main__":
    import sys
    
    # Handle command line arguments for course filter
    if len(sys.argv) > 1:
        if sys.argv[1] == '--set-course':
            if len(sys.argv) > 2:
                set_course_filter(sys.argv[2])
                sys.exit()
            else:
                print("Usage: python specialization_suggestion_realtime.py --set-course <course_id>")
                show_course_options()
                sys.exit()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python specialization_suggestion_realtime.py                    # Run suggestions")
            print("  python specialization_suggestion_realtime.py --set-course 12   # Set course filter")
            print("  python specialization_suggestion_realtime.py --help            # Show help")
            show_course_options()
            sys.exit()
    
    course_filter = f"COURSE_ID: {HARDCODED_COURSE_ID}" if HARDCODED_COURSE_ID else "ALL COURSES"
    print(f"=== MYSQL SPECIALIZATION SYSTEM - {course_filter} ===")
    
    try:
        # Load or train models
        load_models()
        
        # Start auto-retrain daemon
        start_auto_retrain()
        
        print(f"Loaded {len(df)} total specializations")
        if HARDCODED_COURSE_ID:
            course_count = len(df[df['course_id'] == HARDCODED_COURSE_ID])
            print(f"Course {HARDCODED_COURSE_ID} has {course_count} specializations")
            
            # Debug: Show some examples
            course_specs = df[df['course_id'] == HARDCODED_COURSE_ID]['specilization_name'].unique()[:5]
            print(f"Sample specializations: {list(course_specs)}")
        else:
            print("Showing suggestions from all courses")
        
        print("[MYSQL-POWERED] Auto-retrains every 5 minutes")
        print("\nCourse Filter Commands:")
        print("  python specialization_suggestion_realtime.py --set-course 12   # Engineering")
        print("  python specialization_suggestion_realtime.py --set-course 37   # Medical")
        print("  python specialization_suggestion_realtime.py --set-course none # All courses")
        print()
        
        realtime_mode()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MySQL is running and database is set up!")
        print("Run: python comprehensive_model_trainer.py --verify")