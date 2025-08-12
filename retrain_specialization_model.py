import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===== Paths =====
BASE_DIR = Path("ml_specialization_module")
VECTOR_FILE = BASE_DIR / "spec_vectorizer.pkl"
MODEL_FILE = BASE_DIR / "spec_nn_model.pkl"
LOOKUP_FILE = BASE_DIR / "specializations_lookup.csv"

# ===== Config =====
SPEC_COL = "specilization_name"
CSV_FILE = "specalization.csv"  # Your uploaded CSV

# ===== Load from CSV =====
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} total records from {CSV_FILE}")
print(f"Available course_ids: {sorted(df['course_id'].unique())}")

# Train on ALL specializations (no filtering by course_id)
print(f"Training model on all {len(df)} specializations across all courses")

# Generate abbreviation variants for ML training
def normalize_text(text):
    return " ".join(str(text).lower().replace(".", "").strip().split())

def generate_abbreviation_variants(text):
    """Generate comprehensive abbreviation patterns including complex combinations"""
    variants = [text]  # Original text
    
    # Generate initials (CS for Computer Science)
    words = text.split()
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
        
        # Generate partial initials (C S for Computer Science)
        spaced_initials = ' '.join([w[0] for w in words if w])
        variants.append(spaced_initials)
    
    # Generate comprehensive short forms for ALL domains
    short_forms = {
        # Engineering
        'computer science': ['cs', 'comp sci'],
        'artificial intelligence': ['ai'],
        'machine learning': ['ml'],
        'information technology': ['it'],
        'electronics and communication': ['ece', 'ec'],
        'electronics and telecommunications': ['extc', 'etc'],
        'electronics & telecommunications': ['extc', 'etc'],
        'electronics and telecommunication': ['extc', 'etc'],
        'electrical engineering': ['ee'],
        'mechanical engineering': ['me'],
        'civil engineering': ['ce'],
        'computer aided design': ['cad'],
        'very large scale integration': ['vlsi'],
        'internet of things': ['iot'],
        
        # Medical & Healthcare
        'bachelor of medicine bachelor of surgery': ['mbbs', 'md'],
        'master of surgery': ['ms', 'mch'],
        'bachelor of dental surgery': ['bds'],
        'master of dental surgery': ['mds'],
        'bachelor of pharmacy': ['bpharm'],
        'master of pharmacy': ['mpharm'],
        'doctor of pharmacy': ['pharmd'],
        'bachelor of nursing': ['bsc nursing'],
        'medical laboratory technology': ['mlt'],
        'radiology and imaging technology': ['rit'],
        
        # Management & Business
        'master of business administration': ['mba'],
        'bachelor of business administration': ['bba'],
        'master of commerce': ['mcom'],
        'bachelor of commerce': ['bcom'],
        'chartered accountancy': ['ca'],
        'human resources': ['hr'],
        'human resource management': ['hrm'],
        'marketing management': ['marketing'],
        'financial management': ['finance'],
        'supply chain management': ['scm'],
        
        # Arts & Sciences
        'bachelor of arts': ['ba'],
        'master of arts': ['ma'],
        'bachelor of science': ['bsc'],
        'master of science': ['msc'],
        'bachelor of computer applications': ['bca'],
        'master of computer applications': ['mca'],
        'biotechnology': ['biotech'],
        'journalism and mass communication': ['jmc'],
        'mass communication': ['mass comm'],
        'political science': ['pol sci'],
        
        # Law & Education
        'bachelor of laws': ['llb'],
        'master of laws': ['llm'],
        'bachelor of education': ['bed'],
        'master of education': ['med'],
        
        # Architecture & Design
        'bachelor of architecture': ['b arch'],
        'interior design': ['id'],
        'fashion design': ['fashion'],
        'graphic design': ['graphics'],
        
        # Others
        'hotel management': ['hm'],
        'physical education': ['pe']
    }
    
    text_lower = text.lower()
    for full_form, abbrevs in short_forms.items():
        if full_form in text_lower:
            for abbrev in abbrevs:
                variants.append(text_lower.replace(full_form, abbrev))
    
    # Add domain-specific patterns
    if 'telecommunication' in text_lower:
        variants.extend(['extc', 'etc', 'telecom'])
    if 'electronic' in text_lower and 'communication' in text_lower:
        variants.extend(['extc', 'ece', 'ec'])
    if 'medicine' in text_lower and 'surgery' in text_lower:
        variants.extend(['mbbs', 'md'])
    if 'business' in text_lower and 'administration' in text_lower:
        variants.extend(['mba', 'bba'])
    if 'computer' in text_lower and 'application' in text_lower:
        variants.extend(['bca', 'mca'])
    if 'pharmacy' in text_lower:
        variants.extend(['bpharm', 'mpharm', 'pharmd'])
    if 'commerce' in text_lower:
        variants.extend(['bcom', 'mcom'])
    if 'nursing' in text_lower:
        variants.extend(['gnm', 'bsc nursing'])
    if 'management' in text_lower:
        variants.extend(['mgmt'])
    if 'technology' in text_lower:
        variants.extend(['tech'])
    
    # Generate complex combinations for AI/ML specializations
    if 'computer science' in text_lower and ('artificial intelligence' in text_lower or 'machine learning' in text_lower):
        # Generate csaiml, csai, csml patterns
        variants.extend(['csaiml', 'csai', 'csml', 'cs ai ml', 'cs artificial intelligence machine learning'])
        
    if 'artificial intelligence' in text_lower and 'machine learning' in text_lower:
        variants.extend(['aiml', 'ai ml', 'artificial intelligence ml', 'ai machine learning'])
    
    return list(set(variants))  # Remove duplicates

# Expand training data with abbreviation variants
df[SPEC_COL] = df[SPEC_COL].astype(str).str.strip()
df["spec_normalized"] = df[SPEC_COL].apply(normalize_text)

# Generate expanded training data with mapping back to original records
expanded_data = []
for idx, row in df.iterrows():
    spec = row["spec_normalized"]
    variants = generate_abbreviation_variants(spec)
    for variant in variants:
        # Create new row for each variant but keep original metadata
        new_row = row.copy()
        new_row["spec_normalized"] = variant
        expanded_data.append(new_row)

# Create expanded dataframe
expanded_df = pd.DataFrame(expanded_data)
specializations = expanded_df["spec_normalized"].tolist()

print(f"Original specializations: {len(df)}")
print(f"Expanded with abbreviations: {len(expanded_df)}")

# Update df to expanded version for consistent indexing
df = expanded_df.reset_index(drop=True)

# ===== Train model =====
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(specializations)

nn_model = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
nn_model.fit(X)

# ===== Save artifacts =====
BASE_DIR.mkdir(parents=True, exist_ok=True)
with open(VECTOR_FILE, "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(nn_model, f)

df.to_csv(LOOKUP_FILE, index=False)
print(f"SUCCESS: Specialization model trained on {len(specializations)} entries from all courses.")
