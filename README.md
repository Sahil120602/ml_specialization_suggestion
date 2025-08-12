# Specialization Suggestion System

A machine learning-powered system that provides real-time specialization suggestions based on user queries.

## Features

- **Trains on ALL specializations** from the CSV data (1000+ entries)
- **Filters suggestions by course_id** when displaying results
- **Real-time suggestions** as you type
- **Fuzzy matching** using TF-IDF and character n-grams
- **Similarity scoring** with percentage match

## Files

- `retrain_specialization_model.py` - Trains ML model on all specializations
- `specialization_suggestion_realtime.py` - Interactive real-time interface
- `test_suggestions.py` - Test script with examples
- `specalization.csv` - Source data with 1000+ specializations
- `ml_specialization_module/` - Generated ML models and lookup data

## Usage

### 1. Train the Model
```bash
python retrain_specialization_model.py
```
This creates ML models trained on ALL 1000+ specializations.

### 2. Test Suggestions
```bash
python test_suggestions.py
```
Shows example suggestions for different queries and course filters.

### 3. Real-time Interface

**All courses:**
```bash
python specialization_suggestion_realtime.py
```

**Specific course (e.g., Engineering - course_id 12):**
```bash
python specialization_suggestion_realtime.py 12
```

**Other course examples:**
- Course 37 (Medical): `python specialization_suggestion_realtime.py 37`
- Course 46 (Management): `python specialization_suggestion_realtime.py 46`

## Course IDs

Common course IDs in the dataset:
- **1**: Commerce/Business
- **2**: Arts & Humanities  
- **7**: Sciences
- **12**: Engineering (285+ specializations)
- **17**: Technical/Diploma
- **37**: Medical (28+ specializations)
- **46**: Management/MBA (20+ specializations)

## How It Works

1. **Training**: Model learns from ALL specializations using TF-IDF vectorization
2. **Query Processing**: User input is normalized and vectorized
3. **Similarity Search**: Nearest neighbors algorithm finds similar specializations
4. **Filtering**: Results are filtered by course_id if specified
5. **Scoring**: Cosine similarity converted to percentage match

## Example Output

```
Query: 'computer'
Engineering Course (ID: 12):
  • COMPUTER ENGINEERING (77%) [ID: 1738, Course: 12]
  • COMPUTER TECHNOLOGY (75%) [ID: 1681, Course: 12]
  • COMPUTER SCIENCE (72%) [ID: 1255, Course: 12]
```

The system provides intelligent, context-aware specialization suggestions with course-specific filtering!