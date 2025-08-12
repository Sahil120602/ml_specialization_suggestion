import mysql.connector
import pandas as pd

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',  # Change this
    'database': 'specializations_db'
}

def setup_database():
    """Create database and import CSV data"""
    # Connect without database first
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    cursor = conn.cursor()
    
    # Create database
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.execute(f"USE {DB_CONFIG['database']}")
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS specializations (
            id INT PRIMARY KEY,
            specilization_name VARCHAR(500),
            course_id INT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    
    # Import CSV data
    df = pd.read_csv('specalization.csv')
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT IGNORE INTO specializations (id, specilization_name, course_id) 
            VALUES (%s, %s, %s)
        """, (int(row['id']), row['specilization_name'], int(row['course_id'])))
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Database setup complete. Imported {len(df)} records.")

if __name__ == "__main__":
    setup_database()