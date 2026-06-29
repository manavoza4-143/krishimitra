import os
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "krishimitra_secret_gold_2026")
DB_PATH = os.path.join("database", "krishimitra.db")

# Ensure database directory exists
os.makedirs("database", exist_ok=True)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT NOT NULL,
            state TEXT,
            district TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    # Crop queries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crop_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            state TEXT,
            district TEXT,
            season TEXT,
            n REAL, p REAL, k REAL, ph REAL,
            temperature REAL,
            humidity REAL,
            rainfall REAL,
            top_crop TEXT,
            results_json TEXT,
            queried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    
    # Soil queries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS soil_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            crop TEXT,
            soil_type TEXT,
            n REAL, p REAL, k REAL, ph REAL,
            organic_carbon REAL,
            ec REAL,
            health_score REAL,
            suggestions_json TEXT,
            queried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.close()

# Initialize tables on startup
init_db()

# -------------------------------------------------------------------------
# HTML PAGE ROUTING
# -------------------------------------------------------------------------

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash('Welcome back to KrishiMitra!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    phone = request.form.get('phone')
    email = request.form.get('email')
    password = request.form.get('password')
    state = request.form.get('state')
    district = request.form.get('district')
    
    hashed_pw = generate_password_hash(password)
    
    try:
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO users (name, phone, email, password_hash, state, district)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, phone, email, hashed_pw, state, district))
        conn.commit()
        conn.close()
        flash('Registration successful! Please login.', 'success')
    except sqlite3.IntegrityError:
        flash('Email or Phone number already registered.', 'danger')
        
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('landing'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', name=session.get('user_name'))

@app.route('/crop')
def crop_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('crop.html')

@app.route('/soil')
def soil_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('soil.html')

@app.route('/weather')
def weather_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('weather.html')


# -------------------------------------------------------------------------
# API ROUTING FOR ML MODULES (Returns JSON responses)
# -------------------------------------------------------------------------

@app.route('/api/crop/recommend', methods=['POST'])
def api_crop_recommend():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.json or request.form
    # Fallback placeholders replicating backend ML calculations
    # In a fully deployed context, you would load your models/crop_model.pkl here
    top_crops = [
        {"crop": "Rice 🌾", "confidence": 94.5, "tips": "Requires abundant watering and structural clayey soils."},
        {"crop": "Maize 🌽", "confidence": 88.2, "tips": "Ensure good drainage system and timely Nitrogen applications."},
        {"crop": "Cotton 🌱", "confidence": 76.1, "tips": "Thrives best in deep black soils with moderate rainfall."},
        {"crop": "Moong Jowar 🌾", "confidence": 65.4, "tips": "Great low-moisture alternative for drought periods."},
        {"crop": "Wheat 🌾", "confidence": 58.0, "tips": "Optimal choices during winter cycles with light irrigation."}
    ]
    
    # Log query meta to database
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO crop_queries (user_id, state, district, season, n, p, k, ph, temperature, humidity, rainfall, top_crop, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session['user_id'], data.get('state'), data.get('district'), data.get('season'),
          data.get('n'), data.get('p'), data.get('k'), data.get('ph'),
          data.get('temperature'), data.get('humidity'), data.get('rainfall'), "Rice", str(top_crops)))
    conn.commit()
    conn.close()

    return jsonify(top_crops)


@app.route('/api/soil/analyse', methods=['POST'])
def api_soil_analyse():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.json or request.form
    
    # Replicating Deep Learning model categorization output dynamically
    health_score = 78  # Mock calculation
    suggestions = [
        "Incorporate organic compost to enhance the overall Organic Carbon profile.",
        "Apply Gypsum if the alkaline EC metrics drift outside normal bands.",
        "Balance localized Nitrogen deficiency using targeted Neem-coated Urea feeds.",
        "Utilize drip irrigation methods to consistently handle light moisture retention setups.",
        "Schedule a micro-booster crop feed containing Zinc Sulphate before the active sowing stage."
    ]
    
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO soil_queries (user_id, crop, soil_type, n, p, k, ph, organic_carbon, ec, health_score, suggestions_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session['user_id'], data.get('crop'), data.get('soil_type'),
          data.get('n'), data.get('p'), data.get('k'), data.get('ph'),
          data.get('organic_carbon'), data.get('ec'), health_score, str(suggestions)))
    conn.commit()
    conn.close()

    return jsonify({
        "health_score": health_score,
        "deficiencies": ["Zinc (Zn) - Mild", "Organic Carbon - Low"],
        "suggestions": suggestions,
        "timeline": "3-4 Weeks before next sowing cycle"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)