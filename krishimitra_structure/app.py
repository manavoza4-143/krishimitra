import os
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

# Import blueprint route modules for ML predictions
from routes.crop import crop_bp
from routes.soil import soil_bp

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
# HOOKING THE MODEL BLUEPRINTS
# -------------------------------------------------------------------------
# This automatically registers the real predictive routes /api/crop/recommend
# and /api/soil/analyse directly into your web context layout
app.register_blueprint(crop_bp)
app.register_blueprint(soil_bp)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)