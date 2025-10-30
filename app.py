# app.py
import os
import sqlite3
import json
from flask import Flask, render_template, request, redirect, url_for, session, g, flash
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

from model_utils import (
    load_and_prepare_data,
    train_multiple_models,
    load_model_and_metrics,
    generate_permutation_importance,
    create_data_viz_charts,
    explain_prediction_alt
)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
DATABASE = 'users.db'

# -----------------------------
# DB helpers
# -----------------------------
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db:
        db.close()

def init_db():
    """
    Initialize users table and history table.
    """
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT UNIQUE NOT NULL,
                     email TEXT UNIQUE NOT NULL,
                     password TEXT NOT NULL)''')
        db.execute('''CREATE TABLE IF NOT EXISTS history
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id INTEGER NOT NULL,
                     input_data TEXT NOT NULL,
                     prediction TEXT NOT NULL,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (user_id) REFERENCES users (id))''')
        db.commit()

# -----------------------------
# Auth routes
# -----------------------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm_password")

        if not username or not email or not password or not confirm:
            flash("⚠️ All fields are required.", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("❌ Passwords do not match.", "danger")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect(DATABASE)
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, hashed))
            conn.commit()
            user_id = cur.lastrowid
            conn.close()
            flash("✅ Registered! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("⚠️ Username/email exists.", "danger")
            return redirect(url_for("register"))
        except Exception as e:
            flash(f"Registration error: {e}", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        user_input = request.form.get("user_input")
        password = request.form.get("password")
        db = get_db()
        user = db.execute("SELECT id, username, email, password FROM users WHERE username=? OR email=?",
                          (user_input, user_input)).fetchone()
        if user and check_password_hash(user["password"], password):
            session['logged_in'] = True
            session['username'] = user["username"]
            session['user_id'] = user["id"]
            return redirect(url_for('home'))
        flash("❌ Invalid username/email or password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))


# -----------------------------
# Main routes
# -----------------------------
@app.route("/")
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template("home.html", username=session.get('username'))


@app.route("/dashboard")
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    data = load_and_prepare_data()
    if data is None or data.empty:
        return "No data available.", 500

    # --- Dataset preview ---
    data_head = data.head().to_html(classes='table table-striped table-dark', index=False)

    # --- Model + metrics ---
    model, features, metrics_df, tree_model = load_model_and_metrics()
    if model is None or metrics_df is None:
        model, metrics_df, features = train_multiple_models(data)
        _, _, _, tree_model = load_model_and_metrics()

    metrics_html = metrics_df.to_html(classes='table table-striped table-dark', index=False) if metrics_df is not None else "No model data."

    # --- Permutation Importance Plot ---
    permutation_importance_plot = generate_permutation_importance(model, data) if model else None

    # --- Feature importance (for table + chart) ---
    if tree_model is not None and hasattr(tree_model, "feature_importances_"):
        importances = pd.DataFrame({
            "Feature": features,
            "Importance": tree_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        importance_html = importances.to_html(classes="table table-striped table-dark", index=False)

        chart_labels = importances["Feature"].tolist()
        chart_data = importances["Importance"].round(4).tolist()
    else:
        importance_html = "No feature importance available."
        chart_labels = []
        chart_data = []

    return render_template(
        "dashboard.html",
        data_head=data_head,
        metrics_html=metrics_html,
        permutation_importance_plot=permutation_importance_plot,
        importance_html=importance_html,
        chart_labels=chart_labels,
        chart_data=chart_data
    )


@app.route("/test", methods=["GET","POST"])
def test():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == "POST":
        try:
            sample_data = {col: float(request.form.get(col)) for col in
                           ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']}
            session['sample_data'] = sample_data
            return redirect(url_for('result'))
        except Exception as e:
            flash(f"Invalid input: {e}", "danger")
            return redirect(url_for('test'))
    return render_template("test.html")


@app.route("/result")
def result():
    if not session.get('logged_in') or 'sample_data' not in session:
        return redirect(url_for('test'))

    sample_data = session['sample_data']
    model, features, _, _ = load_model_and_metrics()

    if model is None:
        data = load_and_prepare_data()
        model, _, features = train_multiple_models(data)

    sample_df = pd.DataFrame([sample_data], columns=features)
    prediction = model.predict(sample_df)[0]
    result_text = "Potable" if prediction == 1 else "Not Potable"

    # Save to history
    try:
        db = get_db()
        db.execute("INSERT INTO history (user_id, input_data, prediction) VALUES (?, ?, ?)",
                   (session.get('user_id'), json.dumps(sample_data), result_text))
        db.commit()
    except Exception as e:
        print(f"[app] Failed to save history: {e}")

    # simple textual explanation (no SHAP)
    explanation_text = explain_prediction_alt(model, sample_df)

    return render_template("result.html",
                           result=result_text,
                           input_data=sample_data,
                           explanation_text=explanation_text)


@app.route("/history")
def history():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    db = get_db()
    rows = db.execute("SELECT id, input_data, prediction, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC",
                      (session.get('user_id'),)).fetchall()

    # Convert sqlite rows to list of dicts for easier template rendering
    history = []
    for r in rows:
        try:
            input_data = json.loads(r["input_data"])
        except Exception:
            input_data = r["input_data"]
        history.append({
            "id": r["id"],
            "input_data": input_data,
            "prediction": r["prediction"],
            "timestamp": r["timestamp"]
        })

    return render_template("history.html", history=history)


@app.route("/explanation")
def explanation():
    if not session.get('logged_in') or 'sample_data' not in session:
        return redirect(url_for('test'))

    sample_data = session['sample_data']
    model, features, _, _ = load_model_and_metrics()

    if model is None:
        data = load_and_prepare_data()
        model, _, features = train_multiple_models(data)

    sample_df = pd.DataFrame([sample_data], columns=features)
    explanation_text = explain_prediction_alt(model, sample_df)

    return render_template("explanation.html", explanation_text=explanation_text)


@app.route('/data_viz')
def data_viz():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    data = load_and_prepare_data()
    if data is None:
        return "Error: Could not load data.", 500
    ph_dist, hardness_dist, solids_dist = create_data_viz_charts(data)
    return render_template('data_viz.html',
                           ph_dist=ph_dist,
                           hardness_dist=hardness_dist,
                           solids_dist=solids_dist)


# Run app
if __name__ == "__main__":
    init_db()
    if not os.path.exists("best_model.joblib"):
        data = load_and_prepare_data()
        if data is not None:
            train_multiple_models(data)
    app.run(debug=True)
