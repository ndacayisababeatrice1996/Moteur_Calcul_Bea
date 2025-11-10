from flask import Flask, jsonify, request, session, redirect, url_for, render_template
from flask_session import Session
import json
from math import exp
import random
import os
from openpyxl import Workbook
from functools import wraps
import sqlite3
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here-update-it"  # Mettez une clé secrète unique
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 heures (ajustable)
Session(app)

HISTORY_DB = "history.db"


def init_db():
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (user_id TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, calculation_type TEXT, start_age INTEGER, end_age INTEGER, population REAL, model TEXT, root REAL, timestamp TEXT, result TEXT)''')
    conn.commit()
    conn.close()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            logger.debug("Utilisateur non connecté, renvoi d'une erreur JSON")
            return jsonify({"error": "Non autorisé, veuillez vous connecter"}), 401
        logger.debug(f"Utilisateur connecté : {session['user_id']}")
        return f(*args, **kwargs)

    return decorated_function


def register_user(user_id, password):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (user_id, password) VALUES (?, ?)", (user_id, password))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True


def check_user(user_id, password):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result and result[0] == password else None


def save_history(user_id, calculation_type, start_age, end_age, population, model, root, result):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (user_id, calculation_type, start_age, end_age, population, model, root, timestamp, result) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)",
        (user_id, calculation_type, start_age, end_age, population, model, root, json.dumps(result)))
    conn.commit()
    conn.close()


def delete_history_entry(entry_id):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (entry_id, session["user_id"]))
    conn.commit()
    conn.close()


def get_history(page=1, per_page=5, sort_by="timestamp", order="desc", filter_model=None):
    conn = sqlite3.connect(HISTORY_DB)
    c = conn.cursor()
    query = "SELECT id, calculation_type, start_age, end_age, population, model, root, timestamp, result FROM history WHERE user_id = ?"
    params = [session["user_id"]]
    if filter_model:
        query += " AND model = ?"
        params.append(filter_model)
    query += " ORDER BY {} {}".format(sort_by, order)
    c.execute(query, params)
    rows = c.fetchall()
    total_entries = len(rows)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_rows = rows[start_idx:end_idx]
    history = [{
        "id": row[0],
        "calculation_type": row[1],
        "start_age": row[2],
        "end_age": row[3],
        "population": row[4],
        "model": row[5],
        "root": row[6],
        "timestamp": row[7],
        "result": json.loads(row[8])
    } for row in paginated_rows]
    conn.close()
    return {"data": history, "total_pages": (total_entries + per_page - 1) // per_page, "current_page": page,
            "total_entries": total_entries}


def normalize_probability(prob):
    return min(0.99, max(0.0, prob))


def lee_carter_mortality(start_age, end_age, population, root=1.0):
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        logger.debug(
            f"Paramètres pour Lee-Carter : start_age={start_age}, end_age={end_age}, population={population}, root={root}")
        if start_age < 0 or end_age <= start_age or population <= 0:
            raise ValueError("Âge de départ non négatif, âge final > âge de départ, population > 0.")

        ages = np.arange(start_age, end_age + 1, dtype=np.float64)
        ax = 0.05 + 0.001 * (ages / 100)
        kt = np.linspace(0, 0.01 * (end_age - start_age), len(ages))
        bx = np.ones(len(ages)) * 0.5
        data = []
        lx = population
        for i in range(len(ages)):
            base_rate = exp(ax[i] + bx[i] * kt[i]) * root
            qx = normalize_probability(base_rate * (1 + random.uniform(-0.1, 0.1)) / 100)
            px = 1 - qx
            dx = lx * qx
            data.append({
                "age": int(ages[i]),
                "lx": round(lx, 2),
                "dx": round(dx, 2),
                "qx": round(qx, 6),
                "px": round(px, 6)
            })
            lx -= dx
        logger.debug(f"Lee-Carter calculé avec succès")
        return data
    except ValueError as e:
        logger.error(f"Erreur de valeur dans lee_carter_mortality : {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Erreur inattendue dans lee_carter_mortality : {str(e)}", exc_info=True)
        return {"error": f"Erreur dans le calcul : {str(e)}"}


def makeham_mortality(start_age, end_age, population, root=1.0):
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        logger.debug(
            f"Paramètres pour Makeham : start_age={start_age}, end_age={end_age}, population={population}, root={root}")
        if start_age < 0 or end_age <= start_age or population <= 0:
            raise ValueError("Âge de départ non négatif, âge final > âge de départ, population > 0.")

        ages = np.arange(start_age, end_age + 1, dtype=np.float64)
        a, b, c = 0.0001, 0.0001, 0.09
        data = []
        lx = population
        for i in range(len(ages)):
            base_rate = (a + b * exp(c * ages[i])) * root
            qx = normalize_probability(base_rate * (1 + random.uniform(-0.05, 0.05)))
            px = 1 - qx
            dx = lx * qx
            data.append({
                "age": int(ages[i]),
                "lx": round(lx, 2),
                "dx": round(dx, 2),
                "qx": round(qx, 6),
                "px": round(px, 6)
            })
            lx -= dx
        logger.debug(f"Makeham calculé avec succès")
        return data
    except ValueError as e:
        logger.error(f"Erreur de valeur dans makeham_mortality : {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Erreur inattendue dans makeham_mortality : {str(e)}", exc_info=True)
        return {"error": f"Erreur dans le calcul : {str(e)}"}


def gompertz_mortality(start_age, end_age, population, root=1.0):
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        logger.debug(
            f"Paramètres pour Gompertz : start_age={start_age}, end_age={end_age}, population={population}, root={root}")
        if start_age < 0 or end_age <= start_age or population <= 0:
            raise ValueError("Âge de départ non négatif, âge final > âge de départ, population > 0.")

        ages = np.arange(start_age, end_age + 1, dtype=np.float64)
        a, b = 0.0001, 0.1
        data = []
        lx = population
        for i in range(len(ages)):
            base_rate = a * exp(b * ages[i]) * root
            qx = normalize_probability(base_rate * (1 + random.uniform(-0.05, 0.05)))
            px = 1 - qx
            dx = lx * qx
            data.append({
                "age": int(ages[i]),
                "lx": round(lx, 2),
                "dx": round(dx, 2),
                "qx": round(qx, 6),
                "px": round(px, 6)
            })
            lx -= dx
        logger.debug(f"Gompertz calculé avec succès")
        return data
    except ValueError as e:
        logger.error(f"Erreur de valeur dans gompertz_mortality : {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Erreur inattendue dans gompertz_mortality : {str(e)}", exc_info=True)
        return {"error": f"Erreur dans le calcul : {str(e)}"}


def heligman_lorenz_mortality(start_age, end_age, population, root=1.0):
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        logger.debug(
            f"Paramètres pour Heligman-Lorenz : start_age={start_age}, end_age={end_age}, population={population}, root={root}")
        if start_age < 0 or end_age <= start_age or population <= 0:
            raise ValueError("Âge de départ non négatif, âge final > âge de départ, population > 0.")

        ages = np.arange(start_age, end_age + 1, dtype=np.float64)
        A, B, C = 0.007, 0.0001, 2.7
        D, E, F = 0.00002, 0.0001, 0.5
        G, H = 0.001, 0.08
        data = []
        lx = population
        for i in range(len(ages)):
            age = ages[i]
            qx_base = (A * (B ** (C / (age + 1)))) + (D * exp(-E * ((age - 70) ** 2))) + (F * exp(G * (age ** H)))
            qx = normalize_probability(qx_base * root * (1 + random.uniform(-0.05, 0.05)) / 10)
            px = 1 - qx
            dx = lx * qx
            data.append({
                "age": int(age),
                "lx": round(lx, 2),
                "dx": round(dx, 2),
                "qx": round(qx, 6),
                "px": round(px, 6)
            })
            lx -= dx
        logger.debug(f"Heligman-Lorenz calculé avec succès")
        return data
    except ValueError as e:
        logger.error(f"Erreur de valeur dans heligman_lorenz_mortality : {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Erreur inattendue dans heligman_lorenz_mortality : {str(e)}", exc_info=True)
        return {"error": f"Erreur dans le calcul : {str(e)}"}


def coale_demeny_mortality(start_age, end_age, population, root=1.0):
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        logger.debug(
            f"Paramètres pour Coale-Demeny : start_age={start_age}, end_age={end_age}, population={population}, root={root}")
        if start_age < 0 or end_age <= start_age or population <= 0:
            raise ValueError("Âge de départ non négatif, âge final > âge de départ, population > 0.")

        ages = np.arange(start_age, end_age + 1, dtype=np.float64)
        qx_table = np.zeros(len(ages))
        for i in range(len(ages)):
            age = ages[i]
            if age < 1:
                qx_table[i] = 0.02 * root / 100
            elif 1 <= age < 5:
                qx_table[i] = 0.001 * root / 100
            elif 5 <= age < 15:
                qx_table[i] = 0.0005 * root / 100
            elif 15 <= age < 30:
                qx_table[i] = 0.0008 * root / 100
            elif 30 <= age < 50:
                qx_table[i] = 0.0012 * root / 100
            elif 50 <= age < 70:
                qx_table[i] = 0.005 * root / 100
            else:  # 70+
                qx_table[i] = 0.03 * (1 + 0.05 * (age - 70)) * root / 100

        qx_table = np.clip(qx_table, 0, 0.99)
        data = []
        lx = population
        for i in range(len(ages)):
            qx = normalize_probability(qx_table[i] * (1 + random.uniform(-0.05, 0.05)))
            px = 1 - qx
            dx = lx * qx
            data.append({
                "age": int(ages[i]),
                "lx": round(lx, 2),
                "dx": round(dx, 2),
                "qx": round(qx, 6),
                "px": round(px, 6)
            })
            lx -= dx
        logger.debug(f"Coale-Demeny calculé avec succès")
        return data
    except ValueError as e:
        logger.error(f"Erreur de valeur dans coale_demeny_mortality : {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Erreur inattendue dans coale_demeny_mortality : {str(e)}", exc_info=True)
        return {"error": f"Erreur dans le calcul : {str(e)}"}


def calculate_life_expectancy(data):
    total_lx = sum(d["lx"] for d in data)
    if total_lx == 0:
        return 0.0
    weighted_sum = sum(d["age"] * d["lx"] for d in data)
    return round(weighted_sum / total_lx, 2)


@app.route('/')
def login():
    logger.debug("Accès à la page de login")
    return render_template('login.html') if "user_id" not in session else redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
def login_post():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        logger.debug(f"Tentative de connexion : user_id={user_id}")
        stored_password = check_user(user_id, password)
        if stored_password:
            session["user_id"] = user_id
            logger.debug(f"Connexion réussie pour {user_id}")
            return redirect(url_for('index'))
        logger.debug("Échec de la connexion")
        return "Identifiants invalides", 401
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        password = request.form.get('password')
        logger.debug(f"Tentative d'inscription : user_id={user_id}")
        if register_user(user_id, password):
            session["user_id"] = user_id
            logger.debug(f"Inscription réussie pour {user_id}")
            return redirect(url_for('index'))
        return "Utilisateur déjà existant", 400
    return render_template('register.html')


@app.route('/logout')
def logout():
    user_id = session.get("user_id")
    logger.debug(f"Déconnexion de {user_id}")
    session.pop("user_id", None)
    return redirect(url_for('login'))


@app.route('/index')
@login_required
def index():
    return render_template('index.html')


@app.route('/calculate_mortality', methods=['GET'])
@login_required
def calculate_mortality():
    start_age = request.args.get('start_age')
    end_age = request.args.get('end_age')
    population = request.args.get('population')
    model = request.args.get('model', 'lee_carter')
    root = request.args.get('root', '1.0').replace(',', '.')

    logger.debug(
        f"Requête /calculate_mortality reçue : start_age={start_age}, end_age={end_age}, population={population}, model={model}, root={root}")
    if not all([start_age, end_age, population, model, root]):
        logger.error("Paramètres manquants")
        return jsonify({"error": "Tous les paramètres sont requis"}), 400

    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        root = float(root)
    except (ValueError, TypeError) as e:
        logger.error(f"Paramètres invalides : {str(e)}")
        return jsonify({"error": "Paramètres invalides (âges, population et racine doivent être numériques)"}), 400

    if start_age < 0 or end_age <= start_age or population <= 0 or root <= 0:
        logger.error("Paramètres hors limites")
        return jsonify({"error": "Âge de départ non négatif, âge final > âge de départ, population et racine > 0"}), 400

    models = {
        'lee_carter': lee_carter_mortality,
        'makeham': makeham_mortality,
        'gompertz': gompertz_mortality,
        'heligman_lorenz': heligman_lorenz_mortality,
        'coale_demeny': coale_demeny_mortality
    }
    if model not in models:
        logger.error(f"Modèle invalide : {model}")
        return jsonify({"error": "Modèle invalide"}), 400

    try:
        data = models[model](start_age, end_age, population, root)
        if "error" in data:
            logger.error(f"Erreur dans le calcul : {data['error']}")
            return jsonify(data), 400
        save_history(session["user_id"], "mortality", start_age, end_age, population, model, root, data)
        logger.debug(
            f"Réponse /calculate_mortality : {json.dumps(data[:5]) + '...' if len(data) > 5 else json.dumps(data)}")
        return jsonify(data)
    except Exception as e:
        logger.error(f"Exception inattendue dans /calculate_mortality : {str(e)}", exc_info=True)
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500


@app.route('/calculate_life_expectancy', methods=['GET'])
@login_required
def calculate_life_expectancy_endpoint():
    start_age = request.args.get('start_age')
    end_age = request.args.get('end_age')
    population = request.args.get('population')
    model = request.args.get('model', 'lee_carter')
    root = request.args.get('root', '1.0').replace(',', '.')

    logger.debug(
        f"Requête /calculate_life_expectancy reçue : start_age={start_age}, end_age={end_age}, population={population}, model={model}, root={root}")
    if not all([start_age, end_age, population, model, root]):
        logger.error("Paramètres manquants")
        return jsonify({"error": "Tous les paramètres sont requis"}), 400

    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        root = float(root)
    except (ValueError, TypeError) as e:
        logger.error(f"Paramètres invalides : {str(e)}")
        return jsonify({"error": "Paramètres invalides"}), 400

    if start_age < 0 or end_age <= start_age or population <= 0 or root <= 0:
        logger.error("Paramètres hors limites")
        return jsonify({"error": "Paramètres invalides"}), 400

    models = {
        'lee_carter': lee_carter_mortality,
        'makeham': makeham_mortality,
        'gompertz': gompertz_mortality,
        'heligman_lorenz': heligman_lorenz_mortality,
        'coale_demeny': coale_demeny_mortality
    }
    if model not in models:
        logger.error(f"Modèle invalide : {model}")
        return jsonify({"error": "Modèle invalide"}), 400

    try:
        data = models[model](start_age, end_age, population, root)
        if "error" in data:
            logger.error(f"Erreur dans le calcul : {data['error']}")
            return jsonify(data), 400
        life_expectancy = calculate_life_expectancy(data)
        save_history(session["user_id"], "life_expectancy", start_age, end_age, population, model, root,
                     {"life_expectancy": life_expectancy})
        logger.debug(f"Réponse /calculate_life_expectancy : {life_expectancy}")
        return jsonify({"life_expectancy": life_expectancy})
    except Exception as e:
        logger.error(f"Exception inattendue dans /calculate_life_expectancy : {str(e)}", exc_info=True)
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500


@app.route('/download_xlsx', methods=['GET'])
@login_required
def download_xlsx():
    start_age = request.args.get('start_age')
    end_age = request.args.get('end_age')
    population = request.args.get('population')
    model = request.args.get('model', 'lee_carter')
    root = request.args.get('root', '1.0').replace(',', '.')

    logger.debug(
        f"Requête /download_xlsx reçue : start_age={start_age}, end_age={end_age}, population={population}, model={model}, root={root}")
    try:
        start_age = int(start_age)
        end_age = int(end_age)
        population = float(population)
        root = float(root)
    except (ValueError, TypeError) as e:
        logger.error(f"Paramètres invalides : {str(e)}")
        return jsonify({"error": "Paramètres invalides"}), 400

    if start_age < 0 or end_age <= start_age or population <= 0 or root <= 0:
        logger.error("Paramètres hors limites")
        return jsonify({"error": "Paramètres invalides"}), 400

    data = {
        'lee_carter': lee_carter_mortality,
        'makeham': makeham_mortality,
        'gompertz': gompertz_mortality,
        'heligman_lorenz': heligman_lorenz_mortality,
        'coale_demeny': coale_demeny_mortality
    }[model](start_age, end_age, population, root)
    if "error" in data:
        logger.error(f"Erreur dans le calcul : {data['error']}")
        return jsonify(data), 400

    wb = Workbook()
    ws = wb.active
    ws.append(["Âge", "Lx (Vivants)", "Dx (Décès)", "qx (Prob. Décès)", "px (Prob. Survie)"])
    for row in data:
        ws.append([row["age"], row["lx"], row["dx"], row["qx"], row["px"]])
    from io import BytesIO
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    logger.debug(f"Fichier Excel généré pour {model}")
    return output.getvalue(), 200, {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Content-Disposition': f'attachment; filename=mortality_table_{model}.xlsx'
    }


@app.route('/history', methods=['GET'])
@login_required
def get_history_endpoint():
    page = request.args.get('page', default=1, type=int)
    per_page = request.args.get('per_page', default=5, type=int)
    sort_by = request.args.get('sort_by', default="timestamp", type=str)
    order = request.args.get('order', default="desc", type=str)
    filter_model = request.args.get('filter_model', type=str)

    logger.debug(
        f"Requête /history reçue : page={page}, per_page={per_page}, sort_by={sort_by}, order={order}, filter_model={filter_model}")
    if page < 1 or per_page < 1:
        logger.error("Paramètres de pagination invalides")
        return jsonify({"error": "Page et éléments par page doivent être positifs"}), 400
    if sort_by not in ["timestamp", "model", "start_age"]:
        logger.error(f"Tri invalide : {sort_by}")
        return jsonify({"error": "Tri invalide"}), 400
    if order not in ["asc", "desc"]:
        logger.error(f"Ordre invalide : {order}")
        return jsonify({"error": "Ordre invalide"}), 400

    history = get_history(page, per_page, sort_by, order, filter_model)
    logger.debug(f"Réponse /history : {len(history['data'])} entrées")
    return jsonify(history)


@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history_endpoint():
    entry_id = request.json.get('id')
    logger.debug(f"Requête /delete_history reçue : id={entry_id}")
    if not entry_id:
        logger.error("ID requis manquant")
        return jsonify({"error": "ID requis"}), 400
    delete_history_entry(entry_id)
    logger.debug(f"Entrée {entry_id} supprimée")
    return jsonify({"message": "Entrée supprimée"}), 200


if __name__ == "__main__":
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)