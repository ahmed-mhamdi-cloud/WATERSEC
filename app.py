import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import json
import requests
import re
import time
import traceback
import hashlib
from groq import Groq
from collections import Counter
from datetime import datetime, date, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — SINGLE FILE WITH API KEYS
# ═══════════════════════════════════════════════════════════════════════════════

DOSSIER         = r"C:\\Users\\mhamd\\Downloads"
GROQ_API_KEY    = "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"
OPENWEATHER_KEY = "5662feedf185c83f056260dcdb5f9a76"
OLLAMA_URL      = "http://localhost:11434/api/generate"

# ── Mapping UUID → Readable Names ───────────────────────────────────────────
DEVICE_MAP = {
    "8161ea40-4a9c-11ef-82d3-2ffa8384e699": "cabin_1_cold",
    "81643740-4a9c-11ef-a595-f9c8dc6ae4ad": "cabin_1_hot",
    "8164cb10-4a9c-11ef-b24b-055224f3eeba": "cabin_2_cold",
    "81653450-4a9c-11ef-a236-3bbfdd38639d": "cabin_2_hot",
    "8165a900-4a9c-11ef-8393-69e55944ae1e": "cabin_3_cold",
    "81689b30-4a9c-11ef-ad50-1d2daba32da0": "cabin_3_hot",
    "816932d0-4a9c-11ef-b3c5-85b5ce342d21": "cabin_4_cold",
    "8170c860-4a9c-11ef-a57d-1978fe7d9161": "cabin_4_hot",
}

# ── Tunisia Calendar ─────────────────────────────────────────────────────────
CALENDRIER_TN = {
    "2024-03-11": "Début Ramadan 2024", "2024-04-10": "Aïd el-Fitr 2024",
    "2024-06-17": "Aïd el-Adha 2024", "2025-03-01": "Début Ramadan 2025",
    "2025-03-30": "Aïd el-Fitr 2025", "2025-06-06": "Aïd el-Adha 2025",
    "2025-07-25": "Fête République Tunisie", "2025-08-13": "Fête Femme Tunisie",
    "2026-02-18": "Début Ramadan 2026", "2026-03-20": "Fête Indépendance Tunisie",
}

st.set_page_config(page_title="WaterSec AI Agent", page_icon="💧", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES + MÉTADONNÉES DYNAMIQUES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    df_core = pd.read_csv(DOSSIER + r"\\watersec_core.csv", encoding="utf-8", low_memory=False)
    df_gym  = pd.read_csv(DOSSIER + r"\\watersec_gym.csv",  encoding="utf-8", low_memory=False)
    df_gym["device_name"] = df_gym["device"].map(DEVICE_MAP).fillna(df_gym["device"])
    df_core["device_name"] = df_core["device"].copy()
    for df in [df_core, df_gym]:
        if "jour" in df.columns:
            df["jour"] = pd.to_datetime(df["jour"], errors="coerce").dt.date
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df_core, df_gym

@st.cache_resource
def charger_db():
    return duckdb.connect(DOSSIER + r"\\watersec.db")

def get_dataset_metadata(df_core, df_gym):
    """Extrait les métadonnées réelles du dataset pour les guardrails et prompts."""
    meta = {
        "customers": [],
        "sites": [],
        "devices_gym": [],
        "date_min": None,
        "date_max": None,
        "sub_categories": [],
        "moments": [],
        "saisons": [],
    }

    # Customers
    for df, prefix in [(df_core, "core"), (df_gym, "gym")]:
        if "customer_id" in df.columns:
            meta["customers"].extend(df["customer_id"].dropna().unique().tolist())
    meta["customers"] = sorted(list(set(meta["customers"])))

    # Sites
    if "site_type" in df_core.columns:
        meta["sites"] = df_core["site_type"].dropna().unique().tolist()

    # Devices gym
    if "device" in df_gym.columns:
        meta["devices_gym"] = sorted(df_gym["device"].dropna().unique().tolist())

    # Dates
    all_dates = []
    for df in [df_core, df_gym]:
        if "jour" in df.columns:
            valid = pd.to_datetime(df["jour"], errors="coerce").dropna()
            if len(valid) > 0:
                all_dates.extend(valid.tolist())
    if all_dates:
        meta["date_min"] = min(all_dates).date() if hasattr(min(all_dates), 'date') else min(all_dates)
        meta["date_max"] = max(all_dates).date() if hasattr(max(all_dates), 'date') else max(all_dates)

    # Sub-categories
    if "sub_category_name" in df_core.columns:
        meta["sub_categories"] = df_core["sub_category_name"].dropna().unique().tolist()

    # Moments
    if "moment_journee" in df_core.columns:
        meta["moments"] = df_core["moment_journee"].dropna().unique().tolist()

    # Saisons
    if "saison" in df_core.columns:
        meta["saisons"] = df_core["saison"].dropna().unique().tolist()

    return meta

try:
    df_core, df_gym = charger_donnees()
    con = charger_db()
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and "TA_NOUVELLE" not in GROQ_API_KEY else None
    DATA_OK = True
    DATASET_META = get_dataset_metadata(df_core, df_gym)
except Exception as e:
    st.error(f"❌ Erreur chargement données : {e}")
    df_core, df_gym, con, client = None, None, None, None
    DATA_OK = False
    DATASET_META = {}

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTÉO & UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_meteo():
    if not OPENWEATHER_KEY:
        return None
    try:
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": "Tunis,TN", "appid": OPENWEATHER_KEY, "units": "metric", "lang": "fr"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if r.status_code != 200:
            return None
        return {
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"],
            "humidite": data["main"]["humidity"],
            "vent_kmh": round(data["wind"]["speed"] * 3.6, 1)
        }
    except:
        return None

def get_date_range_from_question(q):
    """Extrait une plage de dates depuis la question en langage naturel."""
    q_lower = q.lower()
    today = date.today()

    # Patterns de dates explicites (DD/MM/YYYY, YYYY-MM-DD, etc.)
    date_patterns = [
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', 'dmy'),
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),
    ]

    # Périodes relatives
    if any(k in q_lower for k in ["aujourd'hui", "today", "ce jour"]):
        return (today, today)
    if any(k in q_lower for k in ["hier", "yesterday"]):
        return (today - timedelta(1), today - timedelta(1))
    if any(k in q_lower for k in ["cette semaine", "this week"]):
        return (today - timedelta(7), today)
    if any(k in q_lower for k in ["semaine dernière", "last week"]):
        return (today - timedelta(14), today - timedelta(7))
    if any(k in q_lower for k in ["ce mois", "this month", "mois ci"]):
        return (today.replace(day=1), today)
    if any(k in q_lower for k in ["mois dernier", "last month"]):
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(1)
        last_month_start = last_month_end.replace(day=1)
        return (last_month_start, last_month_end)
    if any(k in q_lower for k in ["2 semaine", "2 semaines", "2 weeks", "deux semaines"]):
        return (today - timedelta(14), today)
    if any(k in q_lower for k in ["30 jour", "30 jours", "30 days", "dernier mois"]):
        return (today - timedelta(30), today)
    if any(k in q_lower for k in ["7 jour", "7 jours", "7 days"]):
        return (today - timedelta(7), today)

    # Recherche de dates explicites
    dates_found = []
    for pattern, fmt in date_patterns:
        matches = re.findall(pattern, q)
        for m in matches:
            try:
                if fmt == 'dmy':
                    day, month, year_str = int(m[0]), int(m[1]), m[2]
                    year = int("20" + year_str) if len(year_str) == 2 else int(year_str)
                    d = date(year, month, day)
                else:  # ymd
                    year, month, day = int(m[0]), int(m[1]), int(m[2])
                    d = date(year, month, day)
                dates_found.append(d)
            except:
                pass

    if len(dates_found) >= 2:
        return (min(dates_found), max(dates_found))
    elif len(dates_found) == 1:
        return (dates_found[0], today)

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL ANTI-HALLUCINATION
# ═══════════════════════════════════════════════════════════════════════════════

def guardrail_check(question, meta):
    """
    Vérifie si la question contient des références hors périmètre.
    Retourne (is_valid, error_message) ou (True, "") si OK.
    """
    if not meta:
        return True, ""  # Pas de métadonnées = pas de guardrail possible

    q_lower = question.lower()

    # 1. Vérification des dates
    date_range = get_date_range_from_question(question)
    if date_range and meta.get("date_min") and meta.get("date_max"):
        req_min, req_max = date_range
        data_min = meta["date_min"] if isinstance(meta["date_min"], date) else meta["date_min"].date()
        data_max = meta["date_max"] if isinstance(meta["date_max"], date) else meta["date_max"].date()

        # Tolérance de 1 jour
        if req_min < data_min - timedelta(1) or req_max > data_max + timedelta(1):
            return False, f"📅 **Hors périmètre temporel**\n\nJe ne possède pas ces informations dans les datasets actuels.\n\n📊 **Période couverte** : {data_min} → {data_max}\n❌ **Votre demande** : {req_min} → {req_max}\n\nVeuillez reformuler votre question avec des dates comprises dans cette période."

    # 2. Vérification des clients
    known_customers = [c.lower() for c in meta.get("customers", [])]
    # Extraire les mentions de clients dans la question
    client_mentions = re.findall(r'customer\s*([a-zA-Z0-9]+)', q_lower)
    client_mentions += re.findall(r'client\s*([a-zA-Z0-9]+)', q_lower)
    for mention in client_mentions:
        full = f"customer{mention}".lower()
        if full not in known_customers and mention.lower() not in [c.replace("customer", "") for c in known_customers]:
            return False, f"🏢 **Client inconnu**\n\nJe ne possède pas ces informations dans les datasets actuels.\n\n📊 **Clients disponibles** : {', '.join(meta.get('customers', []))}\n❌ **Client demandé** : customer{mention}\n\nVeuillez utiliser un client existant."

    # 3. Vérification des cabines (1-4 uniquement)
    cabin_nums = re.findall(r'cabine?\s*(\d+)|cabin\s*(\d+)', q_lower)
    for match in cabin_nums:
        num = int(match[0] or match[1])
        if num < 1 or num > 4:
            return False, f"🚿 **Cabine inexistante**\n\nJe ne possède pas ces informations dans les datasets actuels.\n\n📊 **Cabines disponibles** : 1, 2, 3, 4\n❌ **Cabine demandée** : {num}\n\nLe gym ne dispose que de 4 cabines."

    # 4. Vérification des sites
    if "site" in q_lower and meta.get("sites"):
        # Si mention de site spécifique non existant
        pass  # Trop complexe à parser, laisser le SQL échouer proprement

    return True, ""

# ═══════════════════════════════════════════════════════════════════════════════
# 7 FONCTIONS PATTERNS PANDAS (Calculs 100% exacts)
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_moment_journee(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("moment_journee")["consommation_ml"].agg(total="sum", moyenne="mean", nb_mesures="count").reset_index()
    result["total_litres"] = (result["total"] / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    return result[["moment_journee","total_litres","moyenne_litres","nb_mesures"]].sort_values("total_litres", ascending=False)

def pattern_jour_semaine(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("jour")["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["total_litres"] = (result["total"] / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    return result[["jour","total_litres","moyenne_litres"]]

def detecter_anomalies(customer_id=None, seuil=2.5):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    df["mean"] = df.groupby("device")["consommation_ml"].transform("mean")
    df["std"] = df.groupby("device")["consommation_ml"].transform("std")
    df["zscore"] = (df["consommation_ml"] - df["mean"]) / df["std"].replace(0, 1)
    anomalies = df[df["zscore"].abs() > seuil].copy()
    anomalies["consommation_litres"] = (anomalies["consommation_ml"] / 1000).round(2)
    cols = ["customer_id","device","consommation_litres","statut_alerte","jour","moment_journee","zscore"]
    return anomalies[[c for c in cols if c in anomalies.columns]].head(20)

def pattern_sequences_residentielles():
    df = df_core[df_core["customer_id"] == "customerC"].copy()
    if "sub_category_name" not in df.columns:
        return pd.DataFrame([{"info": "Colonne sub_category_name absente pour customerC"}])
    sequences = []
    for device in df["device"].unique():
        cats = df[df["device"] == device]["sub_category_name"].dropna().tolist()
        for i in range(len(cats) - 1):
            sequences.append(f"{cats[i]} → {cats[i+1]}")
    compteur = Counter(sequences)
    return pd.DataFrame(compteur.most_common(10), columns=["sequence","occurrences"])

def pattern_saison(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("saison")["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["total_litres"] = (result["total"] / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    return result[["saison","total_litres","moyenne_litres"]].sort_values("total_litres", ascending=False)

def pattern_gym_frequentation():
    df = df_gym.copy()
    if "frequentation_gym" in df.columns:
        df["frequentation_gym"] = pd.to_numeric(df["frequentation_gym"], errors="coerce")
        result = df.groupby("moment_journee").agg(
            conso_totale_litres=("consommation_ml", lambda x: round(x.sum()/1000, 2)),
            freq_moyenne=("frequentation_gym", "mean"),
            nb_sessions=("consommation_ml", "count")
        ).reset_index()
        result["freq_moyenne"] = result["freq_moyenne"].round(1)
    else:
        result = df.groupby("moment_journee").agg(
            conso_totale_litres=("consommation_ml", lambda x: round(x.sum()/1000, 2)),
            nb_sessions=("consommation_ml", "count")
        ).reset_index()
    return result.sort_values("conso_totale_litres", ascending=False)

def detecter_fuites():
    df_all = pd.concat([df_core, df_gym], ignore_index=True)
    fuites = df_all[df_all["statut_alerte"].str.contains("Fuite", na=False, case=False)]
    result = fuites.groupby(["customer_id","device","statut_alerte"]).agg(
        nb_occurrences=("consommation_ml","count"),
        conso_totale_litres=("consommation_ml", lambda x: round(x.sum()/1000, 2))
    ).reset_index()
    result["device"] = result["device"].map(DEVICE_MAP).fillna(result["device"])
    return result.sort_values("nb_occurrences", ascending=False)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS GROQ — DÉFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

tools = [
    {
        "type": "function",
        "function": {
            "name": "query_residential",
            "description": "SQL libre sur customerA/B/C. Colonnes: customer_id, site_type, device, consommation_ml, statut_alerte, cout_estime, argent_gaspille, mode_de_vie, intensite_debit, impact_energie, temperature_celsius, jour, moment_journee, saison, annee, mois, heure, debit_par_periode, sub_category_name. Diviser consommation_ml par 1000 pour litres.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_gym",
            "description": "SQL libre sur customerKF Gym. Colonnes: customer_id, site_type, device, consommation_ml, statut_alerte, timestamp, heure, jour, jour_semaine, mois, annee, moment_journee, saison, frequentation_gym, type_utilisation_douche, utilisateurs_actifs_30min, temps_pause_cabine_sec, statut_rotation, nb_utilisations_jour, debit_par_periode. Mapper UUID gym avec DEVICE_MAP. Diviser consommation_ml par 1000.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_all_sites",
            "description": "SQL libre sur tous les sites (comparaisons globales). Colonnes: customer_id, site_type, device, consommation_ml, statut_alerte, jour, moment_journee, saison, annee, mois, heure, debit_par_periode. Diviser consommation_ml par 1000.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_graph",
            "description": "Génère un graphique Plotly interactif. OBLIGATOIRE quand utilisateur demande graphique/plot/chart/courbe/histogramme/camembert/visualisation/tendance/évolution/montre-moi/compare/figure. Le LLM écrit le SQL et choisit chart_type. NE JAMAIS répondre en texte quand generate_graph est appelé.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT. Ex: SELECT jour, SUM(consommation_ml)/1000.0 as total_L FROM gym GROUP BY jour"},
                    "chart_type": {"type": "string", "enum": ["line", "bar", "pie", "scatter", "box"]},
                    "title": {"type": "string"},
                    "x_column": {"type": "string"},
                    "y_column": {"type": "string"},
                    "color_column": {"type": "string", "description": "Optionnel: colonne pour distinguer les séries"}
                },
                "required": ["sql", "chart_type", "title", "x_column", "y_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_patterns_comportementaux",
            "description": "Détecte patterns: sequences, moment, jour, saison, gym. Pour habitudes et comportements d'utilisation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_pattern": {"type": "string", "enum": ["sequences", "moment", "jour", "saison", "gym"]},
                    "customer_id": {"type": "string"}
                },
                "required": ["type_pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies_avancees",
            "description": "Détecte anomalies Z-score et fuites. Pour anomalies, fuites, alertes de consommation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_detection": {"type": "string", "enum": ["anomalies", "fuites"]},
                    "customer_id": {"type": "string"}
                },
                "required": ["type_detection"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meteo_correlation",
            "description": "Météo Tunis temps réel + calendrier tunisien. Pour météo, température, Ramadan, saisons, corrélations.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# EXÉCUTION DES TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def executer_tool(nom, arguments, return_typed=False):
    """
    Exécute un tool.
    - return_typed=False (défaut) : retourne string pour Groq tool_call
    - return_typed=True           : retourne dict {"df": DataFrame, "text": str, "sql": str}
    """
    if nom in ["query_residential", "query_gym", "query_all_sites"]:
        sql = arguments.get("sql", "")
        try:
            result = con.execute(sql).df()
            if "device" in result.columns:
                result["device"] = result["device"].map(DEVICE_MAP).fillna(result["device"])
            if return_typed:
                return {"df": result, "text": result.to_string(index=False), "sql": sql}
            return result.to_string(index=False)
        except Exception as e:
            err = f"❌ Erreur SQL : {str(e)}\nSQL : {sql}"
            if return_typed:
                return {"df": None, "text": err, "sql": sql}
            return err

    elif nom == "generate_graph":
        sql = arguments.get("sql", "")
        chart_type = arguments.get("chart_type", "bar")
        title = arguments.get("title", "Graphique")
        x_col = arguments.get("x_column", "")
        y_col = arguments.get("y_column", "")
        color_col = arguments.get("color_column", None)

        try:
            df_result = con.execute(sql).df()
            if "device" in df_result.columns:
                df_result["device"] = df_result["device"].map(DEVICE_MAP).fillna(df_result["device"])

            if chart_type == "line":
                fig = px.line(df_result, x=x_col, y=y_col, color=color_col, title=title, markers=True)
            elif chart_type == "bar":
                fig = px.bar(df_result, x=x_col, y=y_col, color=color_col, title=title, text=y_col)
            elif chart_type == "pie":
                fig = px.pie(df_result, names=x_col, values=y_col, title=title)
            elif chart_type == "scatter":
                fig = px.scatter(df_result, x=x_col, y=y_col, color=color_col, title=title)
            elif chart_type == "box":
                fig = px.box(df_result, x=x_col, y=y_col, title=title)
            else:
                fig = px.bar(df_result, x=x_col, y=y_col, color=color_col, title=title)

            fig.update_layout(template="plotly_white", height=450, title_x=0.5)
            return "__GRAPH__" + json.dumps(fig.to_dict(), ensure_ascii=False)
        except Exception as e:
            return f"❌ Erreur graphique : {str(e)}\nSQL : {sql}"

    elif nom == "detect_patterns_comportementaux":
        type_pattern = arguments.get("type_pattern", "sequences")
        customer_id = arguments.get("customer_id", None)
        try:
            if type_pattern == "sequences": result = pattern_sequences_residentielles()
            elif type_pattern == "moment": result = pattern_moment_journee(customer_id)
            elif type_pattern == "jour": result = pattern_jour_semaine(customer_id)
            elif type_pattern == "saison": result = pattern_saison(customer_id)
            elif type_pattern == "gym": result = pattern_gym_frequentation()
            else: result = pattern_moment_journee(customer_id)
            if return_typed:
                return {"df": result, "text": result.to_string(index=False), "sql": None}
            return result.to_string(index=False)
        except Exception as e:
            err = f"❌ Erreur pattern : {str(e)}"
            if return_typed:
                return {"df": None, "text": err, "sql": None}
            return err

    elif nom == "detect_anomalies_avancees":
        type_detection = arguments.get("type_detection", "anomalies")
        customer_id = arguments.get("customer_id", None)
        try:
            if type_detection == "fuites": result = detecter_fuites()
            else: result = detecter_anomalies(customer_id)
            if return_typed:
                return {"df": result, "text": result.to_string(index=False), "sql": None}
            return result.to_string(index=False)
        except Exception as e:
            err = f"❌ Erreur détection : {str(e)}"
            if return_typed:
                return {"df": None, "text": err, "sql": None}
            return err

    elif nom == "get_meteo_correlation":
        try:
            meteo = get_meteo()
            aujourd_hui = str(date.today())
            evenement = CALENDRIER_TN.get(aujourd_hui, "Jour ordinaire")
            prochains_events = {k: v for k, v in CALENDRIER_TN.items() if k >= aujourd_hui}
            prochain = list(prochains_events.items())[0] if prochains_events else None

            if not meteo:
                return f"⚠️ Météo indisponible.\n📅 Événement : {evenement}"

            insight = "Forte demande eau chaude attendue 🌡️" if meteo.get("temperature", 20) < 18 else "Consommation eau froide élevée probable 💧"

            return f"""🌤️ Météo Tunis ({aujourd_hui}) :
🌡️ Température : {meteo.get('temperature', 'N/A')}°C — {meteo.get('description', 'N/A')}
💧 Humidité : {meteo.get('humidite', 'N/A')}%
🌬️ Vent : {meteo.get('vent_kmh', 'N/A')} km/h
📅 Événement : {evenement}
📆 Prochain : {prochain[1] if prochain else 'Aucun'}
💡 Insight : {insight}"""
        except Exception as e:
            return f"⚠️ Erreur météo : {str(e)}"
    return "❌ Tool non reconnu"

# ═══════════════════════════════════════════════════════════════════════════════
# PARSER INTENT V2 — Avec extraction de dates et filtres
# ═══════════════════════════════════════════════════════════════════════════════

def parser_intent(question):
    q = question.lower()
    intent = "general"
    tool = "query_all_sites"
    params = {}

    # Détection graphique
    graph_keywords = ["graphique","graph","plot","chart","courbe","histogramme","camembert",
                      "visualisation","visualise","montre-moi","montre moi","tendance","évolution","evolution",
                      "compare","comparaison","comparison","figure","diagramme","dessine","dessiner"]
    wants_graph = any(k in q for k in graph_keywords)

    # Détection tableau
    table_keywords = ["tableau","table","liste","données brutes","raw data","affiche moi","show me the data",
                      "donne moi","liste-moi","lister","énumère","enumerate"]
    wants_table = any(k in q for k in table_keywords)

    # Extraction cabines
    cabin_match = re.search(r'cabine?\s*(\d)|cabin\s*(\d)', q)
    if cabin_match:
        num = cabin_match.group(1) or cabin_match.group(2)
        params["cabin_id"] = f"cabin_{num}"

    # Extraction multiple cabines
    all_cabins = re.findall(r'cabine?\s*(\d)|cabin\s*(\d)', q)
    if len(all_cabins) > 1:
        params["cabin_ids"] = [f"cabin_{n[0] or n[1]}" for n in all_cabins]

    # Type d'eau
    if any(k in q for k in ["chaude", "hot", "eau chaude", "warm"]):
        params["water_type"] = "hot"
    elif any(k in q for k in ["froide", "cold", "eau froide", "cool"]):
        params["water_type"] = "cold"

    # Client
    if "customera" in q or "customer a" in q:
        params["customer_id"] = "customerA"
    elif "customerb" in q or "customer b" in q:
        params["customer_id"] = "customerB"
    elif "customerc" in q or "customer c" in q:
        params["customer_id"] = "customerC"
    elif "customerkf" in q or "customer kf" in q or "kf" in q:
        params["customer_id"] = "customerKF"

    # Période
    date_range = get_date_range_from_question(question)
    if date_range:
        params["date_start"] = date_range[0]
        params["date_end"] = date_range[1]

    # Détection intent principal
    if any(k in q for k in ["gym","cabine","douche","cabin","kf","shower"]):
        intent = "gym"; tool = "query_gym"
    elif any(k in q for k in ["residential","customer","residentiel","maison","toilet","bathroom","kitchen","wc","flush","sink"]):
        intent = "residential"; tool = "query_residential"
    elif any(k in q for k in ["anomalie","fuite","alerte","anormal","anomaly","leak","détection","detect"]):
        intent = "anomalie"; tool = "detect_anomalies_avancees"
    elif any(k in q for k in ["pattern","comportement","sequence","habitude","behavior","habitude","usage pattern"]):
        intent = "pattern"; tool = "detect_patterns_comportementaux"
    elif any(k in q for k in ["meteo","temperature","temps","chaud","froid","saison","ramadan","weather","climate"]):
        intent = "meteo"; tool = "get_meteo_correlation"
    elif any(k in q for k in ["compare","tous","sites","global","total","all","highest","maximum","source","classement","ranking"]):
        intent = "all_sites"; tool = "query_all_sites"

    # Type de sortie demandé
    if wants_graph:
        params["wants_graph"] = True
    if wants_table:
        params["wants_table"] = True

    return {"intent": intent, "tool": tool, "params": params}

# ═══════════════════════════════════════════════════════════════════════════════
# SQL EXTRACTOR — Extrait et exécute le SQL d'une réponse texte Groq
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sql_from_text(text):
    """Extrait le bloc SQL d'un texte narratif."""
    # Pattern 1: Code block ```sql ... ```
    pattern1 = r'```sql\s*(.*?)```'
    match = re.search(pattern1, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: Code block ``` ... ```
    pattern2 = r'```\s*(SELECT.*?)(?:```|$)'
    match = re.search(pattern2, text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        if sql.upper().startswith("SELECT"):
            return sql

    # Pattern 3: SELECT ... ; (sans code block)
    pattern3 = r'(SELECT\s+.*?;?)'
    match = re.search(pattern3, text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        # Nettoyer
        sql = sql.split("\n\n")[0]  # Prendre seulement le premier bloc
        sql = sql.rstrip(";")
        return sql

    return None

def detect_chart_type_from_text(text, question):
    """Détecte le type de graphique depuis le texte ou la question."""
    q = question.lower()
    t = text.lower()

    if any(k in t or k in q for k in ["camembert", "pie", "proportion", "répartition"]):
        return "pie"
    if any(k in t or k in q for k in ["barre", "bar", "histogramme", "colonne", "compare"]):
        return "bar"
    if any(k in t or k in q for k in ["nuage", "scatter", "corrélation", "vs"]):
        return "scatter"
    if any(k in t or k in q for k in ["boîte", "box", "distribution"]):
        return "box"
    if any(k in t or k in q for k in ["ligne", "line", "courbe", "tendance", "evolution"]):
        return "line"
    return "bar"

def auto_generate_graph_from_sql(sql, question, text_response=""):
    """Génère un graphique Plotly à partir d'un SQL extrait."""
    try:
        df_result = con.execute(sql).df()
        if "device" in df_result.columns:
            df_result["device"] = df_result["device"].map(DEVICE_MAP).fillna(df_result["device"])

        if len(df_result.columns) < 2:
            return None, "Pas assez de colonnes pour un graphique"

        x_col = df_result.columns[0]
        y_col = df_result.columns[1]
        chart_type = detect_chart_type_from_text(text_response, question)

        # Détection titre
        title = "Analyse WaterSec"
        if "gym" in question.lower():
            title = "Analyse Gym"
        elif "customer" in question.lower():
            title = "Analyse Client"
        elif "conso" in question.lower() or "consommation" in question.lower():
            title = "Consommation d'eau"

        if chart_type == "line":
            fig = px.line(df_result, x=x_col, y=y_col, title=title, markers=True)
        elif chart_type == "bar":
            fig = px.bar(df_result, x=x_col, y=y_col, title=title, text=y_col, color=x_col)
        elif chart_type == "pie":
            fig = px.pie(df_result, names=x_col, values=y_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(df_result, x=x_col, y=y_col, title=title)
        elif chart_type == "box":
            fig = px.box(df_result, x=x_col, y=y_col, title=title)
        else:
            fig = px.bar(df_result, x=x_col, y=y_col, title=title, text=y_col)

        fig.update_layout(template="plotly_white", height=500, title_x=0.5)
        return fig, None
    except Exception as e:
        return None, str(e)

def auto_generate_table_from_sql(sql):
    """Génère un DataFrame à partir d'un SQL extrait."""
    try:
        df_result = con.execute(sql).df()
        if "device" in df_result.columns:
            df_result["device"] = df_result["device"].map(DEVICE_MAP).fillna(df_result["device"])
        return df_result, None
    except Exception as e:
        return None, str(e)

# ═══════════════════════════════════════════════════════════════════════════════
# NARRATION OLLAMA (Fallback Local)
# ═══════════════════════════════════════════════════════════════════════════════

def narrer_avec_ollama(model, question, data, context=""):
    try:
        prompt = f"""Tu es WaterSec AI, expert en analyse de consommation d'eau IoT en Tunisie.
Voici les données calculées (100% exactes, zéro hallucination) :
{data}

{context}

Question : {question}

RÈGLES ABSOLUES — ZÉRO TOLÉRANCE :
1. Tu ne peux utiliser QUE les chiffres présents dans les données ci-dessus.
2. Si une donnée n'est PAS dans le tableau, réponds EXACTEMENT : "Je ne possède pas ces informations dans les datasets actuels."
3. Ne jamais extrapoler, deviner, ou inventer de valeurs.
4. Ne jamais répondre à une question sur un site, client, cabine, ou période non présent dans les données.

Règles métier eau :
- Douche standard : 6-10 L/min, 5-8 min | Longue : >10 min = gaspillage
- Eau chaude : ~40% hiver, ~60% été | Wudu : ~5L/utilisation
- Gym benchmark : 120L/cabine/jour | Flush : 6L standard, 3L éco
- Lavabo : 15L/jour/personne

Rédige une réponse en FRANÇAIS (max 5 phrases) :
1. Réponds directement à la question avec les chiffres exacts du tableau
2. Donne UNE recommandation actionnable
3. Cite UNIQUEMENT les données fournies ci-dessus"""

        r = requests.post(OLLAMA_URL,
                          json={"model": model, "prompt": prompt, "stream": False,
                                "options": {"temperature": 0.1, "num_ctx": 4096}},
                          timeout=60)
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        raise Exception(f"Ollama {model} indisponible : {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT GROQ — Allégé, Impératif, avec Guardrails
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(meta):
    """Construit le system prompt dynamiquement avec les métadonnées réelles."""
    date_range = ""
    if meta.get("date_min") and meta.get("date_max"):
        dmin = meta["date_min"] if isinstance(meta["date_min"], str) else str(meta["date_min"])
        dmax = meta["date_max"] if isinstance(meta["date_max"], str) else str(meta["date_max"])
        date_range = f"\nPÉRIODE DONNÉES: {dmin} → {dmax}"

    customers = ", ".join(meta.get("customers", ["customerA", "customerB", "customerC", "customerKF"]))

    return f"""Tu es WaterSec AI, expert analyse consommation eau IoT Tunisie.
Détecte la langue de la question et réponds DANS LA MÊME LANGUE.

DONNÉES DISPONIBLES:{date_range}
- Clients: {customers}
- customerA/B/C : résidentiels
- customerKF : Gym 4 cabines hot/cold

UUID→CABINES GYM:
cabin_1_cold=8161ea40..., cabin_1_hot=81643740...
cabin_2_cold=8164cb10..., cabin_2_hot=81653450...
cabin_3_cold=8165a900..., cabin_3_hot=81689b30...
cabin_4_cold=816932d0..., cabin_4_hot=8170c860...

BENCHMARKS:
- Gym: 120L/cabine/jour, >150L=surconso
- Flush: 6L standard, 3L éco
- Lavabo: 15L/jour/personne
- Wudu: ~5L/utilisation

OUTILS:
→ query_residential/query_gym/query_all_sites: SQL libre
→ generate_graph: GRAPHIQUE Plotly — OBLIGATOIRE si graphique demandé
→ detect_patterns_comportementaux: patterns
→ detect_anomalies_avancees: anomalies/fuites
→ get_meteo_correlation: météo Tunis

RÈGLES ABSOLUES:
1. SI graphique demandé (graphique/plot/chart/courbe/histogramme/camembert/visualisation/tendance/évolution/montre-moi/compare/figure/dessine) → OBLIGATOIREMENT appeler generate_graph. JAMAIS répondre en texte.
2. generate_graph prend: sql + chart_type + title + x_column + y_column
3. SQL: toujours diviser consommation_ml par 1000
4. Mapper UUID gym avec DEVICE_MAP
5. ZÉRO HALLUCINATION: Si les données SQL retournent un résultat vide, réponds EXACTEMENT : "Je ne possède pas ces informations dans les datasets actuels." Ne jamais inventer de chiffre.
6. Si une date, un site, un client, ou une cabine n'existe pas dans les données → "Je ne possède pas ces informations dans les datasets actuels."
7. Réponds dans la même langue que la question
8. Pour comparaisons → bar chart, tendances temporelles → line chart, proportions → pie chart

EXEMPLES SQL:
-- Tendance gym
SELECT jour, ROUND(SUM(consommation_ml)/1000.0,2) as total_L FROM gym GROUP BY jour ORDER BY jour
-- Comparer cabines
SELECT device, ROUND(SUM(consommation_ml)/1000.0,2) as total_L FROM gym WHERE device LIKE '%cabin_1%' OR device LIKE '%cabin_2%' GROUP BY device
-- Par type résidentiel
SELECT sub_category_name, ROUND(SUM(consommation_ml)/1000.0,2) as total_L FROM residential WHERE customer_id='customerC' GROUP BY sub_category_name
-- Global
SELECT customer_id, ROUND(SUM(consommation_ml)/1000.0,2) as total_L FROM all_sites GROUP BY customer_id ORDER BY total_L DESC"""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT GROQ — Avec retry, extraction SQL fallback, et gestion narratif
# ═══════════════════════════════════════════════════════════════════════════════

def agent_groq_call(question, historique=[], max_retries=2):
    """
    Appelle Groq avec function calling.
    Si Groq répond en texte avec du SQL (bug), extrait et exécute le SQL.
    Retourne: (content, outils_utilises, is_text_response)
    """
    system_prompt = build_system_prompt(DATASET_META)
    messages = [{"role": "system", "content": system_prompt}]
    for h in historique[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": question})

    outils_utilises = []

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=2048
            )
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tc in message.tool_calls:
                    nom = tc.function.name
                    args = json.loads(tc.function.arguments)
                    resultat = executer_tool(nom, args)
                    outils_utilises.append({"tool": nom, "args": args, "result_preview": str(resultat)[:200]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": resultat})

                # Deuxième appel pour narration
                response2 = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0,
                    max_tokens=2048
                )
                return response2.choices[0].message.content, outils_utilises, False
            else:
                # Groq a répondu en texte (pas de tool call)
                return message.content, outils_utilises, True

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise e

    return "⚠️ Trop d'itérations de tools. Veuillez reformuler.", outils_utilises, True

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT PRINCIPAL — Routeur Intelligent Universel avec Structure Typée
# ═══════════════════════════════════════════════════════════════════════════════

def agent(question, historique=[]):
    """
    Routeur intelligent universel avec retour structuré:
    {"type": "text", "content": str}
    {"type": "table", "dataframe": pd.DataFrame, "sql": str}
    {"type": "graph", "figure": go.Figure, "sql": str}
    {"type": "error", "message": str}

    Fallback: Groq → SQL Extractor → Offline SQL → Ollama
    """
    # ── ÉTAPE 0: Guardrail ───────────────────────────────────────────────
    is_valid, error_msg = guardrail_check(question, DATASET_META)
    if not is_valid:
        return {"type": "guardrail", "message": error_msg}, [], "🛡️ Guardrail Anti-Hallucination"

    parsed = parser_intent(question)
    intent = parsed["intent"]
    params = parsed["params"]
    wants_graph = params.get("wants_graph", False)
    wants_table = params.get("wants_table", False)

    # ── ÉTAPE 1: GROQ (si disponible) ────────────────────────────────────
    if client:
        try:
            content, outils, is_text_only = agent_groq_call(question, historique)

            # Cas A: Groq a exécuté un tool generate_graph
            if outils and any(o["tool"] == "generate_graph" for o in outils):
                for o in outils:
                    if o["tool"] == "generate_graph":
                        result = executer_tool("generate_graph", o["args"])
                        if isinstance(result, str) and result.startswith("__GRAPH__"):
                            fig_dict = json.loads(result[9:])
                            fig = go.Figure(fig_dict)
                            return {"type": "graph", "figure": fig, "sql": o["args"].get("sql", "")}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas B: Groq a exécuté un tool SQL → retourner narration + DataFrame (type dual)
            if outils and any(o["tool"].startswith("query_") for o in outils):
                # Récupérer le DataFrame depuis le premier tool SQL
                sql_tool = next(o for o in outils if o["tool"].startswith("query_"))
                typed = executer_tool(sql_tool["tool"], sql_tool["args"], return_typed=True)
                df_result = typed.get("df") if typed else None
                sql_used  = typed.get("sql", "") if typed else ""
                # Décision d'affichage : si graphique demandé → graph, sinon dual (table + narration)
                if wants_graph and df_result is not None:
                    fig, err = auto_generate_graph_from_sql(sql_used, question, content)
                    if fig:
                        return {"type": "graph", "figure": fig, "sql": sql_used, "caption": content}, outils, "🚀 Groq LLaMA 3.3-70B"
                if df_result is not None and len(df_result) > 0:
                    return {"type": "dual", "content": content, "dataframe": df_result, "sql": sql_used}, outils, "🚀 Groq LLaMA 3.3-70B"
                return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas B2: Groq a exécuté un tool patterns ou anomalies → dual aussi
            if outils and any(o["tool"] in ["detect_patterns_comportementaux", "detect_anomalies_avancees"] for o in outils):
                pattern_tool = next(o for o in outils if o["tool"] in ["detect_patterns_comportementaux", "detect_anomalies_avancees"])
                typed = executer_tool(pattern_tool["tool"], pattern_tool["args"], return_typed=True)
                df_result = typed.get("df") if typed else None
                if df_result is not None and len(df_result) > 0:
                    return {"type": "dual", "content": content, "dataframe": df_result, "sql": None}, outils, "🚀 Groq LLaMA 3.3-70B"
                return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas C: Groq a répondu en texte (pas de tool call)
            if is_text_only:
                # Vérifier si du SQL est présent dans le texte
                extracted_sql = extract_sql_from_text(content)

                if extracted_sql:
                    # Groq a donné du SQL dans le texte → l'exécuter!
                    if wants_graph:
                        fig, err = auto_generate_graph_from_sql(extracted_sql, question, content)
                        if fig:
                            return {"type": "graph", "figure": fig, "sql": extracted_sql}, outils + [{"tool": "sql_extractor", "args": {"sql": extracted_sql}}], "🚀 Groq + Auto-Graph"
                        else:
                            # Fallback: exécuter comme tableau
                            df, err = auto_generate_table_from_sql(extracted_sql)
                            if df is not None:
                                return {"type": "table", "dataframe": df, "sql": extracted_sql}, outils + [{"tool": "sql_extractor", "args": {"sql": extracted_sql}}], "🚀 Groq + Auto-Table"

                    if wants_table:
                        df, err = auto_generate_table_from_sql(extracted_sql)
                        if df is not None:
                            return {"type": "table", "dataframe": df, "sql": extracted_sql}, outils + [{"tool": "sql_extractor", "args": {"sql": extracted_sql}}], "🚀 Groq + Auto-Table"

                    # Par défaut: exécuter le SQL et narrer
                    df, err = auto_generate_table_from_sql(extracted_sql)
                    if df is not None:
                        data_str = df.to_string(index=False)
                        narration = f"{content}\n\n**Résultat de l'analyse:**\n```\n{data_str}\n```"
                        return {"type": "text", "content": narration}, outils + [{"tool": "sql_extractor", "args": {"sql": extracted_sql}}], "🚀 Groq + SQL Auto-Exec"

                # Pas de SQL trouvé → retourner le texte tel quel
                return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas D: Autre réponse texte
            return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "quota" in error_msg.lower() or "429" in error_msg:
                st.warning("⚠️ Quota Groq atteint, passage en mode offline...")
            else:
                st.error(f"❌ Erreur Groq: {error_msg}")

    # ── ÉTAPE 2: FALLBACK OFFLINE (SQL Intelligent) ──────────────────────

    if wants_graph:
        # Générer SQL automatiquement
        sql = build_smart_sql(question, intent, params)
        fig, err = auto_generate_graph_from_sql(sql, question, "")
        if fig:
            return {"type": "graph", "figure": fig, "sql": sql}, [{"tool": "offline_sql_builder", "args": {"sql": sql}}], "📊 Génération Directe (Offline)"

    if wants_table:
        sql = build_smart_sql(question, intent, params)
        df, err = auto_generate_table_from_sql(sql)
        if df is not None:
            return {"type": "table", "dataframe": df, "sql": sql}, [{"tool": "offline_sql_builder", "args": {"sql": sql}}], "📊 Tableau Direct (Offline)"

    # ── ÉTAPE 3: FALLBACK OLLAMA ─────────────────────────────────────────
    try:
        # Récupérer des données génériques pour la narration
        sql = "SELECT customer_id, ROUND(SUM(consommation_ml)/1000.0, 2) as total_L FROM all_sites GROUP BY customer_id ORDER BY total_L DESC"
        data = executer_tool("query_all_sites", {"sql": sql})
        content = narrer_avec_ollama("qwen2.5:14b", question, data)
        return {"type": "text", "content": content}, [], "🛡️ Local Qwen2.5:14b (Offline)"
    except Exception as e:
        pass

    try:
        sql = "SELECT customer_id, ROUND(SUM(consommation_ml)/1000.0, 2) as total_L FROM all_sites GROUP BY customer_id ORDER BY total_L DESC"
        data = executer_tool("query_all_sites", {"sql": sql})
        content = narrer_avec_ollama("qwen2.5:7b", question, data)
        return {"type": "text", "content": content}, [], "⚠️ Local Qwen2.5:7b (Fallback)"
    except Exception as e:
        return {"type": "error", "message": f"❌ Tous les modèles sont indisponibles.\nErreur : {str(e)}"}, [], "❌ Offline"

# ═══════════════════════════════════════════════════════════════════════════════
# SQL BUILDER INTELLIGENT (Offline)
# ═══════════════════════════════════════════════════════════════════════════════

def build_smart_sql(question, intent, params):
    """Construit un SQL adapté à la question sans LLM."""
    q = question.lower()

    # Détection table
    if any(k in q for k in ["gym", "cabine", "douche", "cabin", "kf"]):
        table = "gym"
    elif any(k in q for k in ["residential", "customer", "residentiel", "toilet", "bathroom", "flush", "sink"]):
        table = "residential"
    else:
        table = "all_sites"

    # Détection agrégation
    agg = "SUM"
    if any(k in q for k in ["moyenne", "average", "avg", "mean"]):
        agg = "AVG"
    elif any(k in q for k in ["maximum", "max", "highest", "plus haut"]):
        agg = "MAX"
    elif any(k in q for k in ["minimum", "min", "lowest", "plus bas"]):
        agg = "MIN"
    elif any(k in q for k in ["count", "nombre", "combien", "nb "]):
        agg = "COUNT"

    # Détection dimensions
    group_by = "jour"
    x_col = "jour"
    if any(k in q for k in ["cabine", "cabin", "device", "source"]):
        group_by = "device"
        x_col = "device"
    elif any(k in q for k in ["client", "customer", "profil"]):
        group_by = "customer_id"
        x_col = "customer_id"
    elif any(k in q for k in ["moment", "matin", "midi", "soir", "nuit"]):
        group_by = "moment_journee"
        x_col = "moment_journee"
    elif any(k in q for k in ["saison", "été", "hiver", "printemps", "automne"]):
        group_by = "saison"
        x_col = "saison"

    # Filtres WHERE
    where_clauses = ["consommation_ml > 0"]

    if params.get("cabin_id") and table == "gym":
        where_clauses.append(f"device LIKE '%{params['cabin_id']}%'")
    if params.get("water_type") and table == "gym":
        where_clauses.append(f"device LIKE '%{params['water_type']}%'")
    if params.get("customer_id"):
        where_clauses.append(f"customer_id = '{params['customer_id']}'")

    date_range = get_date_range_from_question(question)
    if date_range:
        start, end = date_range
        where_clauses.append(f"jour >= '{start}' AND jour <= '{end}'")

    where_sql = " AND ".join(where_clauses)

    if agg == "COUNT":
        value_expr = "COUNT(*)"
        y_col = "nb_events"
    else:
        value_expr = f"ROUND({agg}(consommation_ml)/1000.0, 2)"
        y_col = "total_L" if agg == "SUM" else "avg_L" if agg == "AVG" else "max_L" if agg == "MAX" else "min_L"

    sql = f"""SELECT {x_col}, {value_expr} AS {y_col}
FROM {table}
WHERE {where_sql}
GROUP BY {group_by}
ORDER BY {y_col} DESC
LIMIT 50"""

    return sql

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES DASHBOARD — 4 Graphiques Fixes
# ═══════════════════════════════════════════════════════════════════════════════

def graph_conso_par_client():
    df_all = pd.concat([df_core, df_gym], ignore_index=True)
    result = df_all.groupby("customer_id")["consommation_ml"].sum().reset_index()
    result["Litres"] = (result["consommation_ml"] / 1000).round(2)
    return px.bar(result, x="customer_id", y="Litres",
                  title="Consommation totale par client",
                  color="customer_id", text="Litres",
                  color_discrete_sequence=px.colors.qualitative.Bold)

def graph_moment_journee():
    result = df_core.groupby("moment_journee")["consommation_ml"].mean().reset_index()
    result["Litres"] = (result["consommation_ml"] / 1000).round(2)
    return px.bar(result, x="moment_journee", y="Litres",
                  title="Consommation moyenne par moment de journée",
                  color="moment_journee",
                  color_discrete_sequence=px.colors.qualitative.Pastel)

def graph_saison():
    result = df_core.groupby("saison")["consommation_ml"].sum().reset_index()
    result["Litres"] = (result["consommation_ml"] / 1000).round(2)
    return px.pie(result, names="saison", values="Litres",
                  title="Répartition de consommation par saison",
                  color_discrete_sequence=px.colors.qualitative.Safe)

def graph_cabines_gym():
    col = "device_name" if "device_name" in df_gym.columns else "device"
    result = df_gym.groupby(col)["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["Total (L)"] = (result["total"] / 1000).round(2)
    result["Moyenne (L)"] = (result["moyenne"] / 1000).round(2)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Total par cabine", "Moyenne par cabine"))
    fig.add_trace(go.Bar(x=result[col], y=result["Total (L)"],
                         marker_color="#1E88E5", name="Total"), row=1, col=1)
    fig.add_trace(go.Bar(x=result[col], y=result["Moyenne (L)"],
                         marker_color="#43A047", name="Moyenne"), row=1, col=2)
    fig.update_layout(title="Comparaison cabines Gym", showlegend=False,
                      xaxis_tickangle=-45, xaxis2_tickangle=-45)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS RAPIDES DYNAMIQUES (avec vraies dates du dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def get_quick_prompts(meta):
    """Génère 10 prompts rapides avec les vraies dates du dataset."""
    dmin = meta.get("date_min")
    dmax = meta.get("date_max")

    if dmin and dmax:
        dmax_dt = dmax if isinstance(dmax, date) else datetime.strptime(str(dmax), "%Y-%m-%d").date()
        dmin_dt = dmin if isinstance(dmin, date) else datetime.strptime(str(dmin), "%Y-%m-%d").date()
        # Dates relatives basées sur les VRAIES dates du dataset
        two_weeks_ago = max(dmax_dt - timedelta(14), dmin_dt)
        one_month_ago = max(dmax_dt - timedelta(30), dmin_dt)
        mid_point     = dmin_dt + (dmax_dt - dmin_dt) // 2
        tw_str  = two_weeks_ago.strftime("%d/%m/%Y")
        om_str  = one_month_ago.strftime("%d/%m/%Y")
        dmax_s  = dmax_dt.strftime("%d/%m/%Y")
        dmin_s  = dmin_dt.strftime("%d/%m/%Y")
        mid_s   = mid_point.strftime("%d/%m/%Y")
    else:
        tw_str = om_str = dmax_s = dmin_s = mid_s = "disponible"

    return [
        f"Compare cabine 1 et 2 du gym entre {tw_str} et {dmax_s}",
        f"Graphique tendance eau froide gym du {tw_str} au {dmax_s}",
        f"Anomalies chasse d'eau résidentielle depuis {om_str}",
        "Quelle source a la plus haute conso eau chaude tous sites ?",
        "Conso gym corrélée à la météo cette semaine ?",
        "Patterns comportementaux séquentiels customerC",
        f"Tableau comparatif clients entre {mid_s} et {dmax_s}",
        "Camembert répartition consommation par saison",
        f"Liste fuites détectées depuis {tw_str}",
        "Météo Tunis et impact sur consommation aujourd'hui",
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT — Complète et Professionnelle
# ═══════════════════════════════════════════════════════════════════════════════

if not DATA_OK:
    st.error("❌ Impossible de charger les données. Vérifiez les fichiers CSV et DuckDB.")
    st.stop()

st.title("💧 WaterSec AI Agent")
st.markdown("**Agent conversationnel intelligent pour l'analyse de consommation d'eau IoT**")

# ── Badge Online/Offline + Métadonnées ───────────────────────────────────────
status_cols = st.columns([1, 2, 2])
with status_cols[0]:
    if client:
        st.success("🟢 Groq Online")
    else:
        st.warning("🟡 Mode Offline — Ollama actif")
with status_cols[1]:
    if DATASET_META.get("date_min") and DATASET_META.get("date_max"):
        dmin = DATASET_META["date_min"]
        dmax = DATASET_META["date_max"]
        st.info(f"📅 Données: {dmin} → {dmax}")
with status_cols[2]:
    if DATASET_META.get("customers"):
        st.info(f"🏢 Clients: {', '.join(DATASET_META['customers'])}")

st.divider()

# ── Métriques temps réel ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    total = (df_core["consommation_ml"].sum() + df_gym["consommation_ml"].sum()) / 1000
    st.metric("💧 Conso totale", f"{total:,.0f} L")
with col2:
    st.metric("📊 Total lignes", f"{len(df_core) + len(df_gym):,}")
with col3:
    nb_fuites = df_core["statut_alerte"].str.contains("Fuite", na=False).sum()
    st.metric("🚨 Fuites détectées", f"{nb_fuites:,}")
with col4:
    meteo = get_meteo()
    if meteo:
        st.metric("🌡️ Tunis maintenant", f"{meteo['temperature']}°C", meteo["description"])
    else:
        st.metric("🌡️ Tunis", "N/A")

st.divider()

col_chat, col_graphs = st.columns([1.2, 1])

# ── CHAT CONVERSATIONNEL ───────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Chat avec WaterSec AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.historique = []

    # Prompts rapides dynamiques
    st.markdown("**Prompts rapides :**")
    prompts_rapides = get_quick_prompts(DATASET_META)
    cols = st.columns(5)
    for i, prompt in enumerate(prompts_rapides):
        if cols[i % 5].button(prompt[:35] + "..." if len(prompt) > 35 else prompt, 
                               key=f"btn_{i}", use_container_width=True):
            st.session_state.input_rapide = prompt

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "graph" and "figure" in msg:
                st.plotly_chart(msg["figure"], use_container_width=True)
            elif msg.get("type") == "table" and "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
            elif msg.get("type") == "dual":
                if msg.get("content"):
                    st.markdown(msg["content"])
                if "dataframe" in msg and msg["dataframe"] is not None:
                    st.dataframe(msg["dataframe"], use_container_width=True)
            elif msg.get("type") == "guardrail":
                st.info(msg["content"])
            elif msg.get("type") == "error":
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])

    input_val = st.session_state.get("input_rapide", "")
    question = st.chat_input("Posez votre question sur les données water...")

    if not question and input_val:
        question = input_val
        st.session_state.input_rapide = ""

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyse en cours..."):
                result, outils, mode = agent(question, st.session_state.historique)

            # Badge du mode utilisé
            if "❌" in mode or result.get("type") == "error":
                st.error(f"📡 {mode}")
            elif "Offline" in mode or "Fallback" in mode or "Directe" in mode:
                st.warning(f"📡 {mode}")
            elif result.get("type") == "guardrail":
                st.info(f"🛡️ {mode}")
            else:
                st.success(f"📡 {mode}")

            # ── AFFICHAGE INTELLIGENT SELON LE TYPE ──
            result_type = result.get("type", "text")

            if result_type == "graph":
                fig = result.get("figure")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("📊 Graphique généré dynamiquement")
                    # Afficher la narration si présente
                    if result.get("caption"):
                        with st.expander("💬 Analyse"):
                            st.markdown(result["caption"])
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")
                else:
                    st.error("Erreur lors de la génération du graphique")

            elif result_type == "table":
                df = result.get("dataframe")
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"📋 {len(df)} lignes")
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")
                else:
                    st.error("Erreur lors de la génération du tableau")

            elif result_type == "dual":
                # Narration + DataFrame côte à côte
                narration = result.get("content", "")
                df = result.get("dataframe")
                if narration:
                    st.markdown(narration)
                if df is not None and len(df) > 0:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"📋 Données brutes : {len(df)} lignes")
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")

            elif result_type == "guardrail":
                # Message anti-hallucination — afficher en warning (pas en error)
                st.info(result.get("message", "Je ne possède pas ces informations dans les datasets actuels."))

            elif result_type == "error":
                st.error(result.get("message", "Erreur inconnue"))

            else:  # text
                st.markdown(result.get("content", ""))

            # ── OUTILS UTILISÉS ──
            if outils:
                with st.expander("🔧 Outils utilisés"):
                    for o in outils:
                        st.write(f"**{o['tool']}**")
                        if 'args' in o:
                            st.code(json.dumps(o['args'], indent=2, ensure_ascii=False), language="json")
                        if 'result_preview' in o:
                            st.caption(f"Résultat: {o['result_preview']}")

        # Mise à jour historique
        msg_to_store = {
            "role": "assistant",
            "type": result_type,
        }
        if result_type == "graph":
            msg_to_store["figure"] = result.get("figure")
            msg_to_store["content"] = "[Graphique généré]"
        elif result_type == "table":
            msg_to_store["dataframe"] = result.get("dataframe")
            msg_to_store["content"] = f"[Tableau: {len(result.get('dataframe', pd.DataFrame()))} lignes]"
        elif result_type == "dual":
            msg_to_store["dataframe"] = result.get("dataframe")
            msg_to_store["content"] = result.get("content", "")
        elif result_type == "guardrail":
            msg_to_store["content"] = result.get("message", "")
        elif result_type == "error":
            msg_to_store["content"] = result.get("message", "Erreur")
        else:
            msg_to_store["content"] = result.get("content", "")

        st.session_state.messages.append(msg_to_store)
        st.session_state.historique.append({"role": "user", "content": question})
        st.session_state.historique.append({"role": "assistant", "content": result.get("content", str(result))})

        # Garder seulement les 10 derniers échanges
        if len(st.session_state.historique) > 20:
            st.session_state.historique = st.session_state.historique[-20:]

# ── GRAPHIQUES DASHBOARD ───────────────────────────────────────────────────
with col_graphs:
    st.subheader("📊 Tableaux de bord")
    tab1, tab2, tab3, tab4 = st.tabs(["Par client", "Par moment", "Par saison", "Gym"])
    with tab1:
        st.plotly_chart(graph_conso_par_client(), use_container_width=True)
    with tab2:
        st.plotly_chart(graph_moment_journee(), use_container_width=True)
    with tab3:
        st.plotly_chart(graph_saison(), use_container_width=True)
    with tab4:
        st.plotly_chart(graph_cabines_gym(), use_container_width=True)