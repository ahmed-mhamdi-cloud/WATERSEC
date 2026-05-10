# ═══════════════════════════════════════════════════════════════════════════════
#  WaterSec AI Agent — Production Final
#  Architecture : Groq LLaMA 3.3-70B → Parser Intent → Ollama 14b → Ollama 7b
#  Features     : Dynamic Rendering · Smart Prompts · Anti-Hallucination
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import json
import re
import requests
import traceback
from groq import Groq
from collections import Counter
from datetime import datetime, date, timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DOSSIER         = r"C:\Users\mhamd\Downloads"
GROQ_API_KEY    = "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"
OPENWEATHER_KEY = "5662feedf185c83f056260dcdb5f9a76"
OLLAMA_URL      = "http://localhost:11434/api/generate"

# ── UUID → Noms lisibles ─────────────────────────────────────────────────────
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

# ── Calendrier Tunisie ────────────────────────────────────────────────────────
CALENDRIER_TN = {
    "2024-03-11": "Début Ramadan 2024", "2024-04-10": "Aïd el-Fitr 2024",
    "2024-06-17": "Aïd el-Adha 2024",  "2025-03-01": "Début Ramadan 2025",
    "2025-03-30": "Aïd el-Fitr 2025",  "2025-06-06": "Aïd el-Adha 2025",
    "2025-07-25": "Fête République",   "2025-08-13": "Fête Femme Tunisie",
    "2026-02-18": "Début Ramadan 2026","2026-03-20": "Fête Indépendance",
}

st.set_page_config(page_title="WaterSec AI Agent", page_icon="💧", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees():
    df_core = pd.read_csv(DOSSIER + r"\watersec_core.csv", encoding="utf-8", low_memory=False)
    df_gym  = pd.read_csv(DOSSIER + r"\watersec_gym.csv",  encoding="utf-8", low_memory=False)
    df_gym["device_name"]  = df_gym["device"].map(DEVICE_MAP).fillna(df_gym["device"])
    df_core["device_name"] = df_core["device"].copy()
    return df_core, df_gym

@st.cache_resource
def charger_db():
    return duckdb.connect(DOSSIER + r"\watersec.db")

@st.cache_data
def get_dataset_metadata(_df_core, _df_gym):
    """Extrait les métadonnées réelles — dates min/max, clients, cabines."""
    meta = {"customers": [], "date_min": None, "date_max": None,
            "cabines": ["cabin_1","cabin_2","cabin_3","cabin_4"]}
    # Clients
    customers = []
    for df in [_df_core, _df_gym]:
        if "customer_id" in df.columns:
            customers.extend(df["customer_id"].dropna().unique().tolist())
    meta["customers"] = sorted(list(set(customers)))
    # Dates (colonne 'jour' dans les deux datasets)
    all_dates = []
    for df in [_df_core, _df_gym]:
        col = "jour" if "jour" in df.columns else None
        if col:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(parsed):
                all_dates.extend(parsed.tolist())
    if all_dates:
        meta["date_min"] = min(all_dates).date()
        meta["date_max"] = max(all_dates).date()
    return meta

try:
    df_core, df_gym = charger_donnees()
    con             = charger_db()
    client          = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and "COLLE" not in GROQ_API_KEY else None
    DATA_OK         = True
    META            = get_dataset_metadata(df_core, df_gym)
except Exception as e:
    st.error(f"❌ Erreur chargement données : {e}")
    df_core = df_gym = con = client = None
    DATA_OK = False
    META    = {"customers": [], "date_min": None, "date_max": None,
               "cabines": ["cabin_1","cabin_2","cabin_3","cabin_4"]}

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTÉO
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_meteo():
    try:
        url    = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": "Tunis,TN", "appid": OPENWEATHER_KEY, "units": "metric", "lang": "fr"}
        r      = requests.get(url, params=params, timeout=5)
        data   = r.json()
        if r.status_code != 200:
            return None
        return {"temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidite":    data["main"]["humidity"],
                "vent_kmh":    round(data["wind"]["speed"] * 3.6, 1)}
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL ANTI-HALLUCINATION
# ═══════════════════════════════════════════════════════════════════════════════

def guardrail_check(question, meta):
    """
    Vérifie que la question est dans le périmètre des données disponibles.
    Retourne (ok: bool, message: str).
    """
    if not meta or not meta.get("date_min"):
        return True, ""

    q = question.lower()
    data_min = meta["date_min"]
    data_max = meta["date_max"]

    # ── Vérification des dates ────────────────────────────────────────────────
    year_mentions = re.findall(r'\b(20\d{2})\b', question)
    for yr_str in year_mentions:
        yr = int(yr_str)
        if yr < data_min.year or yr > data_max.year:
            return False, (
                f"⚠️ **Hors périmètre temporel**\n\n"
                f"Je ne possède pas ces informations dans les datasets actuels.\n\n"
                f"📅 **Période disponible :** {data_min} → {data_max}\n"
                f"❌ **Année demandée :** {yr}\n\n"
                f"Veuillez reformuler avec une date dans la plage disponible."
            )

    # ── Vérification des numéros de cabines (1-4 uniquement) ─────────────────
    cabin_nums = re.findall(r'cabine?\s*(\d+)|cabin\s*(\d+)', q)
    for match in cabin_nums:
        num = int(match[0] or match[1])
        if num < 1 or num > 4:
            return False, (
                f"⚠️ **Cabine inexistante**\n\n"
                f"Je ne possède pas ces informations dans les datasets actuels.\n\n"
                f"🚿 **Cabines disponibles :** 1, 2, 3, 4\n"
                f"❌ **Cabine demandée :** {num}"
            )

    # ── Vérification des clients ──────────────────────────────────────────────
    known = [c.lower() for c in meta.get("customers", [])]
    cust_mentions = re.findall(r'customer\s*([a-zA-Z0-9]+)', q)
    for m in cust_mentions:
        full = f"customer{m}".lower()
        if full not in known:
            return False, (
                f"⚠️ **Client inconnu**\n\n"
                f"Je ne possède pas ces informations dans les datasets actuels.\n\n"
                f"🏢 **Clients disponibles :** {', '.join(meta.get('customers', []))}\n"
                f"❌ **Client demandé :** customer{m}"
            )

    return True, ""

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS RAPIDES DYNAMIQUES (dates réelles du dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def get_quick_prompts(meta):
    """Génère des prompts rapides cadrés sur les vraies dates du dataset."""
    dmin = meta.get("date_min")
    dmax = meta.get("date_max")

    if dmin and dmax:
        # Dates relatives ancrées sur les vraies données
        tw_ago = max(dmax - timedelta(14), dmin)
        mo_ago = max(dmax - timedelta(30), dmin)
        tw_s   = tw_ago.strftime("%d/%m/%Y")
        mo_s   = mo_ago.strftime("%d/%m/%Y")
        dx_s   = dmax.strftime("%d/%m/%Y")
    else:
        tw_s = mo_s = dx_s = "disponible"

    return [
        f"Compare cabine 1 et 2 du gym entre {tw_s} et {dx_s}",
        "Quelle source a la plus haute conso eau chaude ?",
        "Patterns comportementaux résidentiel customerC",
        f"Anomalies chasse d'eau résidentielle depuis {mo_s}",
        "Y a-t-il des fuites détectées ?",
        f"Graphique tendance eau froide gym du {tw_s} au {dx_s}",
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION D'INTENT — PARSER DÉTERMINISTE
# ═══════════════════════════════════════════════════════════════════════════════

def parser_intent(question):
    """
    Analyse la question et retourne un dict d'intent.
    Utilisé comme fallback quand Groq est indisponible.
    """
    q = question.lower()

    # Détection graphique
    graph_kw = ["graphique","graph","plot","chart","courbe","histogramme","camembert",
                "visualis","tendance","évolution","evolution","montre","compare","diagramme",
                "figure","dessine","visualize","show me","plot me"]
    wants_graph = any(k in q for k in graph_kw)

    # Détection tableau
    table_kw = ["tableau","table","liste","données brutes","affiche","show","données"]
    wants_table = any(k in q for k in table_kw) and not wants_graph

    # Intent principal
    if any(k in q for k in ["gym","cabine","douche","cabin","kf","shower"]):
        intent = "gym"
    elif any(k in q for k in ["anomalie","fuite","alerte","anormal","inhabituel","suspect","leak"]):
        intent = "anomalie"
    elif any(k in q for k in ["pattern","comportement","sequence","habitude","recurrent"]):
        intent = "pattern"
    elif any(k in q for k in ["meteo","temperature","chaud","froid","ramadan","weather","saison"]):
        intent = "meteo"
    elif any(k in q for k in ["compare","tous","sites","global","total","all","highest","source"]):
        intent = "all_sites"
    else:
        intent = "residential"

    return {"intent": intent, "wants_graph": wants_graph, "wants_table": wants_table}

# ═══════════════════════════════════════════════════════════════════════════════
# 7 FONCTIONS PATTERNS PANDAS (calculs 100% exacts — zéro hallucination)
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_moment_journee(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("moment_journee")["consommation_ml"].agg(
        total="sum", moyenne="mean", nb_mesures="count").reset_index()
    result["total_litres"]   = (result["total"]   / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    return result[["moment_journee","total_litres","moyenne_litres","nb_mesures"]].sort_values("total_litres", ascending=False)

def pattern_jour_semaine(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("jour")["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["total_litres"]   = (result["total"]   / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    ordre = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    result["ordre"] = result["jour"].map({j: i for i, j in enumerate(ordre)})
    return result[["jour","total_litres","moyenne_litres"]].sort_values("ordre" if "ordre" in result.columns else "total_litres")

def detecter_anomalies(customer_id=None, seuil=2.5):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    df["mean"]   = df.groupby("device")["consommation_ml"].transform("mean")
    df["std"]    = df.groupby("device")["consommation_ml"].transform("std")
    df["zscore"] = (df["consommation_ml"] - df["mean"]) / df["std"].replace(0, 1)
    anomalies    = df[df["zscore"].abs() > seuil].copy()
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
    return pd.DataFrame(Counter(sequences).most_common(10), columns=["sequence","occurrences"])

def pattern_saison(customer_id=None):
    df = df_core.copy()
    if customer_id:
        df = df[df["customer_id"] == customer_id]
    result = df.groupby("saison")["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["total_litres"]   = (result["total"]   / 1000).round(2)
    result["moyenne_litres"] = (result["moyenne"] / 1000).round(2)
    return result[["saison","total_litres","moyenne_litres"]].sort_values("total_litres", ascending=False)

def pattern_gym_frequentation():
    df = df_gym.copy()
    if "frequentation_gym" in df.columns:
        df["frequentation_gym"] = pd.to_numeric(df["frequentation_gym"], errors="coerce")
    result = df.groupby("moment_journee").agg(
        conso_totale_litres=("consommation_ml", lambda x: round(x.sum()/1000, 2)),
        nb_sessions=("consommation_ml", "count")
    ).reset_index()
    if "frequentation_gym" in df.columns:
        freq = df.groupby("moment_journee")["frequentation_gym"].mean().round(1).reset_index()
        result = result.merge(freq, on="moment_journee", how="left")
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
# TOOLS GROQ — 7 tools (+ generate_graph)
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
            "description": "SQL libre sur customerKF (Gym 4 cabines). Colonnes: customer_id, site_type, device, consommation_ml, statut_alerte, timestamp, heure, jour, jour_semaine, mois, annee, moment_journee, saison, frequentation_gym, type_utilisation_douche, utilisateurs_actifs_30min, temps_pause_cabine_sec, statut_rotation, nb_utilisations_jour, debit_par_periode. Noms cabines: cabin_1_cold, cabin_1_hot... Diviser consommation_ml par 1000.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_all_sites",
            "description": "SQL libre sur tous les sites réunis. Colonnes: customer_id, site_type, device, consommation_ml, statut_alerte, jour, moment_journee, saison, annee, mois, heure, debit_par_periode. Diviser consommation_ml par 1000.",
            "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_graph",
            "description": "OBLIGATOIRE quand l'utilisateur demande : graphique, graph, plot, chart, courbe, histogramme, camembert, visualisation, tendance, évolution, montre-moi, compare, figure, dessine. Génère un graphique Plotly interactif. NE JAMAIS répondre en texte quand un graphique est demandé.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql":          {"type": "string", "description": "SQL SELECT. Ex: SELECT jour, ROUND(SUM(consommation_ml)/1000.0,2) as L FROM gym GROUP BY jour ORDER BY jour"},
                    "chart_type":   {"type": "string", "enum": ["line","bar","pie","scatter","box"], "description": "line=tendance temporelle, bar=comparaison, pie=proportion, scatter=corrélation, box=distribution"},
                    "title":        {"type": "string"},
                    "x_column":     {"type": "string"},
                    "y_column":     {"type": "string"},
                    "color_column": {"type": "string", "description": "Optionnel: colonne pour les séries"}
                },
                "required": ["sql","chart_type","title","x_column","y_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_patterns_comportementaux",
            "description": "Patterns comportementaux: séquences flush→sink, moments journée, jours semaine, saisons, fréquentation gym.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_pattern": {"type": "string", "enum": ["sequences","moment","jour","saison","gym"]},
                    "customer_id":  {"type": "string"}
                },
                "required": ["type_pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies_avancees",
            "description": "Anomalies Z-score et fuites d'eau par device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_detection": {"type": "string", "enum": ["anomalies","fuites"]},
                    "customer_id":    {"type": "string"}
                },
                "required": ["type_detection"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meteo_correlation",
            "description": "Météo Tunis temps réel + calendrier tunisien (Ramadan, Aïd, fêtes). Pour météo, température, Ramadan, saisons, corrélations.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# EXÉCUTION DES TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def executer_tool(nom, arguments, return_df=False):
    """
    Exécute un tool.
    - return_df=False : retourne string pour Groq tool_call
    - return_df=True  : retourne dict {"df": DataFrame|None, "text": str}
    """

    # ── SQL tools ────────────────────────────────────────────────────────────
    if nom in ["query_residential", "query_gym", "query_all_sites"]:
        sql = arguments.get("sql", "")
        try:
            df_result = con.execute(sql).df()
            if "device" in df_result.columns:
                df_result["device"] = df_result["device"].map(DEVICE_MAP).fillna(df_result["device"])
            text = df_result.to_string(index=False)
            if return_df:
                return {"df": df_result, "text": text, "sql": sql}
            return text
        except Exception as e:
            err = f"❌ Erreur SQL : {str(e)}"
            if return_df:
                return {"df": None, "text": err, "sql": sql}
            return err

    # ── Graphique ─────────────────────────────────────────────────────────────
    elif nom == "generate_graph":
        sql       = arguments.get("sql", "")
        chart     = arguments.get("chart_type", "bar")
        title     = arguments.get("title", "Graphique WaterSec")
        x_col     = arguments.get("x_column", "")
        y_col     = arguments.get("y_column", "")
        color_col = arguments.get("color_column", None)
        try:
            df_result = con.execute(sql).df()
            if "device" in df_result.columns:
                df_result["device"] = df_result["device"].map(DEVICE_MAP).fillna(df_result["device"])
            # Validation colonnes
            if x_col not in df_result.columns:
                x_col = df_result.columns[0]
            if y_col not in df_result.columns:
                y_col = df_result.columns[1] if len(df_result.columns) > 1 else df_result.columns[0]
            if color_col and color_col not in df_result.columns:
                color_col = None

            if chart == "line":
                fig = px.line(df_result, x=x_col, y=y_col, color=color_col, title=title, markers=True)
            elif chart == "bar":
                fig = px.bar(df_result, x=x_col, y=y_col, color=color_col or x_col, title=title, text=y_col)
            elif chart == "pie":
                fig = px.pie(df_result, names=x_col, values=y_col, title=title)
            elif chart == "scatter":
                fig = px.scatter(df_result, x=x_col, y=y_col, color=color_col, title=title)
            elif chart == "box":
                fig = px.box(df_result, x=x_col, y=y_col, title=title)
            else:
                fig = px.bar(df_result, x=x_col, y=y_col, title=title)

            fig.update_layout(template="plotly_white", height=450, title_x=0.5)
            return "__GRAPH__" + json.dumps(fig.to_dict(), ensure_ascii=False)
        except Exception as e:
            return f"❌ Erreur graphique : {str(e)}"

    # ── Patterns ──────────────────────────────────────────────────────────────
    elif nom == "detect_patterns_comportementaux":
        type_p  = arguments.get("type_pattern", "sequences")
        cust_id = arguments.get("customer_id", None)
        try:
            if   type_p == "sequences": res = pattern_sequences_residentielles()
            elif type_p == "moment":    res = pattern_moment_journee(cust_id)
            elif type_p == "jour":      res = pattern_jour_semaine(cust_id)
            elif type_p == "saison":    res = pattern_saison(cust_id)
            elif type_p == "gym":       res = pattern_gym_frequentation()
            else:                       res = pattern_moment_journee(cust_id)
            if return_df:
                return {"df": res, "text": res.to_string(index=False), "sql": None}
            return res.to_string(index=False)
        except Exception as e:
            err = f"❌ Erreur pattern : {str(e)}"
            if return_df: return {"df": None, "text": err, "sql": None}
            return err

    # ── Anomalies / Fuites ────────────────────────────────────────────────────
    elif nom == "detect_anomalies_avancees":
        type_d  = arguments.get("type_detection", "anomalies")
        cust_id = arguments.get("customer_id", None)
        try:
            res = detecter_fuites() if type_d == "fuites" else detecter_anomalies(cust_id)
            if return_df:
                return {"df": res, "text": res.to_string(index=False), "sql": None}
            return res.to_string(index=False)
        except Exception as e:
            err = f"❌ Erreur détection : {str(e)}"
            if return_df: return {"df": None, "text": err, "sql": None}
            return err

    # ── Météo + Calendrier ────────────────────────────────────────────────────
    elif nom == "get_meteo_correlation":
        try:
            meteo       = get_meteo()
            auj         = str(date.today())
            evenement   = CALENDRIER_TN.get(auj, "Jour ordinaire")
            prochains   = {k: v for k, v in CALENDRIER_TN.items() if k >= auj}
            prochain    = list(prochains.items())[0] if prochains else None
            if meteo:
                insight = "Forte demande eau chaude 🌡️" if meteo["temperature"] < 18 else "Consommation eau froide élevée 💧"
                return (f"🌤️ Météo Tunis ({auj}) :\n"
                        f"🌡️ {meteo['temperature']}°C — {meteo['description']}\n"
                        f"💧 Humidité : {meteo['humidite']}%  |  🌬️ Vent : {meteo['vent_kmh']} km/h\n"
                        f"📅 Événement : {evenement}\n"
                        f"📆 Prochain : {prochain[1] if prochain else 'Aucun'} ({prochain[0] if prochain else ''})\n"
                        f"💡 Insight : {insight}\n"
                        f"📊 Corrélation : En hiver (< 15°C) conso eau chaude +40%")
            return f"Météo indisponible. Événement du jour : {evenement}"
        except Exception as e:
            return f"❌ Erreur météo : {str(e)}"

    return "❌ Tool non reconnu"

# ═══════════════════════════════════════════════════════════════════════════════
# FORÇAGE GRAPHIQUE (bypass si Groq ignore generate_graph)
# ═══════════════════════════════════════════════════════════════════════════════

def forcer_generate_graph(question, intent):
    """
    Génère un graphique Python directement sans passer par Groq.
    Appelé quand Groq répond en texte alors qu'un graphique était demandé.
    """
    q = question.lower()

    # Sélection de la table
    if intent == "gym" or any(k in q for k in ["gym","cabine","cabin"]):
        table, group_col = "gym", "device"
    elif intent == "residential":
        table, group_col = "residential", "customer_id"
    else:
        table, group_col = "all_sites", "customer_id"

    # Dimension de regroupement
    if any(k in q for k in ["tendance","time","jour","journée","daily","evolution","over time"]):
        group_col, chart_type = "jour", "line"
        title = "Tendance journalière de consommation"
    elif any(k in q for k in ["moment","matin","soir","nuit","midi"]):
        group_col, chart_type = "moment_journee", "bar"
        title = "Consommation par moment de la journée"
    elif any(k in q for k in ["saison","été","hiver","printemps","automne"]):
        group_col, chart_type = "saison", "pie"
        title = "Répartition par saison"
    elif any(k in q for k in ["camembert","pie","proportion","répartition"]):
        chart_type = "pie"
        title = f"Répartition consommation — {table}"
    else:
        chart_type = "bar"
        title = f"Consommation par {group_col}"

    sql = (f"SELECT {group_col}, ROUND(SUM(consommation_ml)/1000.0, 2) AS total_L "
           f"FROM {table} WHERE consommation_ml > 0 "
           f"GROUP BY {group_col} ORDER BY total_L DESC LIMIT 30")

    result = executer_tool("generate_graph", {
        "sql": sql, "chart_type": chart_type, "title": title,
        "x_column": group_col, "y_column": "total_L"
    })
    return result, sql

# ═══════════════════════════════════════════════════════════════════════════════
# NARRATION OLLAMA (fallback offline)
# ═══════════════════════════════════════════════════════════════════════════════

def narrer_avec_ollama(model, question, data):
    """Narre des données pré-calculées. Zéro hallucination — données fournies uniquement."""
    prompt = f"""Tu es WaterSec AI, expert en analyse de consommation d'eau IoT en Tunisie.

DONNÉES CALCULÉES (100% exactes — utilise UNIQUEMENT ces chiffres) :
{data}

Question : {question}

RÈGLES ABSOLUES :
1. N'utilise QUE les chiffres présents dans les données ci-dessus.
2. Si une donnée est absente, réponds : "Je ne possède pas ces informations dans les datasets actuels."
3. Ne jamais extrapoler ou inventer.

Benchmarks eau Tunisie :
- Gym : 120L/cabine/jour | Flush : 6L | Lavabo : 15L/jour | Wudu : ~5L
- Eau chaude hiver (< 15°C) : +40% vs été

Réponds en français, maximum 5 phrases, avec 1 recommandation actionnable."""

    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": model, "prompt": prompt, "stream": False,
                                "options": {"temperature": 0.1, "num_ctx": 4096}},
                          timeout=60)
        r.raise_for_status()
        return r.json()["response"]
    except Exception as e:
        raise Exception(f"Ollama {model} indisponible : {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT GROQ (allégé — économie tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def build_system_prompt():
    dmin = str(META.get("date_min", "N/A"))
    dmax = str(META.get("date_max", "N/A"))
    customers = ", ".join(META.get("customers", ["customerA","customerB","customerC","customerKF"]))
    return f"""Tu es WaterSec AI, expert IoT eau Tunisie. Détecte la langue et réponds DANS LA MÊME LANGUE.

DONNÉES : {dmin} → {dmax} | Clients : {customers}
- residential: customerA/B/C | gym: customerKF (4 cabines hot/cold)
- Cabines : cabin_1_cold, cabin_1_hot, cabin_2_cold, cabin_2_hot, cabin_3_cold, cabin_3_hot, cabin_4_cold, cabin_4_hot

OUTILS :
→ query_residential/query_gym/query_all_sites : SQL libre (consommation_ml ÷ 1000 = L)
→ generate_graph : OBLIGATOIRE si graphique/plot/chart/courbe/tendance/visualis/compare
→ detect_patterns_comportementaux : patterns séquentiels
→ detect_anomalies_avancees : anomalies/fuites
→ get_meteo_correlation : météo + calendrier

RÈGLES :
1. Graphique demandé → TOUJOURS appeler generate_graph (jamais de texte)
2. ZÉRO hallucination : si résultat SQL vide → "Je ne possède pas ces informations dans les datasets actuels."
3. SQL → toujours diviser consommation_ml par 1000
4. Tendance temporelle → chart_type=line | Comparaison → bar | Proportion → pie

SQL EXEMPLES :
-- Gym par cabine: SELECT device, ROUND(SUM(consommation_ml)/1000.0,2) as L FROM gym GROUP BY device ORDER BY L DESC
-- Global: SELECT customer_id, ROUND(SUM(consommation_ml)/1000.0,2) as L FROM all_sites GROUP BY 1 ORDER BY 2 DESC
-- Tendance: SELECT jour, ROUND(SUM(consommation_ml)/1000.0,2) as L FROM gym GROUP BY jour ORDER BY jour"""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT GROQ (avec retry)
# ═══════════════════════════════════════════════════════════════════════════════

def agent_groq_call(question, historique=[]):
    """
    Appelle Groq. Retourne (content, outils_utilises, is_text_only).
    is_text_only=True si Groq a répondu sans appeler d'outil.
    """
    messages = [{"role": "system", "content": build_system_prompt()}]
    for h in historique[-6:]:   # Limiter l'historique pour économiser les tokens
        messages.append(h)
    messages.append({"role": "user", "content": question})

    outils_utilises = []
    MAX_LOOPS = 5

    for _ in range(MAX_LOOPS):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            max_tokens=2048
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content, outils_utilises, True

        messages.append(message)
        for tc in message.tool_calls:
            nom      = tc.function.name
            args     = json.loads(tc.function.arguments)
            resultat = executer_tool(nom, args)
            outils_utilises.append({"tool": nom, "args": args,
                                     "result_preview": str(resultat)[:200]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": resultat})

    # Demande de narration finale
    response_final = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0,
        max_tokens=2048
    )
    return response_final.choices[0].message.content, outils_utilises, False

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT PRINCIPAL — Routeur Universel (3 niveaux de fallback)
# ═══════════════════════════════════════════════════════════════════════════════
# Retourne : {"type": str, ...}, outils: list, mode: str
#
#   type="text"     → {"content": str}
#   type="graph"    → {"figure": go.Figure, "sql": str}
#   type="table"    → {"dataframe": DataFrame, "sql": str}
#   type="dual"     → {"content": str, "dataframe": DataFrame, "sql": str}
#   type="guardrail"→ {"message": str}
#   type="error"    → {"message": str}
# ═══════════════════════════════════════════════════════════════════════════════

def agent(question, historique=[]):

    # ── ÉTAPE 0 : Guardrail anti-hallucination ────────────────────────────────
    ok, guard_msg = guardrail_check(question, META)
    if not ok:
        return {"type": "guardrail", "message": guard_msg}, [], "🛡️ Guardrail"

    # ── Analyse d'intent (utilisé par fallback offline) ───────────────────────
    parsed      = parser_intent(question)
    intent      = parsed["intent"]
    wants_graph = parsed["wants_graph"]
    wants_table = parsed["wants_table"]

    # ── ÉTAPE 1 : GROQ LLaMA 3.3-70B ─────────────────────────────────────────
    if client:
        try:
            content, outils, is_text_only = agent_groq_call(question, historique)

            # Cas A : generate_graph appelé → afficher le graphique
            if outils and any(o["tool"] == "generate_graph" for o in outils):
                for o in outils:
                    if o["tool"] == "generate_graph":
                        result_str = executer_tool("generate_graph", o["args"])
                        if isinstance(result_str, str) and result_str.startswith("__GRAPH__"):
                            try:
                                fig = go.Figure(json.loads(result_str[9:]))
                                return {"type": "graph", "figure": fig,
                                        "sql": o["args"].get("sql","")}, outils, "🚀 Groq LLaMA 3.3-70B"
                            except Exception:
                                pass

            # Cas B : tool SQL appelé → dual (narration + DataFrame)
            if outils and any(o["tool"].startswith("query_") for o in outils):
                sql_tool = next(o for o in outils if o["tool"].startswith("query_"))
                typed    = executer_tool(sql_tool["tool"], sql_tool["args"], return_df=True)
                df_res   = typed.get("df") if typed else None
                sql_used = typed.get("sql","") if typed else ""
                if wants_graph and df_res is not None:
                    # Forcer graphique si demandé
                    graph_str, g_sql = forcer_generate_graph(question, intent)
                    if isinstance(graph_str, str) and graph_str.startswith("__GRAPH__"):
                        try:
                            fig = go.Figure(json.loads(graph_str[9:]))
                            return {"type": "graph", "figure": fig, "sql": g_sql}, outils, "🚀 Groq + Auto-Graph"
                        except Exception:
                            pass
                if df_res is not None and len(df_res) > 0:
                    return {"type": "dual", "content": content, "dataframe": df_res,
                            "sql": sql_used}, outils, "🚀 Groq LLaMA 3.3-70B"
                return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas C : tool patterns/anomalies → dual
            if outils and any(o["tool"] in ["detect_patterns_comportementaux",
                                             "detect_anomalies_avancees"] for o in outils):
                pt = next(o for o in outils if o["tool"] in
                          ["detect_patterns_comportementaux","detect_anomalies_avancees"])
                typed  = executer_tool(pt["tool"], pt["args"], return_df=True)
                df_res = typed.get("df") if typed else None
                if df_res is not None and len(df_res) > 0:
                    return {"type": "dual", "content": content, "dataframe": df_res,
                            "sql": None}, outils, "🚀 Groq LLaMA 3.3-70B"
                return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

            # Cas D : Groq a répondu en texte alors qu'un graphique était demandé
            if is_text_only and wants_graph:
                graph_str, g_sql = forcer_generate_graph(question, intent)
                if isinstance(graph_str, str) and graph_str.startswith("__GRAPH__"):
                    try:
                        fig = go.Figure(json.loads(graph_str[9:]))
                        return {"type": "graph", "figure": fig,
                                "sql": g_sql}, [{"tool": "forcer_generate_graph", "args": {}}], "🚀 Groq + Forçage Graph"
                    except Exception:
                        pass

            # Cas E : réponse texte pure
            return {"type": "text", "content": content}, outils, "🚀 Groq LLaMA 3.3-70B"

        except Exception as e:
            err_str = str(e)
            if "rate_limit" in err_str.lower() or "429" in err_str:
                st.warning("⚠️ Quota Groq atteint — passage mode offline...")
            else:
                st.error(f"❌ Groq : {err_str}")

    # ── ÉTAPE 2 : FALLBACK OFFLINE — graphique direct ─────────────────────────
    if wants_graph:
        try:
            graph_str, g_sql = forcer_generate_graph(question, intent)
            if isinstance(graph_str, str) and graph_str.startswith("__GRAPH__"):
                fig = go.Figure(json.loads(graph_str[9:]))
                return {"type": "graph", "figure": fig,
                        "sql": g_sql}, [{"tool": "offline_graph", "args": {}}], "🟡 Offline — Graph Direct"
        except Exception:
            pass

    # ── ÉTAPE 2b : FALLBACK OFFLINE — données + Ollama ───────────────────────
    def offline_data():
        try:
            return executer_tool("query_all_sites", {
                "sql": "SELECT customer_id, ROUND(SUM(consommation_ml)/1000.0,2) AS L FROM all_sites GROUP BY 1 ORDER BY 2 DESC"
            })
        except:
            return "Données indisponibles."

    # ── ÉTAPE 3 : Ollama Qwen2.5:14b ─────────────────────────────────────────
    try:
        data    = offline_data()
        content = narrer_avec_ollama("qwen2.5:14b", question, data)
        return {"type": "text", "content": content}, [], "🟡 Local Qwen2.5:14b"
    except Exception:
        pass

    # ── ÉTAPE 4 : Ollama Qwen2.5:7b ──────────────────────────────────────────
    try:
        data    = offline_data()
        content = narrer_avec_ollama("qwen2.5:7b", question, data)
        return {"type": "text", "content": content}, [], "⚠️ Local Qwen2.5:7b (Fallback)"
    except Exception as e:
        return {"type": "error",
                "message": f"❌ Tous les modèles sont indisponibles.\n\n```\n{traceback.format_exc()}\n```"
                }, [], "❌ Hors ligne"

# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES DASHBOARD (4 graphiques fixes — colonnes droites)
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
                  title="Consommation moyenne par moment",
                  color="moment_journee",
                  color_discrete_sequence=px.colors.qualitative.Pastel)

def graph_saison():
    result = df_core.groupby("saison")["consommation_ml"].sum().reset_index()
    result["Litres"] = (result["consommation_ml"] / 1000).round(2)
    return px.pie(result, names="saison", values="Litres",
                  title="Répartition par saison",
                  color_discrete_sequence=px.colors.qualitative.Safe)

def graph_cabines_gym():
    col = "device_name" if "device_name" in df_gym.columns else "device"
    result = df_gym.groupby(col)["consommation_ml"].agg(total="sum", moyenne="mean").reset_index()
    result["Total (L)"]   = (result["total"]   / 1000).round(2)
    result["Moyenne (L)"] = (result["moyenne"] / 1000).round(2)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Total par cabine","Moyenne par cabine"))
    fig.add_trace(go.Bar(x=result[col], y=result["Total (L)"],
                         marker_color="#1E88E5", name="Total"), row=1, col=1)
    fig.add_trace(go.Bar(x=result[col], y=result["Moyenne (L)"],
                         marker_color="#43A047", name="Moyenne"), row=1, col=2)
    fig.update_layout(title="Comparaison cabines Gym", showlegend=False,
                      xaxis_tickangle=-45, xaxis2_tickangle=-45)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT — Structure identique à l'original
# ═══════════════════════════════════════════════════════════════════════════════

if not DATA_OK:
    st.error("❌ Impossible de charger les données. Vérifiez les fichiers CSV et DuckDB.")
    st.stop()

# ── Titre + badge mode ────────────────────────────────────────────────────────
st.title("💧 WaterSec AI Agent")
st.markdown("**Agent conversationnel intelligent pour l'analyse de consommation d'eau IoT**")

# Badge mode actif + plage de données
badge_cols = st.columns([1, 2, 2])
with badge_cols[0]:
    if client:
        st.success("🟢 Groq Online")
    else:
        st.warning("🟡 Mode Offline")
with badge_cols[1]:
    if META.get("date_min") and META.get("date_max"):
        st.info(f"📅 Données : {META['date_min']} → {META['date_max']}")
with badge_cols[2]:
    if META.get("customers"):
        st.info(f"🏢 Clients : {', '.join(META['customers'])}")

st.divider()

# ── Métriques temps réel ──────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
# CHAT CONVERSATIONNEL
# ═══════════════════════════════════════════════════════════════════════════════

with col_chat:
    st.subheader("💬 Chat avec WaterSec AI")

    if "messages" not in st.session_state:
        st.session_state.messages   = []
        st.session_state.historique = []

    # ── Prompts rapides dynamiques (dates réelles du dataset) ─────────────────
    st.markdown("**Prompts rapides :**")
    prompts_rapides = get_quick_prompts(META)
    cols = st.columns(3)
    for i, prompt in enumerate(prompts_rapides):
        label = prompt[:34] + "…" if len(prompt) > 35 else prompt
        if cols[i % 3].button(label, key=f"btn_{i}", use_container_width=True):
            st.session_state.input_rapide = prompt

    # ── Affichage historique du chat ──────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            mtype = msg.get("type", "text")
            if mtype == "graph" and "figure" in msg:
                st.plotly_chart(msg["figure"], use_container_width=True)
            elif mtype == "table" and "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)
            elif mtype == "dual":
                if msg.get("content"):
                    st.markdown(msg["content"])
                if "dataframe" in msg and msg["dataframe"] is not None:
                    st.dataframe(msg["dataframe"], use_container_width=True)
            elif mtype == "guardrail":
                st.info(msg.get("content", ""))
            elif mtype == "error":
                st.error(msg.get("content", ""))
            else:
                st.markdown(msg.get("content", ""))

    # ── Zone de saisie ────────────────────────────────────────────────────────
    input_val = st.session_state.get("input_rapide", "")
    question  = st.chat_input("Posez votre question sur les données water...")

    if not question and input_val:
        question = input_val
        st.session_state.input_rapide = ""

    # ── Traitement d'une nouvelle question ────────────────────────────────────
    if question:
        st.session_state.messages.append({"role": "user", "content": question, "type": "text"})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyse en cours..."):
                result, outils, mode = agent(question, st.session_state.historique)

            # Badge du mode utilisé
            result_type = result.get("type", "text")
            if result_type == "error":
                st.error(f"📡 {mode}")
            elif result_type == "guardrail":
                st.info(f"🛡️ {mode}")
            elif "Local" in mode or "Offline" in mode or "Fallback" in mode:
                st.warning(f"📡 {mode}")
            else:
                st.success(f"📡 {mode}")

            # ── Rendu adaptatif selon le type ─────────────────────────────────
            if result_type == "graph":
                fig = result.get("figure")
                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("📊 Graphique généré dynamiquement")
                    except Exception as e:
                        st.error(f"Erreur rendu graphique : {e}")
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")
                else:
                    st.error("Erreur lors de la génération du graphique")

            elif result_type == "table":
                df_r = result.get("dataframe")
                if df_r is not None:
                    try:
                        st.dataframe(df_r, use_container_width=True)
                        st.caption(f"📋 {len(df_r)} lignes")
                    except Exception as e:
                        st.error(f"Erreur rendu tableau : {e}")
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")

            elif result_type == "dual":
                # Narration + DataFrame
                if result.get("content"):
                    st.markdown(result["content"])
                df_r = result.get("dataframe")
                if df_r is not None and len(df_r) > 0:
                    try:
                        st.dataframe(df_r, use_container_width=True)
                        st.caption(f"📋 {len(df_r)} lignes")
                    except Exception as e:
                        st.warning(f"Tableau non affichable : {e}")
                    if result.get("sql"):
                        with st.expander("🔍 SQL utilisé"):
                            st.code(result["sql"], language="sql")

            elif result_type == "guardrail":
                st.info(result.get("message", "Je ne possède pas ces informations dans les datasets actuels."))

            elif result_type == "error":
                st.error(result.get("message", "Erreur inconnue"))

            else:  # text
                st.markdown(result.get("content", ""))

            # ── Outils utilisés ───────────────────────────────────────────────
            if outils:
                with st.expander("🔧 Outils utilisés"):
                    for o in outils:
                        st.write(f"**{o['tool']}**")
                        args = o.get("args", {})
                        if isinstance(args, dict) and "sql" in args:
                            st.code(args["sql"], language="sql")
                        elif isinstance(args, dict):
                            st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")

        # Stockage dans l'historique
        msg_store = {"role": "assistant", "type": result_type}
        if result_type == "graph":
            msg_store["figure"]  = result.get("figure")
            msg_store["content"] = "[Graphique généré]"
        elif result_type == "table":
            msg_store["dataframe"] = result.get("dataframe")
            msg_store["content"]   = f"[Tableau : {len(result.get('dataframe', pd.DataFrame()))} lignes]"
        elif result_type == "dual":
            msg_store["content"]   = result.get("content", "")
            msg_store["dataframe"] = result.get("dataframe")
        elif result_type == "guardrail":
            msg_store["content"] = result.get("message", "")
        elif result_type == "error":
            msg_store["content"] = result.get("message", "Erreur")
        else:
            msg_store["content"] = result.get("content", "")

        st.session_state.messages.append(msg_store)
        st.session_state.historique.append({"role": "user",      "content": question})
        st.session_state.historique.append({"role": "assistant", "content": msg_store.get("content", "")})

        # Limiter l'historique Groq (économie tokens)
        if len(st.session_state.historique) > 20:
            st.session_state.historique = st.session_state.historique[-20:]

# ═══════════════════════════════════════════════════════════════════════════════
# TABLEAUX DE BORD — Graphiques fixes (colonne droite)
# ═══════════════════════════════════════════════════════════════════════════════

with col_graphs:
    st.subheader("📊 Tableaux de bord")
    tab1, tab2, tab3, tab4 = st.tabs(["Par client", "Par moment", "Par saison", "Gym"])
    with tab1:
        try:
            st.plotly_chart(graph_conso_par_client(), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")
    with tab2:
        try:
            st.plotly_chart(graph_moment_journee(), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")
    with tab3:
        try:
            st.plotly_chart(graph_saison(), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")
    with tab4:
        try:
            st.plotly_chart(graph_cabines_gym(), use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique : {e}")