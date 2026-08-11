import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página
st.set_page_config(
    page_title="WTA Match Predictor",
    page_icon="🎾",
    layout="centered"
)

# Cargar modelo y datos
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_rf_wta.pkl')
    datos = joblib.load('datos_wta.pkl')
    return modelo, datos

modelo, df = cargar_recursos()

# Función de stats históricas
def stats_historicas(nombre, datos, n_ultimos=20):
    como_w = datos[datos['winner_name'] == nombre]
    como_l = datos[datos['loser_name'] == nombre]
    
    partidos_w = como_w.tail(n_ultimos)
    partidos_l = como_l.tail(n_ultimos)
    
    total = len(partidos_w) + len(partidos_l)
    if total < 5:
        return None
    
    win_rate = len(partidos_w) / total
    aces = (partidos_w['w_ace'].sum() + partidos_l['l_ace'].sum()) / total
    dfs = (partidos_w['w_df'].sum() + partidos_l['l_df'].sum()) / total
    bp_saved = partidos_w['w_bpSaved'].sum() + partidos_l['l_bpSaved'].sum()
    bp_faced = partidos_w['w_bpFaced'].sum() + partidos_l['l_bpFaced'].sum()
    bp_pct = bp_saved / bp_faced if bp_faced > 0 else 0.5
    
    # Ranking más reciente
    rank_w = datos[datos['winner_name'] == nombre]['winner_rank']
    rank_l = datos[datos['loser_name'] == nombre]['loser_rank']
    rank = pd.concat([rank_w, rank_l]).iloc[-1] if len(pd.concat([rank_w, rank_l])) > 0 else 999
    
    return {
        'win_rate': win_rate,
        'aces': aces,
        'df': dfs,
        'bp_pct': bp_pct,
        'rank': int(rank),
        'partidos': total
    }

# Obtener lista de jugadoras con suficientes datos
@st.cache_data
def obtener_jugadoras():
    ganados = df['winner_name'].value_counts()
    perdidos = df['loser_name'].value_counts()
    total = (ganados.add(perdidos, fill_value=0)).astype(int).sort_values(ascending=False)
    return total[total >= 20].index.tolist()

jugadoras = obtener_jugadoras()

# ============================================
# INTERFAZ
# ============================================

st.title("🎾 WTA Match Predictor")
st.markdown("*Predicción de partidos basada en ranking, forma reciente y estadísticas de servicio*")

st.divider()

# Selectores
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    jugadora_1 = st.selectbox("Jugadora 1", jugadoras, index=jugadoras.index('Aryna Sabalenka'))

with col2:
    st.markdown("<h2 style='text-align: center; padding-top: 25px;'>vs</h2>", unsafe_allow_html=True)

with col3:
    jugadora_2 = st.selectbox("Jugadora 2", jugadoras, index=jugadoras.index('Coco Gauff'))

superficie = st.radio("Superficie", ['Hard', 'Clay', 'Grass'], horizontal=True)

st.divider()

# Predicción
if jugadora_1 == jugadora_2:
    st.warning("Seleccioná dos jugadoras diferentes")
else:
    stats_1 = stats_historicas(jugadora_1, df)
    stats_2 = stats_historicas(jugadora_2, df)
    
    if stats_1 is None or stats_2 is None:
        st.error("No hay suficientes datos para alguna de las jugadoras")
    else:
        # Armar features
        features = pd.DataFrame([{
            'rank_1': stats_1['rank'],
            'rank_2': stats_2['rank'],
            'rank_diff': stats_1['rank'] - stats_2['rank'],
            'wr_diff': stats_1['win_rate'] - stats_2['win_rate'],
            'aces_diff': stats_1['aces'] - stats_2['aces'],
            'bp_diff': stats_1['bp_pct'] - stats_2['bp_pct'],
            'surface_Hard': 1 if superficie == 'Hard' else 0,
            'surface_Grass': 1 if superficie == 'Grass' else 0
        }])
        
        prob_1 = modelo.predict_proba(features)[0][1]
        prob_2 = 1 - prob_1
        
        # Mostrar resultado
        nombre_1 = jugadora_1.split()[-1]
        nombre_2 = jugadora_2.split()[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            color_1 = "#0F6E56" if prob_1 > prob_2 else "#94A3B8"
            st.markdown(f"""
            <div style='text-align:center; padding:20px; background:{color_1}; 
                        border-radius:15px; color:white;'>
                <h2>{nombre_1}</h2>
                <h1>{prob_1*100:.1f}%</h1>
                <p>Ranking #{stats_1['rank']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color_2 = "#0F6E56" if prob_2 > prob_1 else "#94A3B8"
            st.markdown(f"""
            <div style='text-align:center; padding:20px; background:{color_2}; 
                        border-radius:15px; color:white;'>
                <h2>{nombre_2}</h2>
                <h1>{prob_2*100:.1f}%</h1>
                <p>Ranking #{stats_2['rank']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Barra de probabilidad
        st.progress(prob_1)
        
        # Comparación de stats
        st.divider()
        st.subheader("📊 Comparación de stats")
        
        stats_df = pd.DataFrame({
            'Stat': ['Ranking', 'Win Rate (últ 20)', 'Aces/partido', 'Doble faltas/partido', 'BP Salvados %'],
            nombre_1: [f"#{stats_1['rank']}", f"{stats_1['win_rate']*100:.1f}%", 
                      f"{stats_1['aces']:.1f}", f"{stats_1['df']:.1f}", f"{stats_1['bp_pct']*100:.1f}%"],
            nombre_2: [f"#{stats_2['rank']}", f"{stats_2['win_rate']*100:.1f}%",
                      f"{stats_2['aces']:.1f}", f"{stats_2['df']:.1f}", f"{stats_2['bp_pct']*100:.1f}%"]
        })
        
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Footer
        st.divider()
        st.caption("📊 Modelo: Random Forest (AUC: 68.0%) | Datos: WTA 2020-2026 | Por: Saray Garcia")