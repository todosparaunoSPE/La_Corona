# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:31:51 2025

@author: jahop
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Simular datos
np.random.seed(42)
categorias = ['Jabón de tocador', 'Detergente', 'Shampoo', 'Suavizante', 'Limpiador']
regiones = ['CDMX', 'Jalisco', 'Nuevo León', 'Puebla', 'Veracruz']
canales = ['Autoservicio', 'Mayoreo', 'Tiendas propias']

data = pd.DataFrame({
    'Mes': pd.date_range(start='2023-01-01', periods=180, freq='M').repeat(1),
    'Categoría': np.random.choice(categorias, 180),
    'Región': np.random.choice(regiones, 180),
    'Canal': np.random.choice(canales, 180),
    'Ventas (Millones MXN)': np.random.uniform(5, 50, 180).round(2)
})

# Título
st.title("📊 Inteligencia Comercial - La Corona")
st.markdown("Presentación de capacidades como *Analista y Científico de Datos* para la industria de productos de limpieza")

# Filtros
col1, col2, col3 = st.columns(3)
with col1:
    categoria_sel = st.selectbox("🧼 Categoría:", data['Categoría'].unique())
with col2:
    region_sel = st.selectbox("🌎 Región:", ['Todas'] + list(data['Región'].unique()))
with col3:
    canal_sel = st.selectbox("🏪 Canal:", ['Todos'] + list(data['Canal'].unique()))

# Aplicar filtros
filtro = data[data['Categoría'] == categoria_sel]
if region_sel != 'Todas':
    filtro = filtro[filtro['Región'] == region_sel]
if canal_sel != 'Todos':
    filtro = filtro[filtro['Canal'] == canal_sel]

# Gráfica de ventas por región
fig = px.bar(filtro, x='Región', y='Ventas (Millones MXN)', color='Canal',
             title=f"📍 Ventas por Región y Canal - {categoria_sel}", barmode='group')
st.plotly_chart(fig)

# Gráfica de pastel por canal
st.subheader("📊 Participación por Canal de Distribución")
pie_data = filtro.groupby('Canal')['Ventas (Millones MXN)'].sum().reset_index()
fig_pie = px.pie(pie_data, names='Canal', values='Ventas (Millones MXN)',
                 title="Distribución de Ventas por Canal", hole=0.4)
st.plotly_chart(fig_pie)

# Predicción de tendencia de ventas
st.subheader("📈 Predicción de Ventas")
df_pred = filtro.copy()
df_pred['Mes_num'] = df_pred['Mes'].dt.month + 12 * (df_pred['Mes'].dt.year - df_pred['Mes'].dt.year.min())
X = df_pred[['Mes_num']]
y = df_pred['Ventas (Millones MXN)']
model = LinearRegression().fit(X, y)
df_pred['Predicción'] = model.predict(X)

fig2 = px.line(df_pred, x='Mes', y=['Ventas (Millones MXN)', 'Predicción'],
               title=f"Tendencia y Predicción de Ventas - {categoria_sel}")
st.plotly_chart(fig2)

# Resumen estadístico
st.subheader("📌 Resumen de Ventas")
total_ventas = filtro['Ventas (Millones MXN)'].sum()
promedio = filtro['Ventas (Millones MXN)'].mean()
maximo = filtro['Ventas (Millones MXN)'].max()
minimo = filtro['Ventas (Millones MXN)'].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Ventas", f"${total_ventas:,.2f} M")
col2.metric("📅 Promedio", f"${promedio:,.2f} M")
col3.metric("📈 Máximo", f"${maximo:,.2f} M")
col4.metric("📉 Mínimo", f"${minimo:,.2f} M")

# Mostrar tabla interactiva
st.subheader("📋 Datos Detallados")
st.dataframe(filtro.sort_values(by='Mes', ascending=False), use_container_width=True)

# Descargar datos
st.markdown("### 📥 Descargar Datos")
st.download_button("Descargar CSV", data.to_csv(index=False), file_name="ventas_la_corona.csv")

# Perfil profesional
st.markdown("---")
st.markdown("### 👨‍💼 Sobre mí")
st.info("""
Mi nombre es **Javier Horacio Pérez Ricárdez**, analista y científico de datos con experiencia en el sector de manufactura y consumo. 
Especialista en:
- Análisis de tendencias de mercado
- Inteligencia de negocios
- Visualización de datos con Python y Streamlit
- Modelado predictivo

Estoy interesado en colaborar con empresas como La Corona para impulsar decisiones basadas en datos.
""")
