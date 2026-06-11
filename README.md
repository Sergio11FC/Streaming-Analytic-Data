# Streaming-Analytic-Data
Análisis de datos, plataformas de Streaming, Netflix, Amazon Prime, HBO, Disney plus
📺 Streaming Financial Dashboard

Dashboard interactivo en Python/Streamlit para análisis comparativo del desempeño financiero de Netflix, Amazon, Disney y HBO.


🚀 Instalación rápida

bash# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación
streamlit run app.py

La app se abre automáticamente en http://localhost:8501


📂 Estructura del proyecto

streaming_dashboard/
├── app.py              ← Aplicación principal
├── requirements.txt    ← Dependencias Python
├── sample_data.csv     ← Datos (2021–2025)
└── README.md           ← Este archivo


📋 Formato del CSV requerido

El archivo CSV debe contener exactamente estas columnas (los nombres deben coincidir):

ColumnaTipoDescripciónPlatformtextoNetflix, Amazon, Disney, HBOYearenteroAño fiscal (ej: 2023)QuarterenteroTrimestre: 1, 2, 3 o 4Total_RevenuedecimalIngresos totales (millones USD)Gross_ProfitdecimalUtilidad brutaOperating_IncomedecimalIngreso operativoEBITDAdecimalEBITDANet_IncomedecimalUtilidad netaOperating_ExpensedecimalGastos operativosPretax_IncomedecimalIngreso antes de impuestosTax_ProvisiondecimalProvisión de impuestosBasic_EPSdecimalGanancias por acción básicaDiluted_EPSdecimalGanancias por acción diluida


Los valores numéricos deben estar en millones de USD (sin comas como separadores de miles).




🧩 Funcionalidades del Dashboard

Pestañas

PestañaContenido💰 RentabilidadMargen Bruto, Operativo y EBITDA · Líneas temporales · Heatmap📈 CrecimientoQoQ Revenue Growth · Net Income Growth · Scatter Revenue vs Net Income🏗️ CostosOpEx Ratio · Tasa Impositiva Efectiva · Composición financiera⚔️ BenchmarkingRevenue Share · EPS Básico/Diluido · Radar competitivo · Tabla🔮 PredicciónRegresión lineal · Proyección 4 trimestres · Métricas R²/MAE/RMSE

Filtros (sidebar)


📺 Filtro por plataforma (multi-selección)
📅 Filtro por año
🗓️ Filtro por trimestre


KPIs superiores

Total Revenue · EBITDA · Net Income · Gross Margin · Operating Margin · Revenue Share Leader

Insights automáticos

5 insights generados dinámicamente basados en los datos filtrados.


☁️ Deploy en Streamlit Cloud


Sube el repositorio a GitHub
Ve a share.streamlit.io
Conecta tu repositorio y selecciona app.py
¡Listo!



📦 Dependencias principales


Streamlit ≥ 1.35 — Framework de interfaz web
Pandas ≥ 2.1 — Manipulación de datos
Plotly ≥ 5.20 — Gráficos interactivos
Scikit-learn ≥ 1.4 — Modelos de regresión lineal
NumPy ≥ 1.26 — Cómputo numérico
