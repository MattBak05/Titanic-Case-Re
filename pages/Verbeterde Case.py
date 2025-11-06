import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium

# Pagina configuratie
st.set_page_config(
    page_title="Verbeterde Titanic Case",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("⚓ Titanic Overlevingsanalyse Dashboard")

# Sidebar navigatie
with st.sidebar:
    st.title("🧭 Inhoudsopgave")
    pagina = option_menu(
        menu_title=None,
        options=["Data Verkenning", "Analyse", "Predictief Model"],
        icons=['search', 'bar-chart-line', 'graph-up-arrow'],
        menu_icon='ship',
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"}
        }
    )


# Data laden functies
@st.cache_data
def laad_training_data():
    url = 'https://raw.githubusercontent.com/MattBak05/Titanic-Case-Re/main/train.csv'
    df = pd.read_csv(url)
    df = df.drop(columns=['Ticket', 'Cabin', 'PassengerId'], errors='ignore')

    # --- Ontbrekende waarden opvullen ---
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # meestal 'S'

    # --- Leeftijdscategorieën ---
    def categoriseer_leeftijd(age):
        if age < 16:
            return 'Kind'
        elif age < 35:
            return 'Jong Volwassene'
        elif age < 50:
            return 'Volwassene'
        elif age < 65:
            return 'Middelbare Leeftijd'
        else:
            return 'Senior'

    df['Leeftijdsgroep'] = df['Age'].apply(categoriseer_leeftijd)

    # --- Familie relaties ---
    df['Familiegrootte'] = df['SibSp'] + df['Parch'] + 1
    df['Reist_Alleen'] = (df['Familiegrootte'] == 1).astype(int)
    df['Heeft_Familie'] = (df['Parch'] + df['SibSp'] > 0).astype(int)
    df['Kleine_Familie'] = (df['Familiegrootte'] < 5).astype(int)

    # --- Titels uit namen ---
    df['Titel'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    titel_mapping = {
        'Mr': 'Meneer', 'Miss': 'Juffrouw', 'Mrs': 'Mevrouw',
        'Master': 'Jongeheer', 'Dr': 'Zeldzaam', 'Rev': 'Zeldzaam',
        'Col': 'Zeldzaam', 'Major': 'Zeldzaam', 'Mlle': 'Juffrouw',
        'Countess': 'Zeldzaam', 'Ms': 'Juffrouw', 'Lady': 'Mevrouw',
        'Jonkheer': 'Zeldzaam', 'Don': 'Zeldzaam', 'Dona': 'Zeldzaam',
        'Mme': 'Mevrouw', 'Capt': 'Zeldzaam', 'Sir': 'Zeldzaam'
    }
    df['Titel'] = df['Titel'].map(titel_mapping).fillna('Zeldzaam')

    # --- Instaplocaties ---
    locatie_mapping = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    df['Instaplocatie'] = df['Embarked'].map(locatie_mapping)

    # --- Passagier type ---
    def bepaal_type(row):
        return 'Kind' if row['Age'] < 16 else row['Sex']

    df['Passagier_Type'] = df.apply(bepaal_type, axis=1)

    # --- Tariefklasse ---
    df['Tariefklasse'] = pd.qcut(df['Fare'], q=4, labels=['Laag', 'Gemiddeld', 'Hoog', 'Luxe'])

    return df



@st.cache_data
def laad_test_data():
    url = 'https://raw.githubusercontent.com/MattBak05/Titanic-Case-Re/main/test.csv'
    df = pd.read_csv(url)

    # Dezelfde transformaties als training data
    df['Age'] = df['Age'].fillna(df['Age'].median())

    def categoriseer_leeftijd(age):
        if age < 16:
            return 'Kind'
        elif age < 35:
            return 'Jong Volwassene'
        elif age < 50:
            return 'Volwassene'
        elif age < 65:
            return 'Middelbare Leeftijd'
        else:
            return 'Senior'

    df['Leeftijdsgroep'] = df['Age'].apply(categoriseer_leeftijd)
    df['Familiegrootte'] = df['SibSp'] + df['Parch'] + 1
    df['Reist_Alleen'] = (df['Familiegrootte'] == 1).astype(int)
    df['Heeft_Familie'] = (df['Parch'] + df['SibSp'] > 0).astype(int)
    df['Kleine_Familie'] = (df['Familiegrootte'] < 5).astype(int)

    df['Titel'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    titel_mapping = {
        'Mr': 'Meneer', 'Miss': 'Juffrouw', 'Mrs': 'Mevrouw',
        'Master': 'Jongeheer', 'Dr': 'Zeldzaam', 'Rev': 'Zeldzaam',
        'Col': 'Zeldzaam', 'Major': 'Zeldzaam', 'Mlle': 'Juffrouw',
        'Countess': 'Zeldzaam', 'Ms': 'Juffrouw', 'Lady': 'Mevrouw',
        'Jonkheer': 'Zeldzaam', 'Don': 'Zeldzaam', 'Dona': 'Zeldzaam',
        'Mme': 'Mevrouw', 'Capt': 'Zeldzaam', 'Sir': 'Zeldzaam'
    }
    df['Titel'] = df['Titel'].map(titel_mapping).fillna('Zeldzaam')

    locatie_mapping = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Instaplocatie'] = df['Embarked'].map(locatie_mapping)

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    def bepaal_type(row):
        if row['Age'] < 16:
            return 'Kind'
        return row['Sex']

    df['Passagier_Type'] = df.apply(bepaal_type, axis=1)

    df['Tariefklasse'] = pd.qcut(df['Fare'], q=4, labels=['Laag', 'Gemiddeld', 'Hoog', 'Luxe'])

    return df


# Data laden
train_data = laad_training_data()
test_data = laad_test_data()

# PAGINA 1: DATA EXPLORATIE
if pagina == "Data Verkenning":
    st.header("📊 Data Verkenning & Voorbereiding")

    # Overzicht metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Totaal Passagiers</h3>
                <h1 style='text-align: center; color: white;'>{}</h1>
            </div>
        """.format(len(train_data)), unsafe_allow_html=True)

    with col2:
        overlevingspercentage = (train_data['Survived'].sum() / len(train_data)) * 100
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Overlevingspercentage</h3>
                <h1 style='text-align: center; color: white;'>{:.1f}%</h1>
            </div>
        """.format(overlevingspercentage), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Gemiddelde Leeftijd</h3>
                <h1 style='text-align: center; color: white;'>{:.0f}</h1>
            </div>
        """.format(train_data['Age'].mean()), unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Klassen</h3>
                <h1 style='text-align: center; color: white;'>3</h1>
            </div>
        """.format(train_data['Pclass'].nunique()), unsafe_allow_html=True)

    st.markdown("---")

    # Dataset preview
    st.subheader("🔍 Dataset Preview")
    st.dataframe(train_data.head(10), use_container_width=True, height=400)

    st.markdown("---")

    # Ontbrekende waarden
    st.subheader("📋 Ontbrekende Waarden Analyse")

    ontbrekend = train_data.isnull().sum()
    ontbrekend = ontbrekend[ontbrekend > 0].reset_index()
    ontbrekend.columns = ['Variabele', 'Aantal Ontbrekend']
    ontbrekend['Percentage'] = (ontbrekend['Aantal Ontbrekend'] / len(train_data) * 100).round(2)

    if len(ontbrekend) > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ontbrekend['Variabele'],
            y=ontbrekend['Percentage'],
            marker=dict(
                color=ontbrekend['Percentage'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Percentage")
            ),
            text=ontbrekend['Percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Percentage Ontbrekende Waarden per Variabele",
            xaxis_title="Variabele",
            yaxis_title="Percentage Ontbrekend (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            height=550,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **Behandeling van ontbrekende waarden:**
        - **Leeftijd**: Opgevuld met de mediaan ({train_data['Age'].median():.1f} jaar) vanwege de symmetrische verdeling
        - **Instaplocatie**: Opgevuld met 'Southampton' als meest voorkomende locatie
        - Deze aanpak minimaliseert bias in de dataset
        """)
    else:
        st.success(f"""
        **Behandeling van ontbrekende waarden:**
        - **Leeftijd**: Opgevuld met de mediaan ({train_data['Age'].median():.1f} jaar) vanwege de symmetrische verdeling
        - **Instaplocatie**: Opgevuld met 'Southampton' als meest voorkomende locatie
        - Deze aanpak minimaliseert bias in de dataset
        """)

    st.markdown("---")

    # Feature Engineering
    st.subheader("🛠️ Feature Engineering Overzicht")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Nieuwe Features:**
        - 🎂 **Leeftijdsgroepen**: Categorieën op basis van levensfase
        - 👨‍👩‍👧‍👦 **Familiegrootte**: Totaal aantal familieleden aan boord
        - 🚶 **Reist Alleen**: Indicator voor solo reizigers
        - 👔 **Titel**: Sociale status uit naam geëxtraheerd
        """)

    with col2:
        st.markdown("""
        **Extra Dimensies:**
        - 🎭 **Passagier Type**: Kind/Man/Vrouw classificatie
        - 💰 **Tariefklasse**: Categorisatie van ticketprijzen
        - 📍 **Instaplocatie**: Haven van vertrek
        - 👥 **Kleine Familie**: Familie kleiner dan 5 personen
        """)

    # Voorbeeld van nieuwe features
    st.dataframe(
        train_data[['Age', 'Leeftijdsgroep', 'Familiegrootte', 'Reist_Alleen',
                    'Titel', 'Passagier_Type', 'Tariefklasse']].head(10),
        use_container_width=True,
        height=400
    )

    st.markdown("---")

    # Correlatie analyse
    st.subheader("🔗 Correlatie Matrix")

    # Numerieke correlaties
    numerieke_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                      'Familiegrootte', 'Reist_Alleen', 'Heeft_Familie', 'Kleine_Familie']
    correlatie_data = train_data[numerieke_cols].corr()['Survived'].sort_values(ascending=False)

    fig = go.Figure()

    colors = ['#00cc96' if x > 0 else '#ef553b' for x in correlatie_data.values]

    fig.add_trace(go.Bar(
        y=correlatie_data.index,
        x=correlatie_data.values,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{x:.3f}' for x in correlatie_data.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Correlatie: %{x:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Correlatie met Overlevingskans",
        xaxis_title="Pearson Correlatie Coëfficiënt",
        yaxis_title="",
        font=dict( size=12),
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Belangrijkste Correlaties:**
    - 🔴 **Negatieve correlatie**: Pclass, Familiegrootte - lagere kans op overleven
    - 🟢 **Positieve correlatie**: Fare, Heeft_Familie - hogere kans op overleven
    - 💡 **Insight**: Sociaal-economische status speelde een cruciale rol
    """)

# PAGINA 2: DIEPGAANDE ANALYSE
elif pagina == "Analyse":
    st.header("🔍 Diepgaande Analyse van Overlevingsfactoren")

    # Vul ontbrekende waarden op
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Instaplocatie'].fillna('Southampton', inplace=True)

    st.subheader("📈 Overlevingsratio's per Categorie")

    # Selectie van variabelen
    variabelen = st.multiselect(
        'Selecteer variabelen voor analyse:',
        ['Passagier_Type', 'Titel', 'Leeftijdsgroep', 'Reist_Alleen',
         'Instaplocatie', 'Pclass', 'Familiegrootte', 'Kleine_Familie', 'Tariefklasse'],
        default=['Passagier_Type', 'Titel', 'Pclass', 'Instaplocatie']
    )

    if len(variabelen) > 0:
        # Bereken aantal rijen en kolommen voor subplots
        n_vars = len(variabelen)
        n_cols = 2
        n_rows = (n_vars + 1) // 2

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=variabelen,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        for idx, var in enumerate(variabelen):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            # Bereken survival rate
            survival_rate = train_data.groupby(var)['Survived'].agg(['sum', 'count'])
            survival_rate['percentage'] = (survival_rate['sum'] / survival_rate['count'] * 100)
            survival_rate = survival_rate.sort_values('percentage', ascending=False)

            # Kleurenschema
            colors = ['#00cc96' if x >= 50 else '#ef553b' for x in survival_rate['percentage']]

            fig.add_trace(
                go.Bar(
                    x=survival_rate.index,
                    y=survival_rate['percentage'],
                    marker=dict(color=colors),
                    text=[f'{x:.1f}%' for x in survival_rate['percentage']],
                    textposition='outside',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>Overleefde: %{y:.1f}%<extra></extra>'
                ),
                row=row, col=col
            )

            fig.update_yaxes(range=[0, 100], row=row, col=col, title_text="Overleving (%)")

        fig.update_layout(
            height=400 * n_rows,

            font=dict( size=12),
            showlegend=False,
            title_text="Overlevingspercentages per Categorie"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Interactieve kruistabel analyse
    st.subheader("🔄 Kruistabel Analyse")

    col1, col2, col3 = st.columns(3)

    beschikbare_vars = ['Passagier_Type', 'Titel', 'Leeftijdsgroep', 'Reist_Alleen',
                        'Instaplocatie', 'Pclass', 'Tariefklasse', 'Kleine_Familie']

    with col1:
        var1 = st.selectbox("Primaire variabele:", beschikbare_vars, index=0)

    with col2:
        beschikbare_vars_2 = [v for v in beschikbare_vars if v != var1]
        var2 = st.selectbox("Secundaire variabele:", beschikbare_vars_2, index=0)

    with col3:
        beschikbare_vars_3 = [v for v in beschikbare_vars if v not in [var1, var2]]
        var3 = st.selectbox("Filter (optioneel):", ['Geen'] + beschikbare_vars_3)

    # Filter data indien nodig
    if var3 != 'Geen':
        filter_waarde = st.selectbox(f"Selecteer waarde voor {var3}:", train_data[var3].unique())
        gefilterde_data = train_data[train_data[var3] == filter_waarde]
    else:
        gefilterde_data = train_data

    # Bereken overlevingspercentages
    overleefden = gefilterde_data[gefilterde_data['Survived'] == 1].groupby([var1, var2]).size().reset_index(
        name='Overleefden')
    totaal = gefilterde_data.groupby([var1, var2]).size().reset_index(name='Totaal')
    merged = pd.merge(overleefden, totaal, on=[var1, var2])
    merged['Percentage'] = (merged['Overleefden'] / merged['Totaal'] * 100).round(1)

    # Visualisatie
    fig = px.bar(
        merged,
        x=var1,
        y='Percentage',
        color=var2,
        barmode='group',
        text='Percentage',
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Overleving: %{y:.1f}%<extra></extra>'
    )

    fig.update_layout(
        title=f"Overlevingspercentage: {var1} vs {var2}",
        xaxis_title=var1,
        yaxis_title="Overlevingspercentage (%)",
        yaxis=dict(range=[0, 105]),
        font=dict( size=14),
        legend=dict(
            title=var2,
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.info(f"""
    **Observaties uit de data:**
    - Hoogste overlevingskans: {merged.loc[merged['Percentage'].idxmax(), var1]} - {merged.loc[merged['Percentage'].idxmax(), var2]} ({merged['Percentage'].max():.1f}%)
    - Laagste overlevingskans: {merged.loc[merged['Percentage'].idxmin(), var1]} - {merged.loc[merged['Percentage'].idxmin(), var2]} ({merged['Percentage'].min():.1f}%)
    - Gemiddelde overlevingskans in deze analyse: {merged['Percentage'].mean():.1f}%
    """)

    st.markdown("---")

    # Distributie visualisaties
    st.subheader("📊 Distributie Analyses")

    tab1, tab2, tab3 = st.tabs(["Leeftijd", "Fare", "Familiegrootte"])

    with tab1:
        fig = go.Figure()

        for survived in [0, 1]:
            fig.add_trace(go.Violin(
                y=train_data[train_data['Survived'] == survived]['Age'],
                name='Overleefde' if survived else 'Niet Overleefde',
                box_visible=True,
                meanline_visible=True,
                fillcolor='#00cc96' if survived else '#ef553b',
                opacity=0.6
            ))

        fig.update_layout(
            title="Leeftijdsverdeling: Overleefden vs Niet-Overleefden",
            yaxis_title="Leeftijd",

            font=dict( size=14),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()

        for survived in [0, 1]:
            fig.add_trace(go.Box(
                y=train_data[train_data['Survived'] == survived]['Fare'],
                name='Overleefde' if survived else 'Niet Overleefde',
                marker_color='#00cc96' if survived else '#ef553b',
                boxmean='sd'
            ))

        fig.update_layout(
            title="Fare Verdeling: Overleefden vs Niet-Overleefden",
            yaxis_title="Fare (£)",

            font=dict( size=14),
            height=500,
            yaxis_type="log",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        familie_survival = train_data.groupby('Familiegrootte')['Survived'].agg(['sum', 'count'])
        familie_survival['rate'] = (familie_survival['sum'] / familie_survival['count'] * 100)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=familie_survival.index,
            y=familie_survival['rate'],
            mode='lines+markers',
            marker=dict(size=10, color='#636EFA'),
            line=dict(width=3, color='#636EFA'),
            text=[f'{x:.1f}%' for x in familie_survival['rate']],
            hovertemplate='<b>Familiegrootte: %{x}</b><br>Overleving: %{y:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title="Overlevingskans per Familiegrootte",
            xaxis_title="Familiegrootte",
            yaxis_title="Overlevingspercentage (%)",

            font=dict( size=14),
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ Overlevingskansen per Haven van Vertrek")

    # Gemiddelde overlevingskans en aantal passagiers per haven
    embarked_stats = (
        train_data.groupby("Instaplocatie")
        .agg(
            survival_rate=("Survived", "mean"),
            total_passengers=("Survived", "count"),
            avg_fare=("Fare", "mean"),
        )
        .reset_index()
    )

    # Coördinaten van de havens
    embarked_coords = {
        "Southampton": {"coords": [50.9097, -1.4044]},
        "Cherbourg": {"coords": [49.65, -1.62]},
        "Queenstown": {"coords": [51.85, -8.3]},
    }

    # Maak Folium-kaart
    m = folium.Map(location=[50.0, -5.0], zoom_start=4, tiles="cartodb positron")

    # Cirkels toevoegen op basis van data
    for _, row in embarked_stats.iterrows():
        locatie = row["Instaplocatie"]
        if locatie in embarked_coords:
            coords = embarked_coords[locatie]["coords"]
            rate = row["survival_rate"] * 100
            passengers = row["total_passengers"]

            popup = folium.Popup(
                f"<b>{locatie}</b><br>"
                f"Overlevingspercentage: {rate:.1f}%<br>"
                f"Aantal passagiers: {passengers}<br>"
                f"Gemiddelde ticketprijs: £{row['avg_fare']:.2f}",
                max_width=250,
            )

            kleur = "green" if rate > 50 else "orange" if rate > 30 else "red"

            folium.CircleMarker(
                location=coords,
                radius=8 + rate / 8,
                color=kleur,
                fill=True,
                fill_opacity=0.7,
                popup=popup,
            ).add_to(m)

    st.markdown("""
    Deze kaart toont de gemiddelde **overlevingskansen per haven van vertrek**.  
    - **Groenere en grotere cirkels** = hogere kans op overleving.  
    - **Roodere cirkels** = lagere kans.  
    """)

    st_folium(m, width=750, height=500)

# PAGINA 3: PREDICTIEF MODEL
elif pagina == "Predictief Model":
    st.header("🤖 Predictief Model - KNN & Lineaire Regressie")

    # Data voorbereiding
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    train_data['Instaplocatie'].fillna('Southampton', inplace=True)

    # One-hot encoding
    train_encoded = pd.get_dummies(train_data,
                                   columns=['Instaplocatie', 'Passagier_Type', 'Titel', 'Leeftijdsgroep',
                                            'Tariefklasse'],
                                   prefix=['Locatie', 'Type', 'Titel', 'Leeftijd', 'Tarief'])
    test_encoded = pd.get_dummies(test_data,
                                  columns=['Instaplocatie', 'Passagier_Type', 'Titel', 'Leeftijdsgroep',
                                           'Tariefklasse'],
                                  prefix=['Locatie', 'Type', 'Titel', 'Leeftijd', 'Tarief'])

    # Features selecteren
    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Familiegrootte',
                    'Reist_Alleen', 'Heeft_Familie', 'Kleine_Familie']

    # Voeg one-hot encoded columns toe
    for col in train_encoded.columns:
        if col.startswith(('Locatie_', 'Type_', 'Titel_', 'Leeftijd_', 'Tarief_')):
            feature_cols.append(col)

    X = train_encoded[feature_cols]
    y = train_encoded['Survived']

    st.subheader("📋 Model Voorbereiding")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Geselecteerde Features:**")
        st.write(X.head())

    with col2:
        st.markdown("**Feature Statistieken:**")
        st.dataframe(X.describe().T, height=350)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    st.info(f"""
    **Dataset Verdeling:**
    - Training set: {len(X_train)} samples
    - Test set: {len(X_test)} samples
    - Features: {len(feature_cols)}
    """)

    st.markdown("---")

    # KNN Model
    st.subheader("🎯 K-Nearest Neighbors (KNN) Model")

    # Schalen van features - GEBRUIK STANDARDSCALER zoals in de referentie repo
    scaler_knn = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler_knn.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_knn.transform(X_test), columns=X_test.columns)

    # Vind optimale K - test range 1-20 zoals in referentie
    k_range = range(1, 21)
    accuracies = []

    with st.spinner('Training KNN modellen met verschillende K-waarden...'):
        for k in k_range:
            # BASIS KNN zonder extra parameters zoals in referentie
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            predictions = knn.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)

    # Visualiseer K-optimalisatie
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=accuracies,
        mode='lines+markers',
        marker=dict(size=8, color='#636EFA'),
        line=dict(width=3, color='#636EFA'),
        hovertemplate='<b>K=%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>',
        showlegend=False
    ))

    # Markeer beste K
    best_k = k_range[accuracies.index(max(accuracies))]
    fig.add_trace(go.Scatter(
        x=[best_k],
        y=[max(accuracies)],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name=f'Best K={best_k}',
        hovertemplate=f'<b>Optimaal: K={best_k}</b><br>Accuracy: {max(accuracies):.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="KNN Model Prestatie voor Verschillende K-waarden",
        xaxis_title="Aantal Neighbors (K)",
        yaxis_title="Accuracy Score",
        font=dict(size=14),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Train finaal KNN model - gebruik beste K gevonden in de loop
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train_scaled, y_train)
    knn_predictions = final_knn.predict(X_test_scaled)
    knn_accuracy = accuracy_score(y_test, knn_predictions)

    # Resultaten weergeven
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Optimale K</h3>
                <h1 style='text-align: center; color: white;'>{best_k}</h1>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>KNN Accuracy</h3>
                <h1 style='text-align: center; color: white;'>{knn_accuracy:.2%}</h1>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        correct = np.sum(knn_predictions == y_test)
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Correcte Voorspellingen</h3>
                <h1 style='text-align: center; color: white;'>{correct}/{len(y_test)}</h1>
            </div>
        """, unsafe_allow_html=True)

    # Voorspellingen op test data
    st.markdown("#### 📤 Voorspellingen Genereren")

    # Zorg voor correcte kolommen in test data
    X_test_final = test_encoded.reindex(columns=feature_cols, fill_value=0)
    X_test_final_scaled = pd.DataFrame(scaler_knn.transform(X_test_final), columns=X_test_final.columns)

    test_predictions_knn = final_knn.predict(X_test_final_scaled)

    # Resultaten CSV - KAGGLE FORMAT
    resultaten_knn = pd.DataFrame({
        'PassengerId': test_data['PassengerId'].values,
        'Survived': test_predictions_knn
    })

    st.success("✅ Voorspellingen gegenereerd in Kaggle competitie formaat (PassengerId, Survived)")

    st.download_button(
        label="📥 Download KNN Voorspellingen (CSV voor Kaggle)",
        data=resultaten_knn.to_csv(index=False),
        file_name="titanic_knn_submission.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Lineaire Regressie
    st.subheader("📈 Lineaire Regressie Model")

    scaler_lr = StandardScaler()
    X_train_lr_scaled = pd.DataFrame(scaler_lr.fit_transform(X_train), columns=X_train.columns)
    X_test_lr_scaled = pd.DataFrame(scaler_lr.transform(X_test), columns=X_test.columns)

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr_scaled, y_train)

    # Voorspellingen
    lr_predictions_raw = lr_model.predict(X_test_lr_scaled)

    # Optimale threshold vinden
    thresholds = np.linspace(0.3, 0.7, 41)
    threshold_accuracies = []

    for threshold in thresholds:
        lr_pred_binary = (lr_predictions_raw >= threshold).astype(int)
        acc = accuracy_score(y_test, lr_pred_binary)
        threshold_accuracies.append(acc)

    # Visualiseer threshold optimalisatie
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=threshold_accuracies,
        mode='lines+markers',
        marker=dict(size=6, color='#EF553B'),
        line=dict(width=3, color='#EF553B'),
        hovertemplate='<b>Threshold: %{x:.2f}</b><br>Accuracy: %{y:.4f}<extra></extra>',
        showlegend=False
    ))

    best_threshold = thresholds[threshold_accuracies.index(max(threshold_accuracies))]
    fig.add_trace(go.Scatter(
        x=[best_threshold],
        y=[max(threshold_accuracies)],
        mode='markers',
        marker=dict(size=20, color='lime', symbol='star'),
        name=f'Best Threshold={best_threshold:.2f}',
        hovertemplate=f'<b>Optimaal: {best_threshold:.2f}</b><br>Accuracy: {max(threshold_accuracies):.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Lineaire Regressie: Threshold Optimalisatie",
        xaxis_title="Classification Threshold",
        yaxis_title="Accuracy Score",
        font=dict(size=14),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Finaal LR model resultaten
    lr_predictions_binary = (lr_predictions_raw >= best_threshold).astype(int)
    lr_accuracy = accuracy_score(y_test, lr_predictions_binary)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Optimale Threshold</h3>
                <h1 style='text-align: center; color: white;'>{best_threshold:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>LR Accuracy</h3>
                <h1 style='text-align: center; color: white;'>{lr_accuracy:.2%}</h1>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        correct_lr = np.sum(lr_predictions_binary == y_test)
        st.markdown(f"""
            <div class="metric-card">
                <h3 style='text-align: center; color: white;'>Correcte Voorspellingen</h3>
                <h1 style='text-align: center; color: white;'>{correct_lr}/{len(y_test)}</h1>
            </div>
        """, unsafe_allow_html=True)

    # Feature importance voor LR
    st.markdown("#### 🔍 Feature Importance (Lineaire Regressie)")

    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': lr_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False).head(15)

    fig = go.Figure()

    colors = ['#00cc96' if x > 0 else '#ef553b' for x in coef_df['Coefficient']]

    fig.add_trace(go.Bar(
        y=coef_df['Feature'],
        x=coef_df['Coefficient'],
        orientation='h',
        marker=dict(color=colors),
        text=coef_df['Coefficient'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Coefficient: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title="Top 15 Belangrijkste Features",
        xaxis_title="Coefficient Waarde",
        yaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # LR voorspellingen op test data
    X_test_final_lr = test_encoded.reindex(columns=feature_cols, fill_value=0)
    X_test_final_lr_scaled = pd.DataFrame(scaler_lr.transform(X_test_final_lr), columns=X_test_final_lr.columns)

    test_predictions_lr_raw = lr_model.predict(X_test_final_lr_scaled)
    test_predictions_lr = (test_predictions_lr_raw >= best_threshold).astype(int)

    # KAGGLE FORMAT
    resultaten_lr = pd.DataFrame({
        'PassengerId': test_data['PassengerId'].values,
        'Survived': test_predictions_lr
    })

    st.success("✅ Voorspellingen gegenereerd in Kaggle competitie formaat (PassengerId, Survived)")

    st.download_button(
        label="📥 Download Lineaire Regressie Voorspellingen (CSV voor Kaggle)",
        data=resultaten_lr.to_csv(index=False),
        file_name="titanic_lr_submission.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Model vergelijking
    st.subheader("⚖️ Model Vergelijking")

    vergelijking = pd.DataFrame({
        'Model': ['KNN', 'Lineaire Regressie'],
        'Accuracy': [knn_accuracy, lr_accuracy],
        'Correct': [correct, correct_lr],
        'Fout': [len(y_test) - correct, len(y_test) - correct_lr]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Accuracy',
        x=vergelijking['Model'],
        y=vergelijking['Accuracy'],
        text=[f'{x:.2%}' for x in vergelijking['Accuracy']],
        textposition='outside',
        marker_color=['#00cc96', '#ab63fa']
    ))

    fig.update_layout(
        title="Model Prestatie Vergelijking",
        yaxis_title="Accuracy Score",
        yaxis=dict(range=[0, 1]),
        font=dict(size=14),
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Conclusie
    beste_model = 'KNN' if knn_accuracy > lr_accuracy else 'Lineaire Regressie'
    beste_acc = max(knn_accuracy, lr_accuracy)

    st.success(f"""
    ### 🏆 Beste Model: **{beste_model}**

    **Performance:**
    - Accuracy: **{beste_acc:.2%}**
    - Dit model presteert **{abs(knn_accuracy - lr_accuracy):.2%}** beter dan het andere model

    **Aanbeveling:** Gebruik het {beste_model} model voor productie-voorspellingen op de Titanic dataset.
    """)


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🚢 Titanic Overlevingsanalyse Dashboard</p>
        <p>Data Bron: Kaggle Titanic Dataset | Gebouwd met Streamlit & Plotly</p>
    </div>

""", unsafe_allow_html=True)
