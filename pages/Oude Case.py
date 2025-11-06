import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

## opzet
st.set_page_config(
    page_title="Eerste vaart",
    page_icon="❄️",
    layout="wide"
)
st.title('Oorspronkelijke Titanic Case')


# data inladen
@st.cache_data
def load_train_old():
    train_old = pd.read_csv('train.csv')
    return train_old

train_old = load_train_old()

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
#--------------------------------------------------------------------------------------------------------#

if pagina == 'Data Verkenning':
    st.header('1. Data verkenning')
    st.write("De originele Titanic dataset:")

    st.dataframe(train_old.head())
    st.write("Statistische beschrijving van de dataset:")
    st.write(train_old.describe())

    # ------------------------------------------------------------
    # 📊 Crosstabs: Overlevingspercentages en frequenties
    # ------------------------------------------------------------
    st.subheader("1.2 Overlevingspercentages per categorie")

    df1 = pd.crosstab(train_old['Sex'], train_old['Survived'], normalize='index', dropna=True)
    df2 = pd.crosstab(train_old['Embarked'], train_old['Pclass'], normalize='index', dropna=True)
    df3 = pd.crosstab([train_old['Sex'], train_old['SibSp']], train_old['Survived'], normalize='index', dropna=True)
    df4 = pd.crosstab([train_old['Sex'], train_old['Parch']], train_old['Survived'], normalize='index', dropna=True)

    df5 = pd.crosstab(train_old['Sex'], train_old['SibSp'], dropna=True)
    df6 = pd.crosstab(train_old['Sex'], train_old['Parch'], dropna=True)

    # Toon tabellen naast elkaar met Streamlit columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sex vs Survival**")
        st.dataframe(df1.style.format("{:.2f}"))
    with col2:
        st.markdown("**Embarked vs Pclass**")
        st.dataframe(df2.style.format("{:.2f}"))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Sex & SibSp vs Survival**")
        st.dataframe(df3.style.format("{:.2f}"))
    with col4:
        st.markdown("**Sex & Parch vs Survival**")
        st.dataframe(df4.style.format("{:.2f}"))

    st.markdown("### Frequenties per combinatie")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Sex & SibSp counts**")
        st.dataframe(df5)
    with col6:
        st.markdown("**Sex & Parch counts**")
        st.dataframe(df6)


elif pagina == 'Analyse':
    st.header('2. Analyse')

    overleden_overleefd_gem = train_old['Survived'].mean() * 100
    st.write(
        f'Wat was uberhaupt de overlevingskans?  \nOveral gezien is er een overlevingskans van: {np.round(overleden_overleefd_gem, 2)}%')
    st.write('Hieronder worden verschillende verbanden tussen kenmerken en overleving bekeken.')

    # --- Leeftijd vs Survival ---
    st.subheader('2.1 Leeftijdsverdeling met survival overlay')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=train_old, x="Age", hue="Survived", palette=["red", "green"], multiple="stack", bins=20, ax=ax)
    ax.set_xlabel("Leeftijd")
    ax.set_ylabel("Aantal passagiers")
    ax.set_title("Leeftijdsverdeling met survival overlay")
    st.pyplot(fig)

    # --- Age & Sex survival ---
    st.subheader('2.2 Overlevingskans per leeftijd en geslacht')
    survival_by_age_gender = train_old.dropna(subset=['Age', 'Sex']).groupby(['Sex', 'Age']).agg(
        SurvivalRate=('Survived', 'mean'), Count=('Survived', 'count')
    ).reset_index().assign(SurvivalRate=lambda df: df['SurvivalRate'] * 100)
    survival_by_age_gender_weighted = survival_by_age_gender.loc[
        survival_by_age_gender.index.repeat(survival_by_age_gender['Count'])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(
        data=survival_by_age_gender_weighted[survival_by_age_gender_weighted['Sex'] == 'female'],
        x='Age', y='SurvivalRate', order=6,
        scatter_kws={'alpha': 0.35}, line_kws={'color': 'purple'}, ci=None, ax=axes[0]
    )
    axes[0].set_title('Overlevingskans per leeftijd (vrouw)')
    axes[0].set_xlabel('Leeftijd')
    axes[0].set_ylabel('Overlevingspercentage (%)')
    axes[0].set_ylim(0, 100)

    sns.regplot(
        data=survival_by_age_gender_weighted[survival_by_age_gender_weighted['Sex'] == 'male'],
        x='Age', y='SurvivalRate', order=6,
        scatter_kws={'alpha': 0.35}, line_kws={'color': 'blue'}, ci=None, ax=axes[1]
    )
    axes[1].set_title('Overlevingskans per leeftijd (man)')
    axes[1].set_xlabel('Leeftijd')
    axes[1].set_ylabel('Overlevingspercentage (%)')
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    st.pyplot(fig)

    # --- Age & Pclass survival ---
    st.subheader('2.3 Leeftijdsverdeling per ticketklasse met survival overlay')
    classes = sorted(train_old["Pclass"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 4))
    for i, c in enumerate(classes):
        subset = train_old[train_old["Pclass"] == c]
        axes[i].hist(
            [subset[subset["Survived"] == 0]["Age"], subset[subset["Survived"] == 1]["Age"]],
            bins=30, stacked=False, label=["Niet overleefd", "Overleefd"], color=["red", "green"]
        )
        axes[i].set_title(f"Ticketklasse {c}")
        axes[i].set_xlabel("Leeftijd")
        if i == 0:
            axes[i].set_ylabel("Aantal passagiers")
        axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)

    # --- Sex & Pclass survival ---
    st.subheader('2.4 Overlevingskans per geslacht en ticketklasse')
    survival_rate = train_old.dropna(subset=['Pclass', 'Sex']).groupby(["Pclass", "Sex"]).agg(
        SurvivalRate=("Survived", "mean")
    ).reset_index().assign(SurvivalRate=lambda df: df['SurvivalRate'] * 100)
    survival_rate['Sex_label'] = survival_rate['Sex'].map({'male': 'Man', 'female': 'Vrouw'})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Pclass", y="SurvivalRate", hue="Sex_label", palette={"Vrouw": "purple", "Man": "blue"},
                data=survival_rate, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Ticketklasse (Pclass)")
    ax.set_ylabel("Overlevingspercentage (%)")
    ax.set_title("Overlevingskans per geslacht en ticketklasse")
    st.pyplot(fig)

    # --- Overleden per leeftijd in ticketklasse 1 ---
    st.subheader('2.5 Overleden per geslacht in ticketklasse 1')
    gestorven = train_old[train_old['Survived'] == 0]
    overleed_1 = gestorven[gestorven['Pclass'] == 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    overleed_1[overleed_1['Sex'] == 'male']['Age'].hist(color='blue', alpha=0.6, ax=ax)
    overleed_1[overleed_1['Sex'] == 'female']['Age'].hist(color='purple', alpha=0.6, ax=ax)
    ax.set_xlabel('Leeftijd')
    ax.set_ylabel('Aantal passagiers')
    ax.set_title('Overleden per geslacht in ticketklasse 1')
    ax.legend(['Man', 'Vrouw'])
    st.pyplot(fig)

    # --- Embarked & Pclass survival ---
    st.subheader('2.6 Overlevingskans per opstapplaats & ticketklasse')
    survival_by_embarked = train_old.dropna(subset=["Embarked", "Pclass"]).groupby(["Embarked", "Pclass"]).agg(
        SurvivalRate=("Survived", "mean")
    ).reset_index().assign(SurvivalRate=lambda df: df['SurvivalRate'] * 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=survival_by_embarked, x='Embarked', y='SurvivalRate', hue='Pclass',
                palette=["gold", "silver", "#CD7F32"], errorbar=None, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Opstapplaats")
    ax.set_ylabel("Overlevingspercentage (%)")
    ax.set_title("Overlevingskans per opstapplaats & ticketklasse")
    st.pyplot(fig)

    # --- Fare mannen survival ---
    st.subheader('2.7 Ticketprijs van mannen vs survival')
    train_old_male = train_old[train_old['Sex'] == 'male'].copy()
    train_old_male['Fare_log'] = np.log1p(train_old_male['Fare'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=train_old_male, x='Survived', y='Fare_log', ax=axes[0])
    axes[0].set_xlabel("Survived")
    axes[0].set_ylabel("Log(Fare+1)")
    axes[0].set_title("Verdeling van Fare per overleving voor mannen")

    sns.violinplot(data=train_old_male, x='Survived', y='Fare', density_norm='width', ax=axes[1])
    axes[1].set_ylim(0, 300)
    axes[1].set_xlabel("Survived")
    axes[1].set_ylabel("Fare")
    axes[1].set_title("Verdeling van Fare per overleving voor mannen")

    plt.tight_layout()
    st.pyplot(fig)


elif pagina == 'Predictief Model':
    st.header('Predictief model met ruleset')

    st.write(
        'Op basis van onze analyse hebben we een eenvoudige ruleset opgesteld om te voorspellen wie de Titanic heeft overleefd. De regels zijn als volgt:')

    st.subheader('Regels voor vrouwen')
    st.markdown("""
    - Vrouwen in de 1e en 2e klas overleven sowieso.
    - Vrouwen in de 3e klas overleven alleen als ze jonger zijn dan 40 **en** opstappen in Cherbourg (C).
    """)

    st.subheader('Regels voor mannen')
    st.markdown("""
    - Mannen onder de 10 jaar overleven sowieso.
    - Mannen in de 1e klas onder de 18 overleven.
    - Mannen die tussen 125 en 200 betalen voor hun ticket overleven.
    - Mannen die meer dan 275 betalen voor hun ticket overleven.
    - Mannen in de 2e klas onder de 16 overleven.
    - Mannen ouder dan 75 overleven.
    """)

    st.write('Met deze keuzes kwam de kaggle score uit op: 78,47%')