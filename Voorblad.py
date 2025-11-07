import streamlit as st
from streamlit_option_menu import option_menu

# ------------------------------------------------------------
# 🌊 CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Titanic verbetercase",
    page_icon="🚢",
    layout="wide"
)

# ------------------------------------------------------------
# 🧭 PAGINA 1 (introductie, géén sidebar-menu)
# ------------------------------------------------------------
st.title("🚢 Titanic Verbetercase — Team 17")
st.caption("Matthijs Bakker — Visual Analytics Eindopdracht")

st.header("🔄 Beknopte verbeteringen")
st.markdown("""
- **Data bewerking en prepocessing**: leeftijdscategorieën, titels, familievariabelen, NAN-waardes invullen, etc.  
- **Data Visualisatie**: Gebruik van Plotly voor interactiviteit.  
- **Statistische onderbouwing**: Twee modellen getest (KNN en lineaire regressie).  
- **Lay-out**: Overzichtelijke structuur met kleur, consistentie en interactiviteit.
""")

st.markdown("---")

# ------------------------------------------------------------
# 🧭 SIDEBAR MENU VOOR OVERIGE PAGINA’S
# ------------------------------------------------------------
with st.sidebar:
    st.title("🧭 Inhoudsopgave")
    pagina = option_menu(
        menu_title=None,
        options=["Nieuwe variabelen", "Visualisaties", "Statistiek en modellen"],
        icons=['search', 'bar-chart-line', 'graph-up-arrow'],
        menu_icon='list-ul',
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"}
        }
    )

# ------------------------------------------------------------
# 📄 PAGINA-INHOUD
# ------------------------------------------------------------
if pagina == "Nieuwe variabelen":
    st.header("🧩 Nieuwe variabelen")
    st.write(
        "- **Originele case:** Geen verdere kolommen toegevoegd.\n"
        "- **Verbeterde versie:** Meerdere nieuwe kolommen toegevoegd:\n"
        "   - **Leeftijdscategorie** (gesplitst in 5)\n"
        "   - **Type** (man/vrouw/kind)\n"
        "   - **Alleen reizen** (wel/geen familie aan boord)\n"
        "   - **Familie** (aantal familieleden)\n"
        "   - **Familiegrootte** (categorie klein/groot)\n"
        "   - **Titel** (afgeleid uit naam)"
    )

elif pagina == "Visualisaties":
    st.header("📊 Visualisaties")
    st.write(
        "- **Originele case:** Statische plots en geen correlatiematrix.\n"
        "- **Verbeterde versie:**\n"
        "   - Correlatiematrix is toegevoegd.\n"
        "   - Interactieve plots met Plotly.\n"
        "   - Groter aantal grafieken.\n"
        "   - Consistent kleuren palette."
    )

elif pagina == "Statistiek en modellen":
    st.header("📈 Statistiek en modellen")
    st.write(
        "Bij de **eerste versie** werd geen model direct toegepast; beslissingen werden gemaakt op basis van eenvoudige grafieken en percentages. "
        "Daarmee werd een **Kaggle-score van 78,47 %** gehaald.\n\n"
        "In de **tweede versie** zijn en twee algoritmen getest:\n"
        "- **K-Nearest Neighbors (KNN)** → 76,08 %\n"
        "- **Lineaire regressie** → 77,03 %\n\n"
        "Hoewel de score uiteindelijk **niet verbeterde**, waren de modellen wél waardevol om beter te begrijpen "
        "**welke variabelen invloed hadden op de overlevingskans** en om de dataset statistisch te valideren."
    )

# ------------------------------------------------------------
# 🏁 AFSLUITING
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 — Team 17 | Matt Bak | Visual Analytics Eindopdracht Titanic")

