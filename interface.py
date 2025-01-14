import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from streamlit_option_menu import option_menu
import plotly.graph_objects as go


 
#section acceuil
def section_acceuil():

    st.markdown(
        """
        <h2 style='color: #003366;'>Valorisation des options européennes et américaines </h2>
        """, unsafe_allow_html=True
    )
    st.write("Bienvenue sur l'interface de présentation de mon projet. Ce projet explore l'évaluation des produits dérivés, avec une attention particulière aux options européennes et américaines. À travers divers modèles, notamment Black-Scholes, le modèle binomial et trinomial, l'objectif est d'analyser les prix d'options en fonction des variations du marché et des différentes hypothèses financières. Vous trouverez dans le rapport ci-dessous une explication détaillée de ces modèles, leurs hypothèses sous-jacentes, ainsi que des exemples d'implémentation pratique pour calculer et visualiser les prix des options. N'hésitez pas à explorer les sections et découvrir les résultats obtenus.")
    # Chemin vers ton rapport
    rapport_path = "C:\\Users\\pc\\Downloads\\rapport (2).pdf"

    # Créer un bouton de téléchargement
    with open(rapport_path, "rb") as file:
      st.download_button(
        label="Télécharger le rapport",
        data=file,
        file_name="rapport (2).pdf",
        mime="application/pdf"
      )


# Section pour le modèle Black-Scholes
def section_modèle_black_scholes():
    st.markdown(
        """
        <h2 style='color: #003366;'>Modèle Black-Scholes (options européennes) </h2>
        """, unsafe_allow_html=True
    )
    # Données d'entrée
    S0 = st.number_input("Entrez le prix initial de l'actif", min_value=0.0)
    K = st.number_input("Entrez le prix d'exercice", min_value=0.0)
    T = st.number_input("Entrez la maturité (en années)", min_value=0.01)
    r = st.number_input("Entrez le taux d'intérêt sans risque", min_value=0.0)
    sigma = st.number_input("Entrez la volatilité du prix de l'actif", min_value=0.0)

    # Sélection du type d'option (call ou put)
    option_type = st.radio("Choisissez le type d'option", ('call', 'put'))

    # Bouton pour calculer le prix de l'option
    if st.button("Calculer le prix de l'option (Black-Scholes)"):
        price = black_scholes(S0, K, T, r, sigma, option_type)
        st.write(f"Le prix de l'option {option_type} est : {price:.2f} €")

    # Bouton pour visualiser l'évolution du cours de l'action
    if st.button("Visualiser l'évolution du prix de l'actif sous-jacent"):
        plot_stock_price_evolution(S0, r, sigma, T)


# Fonction pour calculer le prix d'une option européenne via le modèle Black-Scholes
def black_scholes(S0, K, T, r, sigma, option_type):
    # Calcul des paramètres d1 et d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return price

# Fonction pour visualiser l'évolution du cours de l'actif sous-jacent
def plot_stock_price_evolution(S0, r, sigma, T):
    # Simuler l'évolution du prix de l'actif sous-jacent
    dt = 0.01  # intervalle de temps (échelle discrète)
    time_points = np.arange(0, T, dt)
    stock_prices = S0 * np.exp((r - 0.5 * sigma**2) * time_points + sigma * np.sqrt(time_points) * np.random.randn(len(time_points)))
    
    # Afficher le graphique de l'évolution du prix de l'action
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_points, stock_prices, label='Prix de l\'actif')
    ax.set_title('Évolution du prix de l\'actif sous-jacent')
    ax.set_xlabel('Temps (années)')
    ax.set_ylabel('Prix de l\'actif')
    ax.legend()
    
    st.pyplot(fig)




# Section pour le modèle binomial
def section_modèle_binomial():
    st.markdown(
        """
        <h2 style='color: #003366;'>Modèle binomial </h2>
        """, unsafe_allow_html=True
    )
    # Données d'entrée (simplifiées pour ressembler à Black-Scholes)
    S0 = st.number_input("Entrez le prix initial de l'actif (S0)", min_value=0.0)
    K = st.number_input("Entrez le prix d'exercice (K)", min_value=0.0)
    r = st.number_input("Entrez le taux d'intérêt sans risque (r)", min_value=0.0)
    T = st.number_input("Entrez le temps jusqu'à l'échéance (T, en années)", min_value=0.01)
    sigma = st.number_input("Entrez la volatilité (sigma, en %)", min_value=0.0) / 100  # Convertir en proportion
    T_steps = st.number_input("Entrez le nombre de périodes (1 ou plus)", min_value=1)
    T_steps = int(T_steps)  # Conversion de T_steps en entier

    # Sélection du type d'option (call ou put) et American/European
    option_type = st.radio("Choisissez le type d'option", ('call', 'put'))
    option_style = st.radio("Choisissez le style d'option", ('européenne', 'américaine'))

    # Bouton pour afficher l'arbre
    if st.button("Calculer l'arbre binomial"):
        # Calculer les paramètres du modèle binomial
        delta_t = T / T_steps
        u = np.exp(sigma * np.sqrt(delta_t))
        d = 1 / u
        p = (np.exp(r * delta_t) - d) / (u - d)

        # Calculer et afficher l'arbre
        option_tree, price_tree = price_multi_period_binomial(S0, u, d, p, r, T,T_steps, K, option_type, option_style)
        plot_binomial_tree_plotly(price_tree, option_tree, T_steps)



# Section pour le modèle trinomial
def section_modèle_trinomial():
    st.markdown(
        """
        <h2 style='color: #003366;'>Modèle trinomial </h2>
        """, unsafe_allow_html=True
    )
    # Données d'entrée (simplifiées pour ressembler à Black-Scholes)
    S0 = st.number_input("Entrez le prix initial de l'actif (S0)", min_value=0.0)
    K = st.number_input("Entrez le prix d'exercice (K)", min_value=0.0)
    r = st.number_input("Entrez le taux d'intérêt sans risque (r)", min_value=0.0)
    T = st.number_input("Entrez le temps jusqu'à l'échéance (T, en années)", min_value=0.01)
    sigma = st.number_input("Entrez la volatilité (sigma, en %)", min_value=0.0) / 100  # Convertir en proportion
    T_steps = st.number_input("Entrez le nombre de périodes (1 ou plus)", min_value=1)
    T_steps = int(T_steps)  # Conversion de T_steps en entier

    # Sélection du type d'option (call ou put) et American/European
    option_type = st.radio("Choisissez le type d'option", ('call', 'put'))
    option_style = st.radio("Choisissez le style d'option", ('européenne', 'américaine'))

    # Bouton pour afficher l'arbre
    if st.button("Calculer l'arbre trinomial"):
        # Calculer les paramètres du modèle trinomial
        delta_t = T / T_steps
        u = np.exp(sigma * np.sqrt(2 * delta_t))
        d = 1 / u
        m = 1  # Facteur stable
        p_u = ((np.exp(r * delta_t / 2) - np.exp(-sigma * np.sqrt(delta_t / 2))) / (np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
        p_d = ((np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(r * delta_t / 2)) / (np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2)))) ** 2
        p_m = 1 - p_u - p_d

        # Calculer et afficher l'arbre
        option_tree, price_tree = price_multi_period_trinomial(S0, u, d, m, p_u, p_d, p_m, r, T,T_steps, K, option_type, option_style)
        plot_trinomial_tree_plotly(price_tree, option_tree, T_steps)



# Fonction pour calculer le prix d'une option via le modèle binomial (supporte options américaines)
def price_multi_period_binomial(S0, u, d, p, r, T,T_steps, K, option_type, option_style):
    # Génération de l'arbre des prix
    price_tree = np.zeros((T_steps+1, T_steps+1))
    price_tree[0, 0] = S0
    
    for i in range(1, T_steps+1):
        for j in range(i+1):
            price_tree[j, i] = S0 * (u**(i-j)) * (d**j)
    
    # Calcul des valeurs des options à la dernière période
    option_tree = np.zeros((T_steps+1, T_steps+1))
    for j in range(T_steps+1):
        if option_type == 'call':
            option_tree[j, T_steps] = max(0, price_tree[j, T_steps] - K)
        else:
            option_tree[j, T_steps] = max(0, K - price_tree[j, T_steps])
    
    # Remonter dans l'arbre pour calculer les prix des options
    for i in range(T_steps-1, -1, -1):
        for j in range(i+1):
            hold_value = np.exp(-r * (T / T_steps)) * (p * option_tree[j, i+1] + (1 - p) * option_tree[j+1, i+1])
            if option_style == 'américaine':  # Vérifier l'exercice anticipé
                if option_type == 'call':
                    option_tree[j, i] = max(hold_value, price_tree[j, i] - K)
                else:
                    option_tree[j, i] = max(hold_value, K - price_tree[j, i])
            else:
                option_tree[j, i] = hold_value
    
    return option_tree, price_tree

# Fonction pour afficher l'arbre binomial avec Plotly
def plot_binomial_tree_plotly(price_tree, option_tree, T_steps):
    nodes = []
    edges = []
    labels = []

    for i in range(T_steps + 1):
        for j in range(i + 1):
            x = i
            y = i - 2 * j  # Position verticale

            # Ajouter les noeuds
            nodes.append((x, y))
            labels.append(f"S: {price_tree[j, i]:.2f}\nOpt: {option_tree[j, i]:.2f}")

            # Ajouter les connexions entre les noeuds
            if i < T_steps:
                edges.append(((x, y), (x + 1, y + 1)))  # Connexion vers la montée
                edges.append(((x, y), (x + 1, y - 1)))  # Connexion vers la descente

    # Créer les traces pour les noeuds
    x_nodes = [node[0] for node in nodes]
    y_nodes = [node[1] for node in nodes]

    fig = go.Figure()

    # Ajouter les noeuds
    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=10, color='blue'),
        name='Noeuds'
    ))

    # Ajouter les connexions
    for edge in edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]

        fig.add_trace(go.Scatter(
            x=x_edge,
            y=y_edge,
            mode='lines',
            line=dict(color='black', width=1),
            name='Connexions',
            showlegend=False
        ))

    # Ajuster l'apparence
    fig.update_layout(
        title="Arbre binomial interactif",
        xaxis_title="Période",
        yaxis_title="Valeur relative",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig)




# Fonction pour calculer le prix d'une option via le modèle trinomial (supporte options américaines)
def price_multi_period_trinomial(S0, u, d, m, p_u, p_d, p_m, r, T,T_steps, K, option_type, option_style):
    # Génération de l'arbre des prix
    price_tree = np.zeros((2*T_steps+1, T_steps+1))
    price_tree[T_steps, 0] = S0

    for i in range(1, T_steps+1):
        for j in range(-i, i+1):
            price_tree[T_steps+j, i] = S0 * (u**max(j, 0)) * (d**max(-j, 0))

    # Calcul des valeurs des options à la dernière période
    option_tree = np.zeros((2*T_steps+1, T_steps+1))
    for j in range(-T_steps, T_steps+1):
        if option_type == 'call':
            option_tree[T_steps+j, T_steps] = max(0, price_tree[T_steps+j, T_steps] - K)
        else:
            option_tree[T_steps+j, T_steps] = max(0, K - price_tree[T_steps+j, T_steps])

    # Remonter dans l'arbre pour calculer les prix des options
    for i in range(T_steps-1, -1, -1):
        for j in range(-i, i+1):
            hold_value = np.exp(-r * (T / T_steps)) * (
                p_u * option_tree[T_steps+j+1, i+1] +
                p_m * option_tree[T_steps+j, i+1] +
                p_d * option_tree[T_steps+j-1, i+1]
            )
            if option_style == 'américaine':  # Vérifier l'exercice anticipé
                if option_type == 'call':
                    option_tree[T_steps+j, i] = max(hold_value, price_tree[T_steps+j, i] - K)
                else:
                    option_tree[T_steps+j, i] = max(hold_value, K - price_tree[T_steps+j, i])
            else:
                option_tree[T_steps+j, i] = hold_value

    return option_tree, price_tree

# Fonction pour afficher l'arbre trinomial avec Plotly
def plot_trinomial_tree_plotly(price_tree, option_tree, T_steps):
    nodes = []
    edges = []
    labels = []

    for i in range(T_steps + 1):
        for j in range(-i, i + 1):
            x = i
            y = i - 2 * j  # Position verticale

            # Ajouter les noeuds
            nodes.append((x, y))
            labels.append(f"S: {price_tree[T_steps+j, i]:.2f}\nOpt: {option_tree[T_steps+j, i]:.2f}")

            # Ajouter les connexions entre les noeuds
            if i < T_steps:
                edges.append(((x, y), (x + 1, y + 2)))  # Connexion vers la montée
                edges.append(((x, y), (x + 1, y)))      # Connexion stable
                edges.append(((x, y), (x + 1, y - 2)))  # Connexion vers la baisse

    # Créer les traces pour les noeuds
    x_nodes = [node[0] for node in nodes]
    y_nodes = [node[1] for node in nodes]

    fig = go.Figure()

    # Ajouter les noeuds
    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=10, color='blue'),
        name='Noeuds'
    ))

    # Ajouter les connexions
    for edge in edges:
        x_edge = [edge[0][0], edge[1][0]]
        y_edge = [edge[0][1], edge[1][1]]

        fig.add_trace(go.Scatter(
            x=x_edge,
            y=y_edge,
            mode='lines',
            line=dict(color='black', width=1),
            name='Connexions',
            showlegend=False
        ))

    # Ajuster l'apparence
    fig.update_layout(
        title="Arbre trinomial interactif",
        xaxis_title="Période",
        yaxis_title="Valeur relative",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig)




# Section finale pour comparer les trois modèles
# Section finale pour comparer les deux modèles
def section_comparaison():
    st.markdown(
        """
        <h2 style='color: #003366;'>Comparaison des modèles </h2>
        """, unsafe_allow_html=True
    )
    # Données d'entrée pour la comparaison
    S0 = st.number_input("Entrez le prix initial de l'actif (S0)", min_value=0.0)
    K = st.number_input("Entrez le prix d'exercice (K)", min_value=0.0)
    T = st.number_input("Entrez le temps jusqu'à l'échéance (T, en années)", min_value=0.01)
    sigma = st.number_input("Entrez la volatilité (sigma, en %)", min_value=0.0) / 100
    r = st.number_input("Entrez le taux d'intérêt sans risque (r)", min_value=0.0)
    T_steps = st.number_input("Entrez le nombre de périodes (1 ou plus)", min_value=1)
    T_steps = int(T_steps)

    # Sélection du type d'option (call ou put)
    option_type = st.radio("Choisissez le type d'option", ('call', 'put'))

    # Calcul des paramètres communs pour les modèles
    delta_t = T / T_steps
    u = np.exp(sigma * np.sqrt(2 * delta_t))
    d = 1 / u
    m = 1.0
    p_u = ((np.exp(r * delta_t / 2) - np.exp(-sigma * np.sqrt(delta_t / 2))) / 
           (np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2))))**2
    p_d = ((np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(r * delta_t / 2)) / 
           (np.exp(sigma * np.sqrt(delta_t / 2)) - np.exp(-sigma * np.sqrt(delta_t / 2))))**2
    p_m = 1 - p_u - p_d

    # Calcul du prix avec Black-Scholes
    bs_price = black_scholes(S0, K, T, r, sigma, option_type)

    # Calcul du prix avec le modèle trinomial
    option_tree_tri, price_tree_tri = price_multi_period_trinomial(S0, u, d, m, p_u, p_d, p_m, r, T,T_steps, K, option_type, 'européenne')
    trinomial_price = option_tree_tri[T_steps, 0]

    # Affichage des résultats dans des cartes
    col1, col2, col3 = st.columns(2)
    col1.metric("Black-Scholes", f"{bs_price:.2f}")
    col2.metric("Binomial", f"{binomial_price:.2f}")
    col3.metric("Trinomial", f"{trinomial_price:.2f}")

    # Création du graphique pour visualiser la convergence
    st.subheader("Convergence du modèle trinomial vers Black-Scholes")
    n_periods = list(range(1, T_steps + 1))
    trinomial_prices = [
        price_multi_period_trinomial(S0, u, d, m, p_u, p_d, p_m, r, n, K, option_type, 'européenne')[0][n, 0]
        for n in n_periods
    ]
    bs_prices = [bs_price] * T_steps

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_periods, y=trinomial_prices, mode='lines+markers', name='Trinomial'))
    fig.add_trace(go.Scatter(x=n_periods, y=bs_prices, mode='lines', name='Black-Scholes', line=dict(dash='dash')))

    fig.update_layout(
        title="Convergence du modèle trinomial",
        xaxis_title="Nombre de périodes",
        yaxis_title="Prix de l'option",
        legend=dict(x=0.1, y=1.1, orientation="h"),
    )

    st.plotly_chart(fig)




# Créer un menu latéral avec des icônes et des rectangles
with st.sidebar:
    selected = option_menu(
        "Menu",  # Titre du menu
        ["Acceuil", "Black-Scholes", "Modèle binomial", "Modèle trinomial", "Convergence"],  # Nom des pages
        icons=["house", "calculator", "graph-up", "diagram-3", "arrow-repeat"],  # Icônes des pages
        menu_icon="cast",  # Icône pour le menu
        default_index=0,  # Page par défaut
        styles={
            "container": {"padding": "5px", "background-color": "#f0f0f0"},
            "icon": {"color": "#003366", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "color": "black",
                "border-radius": "5px",
                "background-color": "#ADD8E6",
            },
            "nav-link-selected": {"background-color": "#003366", "color": "white"},
        },
    )
# Appeler la fonction appropriée en fonction de la sélection de l'utilisateur
if selected == 'Acceuil':
    section_acceuil()
elif selected == 'Black-Scholes':
    section_modèle_black_scholes()
elif selected == 'Modèle binomial':
    section_modèle_binomial()
elif selected == 'Modèle trinomial':
    section_modèle_trinomial()
elif selected == 'Convergence':
    section_comparaison()
