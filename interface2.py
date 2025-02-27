import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import math


 
#section acceuil
def section_acceuil():

    st.markdown(
        """
        <h2 style='color: #003366;'>Valorisation des options européennes et américaines </h2>
        """, unsafe_allow_html=True
    )
    st.write("Bienvenue sur l'interface de présentation de mon projet. Ce projet explore l'évaluation des produits dérivés, avec une attention particulière aux options européennes et américaines. À travers divers modèles, notamment Black-Scholes, le modèle binomial et trinomial, l'objectif est d'analyser les prix d'options en fonction des variations du marché et des différentes hypothèses financières. Vous trouverez dans le rapport ci-dessous une explication détaillée de ces modèles, leurs hypothèses sous-jacentes, ainsi que des exemples d'implémentation pratique pour calculer et visualiser les prix des options. N'hésitez pas à explorer les sections et découvrir les résultats obtenus.")


    # Créer un bouton de téléchargement
    with open("rapport_evaluation_options.pdf", "rb") as file:
      st.download_button(
        label="Télécharger le rapport",
        data=file,
        file_name="rapport_evaluation_options.pdf",
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
    
    # Création du graphique interactif avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points, y=stock_prices, mode='lines', name="Prix de l'actif"))
    
    # Mise en forme du graphique
    fig.update_layout(
        title="Évolution du prix de l'actif sous-jacent",
        xaxis_title="Temps (années)",
        yaxis_title="Prix de l'actif",
        template="plotly_dark",
        hovermode="x"
    )
    
    st.plotly_chart(fig)


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

        print(f"u = {u}, d = {d}, m = {m}")




# Fonction pour calculer le prix d'une option via le modèle binomial (supporte options américaines)
def price_multi_period_binomial(S0, u, d, p, r, T, T_steps, K, option_type, option_style):
    delta_t = T / T_steps
    price_tree = np.zeros((T_steps + 1, T_steps + 1))
    option_tree = np.zeros((T_steps + 1, T_steps + 1))

    # Remplissage de l'arbre des prix
    for i in range(T_steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Valeurs finales des options
    for j in range(T_steps + 1):
        if option_type == 'call':
            option_tree[j, T_steps] = max(price_tree[j, T_steps] - K, 0)
        else:
            option_tree[j, T_steps] = max(K - price_tree[j, T_steps], 0)

    # Remontée dans l'arbre pour le calcul des prix des options
    for i in range(T_steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = math.exp(-r * delta_t ) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])

            if option_style == 'américaine':  # Vérification pour option américaine
                if option_type == 'call':
                    option_tree[j, i] = max(hold_value, price_tree[j, i] - K)
                else:
                    option_tree[j, i] = max(hold_value, K - price_tree[j, i])
            else:  # Option européenne
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




# Correction de la fonction pour calculer l'arbre trinomial
def price_multi_period_trinomial(S0, u, d, m, p_u, p_d, p_m, r, T, T_steps, K, option_type, option_style):
    price_tree = np.zeros((2 * T_steps + 1, T_steps + 1))
    price_tree[T_steps, 0] = S0  # Position centrale

    for i in range(1, T_steps + 1):
        for j in range(-i, i + 1):
            if j > 0:
                price_tree[T_steps + j, i] = price_tree[T_steps + j - 1, i - 1] * u
            elif j < 0:
                price_tree[T_steps + j, i] = price_tree[T_steps + j + 1, i - 1] * d
            else:
                price_tree[T_steps + j, i] = price_tree[T_steps + j, i - 1] * m

    option_tree = np.zeros((2 * T_steps + 1, T_steps + 1))
    for j in range(-T_steps, T_steps + 1):
        if option_type == 'call':
            option_tree[T_steps + j, T_steps] = max(0, price_tree[T_steps + j, T_steps] - K)
        else:
            option_tree[T_steps + j, T_steps] = max(0, K - price_tree[T_steps + j, T_steps])

    for i in range(T_steps - 1, -1, -1):
        for j in range(-i, i + 1):
            hold_value = np.exp(-r * (T / T_steps)) * (
                p_u * option_tree[T_steps + j + 1, i + 1] +
                p_m * option_tree[T_steps + j, i + 1] +
                p_d * option_tree[T_steps + j - 1, i + 1]
            )
            if option_style == 'américaine':
                if option_type == 'call':
                    option_tree[T_steps + j, i] = max(hold_value, price_tree[T_steps + j, i] - K)
                else:
                    option_tree[T_steps + j, i] = max(hold_value, K - price_tree[T_steps + j, i])
            else:
                option_tree[T_steps + j, i] = hold_value

    return option_tree, price_tree




# Correction de l'affichage de l'arbre trinomial
def plot_trinomial_tree_plotly(price_tree, option_tree, T_steps):
    nodes = []
    edges = []
    labels = []

    for i in range(T_steps + 1):
        for j in range(-i, i + 1):
            x = i
            y = -j  

            nodes.append((x, y))
            labels.append(f"S: {price_tree[T_steps + j, i]:.2f}\nOpt: {option_tree[T_steps + j, i]:.2f}")

            if i < T_steps:
                edges.append(((x, y), (x + 1, y - 1)))  # Connexion up
                edges.append(((x, y), (x + 1, y)))      # Connexion stable
                edges.append(((x, y), (x + 1, y + 1)))  # Connexion down

    x_nodes = [node[0] for node in nodes]
    y_nodes = [node[1] for node in nodes]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=10, color='blue'),
        name='Noeuds'
    ))

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

    fig.update_layout(
        title="Arbre trinomial interactif",
        xaxis_title="Période",
        yaxis_title="Valeur relative",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"),
        showlegend=False,
        height=600
    )

    st.plotly_chart(fig)





# Section finale pour comparer les trois modèles


# Fonction Black-Scholes
def black_scholes1(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Modèle binomial
def binomial_model(S, K, T, r, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    stock_prices = np.array([S * (u**j) * (d**(N-j)) for j in range(N+1)])
    option_values = np.maximum(0, (stock_prices - K) if option_type == "call" else (K - stock_prices))

    for i in range(N-1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return option_values[0]

# Modèle trinomial
def trinomial_model(S, K, T, r, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    m = 1
    pu = ((np.exp(r * dt / 2) - np.exp(-sigma * np.sqrt(dt/2))) /
          (np.exp(sigma * np.sqrt(dt/2)) - np.exp(-sigma * np.sqrt(dt/2)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt/2)) - np.exp(r * dt / 2)) /
          (np.exp(sigma * np.sqrt(dt/2)) - np.exp(-sigma * np.sqrt(dt/2)))) ** 2
    pm = 1 - pu - pd

    stock_prices = np.array([S * (u**(j - N)) for j in range(2*N+1)])
    option_values = np.maximum(0, (stock_prices - K) if option_type == "call" else (K - stock_prices))

    for i in range(N-1, -1, -1):
        option_values = (pu * option_values[2:] + pm * option_values[1:-1] + pd * option_values[:-2]) * np.exp(-r * dt)

    return option_values[len(option_values)//2]  # Correction



def section_comparaison():
    st.markdown(
        """
        <h2 style='color: #003366;'>Comparaison des modèles </h2>
        """, unsafe_allow_html=True
    )


    S = st.number_input("Entrez le prix initial de l'actif (S0)", value=100.0)
    K = st.number_input("Entrez le prix d'exercice (K)", value=100.0)
    T = st.number_input("Entrez le temps jusqu'à l'échéance (T, en années)", value=1.0)
    r = st.number_input("REntrez le taux d'intérêt sans risque(r) [%]", value=5.0) / 100
    sigma = st.number_input("Entrez la volatilité(σ) [%]", value=20.0) / 100
    N = st.number_input("Entrez le nombre de période (N)", min_value=1, max_value=500, value=100, step=1)
    option_type = st.radio("Choisissez le type d'option", ["Call", "Put"])

    if st.button("Calculer"):
       bs_price = black_scholes1(S, K, T, r, sigma, option_type.lower())

       # Calculer les prix pour toutes les valeurs de 1 à N
       binomial_prices = [binomial_model(S, K, T, r, sigma, i, option_type.lower()) for i in range(1, N+1)]
       trinomial_prices = [trinomial_model(S, K, T, r, sigma, i, option_type.lower()) for i in range(1, N+1)]
    
       # Graphique interactif avec évolution des prix
       fig = go.Figure()

       fig.add_trace(go.Scatter(
         x=list(range(1, N+1)), y=binomial_prices,
         mode="lines+markers", name="Binomial Model",
         marker=dict(color="blue")
       ))

       fig.add_trace(go.Scatter(
         x=list(range(1, N+1)), y=trinomial_prices,
         mode="lines+markers", name="Trinomial Model",
         marker=dict(color="green")
       ))

       # Ajout de la ligne horizontale pour Black-Scholes
       fig.add_trace(go.Scatter(
         x=list(range(1, N+1)), 
         y=[bs_price] * N,  # Ligne horizontale constante
         mode="lines", 
         name="Black-Scholes", 
         line=dict(color="red", dash="dash")
       ))

       fig.update_layout(
         title="Option Pricing Models Evolution",
         xaxis_title="Number of Steps (N)",
         yaxis_title="Option Price",
         template="plotly_white"
       )

       st.plotly_chart(fig)

       # Résumé des résultats
       st.subheader("Resultat d'analyse")
       col1, col2, col3 = st.columns(3)
       col1.metric("Black-Scholes", f"{bs_price:.4f}")
       col2.metric("Binomial", f"{binomial_prices[-1]:.4f}")
       col3.metric("Trinomial", f"{trinomial_prices[-1]:.4f}")



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
