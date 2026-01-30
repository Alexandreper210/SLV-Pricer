import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import norm
from scipy.optimize import brentq

# ==========================================
# 1. Bases
# ==========================================

def bs_price(vol, S, K, T, r=0, option_type='call'):
    # Protection numérique
    if T <= 1e-4 or vol <= 1e-4:
        intrinsic = max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        return intrinsic
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol_solver_smart(target_price, S, K, T, r=0, option_type='call'):
    """
    Solveur robuste qui gère Put/Call pour éviter les erreurs numériques ITM.
    """
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    # Si le prix MC est inférieur à l'intrinsèque (bruit), pas de vol possible
    if target_price <= intrinsic + 1e-6:
        return np.nan
        
    def objective(sigma):
        return bs_price(sigma, S, K, T, r, option_type) - target_price
    
    try:
        # Recherche entre 0.01% et 500%
        return brentq(objective, 1e-4, 5.0)
    except:
        return np.nan

# ==========================================
# 2. MOTEUR BERGOMI (VECTORISÉ & RENORMALISÉ)
# ==========================================

class BergomiSLV:
    def __init__(self, theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol):
        self.theta = theta
        self.rhoXY = rhoXY
        self.kappaX = kappaX
        self.kappaY = kappaY
        self.sigma = vol_of_vol
        self.xi0 = xi0
        self.rhoSVol = rhoSVol

    def simulate(self, S0, T_max, N_steps, N_paths, seed=None): # <--- Ajout de seed=None
        # Ajout de la gestion de la graine pour les Grecs
        if seed is not None:
            np.random.seed(seed)
        dt = T_max / N_steps
        sqrt_dt = np.sqrt(dt)
        t_grid = np.linspace(0, T_max, N_steps + 1)
        
        S = np.zeros((N_steps + 1, N_paths))
        S[0, :] = S0
        X = np.zeros(N_paths)
        Y = np.zeros(N_paths)
        
        # Pré-calculs OU constants
        exp_kX = np.exp(-self.kappaX * dt)
        exp_kY = np.exp(-self.kappaY * dt)
        std_X = np.sqrt((1 - np.exp(-2 * self.kappaX * dt)) / (2 * self.kappaX))
        std_Y = np.sqrt((1 - np.exp(-2 * self.kappaY * dt)) / (2 * self.kappaY))
        alpha = ((1 - self.theta)**2 + self.theta**2 + 2 * self.rhoXY * self.theta * (1 - self.theta))**(-0.5)

        V_store = np.zeros((N_steps + 1, N_paths))
        V_store[0, :] = self.xi0 

        for i in range(N_steps):
            Z_X = np.random.normal(0, 1, N_paths)
            Z_Y = np.random.normal(0, 1, N_paths)
            Z_S = np.random.normal(0, 1, N_paths)
            
            # Corrélation
            eps_vol = (1-self.theta)*Z_X + self.theta*Z_Y 
            eps_vol /= np.std(eps_vol)
            
            dW_S = self.rhoSVol * eps_vol + np.sqrt(1 - self.rhoSVol**2) * Z_S
            
            # Update Facteurs
            X = X * exp_kX + std_X * Z_X
            Y = Y * exp_kY + std_Y * Z_Y
            
            # Update Variance (Renormalisation Empirique)
            x_t = alpha * ((1 - self.theta) * X + self.theta * Y)
            Raw_Noise = np.exp(2 * self.sigma * x_t)
            Avg_Noise = np.mean(Raw_Noise)
            StochVar = self.xi0 * (Raw_Noise / (Avg_Noise + 1e-9))
            
            V_store[i+1, :] = StochVar
            Sigma_Total = np.sqrt(StochVar)
            S[i+1, :] = S[i, :] * np.exp(-0.5 * Sigma_Total**2 * dt + Sigma_Total * sqrt_dt * dW_S)

        return t_grid, S, V_store

# ==========================================
# CALCULATEUR DE PAYOFF & GRECS
# ==========================================

def calculate_payoff(S_paths, K, product_type, barrier_level=None):
    """Calcule le payoff vectorisé selon le type de produit"""
    final_spot = S_paths[-1, :]
    
    if product_type == "Call Vanille":
        return np.maximum(final_spot - K, 0)
        
    elif product_type == "Put Vanille":
        return np.maximum(K - final_spot, 0)
        
    elif product_type == "Call Barrière (Up-and-Out)":
        max_reached = np.max(S_paths, axis=0)
        is_alive = max_reached < barrier_level # Doit rester SOUS la barrière
        return np.maximum(final_spot - K, 0) * is_alive
        
    elif product_type == "Put Barrière (Down-and-Out)":
        min_reached = np.min(S_paths, axis=0)
        is_alive = min_reached > barrier_level # Doit rester AU-DESSUS
        return np.maximum(K - final_spot, 0) * is_alive
        
    # --- NOUVEAU : Put Down-and-In ---
    elif product_type == "Put Barrière (Down-and-In)":
        min_reached = np.min(S_paths, axis=0)
        is_activated = min_reached <= barrier_level # Doit TOUCHER la barrière pour s'activer
        return np.maximum(K - final_spot, 0) * is_activated
        
    # --- NOUVEAU : Digitale (Call Binaire) ---
    elif product_type == "Digitale Call (Cash-or-Nothing)":
        # Paie 1€ si S_T > K, sinon 0
        return 1.0 * (final_spot > K)
        
    elif product_type == "Call Asiatique (Moyenne)":
        average_spot = np.mean(S_paths, axis=0)
        return np.maximum(average_spot - K, 0)
        
    elif product_type == "Call Lookback (Fixed Strike)":
        max_spot = np.max(S_paths, axis=0)
        return np.maximum(max_spot - K, 0)
        
    return np.zeros_like(final_spot)

def compute_greeks(model, S0, K, T, N_steps, N_paths, product_type, barrier_lvl):
    """Calcule Delta, Gamma, Vega par Différences Finies avec Seed Fixe."""
    seed_base = 42 # Seed fixe vital pour réduire la variance
    

    # 1. Simulation Centrale (Prix)
    _, S_0, _ = model.simulate(S0, T, N_steps, N_paths, seed=seed_base)
    P0 = np.mean(calculate_payoff(S_0, K, product_type, barrier_lvl))
    
    # 2. Simulation Up (S + 1%) pour Delta/Gamma
    ds = S0 * 0.005
    _, S_up, _ = model.simulate(S0 + ds, T, N_steps, N_paths, seed=seed_base)
    P_up = np.mean(calculate_payoff(S_up, K, product_type, barrier_lvl))
    
    # 3. Simulation Down (S - 1%) pour Gamma
    _, S_down, _ = model.simulate(S0 - ds, T, N_steps, N_paths, seed=seed_base)
    P_down = np.mean(calculate_payoff(S_down, K, product_type, barrier_lvl))
    
    # 4. Simulation Vega (Xi0 + 1%)
    model_vega = BergomiSLV(model.theta, model.rhoXY, model.kappaX, model.kappaY, 
                            model.sigma, model.xi0 * 1.01, model.rhoSVol)
    _, S_vol, _ = model_vega.simulate(S0, T, N_steps, N_paths, seed=seed_base)
    P_vol = np.mean(calculate_payoff(S_vol, K, product_type, barrier_lvl))
    
    # Calculs
    delta = (P_up - P_down) / (2 * ds)
    gamma = (P_up - 2*P0 + P_down) / (ds**2)
    vega = (P_vol - P0) / 0.01
    
    return P0, delta, gamma, vega, S_0
# ==========================================
# 3. INTERFACE DE STRUCTURATION
# ==========================================

st.set_page_config(page_title="Bergomi SLV Pricer", layout="wide")
# --- AJOUT DU PROFIL LINKEDIN DANS LA SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Contact</h3>
        <p>Projet réalisé par <b>Alexandre Perier</b></p>
        <a href="https://www.linkedin.com/in/alexandre-perier-46510b264/" target="_blank">
            <button style="
                background-color: #0072b1; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 16px; 
                margin: 4px 2px; 
                cursor: pointer; 
                border-radius: 8px;
                width: 100%;">
                LinkedIn 
            </button>
        </a>
    </div>
    <hr>
    """, unsafe_allow_html=True)

st.title("Bergomi SLV - Pricer")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Paramètres Modèle")
    xi0 = st.number_input("Variance Initiale (xi0)", 0.01, 0.25, 0.04, step=0.01)
    target_vol = np.sqrt(xi0)
    st.caption(f"Volatilité Cible : **{target_vol:.2%}**")
    
    vol_of_vol = st.slider("Vol-of-Vol (sigma)", 0.0, 2.0, 1.0)
    rhoSVol = st.slider("Corrélation Spot/Vol", -1.0, 1.0, -0.7)

    with st.expander("Paramètres Facteurs"):
        theta = st.slider("Mix Facteur Y", 0.0, 1.0, 0.4)
        kappaX = st.number_input("Mean Rev X", 0.1, 10.0, 4.0)
        kappaY = st.number_input("Mean Rev Y", 0.1, 5.0, 0.1)
        rhoXY = st.slider("Corr X,Y", -1.0, 1.0, 0.0)

    st.markdown("---")
    # SÉLECTEUR DE MODE
    app_mode = st.radio("Mode :", ["Pricing & Trajectoires", "Surface de Volatilité (3D)", "Pricing & Grecs Multi-Produits"])

# ==========================================
# MODE 1 : PRICING & TRAJECTOIRES
# ==========================================
if app_mode == "Pricing & Trajectoires":
    st.sidebar.header("Paramètres Option")
    S0 = st.sidebar.number_input("Spot", 100.0)
    Strike = st.sidebar.number_input("Strike", 100.0)
    T = st.sidebar.number_input("Maturité (Ans)", 1.0)
    N_paths = st.sidebar.selectbox("Simulations", [1000, 5000, 10000, 20000], index=2)
    N_steps = st.sidebar.slider("Pas de temps", 10, 300, 100)

    if st.button("Lancer Simulation (Single)", type="primary"):
        model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
        
        with st.spinner('Simulation en cours...'):
            start = time.time()
            t_grid, S_paths, V_paths = model.simulate(S0, T, N_steps, N_paths)
            end = time.time()
            
            # Pricing
            payoffs = np.maximum(S_paths[-1, :] - Strike, 0)
            price = np.mean(payoffs)
            std_err = np.std(payoffs) / np.sqrt(N_paths)
            
            # Stats Vol
            avg_vol_out = np.mean(np.sqrt(V_paths))

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix Call", f"{price:.4f} €", f"± {1.96*std_err:.4f}")
        c2.metric("Volatilité Moyenne", f"{avg_vol_out:.2%}", f"Cible: {target_vol:.2%}")
        c3.metric("Temps", f"{end-start:.3f} s")

        
        st.markdown("---")
        st.subheader("Comparaison Black-Scholes")
        bs_price_call = bs_price(target_vol, S0, Strike, T, 0, 'call')
        diff_pct = ((price - bs_price_call) / bs_price_call) * 100 if bs_price_call > 0 else 0

        col_bs1, col_bs2, col_bs3 = st.columns(3)
        col_bs1.metric("Prix BS", f"{bs_price_call:.4f} €")
        col_bs2.metric("Écart Bergomi vs BS", f"{price - bs_price_call:+.4f} €")
        col_bs3.metric("Écart Relatif", f"{diff_pct:+.2f}%", 
                    delta="Plus cher" if diff_pct > 0 else "Moins cher")
        if abs(diff_pct) > 5:
            st.info(f"**Écart significatif ({diff_pct:+.1f}%)** : La volatilité stochastique du modèle Bergomi capture des effets que BS ignore (clustering, skew dynamique)")
        else:
            st.success("Convergence proche de Black-Scholes (vol plate)")

        # --- ONGLETS ---
        tab1, tab2, tab3 = st.tabs(["Trajectoires (Spot)", "Variance Stochastique", "Distribution Terminale"])

        with tab1:
            fig_spot = go.Figure()
            for i in range(min(50, N_paths)):
                fig_spot.add_trace(go.Scatter(x=t_grid, y=S_paths[:, i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
            fig_spot.add_trace(go.Scatter(x=t_grid, y=np.mean(S_paths, axis=1), mode='lines', line=dict(color='white', width=3), name='Moyenne'))
            fig_spot.update_layout(title="Diffusion du Spot", xaxis_title="Temps", yaxis_title="Spot", template="plotly_dark")
            st.plotly_chart(fig_spot, use_container_width=True)

        with tab2:
            fig_vol = go.Figure()
            for i in range(min(50, N_paths)):
                fig_vol.add_trace(go.Scatter(x=t_grid, y=np.sqrt(V_paths[:, i]), mode='lines', line=dict(color='orange', width=1), opacity=0.4, showlegend=False))
            fig_vol.add_trace(go.Scatter(x=t_grid, y=np.mean(np.sqrt(V_paths), axis=1), mode='lines', line=dict(color='red', width=3), name='Vol Moyenne'))
            fig_vol.update_layout(title="Diffusion de la Volatilité", xaxis_title="Temps", yaxis_title="Volatilité", template="plotly_dark")
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab3:
            fig_hist = go.Figure(data=[go.Histogram(x=S_paths[-1, :], nbinsx=60, histnorm='probability', marker_color='#636EFA')])
            fig_hist.add_vline(x=Strike, line_dash="dash", line_color="red", annotation_text="Strike")
            fig_hist.update_layout(title=f"Distribution à T={T}", xaxis_title="Prix Final", template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# MODE 2 : SURFACE DE VOLATILITÉ
# ==========================================
elif app_mode == "Surface de Volatilité (3D)":
    st.sidebar.header("Paramètres Surface")
    S0 = st.sidebar.number_input("Spot", 100.0)
    T_max = st.sidebar.number_input("Maturité Max", 2.0)
    N_paths_surf = st.sidebar.selectbox("Simulations", [5000, 10000, 20000], index=1)

    if st.button("Générer la Surface (Smart OTM)", type="primary"):
        model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
        
        with st.spinner('Construction de la Surface (Mode OTM)...'):
            maturities = np.linspace(0.2, T_max, 5)
            strikes_pct = np.linspace(0.7, 1.3, 13) 
            strikes = S0 * strikes_pct
            N_steps = 100
            
            t_grid, S_paths, V_paths = model.simulate(S0, T_max, N_steps, N_paths_surf)
            
            surface_iv = []
            for T_target in maturities:
                idx = int((T_target / T_max) * N_steps)
                idx = min(idx, N_steps)
                S_at_T = S_paths[idx, :]
                
                row_iv = []
                for K in strikes:
                    is_call = K >= S0
                    option_type = 'call' if is_call else 'put'
                    if is_call:
                        payoff = np.maximum(S_at_T - K, 0)
                    else:
                        payoff = np.maximum(K - S_at_T, 0)
                    price = np.mean(payoff)
                    iv = implied_vol_solver_smart(price, S0, K, T_target, 0, option_type)
                    row_iv.append(iv * 100 if not np.isnan(iv) else None)
                surface_iv.append(row_iv)
            
            Z_surface = np.array(surface_iv)

        st.success("Surface générée avec méthode OTM !")
        tab1, tab2, tab3 = st.tabs(["Surface 3D", "Term Structure", "Skew Correct"])
        
        with tab1:
            fig_3d = go.Figure(data=[go.Surface(z=Z_surface, x=strikes, y=maturities, colorscale='Jet')])
            fig_3d.update_layout(title="Surface Volatilité (OTM Priced)", scene=dict(xaxis_title='Strike', yaxis_title='Maturité', zaxis_title='Vol %'), height=600, template="plotly_dark")
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab2:
            mid = len(strikes)//2
            fig_ts = go.Figure(go.Scatter(x=maturities, y=Z_surface[:, mid], mode='lines+markers', line=dict(color='#FF4B4B')))
            fig_ts.update_layout(title="Term Structure (ATM)", xaxis_title="Maturité", yaxis_title="Vol %", template="plotly_dark")
            st.plotly_chart(fig_ts, use_container_width=True)
            
        with tab3:
            fig_skew = go.Figure(go.Scatter(x=strikes, y=Z_surface[-1, :], mode='lines+markers', line=dict(color='#00CC96')))
            fig_skew.update_layout(title=f"Skew à T={T_max} (Sourire Volatilité)", xaxis_title="Strike", yaxis_title="Vol %", template="plotly_dark")
            st.plotly_chart(fig_skew, use_container_width=True)

# ==========================================
# MODE 3 : PRICING MULTI-PRODUITS & GRECS
# ==========================================
elif app_mode == "Pricing & Grecs Multi-Produits":
    
    st.sidebar.header("2. Structuration Avancée")
    # Liste mise à jour avec les nouveaux produits
    product_type = st.sidebar.selectbox("Type de Produit", 
        ["Call Vanille", "Put Vanille",
         "Call Barrière (Up-and-Out)", "Put Barrière (Down-and-Out)", "Put Barrière (Down-and-In)",
         "Digitale Call (Cash-or-Nothing)",
         "Call Asiatique (Moyenne)", "Call Lookback (Fixed Strike)"])
    
    S0 = st.sidebar.number_input("Spot Initial", 100.0)
    Strike = st.sidebar.number_input("Strike", 100.0)
    T = st.sidebar.number_input("Maturité", 1.0)
    
    # Gestion dynamique de la Barrière
    barrier_lvl = None
    if "Barrière" in product_type:
        # Valeurs par défaut intelligentes
        if "Up" in product_type: def_bar = 130.0
        elif "Down" in product_type: def_bar = 80.0
        else: def_bar = 100.0
        barrier_lvl = st.sidebar.number_input("Niveau Barrière", 0.0, 300.0, def_bar)

    if st.button("Pricer & Calculer Grecs", type="primary"):
        model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
        
        with st.spinner('Simulation & Calculs...'):
            price, delta, gamma, vega, S_paths = compute_greeks(
                model, S0, Strike, T, 100, 50000, product_type, barrier_lvl
            )
            gamma_scaled = gamma * S0
            
        # --- DASHBOARD ---
        st.markdown(f"### Résultats : {product_type}")
        
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("💰 Prix", f"{price:.4f} €") # 4 décimales pour la Digitale (prix petit)
        g2.metric("Δ Delta", f"{delta:.1%}")
        g3.metric("Γ Gamma", f"{gamma_scaled:.2f}%")
        g4.metric("ν Vega", f"{vega:.3f}")
        
        st.markdown("---")

        # Calcul BS uniquement pour vanilles
        if product_type in ["Call Vanille", "Put Vanille"]:
            option_type_bs = 'call' if product_type == "Call Vanille" else 'put'
            bs_price_ref = bs_price(np.sqrt(xi0), S0, Strike, T, 0, option_type_bs)
            diff_pct = ((price - bs_price_ref) / bs_price_ref) * 100 if bs_price_ref > 0 else 0
            
            st.markdown("### Benchmark Black-Scholes")
            bs1, bs2, bs3 = st.columns(3)
            bs1.metric("Prix BS", f"{bs_price_ref:.4f} €")
            bs2.metric("Écart Bergomi - BS", f"{price - bs_price_ref:+.4f} €")
            bs3.metric("Écart Relatif", f"{diff_pct:+.2f}%",
                    delta="Bergomi plus cher" if diff_pct > 0 else "Bergomi moins cher")
            
            st.caption("Black-Scholes ne capture pas la volatilité stochastique ni le skew dynamique du modèle Bergomi")
        else:
            st.info(f"**{product_type}** : Produit exotique sans équivalent Black-Scholes direct")
        
        # --- VISUALISATION ---
        tab1, tab2 = st.tabs(["Scénarios", "Note"])
        
        with tab1:
            fig = go.Figure()
            subset = S_paths[:, :150] # 150 trajectoires
            max_vals = np.max(subset, axis=0)
            min_vals = np.min(subset, axis=0)
            
            for i in range(150):
                color = '#00CC96' # Vert (Actif/Vivant)
                opacity = 0.1
                
                # Logique Barrière OUT (Mort si touche)
                if "Up-and-Out" in product_type and max_vals[i] >= barrier_lvl:
                    color = '#EF553B'; opacity = 0.05
                elif "Down-and-Out" in product_type and min_vals[i] <= barrier_lvl:
                    color = '#EF553B'; opacity = 0.05
                    
                # Logique Barrière IN (Mort si touche PAS)
                elif "Down-and-In" in product_type:
                    if min_vals[i] <= barrier_lvl: 
                        color = '#00CC96' # A touché = Activé = Vert
                        opacity = 0.2
                    else:
                        color = '#EF553B' # N'a pas touché = Inactif = Rouge
                        opacity = 0.05
                        
                fig.add_trace(go.Scatter(y=subset[:, i], mode='lines', line=dict(color=color, width=1), opacity=opacity, showlegend=False))
                
            fig.add_hline(y=Strike, line_dash="dot", line_color="white", annotation_text="Strike")
            if barrier_lvl:
                fig.add_hline(y=barrier_lvl, line_dash="dash", line_color="yellow", annotation_text="Barrière", line_width=3)
                
            fig.update_layout(title="Monte Carlo (Vert = Option Active, Rouge = Option Inactive)", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.info(f"""
            **Analyse spécifique pour {product_type} :**
            
            * **Put Down-and-In :**
                * C'est une assurance contre un krach. L'option n'existe QUE si le marché baisse fort (touche la barrière).
                * **Test de parité :** Prix(Put Down-In) + Prix(Put Down-Out) $\\approx$ Prix(Put Vanille).
                
            * **Digitale (Binaire) :**
                * Le prix représente grossièrement la **Probabilité risque-neutre** de finir au-dessus du strike.
                * *Attention :* Le Gamma est souvent instable (bruit Monte Carlo) car le payoff est une "marche d'escalier" brutale.

            """)
