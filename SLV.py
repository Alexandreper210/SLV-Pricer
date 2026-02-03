import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from scipy.stats import norm
from scipy.optimize import brentq
import yfinance as yf
import pandas as pd
import json
from datetime import datetime
from scipy.optimize import minimize, differential_evolution



# ==========================================
# Fonctions de Base
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
    
    if product_type == "Vanilla Call":
        return np.maximum(final_spot - K, 0)
        
    elif product_type == "Vanilla Put":
        return np.maximum(K - final_spot, 0)
        
    elif product_type == "Barrier Call (Up-and-Out)":
        max_reached = np.max(S_paths, axis=0)
        is_alive = max_reached < barrier_level # Doit rester SOUS la barrière
        return np.maximum(final_spot - K, 0) * is_alive
        
    elif product_type == "Barrier Put (Down-and-Out)":
        min_reached = np.min(S_paths, axis=0)
        is_alive = min_reached > barrier_level # Doit rester AU-DESSUS
        return np.maximum(K - final_spot, 0) * is_alive
        
    # --- NOUVEAU : Put Down-and-In ---
    elif product_type == "Barrier Put (Down-and-In)":
        min_reached = np.min(S_paths, axis=0)
        is_activated = min_reached <= barrier_level # Doit TOUCHER la barrière pour s'activer
        return np.maximum(K - final_spot, 0) * is_activated
        
    # --- NOUVEAU : Digitale (Call Binaire) ---
    elif product_type == "Digital Call (Cash-or-Nothing)":
        # Paie 1€ si S_T > K, sinon 0
        return 1.0 * (final_spot > K)
        
    elif product_type == "Asian Call (Average Price)":
        average_spot = np.mean(S_paths, axis=0)
        return np.maximum(average_spot - K, 0)
        
    elif product_type == "Lookback Call (Fixed Strike)":
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
# FONCTIONS DE CALIBRATION
# ==========================================

def get_market_data_yahoo(ticker='SPY', target_dte=30):
    stock = yf.Ticker(ticker)
    # Récupération sécurisée du Spot
    hist = stock.history(period='5d') # On prend 5j pour être sûr d'avoir un prix
    if hist.empty: raise ValueError(f"Ticker {ticker} non trouvé.")
    S0 = hist['Close'].iloc[-1]
    
    expirations = stock.options
    if not expirations: raise ValueError(f"Pas d'options pour {ticker}")
    
    today = datetime.now()
    target_date = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - today).days - target_dte))
    
    opt_chain = stock.option_chain(target_date)
    T = (datetime.strptime(target_date, '%Y-%m-%d') - today).days / 365.25

    # --- NETTOYAGE ROBUSTE ---
    def clean_df(df):
        df = df.copy()
        # On garde les colonnes essentielles
        cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']
        df = df[cols]
        
        # Fallback pour le prix : si bid/ask sont à 0, on prend lastPrice
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df.loc[df['mid_price'] <= 0, 'mid_price'] = df['lastPrice']
        
        # Filtres minimums (on enlève les IV aberrantes)
        mask = (df['impliedVolatility'] > 0.001) & (df['impliedVolatility'] < 2.5)
        return df[mask]

    calls_clean = clean_df(opt_chain.calls)
    puts_clean = clean_df(opt_chain.puts)
    
    return {
        'ticker': ticker, 'S0': S0, 'T': T,
        'calls': calls_clean, 'puts': puts_clean,
        'expiration': target_date,
        'dte': (datetime.strptime(target_date, '%Y-%m-%d') - today).days
    }

def prepare_calibration_data(market_data, moneyness_range=(0.80, 1.20)):
    """
    Prépare les données pour la calibration
    """
    S0 = market_data['S0']
    T = market_data['T']
    
    calls = market_data['calls'].copy()
    puts = market_data['puts'].copy()
    
    calls['type'] = 'call'
    puts['type'] = 'put'
    
    calls['market_price'] = calls['mid_price']
    puts['market_price'] = puts['mid_price']
    
    calls['market_iv'] = calls['impliedVolatility']
    puts['market_iv'] = puts['impliedVolatility']
    
    # Fusionner
    options = pd.concat([
        calls[['strike', 'market_price', 'market_iv', 'type']],
        puts[['strike', 'market_price', 'market_iv', 'type']]
    ])
    
    # Calculer moneyness
    options['moneyness'] = options['strike'] / S0
    
    # Filtrer
    options = options[
        (options['moneyness'] >= moneyness_range[0]) &
        (options['moneyness'] <= moneyness_range[1]) &
        (options['market_price'] > 0.05) &
        (options['market_iv'] > 0.05) &
        (options['market_iv'] < 2.5)
    ]
    
    options['S0'] = S0
    options['T'] = T
    
    return options.reset_index(drop=True)


def calibration_objective_weighted(params, market_data, N_paths=5000):
    """
    Fonction objectif pour la calibration
    """
    theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol = params
    
    if xi0 <= 0 or vol_of_vol <= 0:
        return 1e10
    
    model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
    
    S0 = market_data['S0'].iloc[0]
    T = market_data['T'].iloc[0]
    
    try:
        t_grid, S_paths, V_paths = model.simulate(S0, T, 100, N_paths, seed=42)
    except:
        return 1e10
    
    total_error = 0
    total_weight = 0
    
    for idx, row in market_data.iterrows():
        K = row['strike']
        market_iv = row['market_iv']
        option_type = row['type']
        moneyness = row['moneyness']
        
        # Pondération gaussienne 
        weight = np.exp(-1.0 * (moneyness - 1.0)**2)+0.4
        
        # Prix modèle
        if option_type == 'call':
            payoff = np.maximum(S_paths[-1, :] - K, 0)
        else:
            payoff = np.maximum(K - S_paths[-1, :], 0)
        
        price_model = np.mean(payoff)
        iv_model = implied_vol_solver_smart(price_model, S0, K, T, 0, option_type)
        
        if not np.isnan(iv_model) and iv_model > 0:
            error = weight * (iv_model - market_iv) ** 2
            total_error += error
            total_weight += weight
    
    if total_weight == 0:
        return 1e10
    
    weighted_rmse = np.sqrt(total_error / total_weight)
    return weighted_rmse


def calibrate_bergomi_hierarchical(market_data, progress_callback=None):
    """
    Calibration hiérarchique corrigée (5 paramètres en Step 2)
    """
    # ÉTAPE 1 : Calibration ATM (xi0, vol_of_vol, rhoSVol initial)
    if progress_callback:
        progress_callback("Step 1/2 : Calibration ATM...")
    
    atm_data = market_data[
        (market_data['moneyness'] >= 0.95) & 
        (market_data['moneyness'] <= 1.05)
    ]
    # SÉCURITÉ : Si vide, on prend les 5 options les plus proches du spot
    if atm_data.empty:
        market_data['dist_to_atm'] = abs(market_data['moneyness'] - 1.0)
        atm_data = market_data.nsmallest(5, 'dist_to_atm')
    
    # On récupère S0 et T depuis les données complètes pour éviter le crash .iloc[0]
    S0_global = market_data['S0'].iloc[0]
    T_global = market_data['T'].iloc[0]
    
    def objective_step1(params_step1):
        xi0, vol_of_vol, rhoSVol = params_step1
        # On utilise des valeurs par défaut pour le reste le temps de caler l'ATM
        full_params = [0.4, 0.0, 4.0, 0.1, vol_of_vol, xi0, rhoSVol]
        return calibration_objective_weighted(full_params, atm_data, 3000)
    
    # x0 : 3 valeurs | bounds : 3 couples -> OK
    result_step1 = minimize(
        objective_step1,
        [0.02, 5.0, -0.75],
        bounds=[(0.001, 0.5), (0.1, 5.0), (-0.99, 0.99)],
        method='L-BFGS-B',
        options={'maxiter': 30}
    )
    
    xi0_opt, vol_of_vol_opt, rhoSVol_atm = result_step1.x
    
    # ÉTAPE 2 : Calibration Skew (On ré-ajuste rhoSVol pour cambrer le smile)
    if progress_callback:
        progress_callback("Step 2/2 : Calibration Skew...")
    
    def objective_step2(params_step2):
        # IMPORTANT : On déballe bien les 5 paramètres
        theta, rhoXY, kappaX, kappaY, rhoSVol = params_step2
        full_params = [theta, rhoXY, kappaX, kappaY, vol_of_vol_opt, xi0_opt, rhoSVol]
        return calibration_objective_weighted(full_params, market_data, 5000)
    
    # x0 : 5 valeurs (On ajoute rhoSVol_atm à la fin)
    x0_step2 = [0.4, 0.0, 4.0, 0.1, -0.75]
    
    # bounds : 5 couples -> DOIT ÊTRE ÉGAL À LA LONGUEUR DE x0
    bounds_step2 = [
        (0.0, 1.0),      # theta
        (-0.99, 0.99),   # rhoXY
        (0.1, 25.0),     # kappaX
        (0.01, 5.0),     # kappaY
        (-0.99, 0.90)      # rhoSVol (Crucial pour le skew)
    ]
    
    result_step2 = minimize(
        objective_step2,
        x0_step2,
        bounds=bounds_step2,
        method='L-BFGS-B',
        options={'maxiter': 50} # On augmente pour plus de précision sur le Skew
    )
    
    theta_opt, rhoXY_opt, kappaX_opt, kappaY_opt, rhoSVol_final = result_step2.x
    
    optimal_params = [theta_opt, rhoXY_opt, kappaX_opt, kappaY_opt, 
                     vol_of_vol_opt, xi0_opt, rhoSVol_final]
    
    return {
        'params': optimal_params,
        'param_dict': {
            'theta': theta_opt,
            'rhoXY': rhoXY_opt,
            'kappaX': kappaX_opt,
            'kappaY': kappaY_opt,
            'vol_of_vol': vol_of_vol_opt,
            'xi0': xi0_opt,
            'rhoSVol': rhoSVol_final
        },
        'error': result_step2.fun,
        'market_data': market_data
    }

def validate_calibration(calibration_result):
    """
    Calcule les métriques de validation
    """
    params = calibration_result['params']
    market_data = calibration_result['market_data']
    
    theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol = params
    model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
    
    S0 = market_data['S0'].iloc[0]
    T = market_data['T'].iloc[0]
    
    t_grid, S_paths, V_paths = model.simulate(S0, T, 100, 10000, seed=42)
    
    results = []
    
    for idx, row in market_data.iterrows():
        K = row['strike']
        market_iv = row['market_iv']
        market_price = row['market_price']
        
        # 1. Standardisation stricte (Extraction du type pur)
        raw_type = str(row['type']).lower()
        clean_type = 'call' if 'call' in raw_type else 'put' #

        # 2. Calcul du Payoff avec le type propre
        if clean_type == 'call':
            payoff = np.maximum(S_paths[-1, :] - K, 0)
        else:
            payoff = np.maximum(K - S_paths[-1, :], 0)
        
        price_model = np.mean(payoff)
        
        # 3. CRUCIAL : Passer clean_type au solveur pour qu'il utilise la bonne formule BS
        iv_model = implied_vol_solver_smart(price_model, S0, K, T, 0, clean_type)
        
        results.append({
            'strike': K,
            'type': clean_type, # On stocke le type propre pour les graphiques Plotly
            'market_price': market_price,
            'model_price': price_model,
            'market_iv': market_iv,
            'model_iv': iv_model if not np.isnan(iv_model) else market_iv,
            'price_error': price_model - market_price,
            'iv_error': (iv_model - market_iv) if not np.isnan(iv_model) else 0
        })
    
    results_df = pd.DataFrame(results)
    
    rmse_iv = np.sqrt(np.mean(results_df['iv_error']**2))
    mae_iv = np.mean(np.abs(results_df['iv_error']))
    
    return results_df, rmse_iv, mae_iv


def plot_calibration_results(validation_results, market_data):
    """
    Crée les graphiques de validation
    """
    S0 = market_data['S0'].iloc[0]
    
    calls_val = validation_results[validation_results['type'] == 'call']
    puts_val = validation_results[validation_results['type'] == 'put']
    
    # GRAPHIQUE 1 : Smile de Volatilité
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=calls_val['strike'], y=calls_val['market_iv'] * 100,
        mode='markers', name='Calls Marché',
        marker=dict(size=10, color='#EF553B', symbol='circle')
    ))
    
    fig1.add_trace(go.Scatter(
        x=calls_val['strike'], y=calls_val['model_iv'] * 100,
        mode='lines+markers', name='Calls Bergomi',
        marker=dict(size=6, color='#FF6B6B', symbol='x'),
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    fig1.add_trace(go.Scatter(
        x=puts_val['strike'], y=puts_val['market_iv'] * 100,
        mode='markers', name='Puts Marché',
        marker=dict(size=10, color='#636EFA', symbol='circle')
    ))
    
    fig1.add_trace(go.Scatter(
        x=puts_val['strike'], y=puts_val['model_iv'] * 100,
        mode='lines+markers', name='Puts Bergomi',
        marker=dict(size=6, color='#00CC96', symbol='x'),
        line=dict(color='#00CC96', width=2, dash='dash')
    ))
    
    fig1.add_vline(x=S0, line_dash="dot", line_color="white", 
                   annotation_text=f"ATM ({S0:.1f})", annotation_position="top")
    
    fig1.update_layout(
        title="Volatility Smile: Market vs Calibrated Model",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        template="plotly_dark",
        height=500,
        hovermode='x unified'
    )
    
    # GRAPHIQUE 2 : Erreurs
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=validation_results['strike'],
        y=validation_results['iv_error'] * 100,
        marker_color=['#EF553B' if e > 0 else '#00CC96' for e in validation_results['iv_error']],
        name='Erreur IV',
        text=[f"{e*100:.2f}%" for e in validation_results['iv_error']],
        textposition='outside'
    ))
    
    fig2.update_layout(
        title="Calibration Residuals (Implied Volatility)",
        xaxis_title="Strike",
        yaxis_title="Error (%)",
        template="plotly_dark",
        height=400
    )
    
    # GRAPHIQUE 3 : Prix Marché vs Modèle
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=validation_results['market_price'],
        y=validation_results['model_price'],
        mode='markers',
        marker=dict(size=10, color='#AB63FA'),
        name='Options',
        text=[f"K={k:.0f}" for k in validation_results['strike']],
        hovertemplate='Marché: %{x:.2f}€<br>Modèle: %{y:.2f}€<br>%{text}'
    ))
    
    max_price = max(validation_results['market_price'].max(), 
                    validation_results['model_price'].max())
    fig3.add_trace(go.Scatter(
        x=[0, max_price], y=[0, max_price],
        mode='lines', line=dict(dash='dash', color='white', width=2),
        name='Fit Parfait', showlegend=True
    ))
    
    fig3.update_layout(
        title="Price: Market vs Model",
        xaxis_title="Market Price (€)",
        yaxis_title="Model Price (€)",
        template="plotly_dark",
        height=500
    )
    
    return fig1, fig2, fig3



# ==========================================
# 3. INTERFACE DE STRUCTURATION
# ==========================================

st.set_page_config(page_title="Bergomi SLV Pricer", layout="wide")
# --- AJOUT DU PROFIL LINKEDIN DANS LA SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>Contact</h3>
        <p>Project developed by<b>Alexandre Perier</b></p>
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
    st.header("1. Model Parameters")
    xi0 = st.number_input("Initial Variance (xi0)", 0.01, 0.25, 0.04, step=0.01)
    target_vol = np.sqrt(xi0)
    st.caption(f"Target Volatility: **{target_vol:.2%}**")
    
    vol_of_vol = st.slider("Vol-of-Vol (sigma)", 0.0, 2.0, 1.0)
    rhoSVol = st.slider("Spot/Vol Correlation", -1.0, 1.0, -0.7)

    with st.expander("Factor Parameters"):
        theta = st.slider("Factor Mixing (theta)", 0.0, 1.0, 0.4)
        kappaX = st.number_input("Mean Reversion X (kappaX)", 0.1, 10.0, 4.0)
        kappaY = st.number_input("Mean Reversion Y (kappaY)", 0.1, 5.0, 0.1)
        rhoXY = st.slider("Corr X,Y (rhoXY)", -1.0, 1.0, 0.0)

    st.markdown("---")
    # SÉLECTEUR DE MODE
    app_mode = st.radio("Mode :", [
    "Pricing & Path Simulation", 
        "Volatility Surface (3D)", 
        "Multi-Asset Pricing & Greeks",
        "Market Data Calibration"
])

# ==========================================
# MODE 1 : PRICING & TRAJECTOIRES
# ==========================================
if app_mode == "Pricing & Path Simulation":
    st.sidebar.header("Option Parameters")
    S0 = st.sidebar.number_input("Spot", 100.0)
    Strike = st.sidebar.number_input("Strike", 100.0)
    T = st.sidebar.number_input("Maturity (Years)", 1.0)
    N_paths = st.sidebar.selectbox("Simulations", [1000, 5000, 10000, 20000], index=2)
    N_steps = st.sidebar.slider("Time Steps", 10, 300, 100)

    if st.button("Run Simulation (Single)", type="primary"):
        model = BergomiSLV(theta, rhoXY, kappaX, kappaY, vol_of_vol, xi0, rhoSVol)
        
        with st.spinner('Simulation in progress...'):
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
        c1.metric("Call Price", f"{price:.4f} €", f"± {1.96*std_err:.4f}")
        c2.metric("Average Volatility", f"{avg_vol_out:.2%}", f"Target: {target_vol:.2%}")
        c3.metric("Execution Time", f"{end-start:.3f} s")

        
        st.markdown("---")
        st.subheader("Black-Scholes Comparison")
        bs_price_call = bs_price(target_vol, S0, Strike, T, 0, 'call')
        diff_pct = ((price - bs_price_call) / bs_price_call) * 100 if bs_price_call > 0 else 0

        col_bs1, col_bs2, col_bs3 = st.columns(3)
        col_bs1.metric("BS Price", f"{bs_price_call:.4f} €")
        col_bs2.metric("Bergomi vs BS Spread", f"{price - bs_price_call:+.4f} €")
        col_bs3.metric("Relative Spread", f"{diff_pct:+.2f}%", 
                    delta="More expensive" if diff_pct > 0 else "Cheaper")
        if abs(diff_pct) > 5:
            st.info(f"**Significant Spread ({diff_pct:+.1f}%)** : The Bergomi model's stochastic volatility captures market effects that Black-Scholes ignores, such as volatility clustering and dynamic skew.")
            st.success("Convergence close to Black-Scholes (flat volatility surface).")

        # --- ONGLETS ---
        tab1, tab2, tab3 = st.tabs(["Spot Paths", "Stochastic Variance", "Terminal Distribution"])

        with tab1:
            fig_spot = go.Figure()
            for i in range(min(50, N_paths)):
                fig_spot.add_trace(go.Scatter(x=t_grid, y=S_paths[:, i], mode='lines', line=dict(width=1), opacity=0.4, showlegend=False))
            fig_spot.add_trace(go.Scatter(x=t_grid, y=np.mean(S_paths, axis=1), mode='lines', line=dict(color='white', width=3), name='Moyenne'))
            fig_spot.update_layout(title="Spot Price Diffusion", xaxis_title="Time (T)", yaxis_title="Spot", template="plotly_dark")
            st.plotly_chart(fig_spot, width='stretch')

        with tab2:
            fig_vol = go.Figure()
            for i in range(min(50, N_paths)):
                fig_vol.add_trace(go.Scatter(x=t_grid, y=np.sqrt(V_paths[:, i]), mode='lines', line=dict(color='orange', width=1), opacity=0.4, showlegend=False))
            fig_vol.add_trace(go.Scatter(x=t_grid, y=np.mean(np.sqrt(V_paths), axis=1), mode='lines', line=dict(color='red', width=3), name='Vol Moyenne'))
            fig_vol.update_layout(title="Volatility Diffusion", xaxis_title="Time (T)", yaxis_title="Volatility", template="plotly_dark")
            st.plotly_chart(fig_vol, width='stretch')

        with tab3:
            fig_hist = go.Figure(data=[go.Histogram(x=S_paths[-1, :], nbinsx=60, histnorm='probability', marker_color='#636EFA')])
            fig_hist.add_vline(x=Strike, line_dash="dash", line_color="red", annotation_text="Strike")
            fig_hist.update_layout(title=f"Terminal Distribution at T={T}", xaxis_title="Final Price", template="plotly_dark")
            st.plotly_chart(fig_hist, width='stretch')

# ==========================================
# MODE 2 : SURFACE DE VOLATILITÉ
# ==========================================
elif app_mode == "Volatility Surface (3D)":
    st.sidebar.header("Surface Parameters")
    # Logique de sélection des paramètres ### ---
    # Par défaut, on prend les valeurs des sliders de la sidebar
    params_to_use = {
        'theta': theta, 'rhoXY': rhoXY, 'kappaX': kappaX, 
        'kappaY': kappaY, 'vol_of_vol': vol_of_vol, 
        'xi0': xi0, 'rhoSVol': rhoSVol
    }

    # Si une calibration a été effectuée, on propose de l'utiliser
    if 'calibration_result' in st.session_state:
        st.sidebar.markdown("---")
        use_calibrated = st.sidebar.checkbox("Use Calibrated Parameters", value=True)
        if use_calibrated:
            params_to_use = st.session_state['calibration_result']['param_dict']
            st.sidebar.success("Mode : Calibrated Parameters (Yahoo) Active")
            # Optionnel : Afficher le RMSE actuel pour rappel
            st.sidebar.caption(f"Fit Quality: RMSE {st.session_state['calibration_result']['error']:.4f}")
    else:
        st.sidebar.info("No calibration found. Use the dedicated tab to calibrate on real market data.")


    S0 = st.sidebar.number_input("Spot", 100.0)
    T_max = st.sidebar.number_input("Max Maturity (Years)", 2.0)
    N_paths_surf = st.sidebar.selectbox("Simulations", [5000, 10000, 20000], index=1)

    if st.button("Generate Surface", type="primary"):
        model = BergomiSLV(params_to_use['theta'], params_to_use['rhoXY'], params_to_use['kappaX'], 
            params_to_use['kappaY'], params_to_use['vol_of_vol'], 
            params_to_use['xi0'], params_to_use['rhoSVol'])
        
        with st.spinner('Building Volatility Surface...'):
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

        st.success("Surface successfully generated")
        tab1, tab2, tab3 = st.tabs(["3D Surface", "Term Structure", "Skew / Smile Evolution"])
        with tab1:
            fig_3d = go.Figure(data=[go.Surface(z=Z_surface, x=strikes, y=maturities, colorscale='Jet')])
            fig_3d.update_layout(title="Volatility Surface", scene=dict(xaxis_title='Strike', yaxis_title='Maturité', zaxis_title='Vol %'), height=600, template="plotly_dark")
            st.plotly_chart(fig_3d, width='stretch')
            
        with tab2:
            mid = len(strikes)//2
            fig_ts = go.Figure(go.Scatter(x=maturities, y=Z_surface[:, mid], mode='lines+markers', line=dict(color='#FF4B4B')))
            fig_ts.update_layout(title="Term Structure (ATM)", xaxis_title="Maturity", yaxis_title="Vol Imp %", template="plotly_dark")
            st.plotly_chart(fig_ts, width='stretch')
            
        with tab3:
            fig_skew_multi = go.Figure()
            # On boucle sur chaque maturité pour tracer tous les smiles
            for i, T in enumerate(maturities):
                fig_skew_multi.add_trace(go.Scatter(
                    x=strikes, 
                    y=Z_surface[i, :], 
                    mode='lines+markers',
                    name=f"T = {T:.2f} ans"
                ))
    
            fig_skew_multi.update_layout(
                title="Skew Evolution (Volatility Smile) by Maturity",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility (%)",
                template="plotly_dark",
                legend_title="Maturities"
                )
            st.plotly_chart(fig_skew_multi, width='stretch')
            

# ==========================================
# MODE 3 : PRICING MULTI-PRODUITS & GRECS
# ==========================================
elif app_mode == "Multi-Asset Pricing & Greeks":
    st.sidebar.header("2. Pricing")

    # --- LOGIQUE DE SÉLECTION & COMPARAISON ---
    manual_params = {
        'theta': theta, 'rhoXY': rhoXY, 'kappaX': kappaX, 
        'kappaY': kappaY, 'vol_of_vol': vol_of_vol, 
        'xi0': xi0, 'rhoSVol': rhoSVol
    }

    calib_exists = 'calibration_result' in st.session_state
    
    # Toggle pour activer la comparaison
    do_comparison = False
    if calib_exists:
        st.sidebar.markdown("---")
        do_comparison = st.sidebar.checkbox("Comparison Mode (Manual vs Market", value=True)
        if not do_comparison:
            mode_source = st.sidebar.radio("Single Source Selection:", ["Manual Sliders", "Last Calibration"], index=1)
            params_final = st.session_state['calibration_result']['param_dict'] if mode_source == "Last Calibration" else manual_params
        else:
            st.sidebar.info("Application will run both models simultaneously")
    else:
        params_final = manual_params
        st.sidebar.info("Calibration missing: Manual mode active.")

    st.sidebar.markdown("---")

    # Inputs classiques
    product_type = st.sidebar.selectbox("Product Type", 
        ["Vanilla Call", "Vanilla Put",
         "Barrier Call (Up-and-Out)", "Barrier Put (Down-and-Out)", "Barrier Put (Down-and-In)",
         "Digital Call (Cash-or-Nothing)",
         "Asian Call (Average Price)", "Lookback Call (Fixed Strike)"])
    
    S0 = st.sidebar.number_input("Initial Spot Price", 100.0)
    Strike = st.sidebar.number_input("Strike", 100.0)
    T = st.sidebar.number_input("Maturity", 1.0)
    
    barrier_lvl = None
    if "Barrier" in product_type:
        def_bar = 130.0 if "Up" in product_type else 80.0
        barrier_lvl = st.sidebar.number_input("Barrier Level", 0.0, 300.0, def_bar)

    if st.button("Price & Calculate Greeks", type="primary"):
        results_dict = {}
        
        with st.spinner('Running Monte Carlo Simulation..'):
            # 1. Calcul Manuel (Toujours effectué pour comparaison ou mode manuel)
            model_man = BergomiSLV(**manual_params)
            p_m, d_m, g_m, v_m, s_m = compute_greeks(model_man, S0, Strike, T, 100, 50000, product_type, barrier_lvl)
            results_dict['Manuel'] = {'price': p_m, 'delta': d_m, 'gamma': g_m * S0, 'vega': v_m, 'paths': s_m}

            # 2. Calcul Calibré (Si activé)
            if calib_exists and (do_comparison or mode_source == "Last Calibration"):
                cal_p = st.session_state['calibration_result']['param_dict']
                model_cal = BergomiSLV(**cal_p)
                p_c, d_c, g_c, v_c, s_c = compute_greeks(model_cal, S0, Strike, T, 100, 50000, product_type, barrier_lvl)
                results_dict['Marché'] = {'price': p_c, 'delta': d_c, 'gamma': g_c * S0, 'vega': v_c, 'paths': s_c}

        # --- AFFICHAGE DES RÉSULTATS ---
        st.markdown(f"### Results : {product_type}")
        
        if do_comparison and 'Marché' in results_dict:
            # Affichage en tableau comparatif
            res_man = results_dict['Manuel']
            res_mar = results_dict['Marché']
            
            comp_df = pd.DataFrame({
                "Métrique": ["Prix", "Δ Delta", "Γ Gamma (%)", "ν Vega"],
                "Mode Manuel": [f"{res_man['price']:.4f} €", f"{res_man['delta']:.1%}", f"{res_man['gamma']:.2f}%", f"{res_man['vega']:.3f}"],
                "Mode Marché (Calibré)": [f"{res_mar['price']:.4f} €", f"{res_mar['delta']:.1%}", f"{res_mar['gamma']:.2f}%", f"{res_mar['vega']:.3f}"]
            })
            
            c1, c2 = st.columns([2, 1])
            c1.table(comp_df)
            
            # Écart de prix
            diff = res_mar['price'] - res_man['price']
            c2.metric("Price Spread (Market - Manual)", f"{diff:+.4f} €", delta_color="inverse")
            c2.caption("A positive spread indicates the market prices in more risk (vol-of-vol or skew) than your manual settings.")
        
        else:
            # Affichage classique en colonnes
            active_res = results_dict['Marché'] if 'Marché' in results_dict else results_dict['Manuel']
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Prix", f"{active_res['price']:.4f} €")
            g2.metric("Δ Delta", f"{active_res['delta']:.1%}")
            g3.metric("Γ Gamma", f"{active_res['gamma']:.2f}%")
            g4.metric("ν Vega", f"{active_res['vega']:.3f}")

        st.markdown("---")
        
        # --- VISUALISATION (Utilise les trajectoires du mode actif) ---
        main_paths = results_dict['Marché']['paths'] if 'Marché' in results_dict else results_dict['Manuel']['paths']
        
        tab1, tab2 = st.tabs(["Path Scenarios", "Analysis Note"])
        with tab1:
            fig = go.Figure()
            subset = main_paths[:, :150]
            max_v, min_v = np.max(subset, axis=0), np.min(subset, axis=0)
            
            for i in range(150):
                color = '#00CC96'
                opacity = 0.1
                if "Up-and-Out" in product_type and max_v[i] >= barrier_lvl: color = '#EF553B'; opacity = 0.05
                elif "Down-and-Out" in product_type and min_v[i] <= barrier_lvl: color = '#EF553B'; opacity = 0.05
                elif "Down-and-In" in product_type:
                    color = '#00CC96' if min_v[i] <= barrier_lvl else '#EF553B'
                    opacity = 0.2 if min_v[i] <= barrier_lvl else 0.05
                
                fig.add_trace(go.Scatter(y=subset[:, i], mode='lines', line=dict(color=color, width=1), opacity=opacity, showlegend=False))
            
            fig.add_hline(y=Strike, line_dash="dot", line_color="white", annotation_text="Strike")
            if barrier_lvl: fig.add_hline(y=barrier_lvl, line_dash="dash", line_color="yellow", annotation_text="Barrière")
            fig.update_layout(title="Monte Carlo Path Diffusion (Primary Model)", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.info("Comparison mode helps quantify how market calibration affects the cost of barrier options and exotic risks.")
# ==========================================
# MODE 4 : CALIBRATION SUR DONNÉES RÉELLES
# ==========================================
elif app_mode == "Market Data Calibration":
    
    st.header("Bergomi Model Calibration on Live Market Data")
    
    st.markdown("""
    This module automatically calibrates the Bergomi model parameters to replicate 
    option prices observed in the market (via Yahoo Finance).
    
    **Process:**
    1. Download live option chain data
    2. Hierarchical calibration (ATM first, then Skew/Smile)
    3. Fit validation and visualization
    """)
    
    # ===== SIDEBAR : PARAMÈTRES =====
    with st.sidebar:
        st.subheader("Calibration Settings")
        ticker = st.text_input("Ticker", "SPY", help="SPY, QQQ, AAPL, TSLA...")
        target_dte = st.slider("Days to Expiration (DTE)", 20, 90, 30)
        moneyness_min = st.slider("Moneyness Min", 0.70, 0.95, 0.85, 0.05)
        moneyness_max = st.slider("Moneyness Max", 1.05, 1.30, 1.15, 0.05)
    
    # ===== ÉTAPE 1 : TÉLÉCHARGEMENT =====
    st.subheader("Step 1: Market Data Acquisition")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Download Market Data", type="primary", width='stretch'):
            with st.spinner(f"Fetching {ticker} option chains from Yahoo Finance..."):
                try:
                    market_data_raw = get_market_data_yahoo(ticker, target_dte)
                    st.session_state['market_data_raw'] = market_data_raw
                    st.session_state['calibration_done'] = False
                    
                    st.success(f"Data successfully retrieved!")
                    
                    # NOUVEAU : Détails des données
                    st.markdown("### Market Data Summary")
                    
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown("**Calls:**")
                        st.write(f"• {len(market_data_raw['calls'])} options")
                        st.write(f"• Strikes: {market_data_raw['calls']['strike'].min():.0f}$ → {market_data_raw['calls']['strike'].max():.0f}$")
                        st.write(f"• Vol IV Avg.: {market_data_raw['calls']['impliedVolatility'].mean():.2%}")
                        st.write(f"• Total volume: {market_data_raw['calls']['volume'].sum():,.0f}")
                    
                    with summary_col2:
                        st.markdown("**Puts:**")
                        st.write(f"• {len(market_data_raw['puts'])} options")
                        st.write(f"• Strikes: {market_data_raw['puts']['strike'].min():.0f}$ → {market_data_raw['puts']['strike'].max():.0f}$")
                        st.write(f"• Vol IV Avg: {market_data_raw['puts']['impliedVolatility'].mean():.2%}")
                        st.write(f"• Total volume: {market_data_raw['puts']['volume'].sum():,.0f}")
                    
                    st.markdown("---")
                    
                    # Métriques
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Spot", f"{market_data_raw['S0']:.2f}$")
                    m2.metric("Expiry Date", market_data_raw['expiration'])
                    m3.metric("DTE", f"{market_data_raw['dte']} jours")
                    m4.metric("Maturity (T)", f"{market_data_raw['T']:.3f} ans")
                    
                except Exception as e:
                    st.error(f"Download Error: {e}")
                    st.info("Troubleshooting Tips:")
                    st.write("• Verify the ticker symbol is valid (e.g., SPY, AAPL)")
                    st.write("• Ensure options are available for this asset")
                    st.write("• Check your internet connection")
        
    with col2:
        if st.button("Reset Engine"):
            if 'market_data_raw' in st.session_state:
                del st.session_state['market_data_raw']
            if 'calibration_result' in st.session_state:
                del st.session_state['calibration_result']
            st.rerun()
    
    # ===== AFFICHAGE DES DONNÉES =====
    if 'market_data_raw' in st.session_state:
        data_raw = st.session_state['market_data_raw']
        
        with st.expander("Raw Data Preview", expanded=False):
            tab1, tab2 = st.tabs(["Calls", "Puts"])
            
            with tab1:
                st.dataframe(
                    data_raw['calls'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head(10),
                    width='stretch'
                )
            
            with tab2:
                st.dataframe(
                    data_raw['puts'][['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head(10),
                    width='stretch'
                )
        
        # ===== ÉTAPE 2 : CALIBRATION =====
        st.markdown("---")
        st.subheader("Step 2: Model Calibration")
        
        if st.button("Run Hierarchical Calibration", type="primary", width='stretch'):
            
            # Préparation des données
            with st.spinner("Filtering and weighting data..."):
                calib_data = prepare_calibration_data(
                    data_raw, 
                    moneyness_range=(moneyness_min, moneyness_max)
                )
                
                if len(calib_data) < 10:
                    st.error(f"Only {len(calib_data)} options selected. Please widen the moneyness range.")
                    st.stop()
                
                st.info(f"{len(calib_data)} options selected for calibration engine.")
            
            # Calibration
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message):
                if "Step 1" in message:
                    progress_bar.progress(30)
                elif "Step 2" in message:
                    progress_bar.progress(60)
                status_text.text(message)
            
            start_time = time.time()
            
            with st.spinner("Calibrating model (may take up to 10 minutes)..."):
                try:
                    result = calibrate_bergomi_hierarchical(calib_data, update_progress)
                    elapsed = time.time() - start_time
                    
                    progress_bar.progress(100)
                    status_text.text(f"Calibration completed in {elapsed:.1f}s")
                    
                    st.session_state['calibration_result'] = result
                    st.session_state['calibration_data'] = calib_data
                    st.session_state['calibration_done'] = True
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Calibration Error: {e}")
                    st.stop()
        
        # ===== ÉTAPE 3 : RÉSULTATS =====
        if st.session_state.get('calibration_done', False):
            st.markdown("---")
            st.subheader("Step 3 Calibration Results")
            
            result = st.session_state['calibration_result']
            calib_data = st.session_state['calibration_data']
            
            # Paramètres optimaux
            st.markdown("### Optimal Parameters")
            
            params_col1, params_col2 = st.columns(2)
            
            with params_col1:
                st.markdown("**Volatility Parameters:**")
                st.metric("xi0 (Initiale Variance)", f"{result['param_dict']['xi0']:.4f}", 
                         f"Vol ≈ {np.sqrt(result['param_dict']['xi0'])*100:.1f}%")
                st.metric("vol_of_vol (σ)", f"{result['param_dict']['vol_of_vol']:.4f}")
                st.metric("rhoSVol (ρ)", f"{result['param_dict']['rhoSVol']:.4f}")
            
            with params_col2:
                st.markdown("**Factor Parameters:**")
                st.metric("theta (θ)", f"{result['param_dict']['theta']:.4f}")
                st.metric("kappaX (κ_X)", f"{result['param_dict']['kappaX']:.4f}")
                st.metric("kappaY (κ_Y)", f"{result['param_dict']['kappaY']:.4f}")
                st.metric("rhoXY", f"{result['param_dict']['rhoXY']:.4f}")
            
            # Validation
            st.markdown("### FitValidation")
            
            with st.spinner("Calculating validation metrics..."):
                validation_df, rmse, mae = validate_calibration(result)
            
            # Métriques de qualité
            qual_col1, qual_col2, qual_col3 = st.columns(3)
            
            with qual_col1:
                delta_color = "normal" if rmse < 0.02 else "inverse"
                st.metric("RMSE", f"{rmse:.4f}", f"{rmse*100:.2f}%", delta_color=delta_color)
            
            with qual_col2:
                st.metric("MAE", f"{mae:.4f}", f"{mae*100:.2f}%")
            
            with qual_col3:
                if rmse < 0.01:
                    quality = "EXCELLENT"
                elif rmse < 0.02:
                    quality = "GOOD"
                elif rmse < 0.05:
                    quality = "ACCEPTABLE"
                else:
                    quality = "POOR"
                st.metric("Fit Quality", quality)
            
            # Graphiques
            st.markdown("### Visualizations")
            
            fig1, fig2, fig3 = plot_calibration_results(validation_df, calib_data)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Volatility Smile", "Residuals", "Prices", "Data Table"])
            
            with tab1:
                st.plotly_chart(fig1, width='stretch')
                st.caption("● markers represent market data; × markers represent model reconstruction.")
            
            with tab2:
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Green bars = Model underestimation | Red bars = Model overestimation")
            
            with tab3:
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Points closer to the dashed line represent a better price fit.")
            
            with tab4:
                st.dataframe(
                    validation_df.style.format({
                        'strike': '{:.1f}',
                        'market_price': '{:.2f}€',
                        'model_price': '{:.2f}€',
                        'market_iv': '{:.2%}',
                        'model_iv': '{:.2%}',
                        'iv_error': '{:.2%}'
                    }).background_gradient(subset=['iv_error'], cmap='RdYlGn_r'),
                    width='stretch'
                )
            
            # Export
            st.markdown("### Parameter Export")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                params_json = json.dumps(result['param_dict'], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=params_json,
                    file_name=f"bergomi_params_{ticker}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    width='stretch'
                )
            
            with export_col2:
                # CSV des résultats
                csv = validation_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name=f"calibration_results_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
            
            # Code pour utiliser les paramètres
            with st.expander("Python Code to use Calibrated Parameters", expanded=False):
                st.code(f"""
# Initialize model with calibrated parameters
model_calibrated = BergomiSLV(
    theta={result['param_dict']['theta']:.6f},
    rhoXY={result['param_dict']['rhoXY']:.6f},
    kappaX={result['param_dict']['kappaX']:.6f},
    kappaY={result['param_dict']['kappaY']:.6f},
    vol_of_vol={result['param_dict']['vol_of_vol']:.6f},
    xi0={result['param_dict']['xi0']:.6f},
    rhoSVol={result['param_dict']['rhoSVol']:.6f}
)

# Price an option using these results
S0 = {data_raw['S0']:.2f}
K = 100  # Desired Strike
T = {data_raw['T']:.4f}

_, S_paths, _ = model_calibrated.simulate(S0, T, 100, 10000)
call_price = np.mean(np.maximum(S_paths[-1, :] - K, 0))
print(f"Call Price: {{call_price:.2f}}€")
                """, language="python")
    
    else:
        st.info("Please download market data first to begin the calibration engine.")


