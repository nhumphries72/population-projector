import numpy as np
import pandas as pd
import requests
import io
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st

st.set_page_config(page_title="Population Projector", layout="wide")
owid_url = "https://ourworldindata.org/grapher/population.csv"

@st.cache_data
def load_data():
    
    try:
        response = requests.get(owid_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        value_col = [c for c in df.columns if 'Population' in c][0]
        df = df.rename(columns={'Year': 'year', value_col: 'population', 'Entity': 'country'})
        return df
    
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
    

def get_country_data(df, country_name):
    
    mask = df['country'] == country_name
    filtered_df = df.loc[mask, ['year', 'population']].copy()
    return filtered_df


def generalized_logistic(t, K, B, M, nu, A):
    
    return A + ((K - A) / ((1 + np.exp(-B * (t - M)))**(1/nu)))


def optimize(x, y, progress_bar, iterations=1000):
    best_popt = None
    best_error = np.inf
    y_max = np.max(y)
    
    bounds = (
        [y_max, 0.01, 0, -np.inf, 0.8*y[0]],
        [5*y_max, 0.05, np.inf, np.inf, 1.2*y[0]]
    )
    
    for i in range(iterations):
        
        if i % 10 == 0:
            progress_bar.progress((i + 1) / iterations)
        
        p0_guess = [
            np.random.uniform(y_max, 2*y_max),
            np.random.uniform(0.01, 0.05),
            np.random.uniform(1900, 2150),
            np.random.uniform(0, 10),
            np.random.uniform(0.8*y[0], 1.2*y[0])
        ]
        
        try:
            popt, _ = curve_fit(
                generalized_logistic, x, y,
                p0=p0_guess,
                sigma=np.sqrt(y),
                absolute_sigma=True,
                bounds=bounds,
                method='trf',
                maxfev=5000
            )
            
            maxerror = np.max(100 * np.abs(generalized_logistic(x, *popt) - y) / y)
            meanerror = np.mean(100 * np.abs(generalized_logistic(x, *popt) - y) / y)
            
            if maxerror < best_error:
                best_error = maxerror
                best_popt = popt
                meanerr = meanerror
                
        except (RuntimeError, ValueError):
            continue
        
    return best_popt, best_error, meanerr


# ----------------------------------------------------------------------------------


st.title("Global Population Projector")
st.markdown("Optimization target: Minimax")

with st.sidebar:
    
    st.header("Model Settings")
    
    data_load_sate = st.text('Loading OWID data...')
    df = load_data()
    data_load_sate.text('')
    
    country_list = sorted(df['country'].unique())
    default_idx = country_list.index('World') if 'World' in country_list else 0
    selected_country = st.selectbox("Select Country", country_list, index=default_idx)
    
    years_course = list(range(0, 1700, 100))
    years_fine = list(range(1700, 2001, 1))
    slider_options = sorted(list(set(years_course + years_fine)))
    start_year = st.select_slider("Start Year", options=slider_options, value=1700)
    
    iterations = st.slider("Optimization Iterations", 100, 5000, 1000, step=100)
    run_btn = st.button("Run Projection Model", type="primary")
    
if run_btn:
    country_data = get_country_data(df, selected_country)
    
    mask = country_data['year'] >= start_year
    xdata = country_data.loc[mask, 'year'].values
    ydata = country_data.loc[mask, 'population'].values
    
    if len(xdata) < 10:
        st.error("Insufficient data. Increase the timeline or select a different region.")
    else:
        status_text = st.empty()
        status_text.write(f"Optimizing {selected_country} model ({iterations} runs)...") 
        prog_bar = st.progress(0)
        
        best_popt, error, avgerr = optimize(xdata, ydata, prog_bar, int(iterations))
        
        if best_popt is not None:
            status_text.write("Optimization Complete.")
            K, B, M, nu, A = best_popt
            
            current_year = datetime.now().year
            x_future = np.linspace(start_year, 2*current_year - start_year, 20*(current_year-start_year))
            y_future = generalized_logistic(x_future, *best_popt)
            stagnation_year = x_future[y_future >= 0.99*K]
            
            if stagnation_year.size > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
                if K >= 1e9:
                    col1.metric("Carrying Capacity", f"{K/1e9:.2f} Billion")
                else:
                    col1.metric("Carrying Capacity", f"{K/1e6:.2f} Million")
                col2.metric("Inflection Year", f"{int(M)}")
                col3.metric("Stagnation Year", f"{int(stagnation_year[0])}")
                col4.metric("Growth Rate", f"{B:.2f}")
                col5.metric("Error", f"{avgerr:.2f}%")
            else:
                col1, col2, col3, col4 = st.columns(4)
                if K >= 1e9:
                    col1.metric("Carrying Capacity", f"{K/1e9:.2f} Billion")
                else:
                    col1.metric("Carrying Capacity", f"{K/1e6:.2f} Million")
                col2.metric("Inflection Year", f"{int(M)}")
                col3.metric("Growth Rate", f"{B:.2f}")
                col4.metric("Error", f"{avgerr:.2f}%")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.scatter(xdata, ydata, color='black', s=10, alpha=0.5, label='Historical Data')
            
            ax.plot(x_future, y_future, color='red', linewidth=2, label='Predicted Population')
            
            current_pred = generalized_logistic(current_year, *best_popt)
            ax.scatter([current_year], [current_pred], color='green', s=50, zorder=5, label=str(current_year))
            
            ax.set_title(f"Population Projection: {selected_country}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Population")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            def billions(x, pos):
                return '%1.1fB' % (x * 1e-9)
            
            def millions(x, pos):
                return '%1.0fM' % (x * 1e-6)
            
            if np.max(ydata >  1e9):
                ax.yaxis.set_major_formatter(plt.FuncFormatter(billions))
            else:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))
                
            st.pyplot(fig)
            
            with st.expander("See Raw Parameters"):
                st.write(f"K: {K}")
                st.write(f"B: {B}")
                st.write(f"M: {M}")
                st.write(f"nu: {nu}")
                st.write(f"A: {A}")
                
        else:
            st.error("Model failed to converge. Try increasing iterations.")
    
    

    