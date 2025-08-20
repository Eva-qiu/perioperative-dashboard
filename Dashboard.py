# Import required libraries
import streamlit as st  
import pandas as pd
import numpy as np
import folium
import json
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium

# -------------------------
# Utility function: Module switching
# -------------------------
def switch_to_module(module_name, **kwargs):
    st.session_state['module'] = module_name
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()  # Re-run the app to apply the change immediately

# Set global page configuration
st.set_page_config(page_title="Perioperative Dashboard", layout="wide",  initial_sidebar_state="expanded")  # App title, Wide-screen layout, # Sidebar expanded by default

# -------------------------
# Set the current module & Sidebar
# -------------------------
if 'module' not in st.session_state:
    st.session_state['module'] = 'Module1'
module = st.session_state['module']

# -------------------------
# Welcome to hint: Sidebar version
# -------------------------
with st.sidebar.expander("👋 Welcome & Disclaimer", expanded=False):
    st.markdown(
        """
        This prototype dashboard allows you to:
        - **View individual surgical risk profiles**
        - **Browse risk distributions across UK regions**
        - **Simulate the impact of prehabilitation interventions**

        🔄 Use the navigation menu below to explore different modules.

        💬 Feedback and suggestions are very welcome!

        ---
        ⚠️ **Disclaimer**  
        This tool is for informational purposes only and is **not a medical device**. It does **not provide clinical diagnosis or treatment recommendations**.  
        All results are based on retrospective data and current academic research.  
        Any use in real-world or commercial settings must comply with GDPR and NHS information       governance standards.  
        Always consult qualified professionals for clinical decisions.
        """
    )

# Sidebar navigation menu
st.sidebar.title('Navigation')
choice = st.sidebar.radio('Navigate to', ['Module1','Module2','Module3'],
                          index=['Module1','Module2','Module3'].index(module))
if choice != module:  # If user selects a different module
    st.session_state['module'] = choice
    st.rerun()

# -------------------------
# Data loading and preprocessing
# -------------------------
@st.cache_data()  # Cache results to avoid reloading on each run
def load_data():
    df = pd.read_csv('cleaned_data_extract.csv')
    
    # Columns representing potential risk factors
    risk_cols = [
        'Patient BMI','Frailty Score','History of smoking',
        'Patient cannot do heavy housework','Heart problem',
        'patient has diabetes','Patient has kidney problem',
        'patient has problem with blood pressure',
        'patient has problem with mental health currently',
        'patient has problem with lungs or breathing',
        'patient taking lung medications',
        'patient has problem stopping breathing during sleep'
    ]

    # Risk Score = number of positive conditions (count of values > 0)
    df['Risk Score'] = df[risk_cols].gt(0).sum(axis=1)

    # Simulate an assessment date: Jan 1, 2024 + random 0–180 days
    df['AssessmentDate'] = (
        pd.to_datetime('2024-01-01') +
        pd.to_timedelta(np.random.RandomState(42).randint(0,180,size=len(df)), unit='D')
    )
    return df

# -------------------------
# Region mapping data for visualization
# -------------------------
group_region_map = pd.DataFrame({
    'groupID': list('abcdefghi'),  # Group IDs for nine sample regions
    'region': ['London','Manchester','Birmingham','Leeds','Bristol',
               'Liverpool','Sheffield','Glasgow','Newcastle'],
    'lat': [51.5074,53.4808,52.4862,53.8008,51.4545,53.4084,53.3811,55.8642,54.9783],  # Latitude
    'lon': [-0.1278,-2.2426,-1.8904,-1.5491,-2.5879,-2.9916,-1.4701,-4.2518,-1.6178]   # Longitude
})

# -------------------------
# Module 1: Patient View
# -------------------------
if module == 'Module1':
    # 1. Load and merge the data
    # Load preprocessed data and join with region mapping
    df = load_data().merge(group_region_map, on='groupID', how='left')

    # 2. Title & Patient Selection
    st.title('📊 Surgical Risk Dashboard')
    st.header('🧍 Patient‑Level View')
    # Dropdown to select a patient by ID
    patient_id = st.selectbox('Select Patient ID', df['ID'].unique())
    patient = df[df['ID'] == patient_id].iloc[0]  # Extract the selected patient row

    # 3. Two columns on the left and right: basic information on the left and radar chart on the right
    col1, col2 = st.columns([2, 3])

    # ---- Left column: patient demographics & comorbidities ----
    with col1:
        # Basic demographics
        sex = 'Female' if patient['Female Patient'] else 'Male'
        bmi = f"{patient['Patient BMI']} ({'Obese' if patient['Patient BMI']>=30 else 'Normal'})"
        frailty = patient['Frailty Score']
        
        # Collect comorbidities
        coms = []
        if patient['Heart problem']:              coms.append("Heart Disease")
        if patient['patient has diabetes']:       coms.append("Diabetes")
        if patient['patient has problem with blood pressure']:
            coms.append("Hypertension")
        if patient['patient has problem with lungs or breathing']:  
            coms.append("Pulmonary Disease")
        if patient['patient has problem with mental health currently']:  
            coms.append("Mental Health Issue")

        # Collect behavioral risk factors
        beh = []
        if patient.get('History of smoking', False): beh.append("Smoking history")
        if not patient.get('Patient can climb flight of stairs', True):
            beh.append("Cannot climb stairs")
        if patient.get('Patient cannot do heavy housework', False):
            beh.append("Cannot do heavy housework")

        # Render formatted patient summary
        st.markdown("<br>", unsafe_allow_html=True)    

        st.markdown(f"""
**Patient ID:** {patient['ID']}  
**Group:** {patient['groupID']}  
**Region:** {patient['region']}  
**Assessment Date:** {patient['AssessmentDate'].date()}

**Sex:** {sex}  
**BMI:** {bmi}  
**Frailty Score:** {frailty}

**Comorbidities:** {', '.join(coms) if coms else 'None'}  
**Behavioral Risks:** {', '.join(beh) if beh else 'None'}
        """)
        st.caption("• Data source: digital pre‑op questionnaire")

    # ---- Right column: Radar chart visualization ----
    with col2:
        # —— Radar chart —— 
        raw = {
            "Obesity": patient["Patient BMI"],
            "Frailty": patient["Frailty Score"],
            "Smoking": int(patient["History of smoking"]),
            "Hypertension": int(patient["patient has problem with blood pressure"]),
            "Heart Disease": int(patient["Heart problem"]),
            "Pulmonary Disease": int(patient["patient has problem with lungs or breathing"]),
            "Mental Health": int(patient["patient has problem with mental health currently"]),
            "Diabetes": int(patient["patient has diabetes"])

        }

        # Normalization
        max_vals = {"Obesity":40, "Frailty":9}
        norm = {k: (raw[k]/max_vals[k] if k in max_vals else raw[k]) for k in raw}

        # Prepare chart values
        labels = list(norm.keys())
        values = [min(max(v,0),1) for v in norm.values()]
        labels.append(labels[0]); values.append(values[0])

        # Build radar chart with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=labels, fill='toself',
            line=dict(color='indianred')
        ))
        fig.update_layout(
        autosize=False,      # don’t stretch to fill
        width=500,           # or whatever px wide you need
        height=400,          # ditto height
        margin=dict(
            t=20,            # small top/bottom
            b=20,
            l=120,           # extra left so “Obesity” has room
            r=120            # extra right so your last label fits
        ),
        showlegend=False,
        polar=dict(
            radialaxis=dict(
                range=[0,1],
                tickvals=[0,0.5,1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        )
    )

    # Display radar chart in right column
    col2.plotly_chart(
        fig,
        use_container_width=False,
        width=500,
        height=400
    )
 
    # 5. Summary & Recommendations
    st.markdown("---")
    st.subheader("📌 Summary & Recommendations")
    recs = []

    # Add tailored recommendations based on risk profile
    if patient['Patient BMI']>=30 or patient['Frailty Score']>=3:
        recs.append('💪 Exercise & Nutrition: Moderate exercise + high‑protein, low‑calorie diet')
    if patient['History of smoking']:
        recs.append('🚭 Smoking Cessation: Start quit program ≥8 weeks preop')
    if patient['patient has problem with blood pressure']:
        recs.append('❤️ Hypertension Control: Optimize antihypertensives')
    if patient['Heart problem']:
        recs.append('❤️‍🩹 Cardiac Rehab: Aerobic & resistance training')
    if patient['patient has problem with mental health currently']:
        recs.append('🧠 Mental Health Support: Counseling/mindfulness')
    if (
        patient['patient has problem stopping breathing during sleep'] or 
        patient['patient has problem with lungs or breathing']
    ):
        recs.append('💨 Pulmonary Prehab: Inspiratory muscle training')
    if patient['patient has diabetes']:
        recs.append('🩸 Diabetes Management: Glycemic optimization with lifestyle + medication')

    # Display recommendations with references
    if recs:
        for i, txt in enumerate(recs,1):
            st.markdown(f"{i}. {txt}")
        with st.expander('📑 References & Links'):
            refs = {
                'Exercise & Nutrition':    ('Waite et al. 2017','https://doi.org/10.1186/s13019-017-0655-8'),
                'Smoking Cessation':       ('Schmidt‑Hansen 2013','https://doi.org/10.1016/j.cllc.2012.07.003'),
                'Hypertension Control':    ('NICE NG107 2020','https://www.nice.org.uk/guidance/ng107'),
                'Cardiac Rehab':           ('Drudi et al. 2019','https://doi.org/10.1016/j.jss.2018.11.042'),
                'Mental Health Support':   ('Whale et al. 2025','https://doi.org/10.1002/msc.70088'),
                'Pulmonary Prehab':        ('Soares 2013','https://doi.org/10.1177/0269215512471063'),
                'Diabetes Management':     ('Zhao et al. 2024', 'https://doi.org/10.2196/53948')
            }
            for key,(c,l) in refs.items():
                st.markdown(f"- **{key}**: {c} ([link]({l}))")
    else:
        st.write("No prehabilitation recommended.")

    # 6. Lower right navigation buttons
    st.markdown("---")
    _, nav = st.columns([3,1])
     # Jump to Module 2 (region view) with region context
    if nav.button("→ Module 2"):
        switch_to_module(
            'Module2',
            selected_region=patient['region']
        )
    # Jump to Module 3 (simulation) with region, date and suggested interventions
    if nav.button("→ Module 3"):
        switch_to_module('Module3',
            selected_region=patient['region'],
            selected_date=patient['AssessmentDate'].date(),
            suggested_interventions=[r.split()[0].lower() for r in recs]
        )

# -------------------------
# Module2: Population-Level View
# -------------------------
elif module == 'Module2':
    import json
    import pandas as pd
    from streamlit_folium import st_folium
    import folium

    st.title('🌍 Population-Level View')

    # 1. Load data and aggregate by region
    df = load_data().merge(group_region_map, on='groupID', how='left')
    summ = (
        df.groupby('region')
          .agg(
              AvgRisk=('Risk Score', 'mean'),                     # Average risk score per region
              HighPct =('Risk Score', lambda x: (x >= 5).mean())  # % of patients with high risk (≥5)
          )
          .reset_index()
          .merge(group_region_map, on='region')  # add lat/lon back for plotting
    )
    
    # Mean risk score with 2 decimal places retained
    summ['AvgRisk'] = summ['AvgRisk'].round(2)  
    
    # Convert the high-risk ratio from 0 to 1 to a percentage format
    summ['HighPct'] = (summ['HighPct'] * 100).round(1)  

    # 2. ▶ Region Summary - Directly fold to display the overview (highlight selected cities)
    # -------------------------
    selected_city = st.session_state.get('selected_region', None)

    with st.expander("▶ Region Summary — All Regions Overview", expanded=False):
        # summary DataFrame
        df_summary = (
            summ[['region','AvgRisk','HighPct']]
              .rename(columns={
                  'region':      'Region',
                  'AvgRisk':     'Average Risk',
                  'HighPct':     'High-Risk %'
              })
              .reset_index(drop=True)
        )
        df_summary.index = df_summary.index + 1  # start index from 1

        # Highlight the row for the currently selected city
        def highlight_patient(row):
            return [
                'background-color: lightblue; color: black'
                if row['Region'] == selected_city else ''
                for _ in row
            ]

        # Generate Stylers with styles
        styled = (
            df_summary
            .style
            .apply(highlight_patient, axis=1)
            .format({
                'Average Risk': '{:.2f}',
                'High-Risk %': '{:.1f}'
            })
        )

        # Display summary table
        st.dataframe(styled, use_container_width=True)
        
    # 3. Choropleth map (Folium)
    # Load simplified UK GeoJSON boundaries
    with open('uk_9cities_simplified.geojson','r',encoding='utf-8') as fp:
        geo = json.load(fp)
        
    # Add aggregated values into GeoJSON properties    
    for feat in geo['features']:
        nm = feat['properties']['name']
        row = summ[summ['region']==nm]
        if not row.empty:
            feat['properties']['AvgRisk'] = float(row['AvgRisk'])
            feat['properties']['HighPct'] = float(row['HighPct'])

    # Create base map
    m = folium.Map(location=[54.5, -3], zoom_start=5)
    
    # Add choropleth layer by average risk
    folium.Choropleth(
        geo_data=geo,
        data=summ,
        columns=['region','AvgRisk'],
        key_on='feature.properties.name',
        fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
        legend_name='Average Risk'
    ).add_to(m)

    # Add tooltips with AvgRisk and HighPct
    folium.GeoJson(
        geo,
        style_function=lambda f: {'fillColor':'transparent','color':'transparent'},
        tooltip=folium.GeoJsonTooltip(
            fields=['name','AvgRisk','HighPct'],
            aliases=['Region:','AvgRisk:','High-Risk %: (0–100%)'],
            localize=True, sticky=True, labels=True
        )
    ).add_to(m)

    # Add city circle markers for clarity
    for _, r in summ.iterrows():
        folium.CircleMarker(
            [r['lat'], r['lon']],
            radius=6, color='black', fill=True, fill_color='gold', fill_opacity=0.8,
            tooltip=(
                f"{r['region']}<br>"
                f"Avg Risk: {r['AvgRisk']:.2f}<br>"
                f"High‑Risk %: {r['HighPct']:.1f}%"
            )
        ).add_to(m)

    st.subheader("🗺️ Click on the map to inspect a city")
    map_data = st_folium(m, width=700, height=300)

    # 4. Condition-level statistics per city
    comorbid_cols = {
        'Heart Disease':   'Heart problem',
        'Diabetes':        'patient has diabetes',
        'Hypertension':    'patient has problem with blood pressure',
        'Lung Disease':    'patient has problem with lungs or breathing',
        'Smoking History': 'History of smoking',
        'Mental Health':    'patient has problem with mental health currently'
    }
    records = []
    for city in summ['region']:
        sub = df[df['region']==city]
        total = len(sub)
        for lbl, col in comorbid_cols.items():
            cnt      = int(sub[col].sum())
            prev_pct = cnt/total*100 if total else 0
            mean_r   = sub.loc[sub[col]==1, 'Risk Score'].mean() if cnt else 0
            records.append({
                'region':       city,
                'Condition':    lbl,
                'Patients':     cnt,
                'Prevalence (%)': round(prev_pct,1),
                'Mean Risk':    round(mean_r,2)
            })
    cond_df = pd.DataFrame(records)

    # Format values
    cond_df['Prevalence (%)'] = cond_df['Prevalence (%)'].round(1)
    cond_df['Mean Risk'] = cond_df['Mean Risk'].round(2)

    # 5. After clicking on the map, a table of all conditions of the city will be displayed
    if map_data and map_data.get("last_clicked"):
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        clicked = summ.loc[((summ['lat']-lat)**2 + (summ['lon']-lng)**2).idxmin(), 'region']
        st.session_state['selected_region'] = clicked
        
        # Find the city closest to the longitude and latitude of the click
        summ['d2'] = (summ['lat'] - lat)**2 + (summ['lon'] - lng)**2
        clicked = summ.loc[summ['d2'].idxmin(), 'region']

        # Extract the statistical information of all conditions in this city
        recs_df = (
            cond_df[cond_df['region'] == clicked]
            .drop(columns='region')
            .rename(columns={
                'Condition': 'Condition',
                'Patients': 'Number of Patients',
                'Prevalence (%)': 'Prevalence (%)',
                'Mean Risk': 'Average Risk Score'
            })
        )

        st.subheader(f"🔍 {clicked} — Condition Breakdown")
        # Style numbers with proper formatting
        recs_df_styled = recs_df.style.format({
            'Prevalence (%)': '{:.1f}',
            'Average Risk Score': '{:.2f}'
        })

        st.dataframe(recs_df_styled, use_container_width=True) 

    else:
        st.info("🔍 Please click on a city marker on the map above to view details")

    # 6. Compare the Nine Cities table by Condition
    st.markdown("---")
    st.subheader("▶️ Choose a Condition to compare across regions:")
    cond = st.selectbox(
        "", options=list(comorbid_cols.keys()), key="compare_condition"
    )

    # Filter selected condition
    comp = (cond_df[cond_df['Condition']==cond]
            [['region','Patients','Prevalence (%)','Mean Risk']]
            .rename(columns={'region':'Region'}))
    comp = comp.reset_index(drop=True)  
    comp.index = comp.index + 1        

     # Highlight selected city row
    selected_city = st.session_state.get('selected_region', None)
    def highlight_row(row):
        return ['background-color: lightblue; color: black' if row['Region']==selected_city else '' 
                for _ in row]

    styled_comp = (
        comp.style
            .apply(highlight_row, axis=1)
            .format({
                'Prevalence (%)': '{:.1f}',
                'Mean Risk': '{:.2f}'
            })
    )

    st.subheader(f"All Regions — {cond} statistics")
    st.dataframe(styled_comp, use_container_width=True)

    # 7. Navigation buttons
    _, nav = st.columns([3, 1])
    with nav:
        if st.button("← Back to Module 1"):
            switch_to_module('Module1')
        if st.button("→ Module 3"):
            switch_to_module('Module3', selected_region=None)

# -------------------------
# Module3: Intervention
# -------------------------
elif module == 'Module3':
    import pandas as pd
    import plotly.graph_objects as go
    from dateutil.relativedelta import relativedelta
    from datetime import date

    # Remember the region the user came from (if any)
    selected_region = st.session_state.get("selected_region", None)

    # Reduce extra top padding so the header sits higher
    st.markdown("<style>main > div > div > div > div {padding-top: 0rem;}</style>", unsafe_allow_html=True)


    st.title("🧪 Intervention-by-Intervention Simulation")
    st.markdown(
        "By default, the simulation spans six months from the assessment date in Module 1." 
        "Below, you can select one or more interventions to view the individual effect of each."
    )

    # ─── Risk Score Explanation ───
    with st.expander("🔎 How the Risk Score Is Calculated", expanded=False):
        st.markdown("""
        **Baseline Risk** is the average of each patient’s **Weighted Baseline Score**  
        over the selected six‑month window in your chosen region.

        The **Weighted Baseline Score** for each assessment is computed as:

        > **WeightedBaseline** = Σ (Factor_flag × ln(OR_Factor))

        - **Factor_flag** is 1 if the risk factor is present (e.g., Obesity, Smoking,  
          Pulmonary Disease) and 0 otherwise.  
        - **OR_Factor** is the Odds Ratio for each risk factor, derived from the following studies:  
          - Obesity (BMI ≥ 30): OR = 2.1  *(Sánchez‑Guillén et al., 2020)*
          - Smoking: 2.7  *(Vaporciyan et al., 2002)*  
          - Pulmonary Disease:   OR = 1.4  *(Sánchez‑Guillén et al., 2020)*
          - Preoperative Anxiety: OR = 12.05  *(Sánchez‑Guillén et al., 2020)* 
          - HeartDisease:   OR = 1.53   *(Larsson et al., 2022 – severe complications after major surgery)*
          - Hypertension:  OR = 1.48  *(Loyst et al., 2024 – reoperation risk in TSA patients)*
          - Diabetes: OR = 1.65  *(Zhang et al., 2022 – meta-analysis in non-cardiac surgeries)*
        - ln(OR_Factor) converts each OR into a weight in the additive score.

        **Interpretation:**  
        - Records with more high‑OR factors yield a larger Weighted Baseline → higher Baseline Risk.  
        - We then apply each intervention’s percentage reduction to this Baseline Risk in the simulation.
        """)
    # ─────────────────────────────────

    # …then continue with your date‑window, region‑select, slider, plotting, etc.

    # Data loading and preprocessing
    df = load_data().merge(group_region_map, on="groupID", how="left")
    df["Date"] = df["AssessmentDate"].dt.date

    # A six-month time window
    sel_date = st.session_state.get("selected_date")
    min_d, max_d = df["Date"].min(), df["Date"].max()
    if sel_date:
        if not isinstance(sel_date, date):
            sel_date = pd.to_datetime(sel_date).date()
        start = sel_date
        end = (pd.to_datetime(sel_date) + relativedelta(months=6)).date()
        end = min(end, max_d)
    else:
        start, end = min_d, max_d

    # Filter the dataset to the time window
    df_window = df[(df["Date"] >= start) & (df["Date"] <= end)]

    # Regional selection
    regions = sorted(df_window["region"].unique())
    default_region = st.session_state.get("selected_region")
    if default_region not in regions:
        default_region = regions[0]  # fallback to the first region

    region = st.selectbox("Select region:", regions, index=regions.index(default_region))

    # Subset to chosen region for the simulation
    df_reg = df_window[df_window["region"] == region]
    
    # ─── Define ORs and convert to additive weights (ln OR) ───
    or_coeffs = {
        'Obesity':          2.1,
        'Smoking':     2.7,
        'PulmonaryDisease': 1.4,
        'Anxiety':          12.05,
        'HeartDisease':   1.53,     # Larsson et al., 2022 – severe postoperative complications
        'Hypertension':   1.48,     # Based on Loyst et al. (2024): OR for reoperation following TSA
        'Diabetes': 1.65,
    }
    # Uses numpy imported at top of file
    weights = {k: np.log(v) for k,v in or_coeffs.items()}

    # ─── Construct binary flags (0/1) for each risk factor ───
    # NOTE: .astype(int) converts booleans to 0/1 integers
    df_reg['Obesity_flag']        = (df_reg['Patient BMI'] >= 30).astype(int)
    df_reg['Smoking_flag']   = df_reg['History of smoking'].astype(int) 
    df_reg['PulmonaryDisease_flag']= df_reg['patient has problem with lungs or breathing'].astype(int)
    df_reg['Anxiety_flag']         = df_reg['patient has problem with mental health currently'].astype(int)
    df_reg['HeartDisease_flag']  = df_reg['Heart problem'].astype(int)
    df_reg['Hypertension_flag']  = df_reg['patient has problem with blood pressure'].astype(int)

    # ─── Compute the Weighted Baseline Risk per record ───
    df_reg['WeightedBaseline'] = (
          df_reg['Obesity_flag']        * weights['Obesity']
        + df_reg['Smoking_flag']   * weights['Smoking']
        + df_reg['PulmonaryDisease_flag']* weights['PulmonaryDisease']
        + df_reg['Anxiety_flag']         * weights['Anxiety']
        + df_reg['HeartDisease_flag']    * weights['HeartDisease'] 
        + df_reg['Hypertension_flag']    * weights['Hypertension']
    )

    # Aggregate to daily mean baseline risk over the window
    daily = (
        df_reg
          .groupby("AssessmentDate")
          .agg(BaselineRisk=("WeightedBaseline","mean"))
          .reset_index()
    )

    # Intervention and its default effects
    default_effects = {
        'Exercise & Nutrition': 0.18,   # frailty-prehab's Research on "Home"
        'Smoking Cessation':    0.30,    # The postoperative complication rate of severe smokers is                                              approximately 30%
        'Cardiac Rehab':        0.25,   # Take the median of the decline range of 13.5-35% ≈25%
        'Pulmonary Prehab':     0.25,   # Take the average of MIP(15%) and 6MWT(65%) as approximately                                           40%, and conservatively use 25% to 30%
        'Mental Health Support': 0.5,   # According to Smith et al. (20XX) RCT, CBT can reduce                                                  preoperative anxiety by 50%
        'Hypertension Control': 0.27,   # Supported by RCT showing 27% reduction in organ                                                       dysfunction via personalized BP management (INPRESS trial)
        'Diabetes Optimization': 0.41  # Derived from RR=1.13 (Zhao et al., 2024): ~41% estimated                                              reduction in adverse outcomes

    }

    # Select the intervention to be simulated
    st.subheader("Choose interventions to simulate:")

    # Create a mapping from the Module1 abbreviation code to full-name
    code_to_name = {
        'bmi':          'Exercise & Nutrition',
        'smoking':      'Smoking Cessation',
        'heart':        'Cardiac Rehab',
        'lung':         'Pulmonary Prehab',
        'mental':       'Mental Health Support',
        'hypertension': 'Hypertension Control',
        'frailty':      'Exercise & Nutrition',
        'diabetes':     'Diabetes Optimization'
    }

    # Prefill selection using suggestions from Module 1 
    raw_suggestions = st.session_state.get("suggested_interventions", [])
    default_selections = [
        code_to_name[c]
        for c in raw_suggestions
        if c in code_to_name and code_to_name[c] in default_effects
    ]

    # Let users choose any subset of interventions to simulate
    interventions = st.multiselect(
        "",
        options=list(default_effects.keys()),
        default=default_selections
    )

    # Display the slider of each intervention and calculate the single-item simulation
    sim_cols = []
    for name in interventions:
        pct_default = int(default_effects[name] * 100)
        eff = st.slider(f"{name} effect (%)", 0, 100, pct_default, key=name) / 100
        col = f"Sim_{name}"
        daily[col] = daily["BaselineRisk"] * (1 - eff)
        sim_cols.append((name, col))

    # Drawing: Baseline + each intervention
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["AssessmentDate"], y=daily["BaselineRisk"],
        mode="lines", name="Baseline"
    ))
    for name, col in sim_cols:
        fig.add_trace(go.Scatter(
            x=daily["AssessmentDate"], y=daily[col],
            mode="lines", name=name
        ))
    fig.update_layout(
        title=f"Risk Over Time in {region} ({start}→{end})",
        xaxis_title="Date", yaxis_title="Risk Score"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Endpoint comparison table
    last = daily.iloc[-1]
    comp = []
    comp.append({"Scenario": "Baseline", "Risk": round(last["BaselineRisk"], 2)})
    for name, col in sim_cols:
        comp.append({"Scenario": name,     "Risk": round(last[col], 2)})
    comp_df = pd.DataFrame(comp)
    st.subheader("Comparison at End Date")
    st.table(comp_df)

    # Explanation and description
    st.subheader("🔍 Interpretation")

    # Prepare corresponding literature descriptions for each intervention
    interpretation_map = {
        'Exercise & Nutrition':
            "assumed 18% improvement in physical function (6MWT, SPPB, CFS from PREHAB RCT).",
        'Smoking Cessation':
            "assumed 30% reduction in postoperative pulmonary complications (Schmidt-Hansen et al. 2013).",
        'Cardiac Rehab':
            "assumed 25% reduction in complications & LOS (Arthur 2000; Branea 2015; Nery 2007; Barakat 2016).",
        'Pulmonary Prehab':
            "assumed 25% average lung & functional gain (MIP↑15%, 6MWT↑65% in RCT).",
        'Mental Health Support':
            "assumed 10% reduction in anxiety/depression (REST TKR sleep intervention qualitative).",
        'Hypertension Control':
        "assumed 27% improvement in postoperative outcomes based on individualized perioperative BP management (INPRESS trial, Futier et al. 2017).",
        'Diabetes Optimization':
        "assumed ~41% reduction in treatment failure based on improved glycemic control (Zhao et al., 2024)."  
    }

    # Only display those explanations selected by the user (or suggested by Module1)
    if interventions:
        for name in interventions:
            text = interpretation_map.get(name, "No literature annotation available.")
            st.markdown(f"- **{name}**: {text}")
    else:
        st.markdown("- No interventions selected; showing baseline only.")

    # Navigation buttons
    st.markdown("---")
    _, nav = st.columns([3, 1])
    with nav:
        if nav.button("← Back to Module 2"):
        
            if selected_region is not None:
                st.session_state["selected_region"] = selected_region
            switch_to_module("Module2")
        if nav.button("← Back to Module 1"):
            switch_to_module("Module1")
