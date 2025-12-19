import streamlit as st
import read_cgm_data as rd
import numpy as np

cgm_data = st.session_state['cgm_data']
pages_master = st.session_state['pages_master']
pages_dict = st.session_state['pages_dict']
units_ = st.session_state['units']
units_dict = {'mg':'mg/dL','mmol':'mmol/L'}
units_val = {'mg':0, 'mmol':1}


if len(cgm_data.names)>2:
    rd.add_page_link(pages_master[4][0],pages_master[4][1])
rd.add_page_link(pages_master[5][0],pages_master[5][1])
#rd.add_page_link(pages_master[6][0],pages_master[6][1])
    
rd.display_page_links(pages_dict)

name = st.sidebar.selectbox("Choose a file:",
                     options = cgm_data.names,
                     index = 0)

idx = units_val[units_]

units = st.sidebar.radio("Units:",
                         options = ['mg/dL','mmol/L'],
                         index=idx,
                         )

cgm_data.data[name].units = units.split('/')[0]
options = ["View Data",
           "Ambulatory Glucose Profile",
           "Glycemia Risk Index",
           "AGP Report",
           "Visualize Data",
           "Events",
           "Markov Analysis",
           "Time In Range",
           ]
select = st.pills("Select a tool:",
                  options = options,
                  default = options[0])
if select == options[0]:
    st.subheader("View Data")
    cgm_data.view_df_series(name)
if select == options[1]:
    st.subheader("Ambulatory Glucose Profile")
    cgm_data.ambulatory_glucose_profile(name)
if select ==options[2]:
    st.subheader("Glycemia Risk Index")
    cgm_data.view_gri(name)

if select == options[3]:
    st.subheader("AGP Report")
    cgm_data.agp_report(name)

if select == options[4]:    
    st.subheader("Visualize Data")
    cgm_data.visualize_data(name)

if select == options[5]:
    cgm_data.episodes(name)

if select == options[6]:
    cgm_data.markov_analysis(name)

if select == options[7]:
    st.subheader("Time In Range Analysis")
    v1,v2,v3,v4,v5,v6 = st.session_state['bins']
    cols = st.columns(4)
    with cols[0]:
        val1 = st.number_input("Very Low",min_value=1,value=v2,key="v1_tir")
    with cols[1]:
        val2 = st.number_input("In-Range Min",min_value=1,value=v3,key='v2_tir')
    with cols[2]:
        val3 = st.number_input("In-Range Max", min_value = 1, max_value = 500,value=v4,key='v3_tir')
    with cols[3]:
        val4 = st.number_input("Very High",min_value = 180, max_value = 500,value = v5,key='v4_tir')
    ok_btn = st.button("OK")
    if ok_btn:
        bins = np.array([0,val1,val2,val3,val4,350])
        st.session_state['bins'] = bins
        df_tir = cgm_data.create_tir_dataframe(units='mg',bins = bins)
        st.write(df_tir)
    

    
