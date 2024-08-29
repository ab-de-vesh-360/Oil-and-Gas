import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(layout="wide")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'updated_df' not in st.session_state:
    st.session_state.updated_df = None
# Initialize session state for graph page inputs
input_keys = ['Pi', 'q', 'phi', 'vis', 'ct', 'rw', 'h', 'Bo', 't_s', 't_e']
for key in input_keys:
    if key not in st.session_state:
        st.session_state[key] = None
with st.sidebar:
    if st.session_state.page != 'home':
        if st.button('Home'):
            st.session_state.page = 'home'
            st.experimental_rerun()

    # Show 'Data Preview' button only if on the 'graph' page
    # if st.session_state.page == 'graph':
    #     if st.button('Data Preview'):
    #         st.session_state.page = 'data'
    #         st.experimental_rerun()

# Home page
if st.session_state.page == 'home':
    st.markdown("<h1 style='text-align: center; font-size: 40px; text-decoration: underline; font-family: Lucida Bright, sans-serif; margin-bottom: 0; color: dark brown;'>Pressure Drawdown Test</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: right; font-size: 20px; font-family: Lucida Bright, sans-serif; margin-bottom: 0; color: green;'>A Derivative Approach!!</h2>", unsafe_allow_html=True)
    st.markdown("*Welcome to the Pressure Drawdown Test Analysis using Pressure Derivative Approach.*")

    uploaded_file = st.file_uploader("Upload your Excel file containg t (hours)|pwf (psi) data:", type=["xlsx"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file  # Store the uploaded file in session state
        st.session_state.page = 'graph'
        st.experimental_rerun()

# Graph page
elif st.session_state.page == 'graph':

    if st.session_state.uploaded_file is not None:
        df = pd.read_excel(st.session_state.uploaded_file)



            # Sidebar inputs
        with st.sidebar:
            st.write('Enter the known parameters:')
            co1, co2 = st.columns([1,1])
            with co1:
                Pi = st.text_input('Pi (psi)', value='')
                if Pi:
                    Pi = float(Pi)
                q = (st.text_input('q (bbl/day)', value=None))
                if q:
                    q = float(q)
                phi = (st.text_input('phi', value=None))
                if phi:
                    phi = float(phi)
                vis = (st.text_input('vis (cp)', value=None))
                if vis:
                    vis = float(vis)
            with co2:
                ct = (st.text_input('ct (1/psi)', value=None))
                if ct:
                    ct = float(ct)
                rw = (st.text_input('rw (ft)', value=None))
                if rw:
                    rw = float(rw)
                h = (st.text_input('h (ft)', value=None))
                if h:
                    h = float(h)
                Bo = (st.text_input('Bo (bbl/STB)', value=None))
                if Bo:
                    Bo = float(Bo)

        if Pi and q and ct and h and vis and phi and Bo and rw is not None:
            # Data processing
            if df['t (hours)'].iloc[0] == 0:
                df.drop(df.index[0], inplace=True)
                df.reset_index(drop=True, inplace=True)
            
            df['del_p'] = Pi - df['pwf (psi)']
            
            df['ddp/dlnt'] = [
                df['del_p'][i] if i == 0 else 
                (df['del_p'][i] - df['del_p'][i-1]) / (np.log(df['t (hours)'][i]) - np.log(df['t (hours)'][i-1])) 
                for i in range(len(df))
            ]

            st.session_state.updated_df = df  # Store the updated dataframe in session state

            fig_con = px.line(df, x = 't (hours)', y = 'pwf (psi)', title = 'Pressure Drawdown Profile', height= 350, width = 600)


            # Log-log and semi-log plots
            fig_loglog = px.scatter(df, x='t (hours)', y=['del_p', 'ddp/dlnt'], title='Log-Log Plot', log_x=True, log_y=True, height= 360, color_discrete_sequence= ['lightskyblue', 'red'])
            fig_loglog.update_xaxes(minor=dict(ticks="inside", showgrid=True))
            fig_loglog.update_yaxes(minor=dict(ticks="inside", showgrid=True), title_text = 'del_ & ddp/dlnt')
            fig_loglog.update_layout(showlegend=False)
                # Calculations and results
            fig_semilog = px.line(df, x='t (hours)', y='pwf (psi)', log_x=True, title='Semi-Log Plot', height= 350)
            fig_semilog.update_xaxes(minor=dict(ticks='inside', showgrid=True, gridcolor='grey'), gridcolor='black')
            col_1, col_2 = st.columns([2,1])
            col1, col2 = st.columns(2)
            with col_2:
                st.markdown("""
                    <p style='text-align: center; font-family: Lucida Bright, sans-serif; padding: 2px; font-size: 16px; border-radius: 15px; border: 2px solid skyblue; color: black; background-color: lightblue;'><strong>Calculated Parameters:</strong></p>
                    """, unsafe_allow_html=True)

            c = st.sidebar.checkbox('Confirm for Radial Flow')
            if c==True:
                t_s = st.sidebar.number_input('Aproximated Start time for radial flow: ', value=None, min_value=0.0, step=0.1)
                t_e = st.sidebar.number_input('Aproximated End time for radial flow: ', value=None, min_value=0.0, step=0.1)
                
                
                if t_s and t_e is not None:
                    
                    # Find the indices corresponding to the times t_s and t_e
                    index_s = df[df['t (hours)'] >= t_s].index[0]
                    index_e = df[df['t (hours)'] <= t_e].index[-1]

                    # Calculate Cs
                    Cs = q * Bo / (24 * (df['del_p'][1]) / (df['t (hours)'][1]))
                

                    # Calculate the average value of 'ddp/dlnt' within the specified time range
                    stb = np.average(df['ddp/dlnt'][index_s:index_e+1])


                    k = 70.6 * q * vis * Bo / (h * stb)
                    # st.markdown(f'<p>Permeability (k): {k} md</p>', unsafe_allow_html=True)

                    t_st = np.mean(df['t (hours)'][index_s:index_e+1])
                    del_p_st = np.interp(t_st, df['t (hours)'][index_s:index_e+1], df['del_p'][index_s:index_e+1])

                    s = 1.151 * (del_p_st / (2.303 * stb) - np.log10(t_st) - np.log10(k / (phi * vis * ct * rw * rw)) + 3.23)
                    # st.markdown(f'<p>Skin (S): {s}</p>', unsafe_allow_html=True)

                    # Semi-log calculations
                    slope, intercept = np.polyfit(np.log10(df['t (hours)'][index_s:index_e+1]), df['pwf (psi)'][index_s:index_e+1], 1)


                    k2 = 162.6 * q * vis * Bo / (h * np.absolute(slope))

                    # from scipy.interpolate import interp1d
                    # interp_func = interp1d(df['t (hours)'][index_s:index_e+1], df['pwf (psi)'][index_s:index_e+1], kind='linear', fill_value='extrapolate')
                    # # P1 = interp_func(1)  # Extrapolation for P1
                    # st.markdown(f'<p>The value of P1 is: {P1}</p>', unsafe_allow_html=True)
                    P1 = slope * np.log10(1) + intercept
                    s2 = 1.151 * ((Pi - P1) / np.absolute(slope) - np.log10(k2 / (phi * vis * ct * rw * rw)) + 3.23)
                
                # fig_loglog.add_scatter(x=df['t (hours)'], y=[stb] * len(df), mode='lines', name=f'STB ({stb:.2f})', line=dict(color='yellow', dash='dash'))
                    fig_loglog.add_hline(y=stb, line=dict(color='yellow', dash='dash', width=2))
                    fig_loglog.add_annotation(
                        
                        # Place annotation at the start of the x-axis
                        x= np.log10(1),
                        y =np.log10(stb),
                        text=f"ΔP'= {stb:.2f}",
                        font= dict(color = 'white', size = 15),
                        showarrow=True,
                        arrowhead=1,
                        bgcolor = 'gray',
                        arrowcolor= 'gray',
                        ax=-40,  # Adjust x position
                        ay=20)
                    t1 = df['t (hours)'][0]  # Very small value close to 0 (logarithmically)
                    t2 = t_e     # Example of a larger value on the x-axis

                    # Calculate pwf values using the slope and intercept
                    pwf1 = slope * np.log10(t1) + intercept
                    pwf2 = slope * np.log10(t2) + intercept
                    fig_semilog.add_trace(go.Scatter(x=[t1, t2], y=[pwf1, pwf2],
                                mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                    fig_semilog.add_annotation(
                        x = np.log10(df['t (hours)'][index_s]),
                        y = slope * np.log10(df['t (hours)'][index_s]) + intercept,
                        text=f'|m|= {-1*slope:.2f}',
                        font= dict(color = 'white', size = 15),
                        showarrow=True,
                        arrowhead=1,
                        bgcolor = 'gray',
                        arrowcolor= 'gray',
                        ax=-20,  # Adjust x position
                        ay=-25

                    )
                    fig_semilog.add_annotation(
                        x = np.log10(1),
                        y = P1,
                        text=f'Pwf<sub>(1hr)</sub>= {P1:.2f} psi',
                        font= dict(color = 'white', size = 15),
                        showarrow=True,
                        arrowhead=1,
                        bgcolor = 'gray',
                        arrowcolor= 'gray',
                        ax=-40,  # Adjust x position
                        ay=20

                    )
                    with col_2:
                        if t_s and t_e is not None:
                            st.markdown("""
                            <div style="
                                padding: 2px;
                                background-color: lightgrey;
                                color: black;
                                border-radius: 5px;
                                font-size: 10px;
                                font-family: Arial, sans-serif;
                                text-align: center;
                                height: 100%;
                                width: 100%;
                            ">
                            <Cs>
                            <stb >
                            <slope >
                            <k>
                            <s>            
                            <div>
                            """
                            .replace('<Cs>', f'<p style="font-size: 14px; margin: 0;">Wellbore Storage constant (Cs): {Cs:.4f} bbl/psi</p>')
                            .replace('<stb>', f'<p style="font-size: 14px; margin: 0;">Stabilized value of IARF from log-log plot is: {stb:.2f}</p>')
                            .replace('<slope>', f'<p style="font-size: 14px; margin: 0;">Slope of the semi-log plot is: {np.absolute(slope):.2f} psi/cycle</p>')
                            .replace('<k>', f'<p style="font-size: 14px; margin: 0;">Permeability (K): {k2:.2f} md</p>')
                            .replace('<s>',f'<p style="font-size: 14px; margin: 0;">Skin (S): {s2:.2f}</p>')
                            , unsafe_allow_html=True)
            
                
                
            
            
            
            with col_1:
                st.plotly_chart(fig_con)
            
            
            
            # with col1:
            #     st.plotly_chart(fig_loglog)
            # with col2:
            #     st.plotly_chart(fig_semilog)
            
            
            with st.sidebar:
                co3, co4 = st.columns(2)
                with co3:
                    c1 = st.checkbox('Check for PSS')
                with co4:
                    c2 = st.checkbox('Confirm')
            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)
            col9, col10 = st.columns(2)
            col11, col12 = st.columns(2)
            col13, col14 = st.columns(2)
            
            if c1==True:
                df['dp/dt'] = [(df['del_p'][0])/(df['t (hours)'][0]) if i == 0 else (df['del_p'][i]-df['del_p'][i-1])/(df['t (hours)'][i]-df['t (hours)'][i-1]) for i in range(len(df))]
                fig_pss = px.scatter(df, x = 't (hours)', y = 'dp/dt', title = 'Check for PSS', log_x = True, log_y = True, color_discrete_sequence= ['orange'])
                fig_pss.update_xaxes(minor=dict(ticks="inside", showgrid=True))
                fig_pss.update_yaxes(minor=dict(ticks="inside", showgrid=True), title_text = 'dp/dt')
                fig_pss.update_layout(showlegend=False)
                fig_con = px.line(df, x = 't (hours)', y = 'pwf (psi)', title = 'PSS Drawdown Profile', color_discrete_sequence= ['orange'])
                
                if c2==True:
                    t_ps = st.sidebar.number_input('Aproximated Start time for PSS flow: ', value=None, min_value=0.0, step=0.1)
                    t_pe = st.sidebar.number_input('Aproximated End time for PSS flow: ', value=None, min_value=0.0, step=0.1)
                    if t_ps and t_pe is not None:
                        index_ps = df[df['t (hours)'] >= t_ps].index[0]
                        index_pe = df[df['t (hours)'] <= t_pe].index[-1]
                        stb2 = np.average(df['dp/dt'][index_ps:index_pe+1])
                        
                        slope_p, intercept_p = np.polyfit(df['t (hours)'][index_ps:index_pe+1], df['pwf (psi)'][index_ps:index_pe+1], 1)
                        A = 0.2339*q*Bo/((np.absolute(slope_p))*phi*ct*h)

                        t1p = df['t (hours)'][0]  # Very small value close to 0 (logarithmically)
                        t2p = t_pe     # Example of a larger value on the x-axis

                        # Calculate pwf values using the slope and intercept
                        pwf1_p = slope_p * t1p + intercept_p
                        pwf2_p = slope_p * t2p + intercept_p
                        
                        fig_con.add_trace(go.Scatter(x=[t1p, t2p], y=[pwf1_p, pwf2_p],
                                    mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                        fig_pss.add_hline(y=stb2, line=dict(color='yellow', dash='dash', width=2))
                        fig_pss.add_annotation(
                            
                            # Place annotation at the start of the x-axis
                            x= np.log10(t_ps),
                            y = np.log10(stb2),
                            text=f"|dp/dt|= {stb2:.2f} psi/hr",
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-40,  # Adjust x position
                            ay=20)
                        fig_con.add_annotation(
                            x = df['t (hours)'][index_ps],
                            y = slope_p * (df['t (hours)'][index_ps]) + intercept_p,
                            text=f'|m|= {-1*slope_p:.2f} psi/hr',
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-20,  # Adjust x position
                            ay=-25

                        )
                        fig_con.add_annotation(
                            x = 0,
                            y = intercept_p,
                            text=f'Pwf(t= 0hr)= {intercept_p:.2f} psi',
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=70,  # Adjust x position
                            ay=40
                        )
                        with col_2:
                            if t_pe and t_ps is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <A>
                                <J>
                                <div>
                                """
                                .replace('<A>', f'<p style="font-size: 14px; margin: 0;">Drainage Area (A): {A/43560:.2f} acre</p>')
                                .replace('<J>', f'<p style="font-size: 14px; margin: 0;">Productivity Index (J): {(q/(Pi - intercept_p)):.2f} STB/day/psi</p>'), unsafe_allow_html=True)
                
                with col5:
                    st.plotly_chart(fig_pss)
                with col6:
                    st.plotly_chart(fig_con)

            with st.sidebar:
                co5, co6 = st.columns(2)
                with co5:
                    c5 = st.checkbox('Check for Linear')
                with co6:
                    c6 = st.checkbox('Confirm ')
            if c5==True:
                df['sqrt(t)'] = np.sqrt(df['t (hours)'])
                df['dp/d(sqrt(t))'] = [(df['del_p'][0])/np.sqrt((df['t (hours)'][0])) if i == 0 else (df['del_p'][i]-df['del_p'][i-1])/(np.sqrt(df['t (hours)'][i])-np.sqrt(df['t (hours)'][i-1])) for i in range(len(df))]
                fig_lin = px.scatter(df, x = 't (hours)', y = 'dp/d(sqrt(t))', title = 'Check for Linear', log_x = True, log_y = True, color_discrete_sequence= ['lightgreen'])
                fig_lin.update_xaxes(minor=dict(ticks="inside", showgrid=True))
                fig_lin.update_yaxes(minor=dict(ticks="inside", showgrid=True), title_text = 'dp/d(sqrt(t))')
                fig_lin.update_layout(showlegend=False)
                fig_lin_root = px.scatter(df, x = 'sqrt(t)', y = 'pwf (psi)', title = 'Linear Conventional Plot', color_discrete_sequence= ['lightgreen'])
                if c6==True:
                    t_ls = st.sidebar.number_input('Aproximated Start time for Linear flow: ', value=None, min_value=0.0, step=0.1)
                    t_le = st.sidebar.number_input('Aproximated End time for Linear flow: ', value=None, min_value=0.0, step=0.1)
                    if t_ls and t_le is not None:
                        index_ls = df[df['t (hours)'] >= t_ls].index[0]
                        index_le = df[df['t (hours)'] <= t_le].index[-1]
                        stb3 = np.average(df['dp/d(sqrt(t))'][index_ls:index_le+1])
                        
                        slope_l, intercept_l = np.polyfit(df['sqrt(t)'][index_ls:index_le+1], df['pwf (psi)'][index_ls:index_le+1], 1)
                        Xf = 4.06*(q*Bo*np.sqrt(vis/(k2*phi*ct)))/(h*(-1*slope_l))
                        t1l = df['sqrt(t)'][0]  # Very small value close to 0 (logarithmically)
                        t2l = df['sqrt(t)'][index_le]    # Example of a larger value on the x-axis

                        # Calculate pwf values using the slope and intercept
                        pwf1_l = slope_l * t1l + intercept_l
                        pwf2_l = slope_l * t2l + intercept_l
                        fig_lin_root.add_trace(go.Scatter(x=[t1l, t2l], y=[pwf1_l, pwf2_l],
                                    mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                        fig_lin.add_hline(y=stb3, line=dict(color='yellow', dash='dash', width=2))
                        fig_lin.add_annotation(
                            
                            # Place annotation at the start of the x-axis
                            x= np.log10(t_ls),
                            y = np.log10(stb3),
                            text=f"|dp/ds(qrt(t))|= {stb3:.2f} psi/hr<sup>(0.5)</sup>",
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-40,  # Adjust x position
                            ay=20)
                        fig_lin_root.add_annotation(
                            x = df['sqrt(t)'][index_ls],
                            y = slope_l * (df['sqrt(t)'][index_ls]) + intercept_l,
                            text=f'|m|= {-1*slope_l:.2f} psi/hr<sup>(0.5)</sup>',
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-20,  # Adjust x position
                            ay=-25

                        )
                        with col_2:
                            if t_le and t_ls is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <Xf>
                                <J>
                                <div>
                                """
                                .replace('<Xf>', f'<p style="font-size: 14px; margin: 0;">Fracture half-length (Xf): {Xf:.2f} ft</p>'), unsafe_allow_html=True)
                
                with col7:
                    st.plotly_chart(fig_lin)
                with col8:
                    st.plotly_chart(fig_lin_root)
            with st.sidebar:
                co7, co8 = st.columns(2)
                with co7:
                    c7 = st.checkbox('Check for Bilinear')
                with co8:
                    c8 = st.checkbox('Confirm  ')
            if c7==True:
                df['(t)<sup>(1/4)</sup>'] = np.sqrt(np.sqrt(df['t (hours)']))
                df['dp/d((t)<sup>(1/4)</sup>'] = [(df['del_p'][0])/np.sqrt(np.sqrt((df['t (hours)'][0]))) if i == 0 else (df['del_p'][i]-df['del_p'][i-1])/(np.sqrt(np.sqrt(df['t (hours)'][i]))-np.sqrt(np.sqrt(df['t (hours)'][i-1]))) for i in range(len(df))]
                fig_bil = px.scatter(df, x = 't (hours)', y = 'dp/d((t)<sup>(1/4)</sup>', title = 'Check for Biliinear', log_x = True, log_y = True, color_discrete_sequence= ['violet'])
                fig_bil.update_xaxes(minor=dict(ticks="inside", showgrid=True))
                fig_bil.update_yaxes(minor=dict(ticks="inside", showgrid=True), title_text = 'dp/d((t)<sup>(1/4)</sup>')
                fig_bil.update_layout(showlegend=False)
                fig_bil_root = px.scatter(df, x = '(t)<sup>(1/4)</sup>', y = 'pwf (psi)', title = 'Biinear Conventional Plot', color_discrete_sequence= ['violet'])
                
                if c8==True:
                    t_bs = st.sidebar.number_input('Aproximated Start time for Bilinear flow: ', value=None, min_value=0.0, step=0.1)
                    t_be = st.sidebar.number_input('Aproximated End time for Bilinear flow: ', value=None, min_value=0.0, step=0.1)
                    if t_bs and t_be is not None:
                        index_bs = df[df['t (hours)'] >= t_bs].index[0]
                        index_be = df[df['t (hours)'] <= t_be].index[-1]
                        # stb3 = np.average(df['dp/d(sqrt(t))'][index_ls:index_le+1])
                        stb4 = np.average(df['dp/d((t)<sup>(1/4)</sup>'][index_bs:index_be+1])
                        slope_b, intercept_b = np.polyfit(df['(t)<sup>(1/4)</sup>'][index_bs:index_be+1], df['pwf (psi)'][index_bs:index_be+1], 1)
                        Fc = ((44.1*q*vis*Bo/(h*np.absolute(slope_b)))**2)*np.sqrt(1/(k2*phi*vis*ct))
                        t1b = df['(t)<sup>(1/4)</sup>'][0]  # Very small value close to 0 (logarithmically)
                        t2b = df['(t)<sup>(1/4)</sup>'][index_be]    # Example of a larger value on the x-axis

                        # Calculate pwf values using the slope and intercept
                        pwf1_b = slope_b * t1b + intercept_b
                        pwf2_b = slope_b * t2b + intercept_b
                        fig_bil_root.add_trace(go.Scatter(x=[t1b, t2b], y=[pwf1_b, pwf2_b],
                                    mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                        # Xf  = 4.06*(q*Bo*np.sqrt(vis/(k2*phi*ct)))/(h*(stb3 from col6))
                        
                        fig_bil.add_hline(y=stb4, line=dict(color='yellow', dash='dash', width=2))
                        fig_bil.add_annotation(
                            
                            # Place annotation at the start of the x-axis
                            x= np.log10(t_bs),
                            y = np.log10(stb4),
                            text=f"|dp/d((t)<sup>(1/4)</sup>| = {stb4:.2f} psi/hr<sup>(0.25)</sup>",
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-40,  # Adjust x position
                            ay=20)
                        fig_bil_root.add_annotation(
                            x = df['(t)<sup>(1/4)</sup>'][index_bs],
                            y = slope_b * (df['(t)<sup>(1/4)</sup>'][index_bs]) + intercept_b,
                            text=f'|m|= {-1*slope_b:.2f} psi/hr<sup>(0.25)</sup>',
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-20,  # Adjust x position
                            ay=-25

                        )
                        with col_2:
                            if t_be and t_bs is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <Xf >
                                <Fc>
                                <div>
                                """
                                .replace('<Fc>', f'<p style="font-size: 14px; margin: 0;">Fracture Conductivity Fc (kfwf): {Fc:.2f} md-ft</p>'), unsafe_allow_html=True)
                
                with col9:
                    st.plotly_chart(fig_bil)
                with col10:
                    st.plotly_chart(fig_bil_root)
            with st.sidebar:
                co9, co10 = st.columns(2)
                with co9:
                    c9 = st.checkbox('Check for Spherical')
                with co10:
                    c10 = st.checkbox('Confirm   ')
            if c9==True:
                df['(t)<sup>(-1/2)</sup>'] = 1/(np.sqrt(df['t (hours)']))
                df['dp/d(t<sup>(-1/2)</sup>)'] = [(df['del_p'][0])/(1/np.sqrt((df['t (hours)'][0]))) if i == 0 else np.absolute((df['del_p'][i]-df['del_p'][i-1])/((1/(np.sqrt(df['t (hours)'][i])))-(1/(np.sqrt(df['t (hours)'][i-1]))))) for i in range(len(df))]
                fig_sph = px.scatter(df, x = 't (hours)', y = 'dp/d(t<sup>(-1/2)</sup>)', title = 'Check for Spherical', log_x = True, log_y = True, color_discrete_sequence= ['red'])
                fig_sph.update_xaxes(minor=dict(ticks="inside", showgrid=True))
                fig_sph.update_yaxes(minor=dict(ticks="inside", showgrid=True), title_text = '|dp/d(t<sup>(-1/2)</sup>)|')
                fig_sph.update_layout(showlegend=False)
                fig_sph_root = px.scatter(df, x = '(t)<sup>(-1/2)</sup>', y = 'pwf (psi)', title = 'Spherical Conventional Plot', color_discrete_sequence= ['red'])
                
                if c10==True:
                    t_ss = st.sidebar.number_input('Aproximated Start time for Spherical flow: ', value=None, min_value=0.0, step=0.1)
                    t_se = st.sidebar.number_input('Aproximated End time for Spherical flow: ', value=None, min_value=0.0, step=0.1)
                    if t_ss and t_se is not None:
                        index_ss = df[df['t (hours)'] >= t_ss].index[0]
                        index_se = df[df['t (hours)'] <= t_se].index[-1]
                        stb5 = np.average(df['dp/d(t<sup>(-1/2)</sup>)'][index_ss:index_se+1])
                        slope_s, intercept_s = np.polyfit(df['(t)<sup>(-1/2)</sup>'][index_ss:index_se+1], df['pwf (psi)'][index_ss:index_se+1], 1)
                        Ks = np.power(((4906*q*Bo*np.sqrt(phi*vis*ct))/(np.absolute(slope_s))),(2/3))
                        Kv = (np.power(Ks,3))/(np.power(k2,2))
                        t1s = df['(t)<sup>(-1/2)</sup>'][0]  # Very small value close to 0 (logarithmically)
                        t2s = df['(t)<sup>(-1/2)</sup>'][index_se]    # Example of a larger value on the x-axis

                        # Calculate pwf values using the slope and intercept
                        pwf1_s = slope_s * t1s + intercept_s
                        pwf2_s = slope_s * t2s + intercept_s
                        fig_sph_root.add_trace(go.Scatter(x=[t1s, t2s], y=[pwf1_s, pwf2_s],
                                    mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                        
                        
                        fig_sph.add_hline(y=stb5, line=dict(color='yellow', dash='dash', width=2))
                        fig_sph.add_annotation(
                            
                            # Place annotation at the start of the x-axis
                            x= np.log10(t_ss),
                            y = np.log10(stb5),
                            text=f"|dp/d(t<sup>(-1/2)</sup>)| = {stb5:.2f} psi/hr<sup>(-0.5)</sup>",
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-40,  # Adjust x position
                            ay=20)
                        fig_sph_root.add_annotation(
                            x = df['(t)<sup>(-1/2)</sup>'][index_ss],
                            y = slope_s * (df['(t)<sup>(-1/2)</sup>'][index_ss]) + intercept_s,
                            text=f'|m|= {slope_s:.2f} psi/hr<sup>(-0.5)</sup>',
                            font= dict(color = 'white', size = 15),
                            showarrow=True,
                            arrowhead=1,
                            bgcolor = 'gray',
                            arrowcolor= 'gray',
                            ax=-20,  # Adjust x position
                            ay=-25

                        )
                        with col_2:
                            if t_se and t_ss is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <Ks>
                                <Kv>
                                <div>
                                """
                                .replace('<Ks>', f'<p style="font-size: 14px; margin: 0;">Spherical Permeability (Ks): {Ks:.2f} md</p>')
                                .replace('<Kv>', f'<p style="font-size: 14px; margin: 0;">Vertical Permeability (Kv): {Kv:.2f} md</p>'), unsafe_allow_html=True)
                
                with col11:
                    st.plotly_chart(fig_sph)
                with col12:
                    st.plotly_chart(fig_sph_root)

            c11 = st.sidebar.checkbox('Confirm for Boundar Faults')
            if c11==True:
                t_fs = st.sidebar.number_input("Fault's start time felt:", value=None, min_value=0.0, step=0.1)
                t_fe = st.sidebar.number_input("Fault's end time felt:", value=None, min_value=0.0, step=0.1)
                if t_fs and t_fe is not None:
                    index_fs = df[df['t (hours)'] >= t_fs].index[0]
                    index_fe = df[df['t (hours)'] <= t_fe].index[-1]
                    # Calculate the average value of 'ddp/dlnt' within the specified time range
                    stb6 = np.average(df['ddp/dlnt'][index_fs:index_fe+1])
                    slope_f, intercept_f = np.polyfit(np.log10(df['t (hours)'][index_fs:index_fe+1]), df['pwf (psi)'][index_fs:index_fe+1], 1)
                    fig_loglog.add_hline(y=stb6, line=dict(color='yellow', dash='dash', width=2))
                    fig_loglog.add_annotation(
                        
                        # Place annotation at the start of the x-axis
                        x= np.log10(t_fs),
                        y =np.log10(stb6),
                        text=f"ΔP'<sub>(fault)</sub>= {stb6:.2f}",
                        font= dict(color = 'white', size = 15),
                        showarrow=True,
                        arrowhead=1,
                        bgcolor = 'gray',
                        arrowcolor= 'gray',
                        ax=-40,  # Adjust x position
                        ay=-20)
                    t1f = df['t (hours)'][0]  # Very small value close to 0 (logarithmically)
                    t2f = t_fe     # Example of a larger value on the x-axis

                    # Calculate pwf values using the slope and intercept
                    pwf1_f = slope_f * np.log10(t1f) + intercept_f
                    pwf2_f = slope_f * np.log10(t2f) + intercept_f
                    
                    fig_semilog.add_trace(go.Scatter(x=[t1f, t2f], y=[pwf1_f, pwf2_f],
                                mode='lines', line=dict(color='yellow', dash='dash', width=2), showlegend=False))
                    fig_semilog.add_annotation(
                        x = np.log10(df['t (hours)'][index_fs]),
                        y = slope_f * np.log10(df['t (hours)'][index_fs]) + intercept_f,
                        text=f'|m<sub>(fault)</sub>|= {-1*slope_f:.2f}',
                        font= dict(color = 'white', size = 15),
                        showarrow=True,
                        arrowhead=1,
                        bgcolor = 'gray',
                        arrowcolor= 'gray',
                        ax=-20,  # Adjust x position
                        ay=-25

                    )
                    if np.absolute(slope)/np.absolute(slope_f) >= 0.45:
                        F = 'Single Fault'
                        d = 0.5*np.power(10, -1*((df['del_p'][index_fs+1]/np.absolute(slope_f))-np.log10(df['t (hours)'][index_fs+1])-np.log10(k2/(phi*vis*ct*rw))+3.23-0.435*s2))
                        with col_2:
                            if t_fe and t_fs is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <F>
                                <d>
                                <div>
                                """
                                .replace('<F>', f'<p style="font-size: 14px; margin: 0;">Fault Type: {F}</p>')
                                .replace('<d>', f'<p style="font-size: 14px; margin: 0;">Distance to Fault: {d} ft</p>'), unsafe_allow_html=True)
                    
                
                    if np.absolute(slope)/np.absolute(slope_f) <= 0.45:
                        F2 = 'Intersecting Fault'
                        theta = 360*slope/slope_f
                        with col_2:
                            if t_fe and t_fs is not None:
                                
                                st.markdown("""
                                <div style="
                                    padding: 2px;
                                    background-color: lightgrey;
                                    color: black;
                                    border-radius: 5px;
                                    font-size: 10px;
                                    font-family: Arial, sans-serif;
                                    text-align: center;
                                    height: 100%;
                                    width: 100%;
                                ">
                                <F2>
                                <theta>
                                <div>
                                """
                                .replace('<F2>', f'<p style="font-size: 14px; margin: 0;">Fault Type: {F2}</p>')
                                .replace('<theta>', f'<p style="font-size: 14px; margin: 0;">Angle B/W Faults: {theta:.2f}°</p>'), unsafe_allow_html=True)
                
                    
                    
            with col1:
                st.plotly_chart(fig_loglog)
            with col2:
                st.plotly_chart(fig_semilog)


                
                
            if st.button('Data Preview'):
                st.write(st.session_state.updated_df)
            



# Data Preview page
# if st.session_state.page == 'data':
#     st.write('Data Preview')
#     st.write(st.session_state.updated_df)

#     if st.sidebar.button('Back to Graphs'):
#         st.session_state.page = 'graph'
#         st.experimental_rerun()
