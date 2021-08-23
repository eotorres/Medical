import streamlit as st 
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_option('deprecation.showPyplotGlobalUse', False)

modelo = load_model('meu-melhor-modelo-para-charges')

st.markdown('# Plano de Saúde Deploy Center')

st.sidebar.title('Barra Lateral')

idade = st.sidebar.number_input('Entre com a idade', 18, 65, 20, 1)
imc = st.sidebar.slider('IMC:', 15, 50, 25, 1)
sexo = st.sidebar.selectbox('Entre com o sexo:', ['male', 'female'])
criancas = st.sidebar.slider('Número de crianças:', 0, 5, 0, 1)
fumante = st.sidebar.selectbox('Fumante?:', ['yes', 'no'])
regiao =  st.sidebar.selectbox('Região:', ['southeast', 'southwest', 'northeast', 'northwest'])


dici = {'age': [idade],
		'sex': [sexo],
		'bmi': [imc],
		'children': [criancas],
		'region': [regiao],
		'smoker': [fumante]}

dados = pd.DataFrame(dici) 

saida = predict_model(modelo, dados)

st.write(dados)

#if st.button('APLICAR O MODELO'):
#	pred = float(saida['Label'].round(2)) 
#	s1 = 'Custo Estimado do Seguro: ${:.2f}'.format(pred) 
#	st.markdown('### **' + s1 + '**')  
