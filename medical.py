import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from pycaret.classification import load_model, predict_model

 
modelo1 = load_model('meu-melhor-modelo-para-charges')
modelo2 = load_model('meu-melhor-modelo-para-smoker')

def classificador(modelo, dados):
	pred = predict_model(estimator = modelo, data = dados) 
	return pred

def smap(x):  
		y = 'male' if x == 'Masculino' else 'female' 
		return y

def rmap(x):
	if x == 'Sudeste':
		return 'southeast'
	elif x == 'Noroeste':
		return 'northwest'
	elif x == 'Sudoeste':
		return 'southwest' 
	else:
		return 'northeast'

def fmap(x):  
	y = 'yes' if x == 'Sim' else 'no' 
	return y

st.sidebar.header('**Custo Médio**') 

opcoes = ['Página Inicial', 
		  'Modelagem de valor do seguro', 
		  'Detectar probabilidade de fraude', 
		  'Observações']

pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)

st.sidebar.markdown('---')


###### PAGINA INICIAL ######

if pagina == 'Página Inicial':
	
	st.write("""
	# Bem-vindo ao Medical Cost (Custo Médio)

	Nesse Web App podemos utilizar em produção os modelos desenvolvidos tanto para
	precificar novos seguros, quanto para buscar por fraudadores do seguro.
	
	A lista abaixo ilustra o que está implementado até o momento. 
	### Funcionalidades no momento
	- [x]  Página Inicial 
	- [x]  Modelo em produção para precificar planos de saúde em novos clientes
	- [x]  Modelo em produção para detectar possíveis fraudadores 

	Os modelos desse web-app foram desenvolvidos utilizando o conjunto de 
	dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/mirichoi0218/insurance).
	
	O referencial sobre os modelos utilizados você pode encontrar nesse [link](https://github.com/eotorres/Medical).
	Foi realizado dois tipos de analises para a construção final da previsão
	Os modelos são desenvolvidos e analisados utilizando a biblioteca [PyCaret](https://pycaret.org/).
	
	Para mais informações sobre o Streamlit, consulte o [site oficial](https://www.streamlit.io/) ou a sua [documentação](https://docs.streamlit.io/_/downloads/en/latest/pdf/).
	[Lista de emojis para markdown](https://gist.github.com/rxaviers/7360908).
        	""")

###### PAGINA: MODELO DE COTACAO DO SEGURO ######

elif pagina == 'Modelagem de valor do seguro':
	
	st.markdown('# Modelagem de valor do seguro')

	st.markdown('Nessa seção é feito o deploy do modelo para cotar o valor do seguro para um indivíduo.\
			Entre com os dados e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	idade = st.number_input('Idade', 18, 65, 30)
	sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
	fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = st.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

	#custos = st.number_input('Custos da pessoa', 1000, 64000, 10000)

	dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 
			'children': [criancas], 'region': [rmap(regiao)], 'smoker': [fmap(fumante)]}
		
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo1, dados)
		pred = float(saida['Label'].round(2))
		valor = round(1.8*pred, 2)
		x = round(1.8,2)

		s1 = 'Custo Estimado do Seguro (Pred): ${:.2f}'.format(pred)
		s2 = 'Valor de Venda do Seguro: ${:.2f}'.format(valor)
		s3 = 'Calculo realizado :'
		
		st.markdown('## Resultados do modelo para as entradas:')
		st.write(dados)
		st.markdown('## **' + s1 + '**') 
		st.markdown('## **' + s2 + '**')
		st.markdown('## **' + s3 + '**')

###### PAGINA: MODELO DE FRAUDE ######

elif pagina == 'Detectar probabilidade de fraude':

	st.markdown('# Detectar probabilidade de fraude')

	st.markdown('Nessa seção é feito o deploy do modelo para detectar probabilidade de fraude na \
		     variável "fumante". Entre com os dados do indivíduo\
		      em análise e clique em APLICAR O MODELO para obter as predições.')

	st.markdown('---')

	idade = st.number_input('Idade', 18, 65, 30)
	sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
	#fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = st.selectbox("Região em que mora", 
								  ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

	custos = st.number_input('Custos da pessoa', 1000, 64000, 10000)
 
	dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 
			'children': [criancas], 'region': [rmap(regiao)], 'charges': [custos]}
		
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo2, dados)
		resp = 'NÃO' if saida['Label'][0] == 'no' else 'SIM' 
		prob = saida['Score'][0] 
		st.markdown('## **O indivíduo em análise é fumante?**')
		s = 'Resposta do modelo: {}, com probabilidade {:.2f}%.'.format(resp, 100*prob)
		st.markdown('## **' + s + '**') 

		if resp == 'NÃO':
			st.success('Probabilidade Baixa de Fraude!')
		elif prob < 0.7:
			st.warning('Probabilidade Moderada de Fraude!')
		else:
			st.error('Probabilidade Alta de Fraude!')
	




###### PAGINA: OBSERVAÇÕES ######
else:
	st.write("""
			 # Observações
             """) 

	st.markdown('Nesse Web App mostramos o poder do streamlit para construir \
		soluções fáceis, rápidas e que permitem uma usabilidade bastante ampla.')
		
	
	st.markdown('---')
	
	if st.button('Comemorar'):
		st.balloons()

 
