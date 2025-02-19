from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from janitor import clean_names
import pandas as pd
import numpy as np
import joblib

pd.options.mode.copy_on_write = True

# ---------------------------- CONFIGURAÇÕES ---------------------------- #

TEMPO_MAP = {
    '1 a 3 meses': 0.17, '3 a 6 meses': 0.375, '6 meses a 1 ano': 0.75,
    '1 a 2 anos': 1.5, '2 a 3 anos': 2.5, '3 a 5 anos': 4, 'acima de 5 anos': 6
}

# ---------------------------- FUNÇÕES AUXILIARES ---------------------------- #

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os arquivos CSV e retorna os DataFrames de vendas e clientes."""
    df_vendas = pd.read_csv(
        'vendas_atual.csv', sep=';',
        usecols=['Data e hora', 'Funcionário', 'Dia da semana', 'Turno', 'Código', 'Tipo do Item', 'Grupo', 'Líquido', 'Venda']
    )
    df_clientes = pd.read_csv(
        'clientes_atual.csv', sep=';',
        usecols=['Ficha', 'Nome', 'Situação no ciclo de vida', 'Comparecimento', 'Tempo de relacionamento', 'Sexo', 'Ranking ABC', 'Animais vivos']
    )
    return df_vendas, df_clientes

def processar_vendas(df_vendas: pd.DataFrame) -> pd.DataFrame:
    """Limpa e formata o DataFrame de vendas."""
    df_vendas = clean_names(df_vendas).dropna(thresh=3)  # Mantém linhas com pelo menos 3 valores
    df_vendas[['data', 'hora']] = df_vendas['data_e_hora'].str.split(' ', expand=True)
    df_vendas = df_vendas.drop(columns=['data_e_hora']).set_index('codigo')

    df_vendas.index = df_vendas.index.astype('Int64')
    df_vendas['venda'] = df_vendas['venda'].astype(int)
    df_vendas['data'] = pd.to_datetime(df_vendas['data'], format='%d/%m/%Y')
    df_vendas['liquido'] = df_vendas['liquido'].str.replace(',', '.').astype(float)

    df_vendas = df_vendas.apply(lambda x: x.str.strip().str.lower().astype('category') if x.dtype == 'object' else x)
    df_vendas['turno'] = df_vendas['turno'].str.split(' ').str[0].str.lower().replace({'manhã': 'manha'})
    
    df_vendas.insert(0, 'data', df_vendas.pop('data'))  # Move coluna 'data' para o início
    
    return df_vendas


def processar_clientes(df_clientes: pd.DataFrame) -> pd.DataFrame:
    """Limpa e formata o DataFrame de clientes."""
    df_clientes = clean_names(df_clientes).dropna(thresh=3)
    
    df_clientes = df_clientes.apply(lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x)
    df_clientes['animais_vivos'] = df_clientes['animais_vivos'].fillna('').apply(lambda x: x.count(',') + 1 if x else 0)
    df_clientes['sexo'] = df_clientes['sexo'].fillna('desconhecido')

    categorias = ['ranking_abc', 'tempo_de_relacionamento', 'sexo', 'comparecimento']
    df_clientes[categorias] = df_clientes[categorias].astype('category')
    
    df_clientes = df_clientes.rename(columns={'ficha': 'codigo'})
    
    return df_clientes


def agregar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os dados dos clientes e calcula estatísticas relevantes."""
    df_clientes = df.groupby('codigo').agg({
        'venda': 'count',
        'liquido': 'sum',
        'sexo': 'first',
        'tempo_de_relacionamento': 'first',
        'animais_vivos': 'first',
        'situacao_no_ciclo_de_vida': 'first',
        'nome': 'first'
    })

    df_clientes['ticket_medio'] = np.round(df_clientes['liquido'] / df_clientes['venda'], 2)

    for col in ['dia_da_semana', 'turno', 'tipo_do_item', 'grupo']:
        df_clientes[col] = df.groupby('codigo')[col].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    df_clientes.reset_index(inplace=True)
    return df_clientes

def transformar_dados(df_clientes: pd.DataFrame) -> pd.DataFrame:
    """Transforma os dados para o modelo de Machine Learning."""
    df_clientes['churn'] = df_clientes.pop('situacao_no_ciclo_de_vida').apply(lambda x: 0 if x == 'clientes ativos' else 1)
    
    df_clientes['tempo_de_relacionamento'] = df_clientes['tempo_de_relacionamento'].map(TEMPO_MAP)
    df_clientes = df_clientes.dropna()

    df_clientes['sexo'] = df_clientes['sexo'].apply(lambda x: 0 if x != 'feminino' else 1)
    df_clientes['tipo_do_item'] = df_clientes['tipo_do_item'].apply(lambda x: 1 if x.strip() == 'serviço' else 0)

    clientes_totais = df_clientes.shape[0]

    return df_clientes, clientes_totais

# ---------------------------- APLICAR MODELO ---------------------------- #

def prever_churn(df_clientes: pd.DataFrame) -> pd.DataFrame:
    """Carrega o modelo e faz previsões de churn."""
    df_final = df_clientes[['nome', 'ticket_medio', 'tempo_de_relacionamento']].copy()
    
    df_clientes = df_clientes.drop(columns=['codigo', 'nome'])
    df_c = pd.get_dummies(df_clientes, columns=['dia_da_semana', 'turno', 'grupo'])

    modelo_carregado = joblib.load('modelo_churn.pkl')
    
    colunas_treino = modelo_carregado.feature_names_in_
    df_c = df_c.reindex(columns=colunas_treino, fill_value=0)

    previsoes = pd.Series(modelo_carregado.predict(df_c), name='previsoes')
    
    df_final = df_final.reset_index(drop=True)
    previsoes = previsoes.reset_index(drop=True)

    df_final['previsoes'] = previsoes
    df_final = df_final[df_final['previsoes'] == 1].drop(columns=['previsoes'])
    df_final = df_final.sort_values(by='nome', ascending=True)
    df_final.to_csv('df_com_previsoes.csv', index=False)

    return df_final

# ---------------------------- EXECUÇÃO ---------------------------- #

if __name__ == "__main__":
    df_vendas, df_clientes = carregar_dados()

    df_vendas = processar_vendas(df_vendas)
    df_clientes = processar_clientes(df_clientes)

    df = df_vendas.merge(df_clientes, on='codigo', how='left')
    df = df[df['situacao_no_ciclo_de_vida'] == 'clientes ativos']

    df_clientes = agregar_dados(df)
    df_clientes, clientes_totais = transformar_dados(df_clientes)

    df_final = prever_churn(df_clientes)
    clientes_churn = df_final.shape[0]

    print("Previsões salvas em 'df_com_previsoes.csv'.")
    print("Métricas do dataset:")
    print("Clientes totais analisados:", clientes_totais)
    print("Clientes previstos à evadir:", clientes_churn)
    print("Taxa de churn: ", np.round(clientes_churn / clientes_totais,2))
