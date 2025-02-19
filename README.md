# random-forest-churn

<h1>📊 Detecção de Churn com Aprendizado de Máquina</h1>

<p>Este projeto utiliza aprendizado de máquina para identificar clientes com alta probabilidade de evasão (<em>churn</em>).</p>
    
<p>O modelo foi treinado com três anos de dados (merge da tabela vendas e clientes), com variáveis como tempo de relacionamento,ticket_medio e numero_animais.</p>
<p>A variável alvo é churn(0,1), definida como probabilidade de evasão no período de 1 ano.</p>

<h2>📊 Relatório de Classificação</h2>
    <pre>
              precision    recall  f1-score   support

           0       0.85      0.82      0.84       115
           1       0.81      0.85      0.83       104

    accuracy                           0.83       219
   macro avg       0.83      0.83      0.83       219
weighted avg       0.83      0.83      0.83       219
    </pre>

<h2>📈 Métricas</h2>
  <ul>
      <li><strong>AUC-ROC:</strong> 0.91</li>
      <li><strong>Acurácia média da validação cruzada:</strong> 0.80</li>
  </ul>
<p align="center">
    <img src="https://github.com/user-attachments/assets/fedbc906-1113-4307-a254-b25d4d0466e6" alt="Imagem 1">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/0481d474-8f91-452f-a61b-3a071f793bda" alt="Imagem 2">
</p>
