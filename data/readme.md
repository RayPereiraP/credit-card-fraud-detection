# Credit Card Fraud Detection

Modelo de machine learning para detectar transações fraudulentas em um dataset altamente desbalanceado (~0,17% de fraudes).

## Dataset
[Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
Baixe o `creditcard.csv` e coloque em `data/`.

## Por que não usar accuracy?
Um modelo que diz "nunca é fraude" teria ~99,8% de accuracy e não detectaria nada. Por isso a avaliação usa **PR-AUC**, **precision** e **recall** da classe fraude.

## Abordagem
- Split em treino, validação e teste (estratificado)
- Comparação entre Regressão Logística e Gradient Boosting, ambos com `class_weight="balanced"`
- Escolha do modelo pela PR-AUC na validação
- Threshold de decisão ajustado na validação para maximizar o F1, sem olhar o teste

## Como rodar
```bash
pip install -r requirements.txt
python train.py
```

## Resultados (conjunto de teste)
| Métrica | Valor |
|---|---|
| PR-AUC | XX |
| Precision | XX |
| Recall | XX |
| F1 | XX |

![Curva Precision-Recall](pr_curve.png)

## Stack
Python, pandas, scikit-learn, matplotlib