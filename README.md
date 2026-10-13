# Avaliação final — IA e Dados

Repositório com as resoluções das questões 4, 7 e 10 da avaliação final. As atividades exploram redução de dimensionalidade, ensembles para classificação tabular e AutoML para classificação de imagens.

## Conteúdo

| Questão | Tema | Arquivos |
| --- | --- | --- |
| 4 — Avançada | Pipeline PCA + LDA e classificação KNN no MNIST | [`Q4/questao4_pca_lda_pipeline.ipynb`](Q4/questao4_pca_lda_pipeline.ipynb) |
| 7 — Intermediária | Classificação de crédito com modelos individuais, Voting e Stacking | [`Q7/questao7_stacking_voting_credito.ipynb`](Q7/questao7_stacking_voting_credito.ipynb) e [`Q7/german_credit.csv`](Q7/german_credit.csv) |
| 10 — Avançada | AutoGluon e CNN manual no CIFAR-10 | [`Q10/questao10_autogluon_cifar10.ipynb`](Q10/questao10_autogluon_cifar10.ipynb) |

## Resumo das questões

### Questão 4 — PCA, LDA e KNN

Usa as 10 classes do MNIST, aplica PCA para reter 95% da variância e, em seguida, LDA com no máximo 9 componentes. Compara o KNN para `k` entre 1 e 10 com uma alternativa que usa PCA sem LDA.

A execução registrada na notebook favorece PCA + KNN sem LDA: acurácia de teste de 94,70%, contra 91,84% para PCA + LDA + KNN. A redução do LDA para 9 dimensões diminui o custo computacional, mas também pode descartar informação útil à classificação.

### Questão 7 — Classificação de crédito

Trabalha com o German Credit (1.000 clientes e 20 atributos). O pré-processamento combina One-Hot Encoding para variáveis categóricas e StandardScaler para variáveis numéricas. Compara DummyClassifier, Regressão Logística, Árvore de Decisão e SVM, além de Voting por maioria e Stacking com Regressão Logística como metamodelo.

A avaliação inclui hold-out estratificado e validação cruzada estratificada de 5 folds, usando acurácia, precisão, recall e F1. A classe positiva é `1` (mau pagador), portanto recall e F1 ajudam a avaliar a detecção dos clientes de risco. A conclusão registrada é que o ganho do Stacking é pequeno e não conclusivo frente aos modelos individuais.

O CSV do German Credit está incluído. A notebook o carrega localmente; se o arquivo não estiver disponível, também contém uma alternativa para obter os dados via OpenML.

### Questão 10 — AutoML e CIFAR-10

Compara o `MultiModalPredictor` do AutoGluon com uma CNN manual no CIFAR-10. A notebook usa uma subamostra de 6.000 imagens de treino e 1.500 de teste e configura ambos os métodos para 10 épocas.

Na execução documentada, o AutoGluon obteve 98,80% de acurácia de teste em cerca de 18 minutos, enquanto a CNN manual obteve 59,47% em cerca de 16 segundos. Esses valores são resultados daquela execução, não garantias: podem variar conforme ambiente, dados e treinamento. O AutoGluon pode exigir recursos computacionais consideráveis; a notebook também cria arquivos de imagem locais para preparar os dados.

## Como executar

1. Abra o repositório em Jupyter Notebook, JupyterLab ou VS Code com suporte a notebooks.
2. Instale as dependências da questão que deseja executar.
3. Execute as células da notebook em ordem.

Para as questões 4 e 7, as dependências principais são NumPy, pandas, Matplotlib e scikit-learn; a obtenção de dados pelo OpenML também requer acesso à internet. Para a questão 10, a notebook registra esta instalação:

```bash
pip install autogluon.multimodal tensorflow matplotlib pandas scikit-learn pillow
```

Recomenda-se usar um ambiente virtual isolado. O CIFAR-10 é carregado pelo TensorFlow e baixado na primeira execução. O tempo de treinamento depende do hardware disponível.
