# Redes neurais: classificação de objetos astronômicos com PyTorch e rede neural do zero com NumPy

Dois estudos de redes neurais feitos no Ramo Estudantil do IEEE Computational Intelligence Society da UnB.

## 1. Classificação de galáxias, estrelas e quasares com PyTorch

Notebook: [`Rede_neural_galaxias.ipynb`](Rede_neural_galaxias.ipynb)

**Problema:** classificar 100 mil objetos observados pelo Sloan Digital Sky Survey (SDSS) em galáxia, estrela ou quasar, a partir das medidas de cada objeto.

**Dados:** [Stellar Classification Dataset – SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17), no Kaggle. Baixe o arquivo `star_classification.csv` e coloque-o na mesma pasta do notebook.

**Modelo base:** rede totalmente conectada com duas camadas ocultas (50 e 40 neurônios, ReLU), otimizador Adam com learning rate 0,001, entropia cruzada e 20 épocas. Os dados foram divididos de forma estratificada em 70% treino, 15% validação e 15% teste, e padronizados com `StandardScaler`.

**Resultado:** 96,9% de acurácia no conjunto de teste.

| Classe | Precisão | Recall | F1 |
| --- | --- | --- | --- |
| Galáxia | 0,97 | 0,97 | 0,97 |
| Estrela | 0,97 | 1,00 | 0,98 |
| Quasar | 0,95 | 0,92 | 0,94 |

### Experimentos

A partir do modelo base, variei uma coisa por vez e medi a acurácia no conjunto de teste.

| Variação | Acurácia no teste |
| --- | --- |
| Modelo base (50 e 40 neurônios, 20 épocas) | 96,88% |
| Mais largo (400 e 320 neurônios) | 96,43% |
| Gargalo de 2 neurônios por camada | 91,36% |
| Mais profundo (4 camadas ocultas) | 96,83% |
| Raso (1 camada oculta) | 96,07% |
| 50 épocas | 96,97% |
| 5 épocas | 95,23% |

O que os experimentos mostraram:

- Aumentar a largura ou a profundidade quase não muda o resultado, porque o problema é relativamente simples de separar. Mesmo comprimida para 2 neurônios, a rede ainda passa de 91%.
- Mais épocas melhoram a acurácia, mas o ganho diminui rápido enquanto o custo de treino cresce na mesma proporção.
- Comparei learning rates de 0,1 (alto demais), 1e-6 (baixo demais, risco de underfitting) e 0,001 para ver como cada um afeta a convergência. As curvas de perda e acurácia de treino e validação estão no notebook.
- Adam com regularização L2 (weight decay) terminou um pouco à frente de SGD com momentum na validação: 96,93% contra 96,65% após 30 épocas.

## 2. Rede neural do zero com NumPy

Notebook: [`Redes_Neurais_from_scratch.ipynb`](Redes_Neurais_from_scratch.ipynb)

Implementação de uma rede neural sem frameworks de deep learning, só com NumPy, para entender o que acontece por baixo do PyTorch:

- camada densa com forward e backpropagation;
- ativação ReLU;
- softmax combinada com entropia cruzada;
- otimizador SGD com decaimento do learning rate e momentum.

A rede tem uma camada oculta de 500 neurônios e foi treinada por 2.000 épocas no conjunto de dados em espiral (3 classes), chegando a 96% de acurácia nos dados de treino.

O código segue o livro *Neural Networks from Scratch in Python*, de Harrison Kinsley e Daniel Kukieła. A biblioteca `nnfs`, que acompanha o livro, é usada para gerar os dados.

## Como rodar

```bash
pip install torch pandas scikit-learn matplotlib seaborn numpy nnfs jupyter
jupyter notebook
```

Os notebooks também rodam no Google Colab.
