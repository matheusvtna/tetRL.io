# tetRL.io

## Sobre o Projeto

O tetRL.io é o projeto final da disciplina de Introdução à Multimídia, do Centro de Informática da Universidade Federal de Pernambuco, o CIn - UFPE. A ideia do projeto é desenvolver um algoritmo de Reinforcement Learning para um single agent jogar o famoso jogo de Tetris.
Nesse projeto, a engine de Tetris usada é o [tetr.io](https://uandersonricardo.github.io/tetr.io/), um outro projeto de nossa equipe para a disciplina de Engenharia de Software do CIn. Mais detalhes da implementação do jogo em si estão disponíveis no repositório onde está hospedado o nosso projeto. Para isso, [clique aqui](https://github.com/uandersonricardo/tetr.io).

## Estrutura
### dqn_agent: 
Métodos e descrições do agente e estruturas utilizadas para o seu treinamento usando DQN. **OK**

### run_model:
Execução de um modelo já treinado no ambiente. **OK**

### engine:
Estrutura do Tetris, integração com o [tetr.io](https://uandersonricardo.github.io/tetr.io/) e estruturas do environment. **EM PROGRESSO**

## Dependências 
Para instalar as dependências, basta executar o script ... **EM CRIAÇÃO**

Ou, de forma manual, instalar as bibliotecas:
- [PyTorch](https://pytorch.org/).
- [NumPy](https://anaconda.org/anaconda/numpy)

## Como Treinar
Para treinar um agente do zero, basta usar a seguinte linha de comando:
~~~bash
$ python dqn_agent
~~~
Para treinar um agente a partir de um Checkpoint, basta usar a seguinte linha de comando:
~~~bash
$ python dqn_agent checkpoint
~~~

## Como Executar
Para executar um agente treinado, basta usar a seguinte linha de comando:
~~~bash
$ python run_model checkpoint
~~~

## Equipe 
- Alexandre Burle    [(aqb)](https://github.com/aqb)
- Danilo Vaz         [(dvma)](https://github.com/danilovazm)
- Humberto Lopes     [(hlfs2)](https://github.com/humbertobz26)
- Matheus Andrade    [(mvtna)](https://github.com/matheusvtna)
- Uanderson Ricardo  [(urfs)](https://github.com/uandersonricardo)
