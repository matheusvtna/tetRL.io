# tetRL.io

## Sobre o Projeto

O tetRL.io é o projeto final da disciplina de Introdução à Multimídia, do Centro de Informática da Universidade Federal de Pernambuco, o CIn - UFPE. A ideia do projeto é desenvolver um algoritmo de Reinforcement Learning para um single agent jogar o famoso jogo de Tetris.
Nesse projeto, a engine de Tetris usada é feita pelo próprio código em Python, porém a renderização da interface é feita no Browser. Para isso, utilizamos da base do [tetr.io](https://uandersonricardo.github.io/tetr.io/), um outro projeto de nossa equipe para a disciplina de Engenharia de Software do CIn, e criamos um repositório mais simples que apenas renderizasse o grid do Tetris e apresentasse alguns dados do treinamento da rede, o [web-tetRL.io](https://github.com/uandersonricardo/web-tetRL.io). 
Mais detalhes da implementação da interface estão disponíveis no repositório onde está hospedado o nosso projeto. Para isso, [clique aqui](https://github.com/uandersonricardo/web-tetRL.io).

## Estrutura
### train: 
Métodos de carregamento e looping de treino da rede neural para a criação de modelos. 

### test:
Execução de um modelo já treinado no ambiente.

### tetris:
Estrutura do Tetris com definição de regras, peças e métodos do environment do aprendizado. 

### web_interface
Integração com o [web-tetRL.io](https://github.com/uandersonricardo/web-tetRL.io).

### deep_q_network
Definição e inicialização da arquitetura da rede neural.

## Dependências 
- [PyTorch](https://pytorch.org/).
- [NumPy](https://anaconda.org/anaconda/numpy)
- [Selenium](https://selenium-python.readthedocs.io/)
- [Webdriver Manager](https://github.com/SergeyPirogov/webdriver_manager)

## Como Treinar
### Um Novo Agente
~~~bash
$ python train.py
~~~
### Agente a partir de um Checkpoint

No arquivo train.py, certifique-se de que os seguintes parâmetros estarão definidos com os valores corretos para o diretório e nome do arquivo, assim como a flag de load definida como True.
~~~python 
parser.add_argument("--saved_path", type=str, default="trained_models")  # Diretório de salvamento 
parser.add_argument("--saved_name", type=str, default="tetris")          # Arquivo de salvamento
parser.add_argument("--checkpoint_name", type=str, default="tetris")     # Arquivo de carregamento
parser.add_argument("--load", type=bool, default=True)                   # Flag para carregamento
~~~
E execute o mesmo comando para um novo treino:
~~~bash
$ python train.py
~~~

## Como Executar
No arquivo test.py, certifique-se de que os seguintes parâmetros estarão definidos com os valores corretos para o diretório e nome do arquivo.
~~~python     
parser.add_argument("--saved_path", type=str, default="trained_models")
parser.add_argument("--checkpoint_file", type=str, default="tetris")
~~~
~~~bash
$ python test.py
~~~

## Equipe 
- Alexandre Burle    [(aqb)](https://github.com/aqb)
- Danilo Vaz         [(dvma)](https://github.com/danilovazm)
- Humberto Lopes     [(hlfs2)](https://github.com/humbertobz26)
- Matheus Andrade    [(mvtna)](https://github.com/matheusvtna)
- Uanderson Ricardo  [(urfs)](https://github.com/uandersonricardo)

## Créditos
O projeto foi fortemente inspirado pelo código disponibilizado por Viet Nguyen, sob a [licença MIT](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/blob/master/LICENSE). Gostaríamos de deixar explícito nossos agradecimentos ao autor. [Clique aqui](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch) para visitar o Github do projeto base.
