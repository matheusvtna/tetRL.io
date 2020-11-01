import numpy as np
import random

# Aqui deverá ser feita a integração com o tetr.io para a execução das ações 
# executadas pelo agente (treinamento ou execução) e recepção do estado de
# observação. 
# Integração do código usando Selenium para pegar o estado do grid, as variáveis
# do tetrominó e aplicar


# Tetrominós possíveis
# shapes = {
#     'T': 
#     'J': 
#     'L': 
#     'Z': 
#     'S': 
#     'I': 
#     'O': 
# }
# shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']

# Funções de movimentação
# def left(shape, board):


# def right(shape, board):


# def rotate(shape, board):


# def idle(shape, board):


# Engine do Tetris
class TetrisEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)

        # Espaço de ações
        self.value_action_map = {
            0: left,
            1: right,
            2: rotate_right,
            3: idle,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # Parâmetros do jogo
        self.time = -1
        self.score = -1
        self.x = None
        self.y = None
        self.shape = None
        self.n_deaths = 0

        # Reset the game 
        self.reset()


    def step(self, action):

        self.x = # Pelo selenium
        self.y = # Pelo selenium

        # 1 linha abaixo a cada step
        self.shape, self.x, self.y = down(self.shape, self.board)

        # Atualizando tempo e recompensa
        self.time += 1
        reward, done = self.reward_and_done_flag()

        state = # get_board()

        return state, reward, done

    def reset(self):
        self.time = 0
        self.score = 0
        self.board = np.zeros_like(self.board) 

        # Reset o site de alguma forma...

        return self.board

    def reward_and_done_flag(self){
        reward = 0
        done = False

        # A partir dos dados do jogo, decide qual será a recompensa do agente

        # Completou linha? Recompensa de width * qtd de linhas completas
        # Perdeu o jogo?   Recompensa de -100 e done = True
        # Nada aconteceu?  Recompensa de 0

        return reward, done
    }