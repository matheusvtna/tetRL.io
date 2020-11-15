import numpy as np
import random
from tetrio_interface import TetrioInterface
from selenium.webdriver.common.keys import Keys

# Aqui deverá ser feita a integração com o tetr.io para a execução das ações 
# executadas pelo agente (treinamento ou execução) e recepção do estado de
# observação. 
# Integração do código usando Selenium para pegar o estado do grid, as variáveis
# do tetrominó e aplicar

# Tetrominós possíveis
shapes = {
    'T': [[0, 1, 0], [1, 1, 1]],
    'J': [[1, 0, 0], [1, 1, 1]],
    'L': [[0, 0, 1], [1, 1, 1]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]]
}

shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']

# Funções de movimentação
def left(interface):
    # Executa o movimento
    interface.press(Keys.ARROW_LEFT)

    # Lê o estado do bloco
    block = interface.get_block()

    # Retorna as variáveis
    return np.array(block['tetromino']), block['x'], block['y']


def right(interface):
    # Executa o movimento
    interface.press(Keys.ARROW_RIGHT)

    # Lê o estado do bloco
    block = interface.get_block()

    # Retorna as variáveis
    return np.array(block['tetromino']), block['x'], block['y']

def down(interface):
    # Executa o movimento
    interface.press(Keys.ARROW_DOWN)

    # Lê o estado do bloco
    block = interface.get_block()

    # Retorna as variáveis
    return np.array(block['tetromino']), block['x'], block['y']

def rotate(interface):
    # Executa o movimento
    interface.press(Keys.ARROW_UP)

    # Lê o estado do bloco
    block = interface.get_block()

    # Retorna as variáveis
    return np.array(block['tetromino']), block['x'], block['y']


def idle(interface):
    # Lê o estado do bloco
    block = interface.get_block()

    # Retorna as variáveis
    return np.array(block['tetromino']), block['x'], block['y']


# Engine do Tetris
class TetrisEngine:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(height, width), dtype=np.float)

        # Espaço de ações
        self.value_action_map = {
            0: left,
            1: right,
            2: down,
            3: rotate,
            4: idle,
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
        self.current_height = 0

        # Reset the game 
        self.interface = TetrioInterface('https://uandersonricardo.github.io/tetr.io/')
        self.interface.navigate()
        self.interface.set_speed(0)


    def step(self, action):
        # Executa um passo
        self.interface.step()

        # Executa a ação
        self.value_action_map[action](self.interface)

        # Lê o estado do bloco
        block = self.interface.get_block()

        # Atualiza as variáveis de estado
        self.shape, self.x, self.y = block['tetromino'], block['x'], block['y']

        # Atualizando tempo
        self.time += 1

        # Lê o estado do grid
        grid = self.interface.get_grid()
        state = np.array([[0 if j == None else j for j in i] for i in grid['matrix']])

        # Atualiza a recompensa
        reward, done = self.reward_and_done_flag(state)

        return state, reward, done

    def start(self):
        self.time = 0
        self.score = 0
        self.board = np.zeros_like(self.board) 

        # Reseta o site
        self.interface.start()

        return self.board

    def reset(self):
        self.time = 0
        self.score = 0
        self.board = np.zeros_like(self.board) 

        # Reseta o site
        self.interface.reset()

        return self.board

    def reward_and_done_flag(self, grid):
        reward = 0
        done = False

        # A partir dos dados do jogo, decide qual será a recompensa do agente
        state = None

        while state == None:
            state = self.interface.get_state()

        # Completou linha? Recompensa de width * qtd de linhas completas
        # Perdeu o jogo?   Recompensa de -100 e done = True
        # Nada aconteceu?  Recompensa de 0
        if state == 0:
            new_height = -1

            for line in range(len(grid)):
                if 1 in grid[line]:
                    new_height = self.height - line
                    break

            if new_height > self.current_height:
                reward = -100
                self.current_height = new_height
            else:
                reward = 0
        elif state == 1:
            reward = 100
        elif state == 2:
            reward = -100
            done = True

        return reward, done
