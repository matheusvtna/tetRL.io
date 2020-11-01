import curses
import sys
import os
import torch
import time
from engine import TetrisEngine
from dqn_agent import DQN, ReplayMemory, Transition
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

# Criação do Tetris
width, height = 10, 20 
engine = TetrisEngine(width, height)

# Carrega o modelo (Pesos da rede)
def load_model(filename):
    model = DQN()
    if use_cuda:
        model.cuda()
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    return model

# Executa o modelo carregado
def run(model):
    state = FloatTensor(engine.clear()[None,None,:,:])
    score = 0

    while True:
        action = model(Variable(state,
            volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1).type(LongTensor)
        
        print( model(Variable(state,
            volatile=True).type(FloatTensor)).data)

        # Executando a ação no ambiente e recebendo estado de observação
        state, reward, done = engine.step(action.item())
        state = FloatTensor(state[None,None,:,:])

        # Acumulando a recompensa
        score += int(reward)

        stdscr = curses.initscr()
        stdscr.clear()
        stdscr.addstr(str(engine))
        stdscr.addstr('\ncumulative reward: ' + str(score))
        stdscr.addstr('\nreward: ' + str(reward))
        time.sleep(.1)

        if done:
            print('score {0}'.format(score))
            break


# Nome do modelo não digitado na execução do código
if len(sys.argv) <= 1:
    print('specify a filename to load the model')
    sys.exit(1)

# Main
if __name__ == '__main__':
    filename = sys.argv[1]

    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        model = load_model(filename).eval()
        run(model)
    else:
        print("=> no file found at '{}'".format(filename))
