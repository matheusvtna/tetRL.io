import math
import random
import sys
import os
import shutil
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from engine import TetrisEngine


# -----------------------------------------------------------------------------

# Criação do Tetris
width, height = 10, 16 
engine = TetrisEngine(width, height)

# CUDA
use_cuda = torch.cuda.is_available()
if use_cuda:print("....Usando GPU...")

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor


# -----------------------------------------------------------------------------

#
# Transição
# Tupla que representa uma transição no ambiente (State Action Reward State) 
#
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#
# Replay Memory
# Um buffer cíclico de tamanho limitado que salva as transições obeservadas 
# recentemente. O método sample aqui desenvolvido seleciona um lote aleatório 
# para treino.
#
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#
# DQN = Q-Learning + Deep Neural Networks
#
# A DQN utiliza duas redes neurais para aproximar a função de valor Q(s,a).
# A primeiro é denominado rede neural principal, representada pelo vetor de 
# peso theta, e é utilizado para estimar os valores Q para o estado atual s 
# e ação a: Q(s, a; theta). A segunda é a rede neural alvo, parametrizada pelo 
# vetor de peso a', e terá exatamente a mesma arquitetura que a rede principal, 
# mas será utilizada para estimar os valores Q do próximo estado s' e ação a'. 
# Toda a aprendizagem tem lugar na rede principal. A rede alvo é congelada 
# (os seus parâmetros são deixados inalterados) durante algumas iterações e 
# depois os pesos da rede principal são copiados para a rede alvo, transferindo 
# assim o conhecimento aprendido de um para o outro. Isto torna as estimativas 
# produzidas pela rede alvo mais precisas após a cópia ter ocorrido.
#
class DQN(nn.Module):

    # Arquitetura da rede
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.lin1 = nn.Linear(576, 256)
        self.head = nn.Linear(256, engine.nb_actions)

    # Forward-propagation 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))


# -----------------------------------------------------------------------------
# Treinamento
#
# Hyperparameters
#
# Variable
# Um wrapper de torch.autograd.Variable que enviará automaticamente os dados 
# para a GPU cada vez que construímos uma Variable.


BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
CHECKPOINT_FILE = 'checkpoint.pth.tar'

steps_done = 0

model = DQN()
print(model)

if use_cuda:
    model.cuda()

loss = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=.001)
memory = ReplayMemory(3000)


#
# Select Action
# Selecionará uma ação de acordo com uma política epsilon-greedy (gananciosa em epsilon)
# Dito de forma simples, por vezes utilizaremos o nosso modelo para escolher a ação, e  
# por vezes apenas colhemos uma amostra uniforme. A probabilidade de escolher uma ação  
# aleatória começará em EPS_START e irá decair exponencialmente em direção a EPS_END. 
# EPS_DECAY controla a taxa de decadência.
#
def select_action(state):
    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            actions = model(Variable(state).type(FloatTensor))
            final_action = actions.data.max(1)[1].view(1, 1)
            
            # Às vezes, printa a distribuição de probabilidade
            if random.random() < 0.001:
                print(actions)

            return final_action
    else:
        return FloatTensor([[random.randrange(engine.nb_actions)]])


episode_durations = []

# -----------------------------------------------------------------------------
# Loop de treino

last_sync = 0

# Optimize Model
# Função que realiza um único passo da otimização. Primeiro recolhe amostras
# de um lote de transições, concatena todos os tensores em um único, calcula
# Q(s_t, a_t) e V(s_t+1) = arg max(Q(s_t+1, a)), e os combina na perda. Por
# definição, temos que V(s) = 0 se s for um estado terminal.
#
def optimize_model():
    global last_sync

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Máscara para manter apenas os estados não-terminais
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    with torch.no_grad():
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]))
    
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Calcular Q(s_t, a)
    # O modelo calcula Q(s_t) e nós selecionamos as colunas de ações tomadas
    state_action_values = model(state_batch).gather(1, action_batch)

    # Calcular V(s_t+1) para todos os próximos estados
    with torch.no_grad():
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
    
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    # Calcular os valores esperados de Q
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Calcular a perda de Huber
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Otimizar o modelo
    optimizer.zero_grad()
    loss.backward()

    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()

    return loss.item()


# Salva os pesos da rede
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# Carrega os pesos da rede
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    try: 
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
    except Exception as e:
        pass


    return checkpoint['epoch'], checkpoint['best_score']

# Main 
if __name__ == '__main__':
    
    start_epoch = 0
    best_score = 0

    # Checa se deve retomar algum treinamento específico
    if len(sys.argv) > 1 and sys.argv[1] == 'resume':

        if len(sys.argv) > 2:
            CHECKPOINT_FILE = sys.argv[2]
        
        if os.path.isfile(CHECKPOINT_FILE):
            print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
            start_epoch, best_score = load_checkpoint(CHECKPOINT_FILE)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(CHECKPOINT_FILE, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))


    # Loop de treino
    f = open('log.out', 'w+')
    for i_episode in count(start_epoch):
        # Inicializa o ambiente e recebe o estado inicial
        state = FloatTensor(engine.start()[None,None,:,:]) if i_episode == 0 else FloatTensor(engine.reset()[None,None,:,:])

        score = 0
        for t in count():
            # Seleciona uma ação
            action = select_action(state).type(LongTensor)

            # Aplica a ação e recebe o estado de observação
            last_state = state
            state, reward, done = engine.step(action.item())
            state = FloatTensor(state[None,None,:,:])
            
            # Acumula a recompensa
            score += int(reward)

            reward = FloatTensor([float(reward)])

            # Salva a transição no Replay Buffer
            if reward > 0:
                memory.push(last_state, action, state, reward)

            # Executa um passo de otimização na rede alvo (target network)
            if done:
                
                # Treino do modelo
                if i_episode % 10 == 0:
                    log = 'epoch {0} score {1}'.format(i_episode, score)
                    print(log)
                    f.write(log + '\n')
                    loss = optimize_model()
                    if loss:
                        print('loss: {:.0f}'.format(loss))
                
                # Salva um Checkpoint a cada 100 episódios
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
                        'memory' : memory
                        }, is_best)
                break

    f.close()
    print('Complete')
