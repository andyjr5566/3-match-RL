import numpy as np
from keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from keras.models import Model, load_model
import keras
from keras.optimizers import RMSprop ,adam
from keras import backend as K
from collections import deque
np.random.seed(1)
from keras.utils.np_utils import to_categorical  
import os
import random
from Condition_method import Condition
# For Dence
def stack_samples(samples):
	array = np.array(samples)
	
	current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
	actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
	rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
	new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
	
	return current_states, actions, rewards, new_states, dones

# For Conv2D
# def stack_samples(samples):
# 	array = np.array(samples)
	
# 	current_states = np.stack(array[:,0]).reshape((np.stack(array[:,0]).shape))
# 	actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
# 	rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
# 	new_states = np.stack(array[:,3]).reshape((np.stack(array[:,3]).shape))
# 	dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
# 	return current_states, actions, rewards, new_states, dones

class DQN:
    def __init__(
            self,
            observation,
            action,
            learning_rate=0.00005,
            reward_decay=0.2,
            e_greedy=0.8,
            replace_target_iter=300,
            memory_size=4000000,
            batch_size=100,
            e_greedy_increment=None,
            output_graph=False,
            epsilon_decay = 0.999995,
    ):
        self.observation = observation   
        self.loc = action[0]
        self.direc = action[1]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon_decay = epsilon_decay
        self.learn_step_counter = 0

        self.memory = deque(maxlen=self.memory_size)
        self.m = Condition(self.loc)

        self._build_net()
        if os.path.isdir('DQN_models'):
            print('=================================================')
            print('                Model loaded')
            print('=================================================')
            self.load_model()
            a = adam(lr=self.lr)
            self.model1.compile(loss='mean_squared_error', optimizer=a, metrics=['accuracy'])
            self.model2.compile(loss='mean_squared_error', optimizer=a, metrics=['accuracy'])

    def state2catagory(self, observation):
        observation = np.array(observation)
        # observation = np.concatenate(observation[:])
        observation  = to_categorical(observation, num_classes=7)
        # observation = observation.reshape(1,observation.shape[0],observation.shape[1])
        # observation = observation.reshape(1,observation.shape[0]*observation.shape[1])
        return observation

    def target_replace_op(self):
        v1 = self.model2.get_weights()
        self.model1.set_weights(v1)
        print("params has changed")

    def _build_net(self):
        # 构建evaluation网络



        # eval_inputs = Input(shape=(self.observation.shape[1],))

        eval_inputs = Input(shape=(self.observation.shape[1],))
        # x = Conv2D(3, 3, padding= 'same', activation='relu')(eval_inputs)
        # x = Dropout(.25)(x)
        # x = Conv2D(3, 3,padding= 'same', activation='relu')(x)
        # x = Flatten()(x)
        x = Dense(400, activation='relu')(eval_inputs)
        x = Dropout(.25)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(.25)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(.25)(x)
        # x = Dense(50, activation='relu')(x)
        # x = Dropout(.25)(x)
        self.q_eval = Dense((self.loc-1) * self.direc)(x)
        # 构建target网络，注意这个target层输出是q_next而不是，算法中的q_target
        # target_inputs = Input(shape=(self.observation.shape[1],))

        target_inputs = Input(shape=(self.observation.shape[1],))
        # x = Conv2D(3, 3, padding= 'same', activation='relu')(target_inputs)
        # x = Dropout(.25)(x)
        # x = Conv2D(3, 3, padding= 'same', activation='relu')(x)
        # x = Flatten()(x)
        x = Dense(400, activation='relu')(target_inputs)
        x = Dropout(.25)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(.25)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(.25)(x)
        # x = Dense(50, activation='relu')(x)
        # x = Dropout(.25)(x)
        self.q_next = Dense((self.loc-1) * self.direc)(x)

        self.model1 = Model(target_inputs, self.q_next)
        self.model2 = Model(eval_inputs, self.q_eval)
        a = adam(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=a, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=a, metrics=['accuracy'])
        print(self.model2.summary())
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r], s_))
            
        #print('transition is:',transition)
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # memory是一个二维列表
        self.memory_counter += 1

    def remember(self, cur_state, action, reward, new_state, done):
        # cur_state = self.state2catagory(cur_state)
        # new_state = self.state2catagory(new_state)
        self.memory.append([cur_state, action, reward, new_state, done])
        
    def act(self, cur_state, posible_actions):
        self.epsilon *= self.epsilon_decay
        c = np.random.random()
        
        if len(posible_actions) == 0:
            action = self.model1.predict(cur_state)
            # print(np.argmax(action[0]))
            return action
        if c < self.epsilon:
            action = random.choice(posible_actions)
            minRow = min(action[0][0], action[1][0])
            minCol = min(action[0][1], action[1][1])
            sameCol = 1 if action[0][1] == action[1][1] else 0
            icon = minCol * int(self.loc**.5) + minRow
            act = icon*2 + sameCol
            actTable = np.zeros(self.loc *2-2)
            actTable[act] = 1
            print(action)
            return actTable[np.newaxis,:]

            # temp = []
            # # for i in range(len(cur_state)):
            # temp.append([np.random.random(),np.random.random()])
            # return np.random.random((self.loc-1) * self.direc)[np.newaxis,:]
            # return self.m.action(cur_state)
            # if c < self.epsilon/2:
            #     return np.random.random((self.loc-1) * self.direc)[np.newaxis,:]
            # else:
            #     return self.m.action(cur_state)
        # cur_state = self.state2catagory(cur_state)
        # cur_state = cur_state[np.newaxis,:]
        # print(cur_state.shape)

        action = self.model1.predict(cur_state)
        # print(np.argmax(action[0]))
        return action

    def learn(self, samples):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        # if self.memory_counter > self.memory_size:
        #     sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # else:
        #     sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # batch_memory = self.memory[sample_index, :]

        cur_states, actions, rewards, new_states, _ =  stack_samples(samples)
        q_next, q_eval = self.model1.predict(cur_states), self.model2.predict(new_states)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = np.argmax(actions, axis=1).astype(int)
        # reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = rewards[0] + self.gamma * np.max(q_next, axis=1)
        
        # rewards += np.add(rewards, self.gamma * np.max(q_next, axis=1) * (1 - dones), out=rewards, casting="unsafe")
        # q_target += np.add(rewards, self.gamma * np.max(q_next, axis=1), out=rewards, casting="unsafe")

        self.model2.fit(new_states, q_target, epochs=10, use_multiprocessing = True, verbose = 1)
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def train(self):
        batch_size = self.batch_size
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.samples = samples
        self.learn(samples)

    def save_model(self):
        try:
            os.makedirs('DQN_models')
        except FileExistsError:
            pass
        self.model1.save('DQN_models/model1.h5')
        self.model2.save('DQN_models/model2.h5')
        
    def load_model(self):
        self.model1 = load_model('DQN_models/model1.h5')
        self.model2 = load_model('DQN_models/model2.h5')