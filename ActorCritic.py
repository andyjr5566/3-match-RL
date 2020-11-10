
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Conv2D, LeakyReLU, Flatten
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical   
import keras.backend as K
import numpy as np
import os
import cv2

# import tensorflow as tf
import tensorflow as tf
tf.disable_v2_behavior() 
import random
from collections import deque

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

class ActorCritic:
    def __init__(self, observation, action):
        sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)
        # K.set_session(sess)
        self.sess = sess
        self.observation = self.state2catagory(observation)
        self.loc = action[0]
        self.direc = action[1]
        self.learning_rate = 0.001
        self.epsilon = .0
        self.epsilon_decay = .9999995
        self.gamma = .90
        self.tau   = .01

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=4000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        # ===================================================================== #
        #                              Load Model                             #
        # ===================================================================== #
        if os.path.isdir('ActorCritic_models'):
            print('=================================================')
            print('                Model loaded')
            print('=================================================')
            self.load_model()

        # ===================================================================== #
        #                               Actor Model                             #
        self.actor_critic_grad = tf.placeholder(tf.float32,
            [None, (self.loc-1) * self.direc]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        self.critic_grads = tf.gradients(self.critic_model.output,
            self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())



    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        
        # state_input = Input(shape=(self.observation.shape[0],self.observation.shape[1],self.observation.shape[2]))

        state_input = Input(shape=(self.observation.shape[1],))
        # cov = Conv2D(3,3,activation='relu')(state_input)
        # f = Flatten()(cov)
        h1 = Dense(500, activation='relu')(state_input)
        h2 = Dense(1000, activation='relu')(h1)
        h3 = Dense(500, activation='relu')(h2)
        output = Dense((self.loc-1) * self.direc, activation='softmax')(h3)

        model = Model(state_input, output)

        adam  = Adam(lr=0.0001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        
        # state_input = Input(shape=(self.observation.shape[0],self.observation.shape[1],self.observation.shape[2]))

        state_input = Input(shape=(self.observation.shape[1],))
        # cov = Conv2D(3,3,activation='relu')(state_input)
        # f = Flatten()(cov)
        
        h1 = Dense(500, activation='relu')(state_input)
        h2 = Dense(1000, activation='relu')(h1)
        h3 = Dense(500, activation='relu')(h2)

        action_input = Input(shape=(((self.loc-1) * self.direc,)))
        action_h1    = Dense(100)(action_input)

        merged    = Concatenate()([h3, action_h1])
        merged_h1 = Dense(10, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([state_input,action_input], output)

        adam  = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #
    def state2catagory(self, observation):
        observation = np.array(observation)
        observation = np.concatenate(observation[:])
        observation  = to_categorical(observation, num_classes=7)
        # observation = observation.reshape(1,observation.shape[0],observation.shape[1])
        observation = observation.reshape(1,observation.shape[0]*observation.shape[1])
        return observation

    def remember(self, cur_state, action, reward, new_state, done):
        cur_state = self.state2catagory(cur_state)
        new_state = self.state2catagory(new_state)
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        
            cur_states, actions, rewards, new_states, _ =  stack_samples(samples)
            predicted_actions = self.actor_model.predict(cur_states)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  cur_states,
                self.critic_action_input: predicted_actions
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_states,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
   
        cur_states, actions, rewards, new_states, dones = stack_samples(samples)

        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states, target_actions])

        rewards += np.add(rewards, self.gamma * future_rewards * (1 - dones), out=rewards, casting="unsafe")
        
        evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
#         print(evaluation.history)
    def train(self):


        batch_size = 400
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self.samples = samples

        self._train_critic(samples)
        self._train_actor(samples)

        self.update_target()
    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            # temp = []
            # # for i in range(len(cur_state)):
            # temp.append([np.random.random(),np.random.random()])
            return np.random.random((self.loc-1) * self.direc)[np.newaxis,:]
        cur_state = self.state2catagory(cur_state)
        # cur_state = cur_state[np.newaxis,:]
        # print(cur_state.shape)
        return self.actor_model.predict(cur_state)

    # ========================================================================= #
    #                         Model Saving & Loading                            #
    # ========================================================================= #

    def save_model(self):
        try:
            os.makedirs('ActorCritic_models')
        except FileExistsError:
            pass

        self.actor_model.save('ActorCritic_models/actor_model.h5')
        self.target_actor_model.save('ActorCritic_models/target_actor_model.h5')
        self.critic_model.save('ActorCritic_models/critic_model.h5')
        self.target_critic_model.save('ActorCritic_models/target_critic_model.h5')

    def load_model(self):
        self.actor_model = load_model('ActorCritic_models/actor_model.h5')
        self.actor_state_input = self.actor_model.input
        self.target_actor_model = load_model('ActorCritic_models/target_actor_model.h5')

        self.critic_model = load_model('ActorCritic_models/critic_model.h5')
        self.critic_state_input, self.critic_action_input = self.critic_model.input
        self.target_critic_model = load_model('ActorCritic_models/target_critic_model.h5')