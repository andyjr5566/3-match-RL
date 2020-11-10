"""
Using NEAT for reinforcement learning.

The detail for NEAT can be find in : http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import neat
import numpy as np
# import visualize
from keras.utils.np_utils import to_categorical  
import BJ_NEAT_TEST
from neat.math_util import softmax

game = BJ_NEAT_TEST

CONFIG = "config"
EP_STEP = 3000           # maximum episode steps
GENERATION_EP = 10      # evaluate by the minimum of 10-episode rewards
TRAINING = True         # training or testing
CHECKPOINT = 9          # test on this checkpoint

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game.main()
        game.runGame()
        ep_r = []
        for ep in range(GENERATION_EP): # run many episodes for the genome in case it's lucky
            accumulative_r = 0.         # stage longer to get a greater episode reward
            observation = game.gameBoard
            observation = state2catagory(observation)
            for t in range(EP_STEP):
                action_values = net.activate(observation)
                # print(np.argmax(action_values))
                action_values = softmax(action_values)
                # print(np.argmax(action_values))
                _, _, reward, observation_, done = game.interact([action_values])
                accumulative_r += reward
                if done:
                    break
                observation = observation_
                observation = state2catagory(observation)
            ep_r.append(accumulative_r)
        genome.fitness = np.min(ep_r)/float(EP_STEP)    # depends on the minimum episode reward


def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    # recode history
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    pop.run(eval_genomes, 100)       # train 10 generations

    # visualize training
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


def evaluation():
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    winner = p.run(eval_genomes, 1)     # find the winner in restored population

    # show winner net
    node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    # visualize.draw_net(p.config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(net.activate(s))
            s, r, done, _ = env.step(a)
            if done: break

def state2catagory(observation):
    observation = np.array(observation)
    observation = np.concatenate(observation[:])
    observation  = to_categorical(observation, num_classes=7)
    # observation = observation.reshape(1,observation.shape[0],observation.shape[1])
    observation = observation.reshape(observation.shape[0]*observation.shape[1])
    return observation


if __name__ == '__main__':
    if TRAINING:
        run()
    else:
        evaluation()