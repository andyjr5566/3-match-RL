import random
import numpy as np
import collections
import json
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.layers.merge import Add, Concatenate
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import time
import BJ_NEAT_TEST

class BJ_Sarsa:
    def __init__(self):

        self.gameMain = BJ_NEAT_TEST
        self.gameTest = BJ_NEAT_TEST

        self.NSARS = 50       # number of SARS' sequences to generate at each turn
        self.DEPTH = 5 	#max self.DEPTH of SARS' sequence
        self.NUMSAMPLES = 10000	# number of total trials to run
        self.NTURNS = 3000		# number of turns in the game
        self.NROWS = 8
        self.NCOLS = 8
        self.DISCOUNT = 0.5		# self.DISCOUNT in computing Q_opt
        self.STEP_SIZE = 0.00000000001		# eta in computing Q_opt
        self.EPSILON = 0		# self.EPSILON in self.EPSILON-greedy (used in generating SARS')
        self.isBulit = False
        self.mainActionID = 0
        self.testActionID = 0
        self.lr = 0.001
        self.mainRead = 'Read'
        self.mainWrite = 'Write'
        self.testRead = 'Read_T'
        self.testWrite = 'Write_T'
    def main(self):
        # initail game board
        # need to open two game simulater, one is for main operation, the other use as experience generation.
        # Require Read and Write json file for both simulation, main one name as 'Read' and 'Write', the other one name as 'Read_T' and 'Write_T', respectively.
        
        self.gameMain.main('Main')
        self.gameMain.runGame()
        # self.gameTest.main('Test')
        # self.gameTest.runGame()
        score = 0
        for turn in range(self.NTURNS):
            
            turnsLeft = self.NTURNS - turn
            print ('')
            print ('Turns left:', self.NTURNS - turn)
            print ('SCORE', score)

            MainCurState = np.array(self.gameMain.gameBoard)

            currState = (self.arrToTuple(MainCurState), turnsLeft)
            print ('choosing action...')
            action = None

            if (random.random() < self.EPSILON):
                print ('choosing random action...')
                action = random.choice(self.actions(currState))
            else:
                print ('choosing optimal action...')
                Vopt, pi_opt = max((self.getQopt(currState, action), action) for action in self.actions(currState))
                action = pi_opt
            
            actTable = self.action4write(action)
            _, _, reward, observation_, done = self.gameMain.interact(actTable)
            
            grid, turnScore = np.array(observation_), reward
            newState = (self.arrToTuple(grid), turnsLeft-1)
            print ('updating net')
            self.updateWeights(currState, action, turnScore, newState)

            print ('')
            print ('Turns left:', 0)
            print (grid)
            print ('FINAL SCORE', score)

        return score

    def state2catagory(self, observation):
        observation = np.array(observation)
        observation = np.concatenate(observation[:])
        observation  = to_categorical(observation, num_classes=7)
        # observation = observation.reshape(1,observation.shape[0],observation.shape[1])
        observation = observation.reshape(observation.shape[0]*observation.shape[1])
        return observation.tolist()

    def updateWeights(self, state, action, reward, newState):
        Vopt, pi_opt = max((self.getQopt(newState, action), action) for action in self.actions(newState))
        # weights = weights - self.STEP_SIZE * (getQopt(state, action) - (reward + self.DISCOUNT*Vopt)) * getFeatureVec(state, action)
        # for i in range(NFEATURES):
        # 	if weights[i] < 0:
        # 		weights[i] = 0.
        print ("residual:", self.net.predict([self.getFeatureVec(state, action)])[0] - (reward + self.DISCOUNT*Vopt),
        self.net.fit([self.getFeatureVec(state, action)], [reward + self.DISCOUNT*Vopt], epochs= 10))
        return

    def buileNet(self, phi):
        print('Build net')
        inputs = Input(shape=(len(phi),))
        x = Dense(4000, activation='relu')(inputs)
        x = Dropout(.25)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dropout(.25)(x)
        x = Dense(250, activation='relu')(x)
        x = Dropout(.25)(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(.25)(x)
        outputs = Dense(1)(x)
        self.net = Model(inputs, outputs)
        self.isBulit = True
        a = Adam(lr = self.lr)
        self.net.compile(loss='mean_squared_error', optimizer=a, metrics=['accuracy'])

    def getQopt(self, state, action):

        if self.isEndState(state): return 0.
        FeatureVec = self.getFeatureVec(state, action)
        
        return self.net.predict(FeatureVec)[0]

    def getFeatureVec(self, state, action):
        minRow = min(action[0][0], action[1][0])
        maxRow = max(action[0][0], action[1][0])
        sameCol = 1 if action[0][1] == action[1][1] else 0
        nValidMoves = len(self.actions(state))
        # maxDelete = testSwitch(state, action)
        medUtil = np.median([self.generateSARSA(state, action) for i in range(self.NSARS)])
        phi = [minRow, maxRow, sameCol, nValidMoves, medUtil]
        # print(phi)

        phi += self.state2catagory(self.gameMain.gameBoard)

        if self.isBulit == False:
            self.buileNet(phi)
        return np.array(phi)[np.newaxis,:]

    def generateSARSA(self, currState, action):
        prevState = currState
        utility = 0
        depth = 0
        while ((not self.isEndState(prevState)) and depth < self.DEPTH):
            if depth != 0:
                action = random.choice(self.actions(prevState))
            actTable = self.action4write(action)
            if depth == 0:
                print('first depth')
                _, _, reward, observation_, done = self.gameTest.interact_grid(self.gameMain.gameBoard, actTable)
            else:
                _, _, reward, observation_, done = self.gameTest.interact_grid(observation_, actTable)

            grid = np.array(observation_)
            newState = (self.arrToTuple(grid), prevState[1]-1)
            prevState = newState
            utility += reward * (self.DISCOUNT**depth)
            depth += 1
        return utility

    def action4write(self, action):
        minRow = min(action[0][0], action[1][0])
        minCol = min(action[0][1], action[1][1])
        sameCol = 1 if action[0][1] == action[1][1] else 0
        icon = minRow * self.NROWS + minCol
        if sameCol == 1:
            movement = 'Right'
        elif sameCol == 0:
            movement = 'Bottom'
        
        act = icon*2 + sameCol
        actTable = np.zeros(self.NROWS * self.NCOLS *2)
        actTable[act] = 1
        return [actTable]

    def isEndState(self, state):
        return state[1] == 0

    def actions(self, state):
        actions = []
        grid = state[0]
        for i in range(self.NROWS):
            for j in range(self.NCOLS):
                if self.isValidMove(grid,(i,j),(i,j+1)):
                    coord1 = (i,j)
                    coord2 = (i,j+1)
                    actions.append((coord1, coord2))
                elif self.isValidMove(grid,(i,j),(i+1,j)):
                    coord1 = (i,j)
                    coord2 = (i+1,j)
                    actions.append((coord1, coord2))
                if grid[i][j] == 'choco':
                    coord1 = (i,j)
                    if self.isValidCoord((i+1,j)):
                        coord2 = (i+1,j)
                        actions.append((coord1, coord2))
                    if self.isValidCoord((i-1,j)):
                        coord2 = (i-1,j)
                        actions.append((coord1, coord2))
                    if self.isValidCoord((i,j+1)):
                        coord2 = (i,j+1)
                        actions.append((coord1, coord2))                
                    if self.isValidCoord((i,j-1)):
                        coord2 = (i,j-1)
                        actions.append((coord1, coord2))
        return actions

    def isValidCoord(self, coord):
        if (coord[0] < 0 or coord[0] >= self.NROWS) or (coord[1] < 0 or coord[1] >= self.NCOLS):
            return False
        else:
            return True

    def arrToTuple(self, arr):
        tupArr = [tuple(elem) for elem in arr]
        return tuple(tupArr)

    def initailGameBoard(self):
        self.mainActionID = -1
        readF = {
            "ActionId": self.mainActionID,
            "PatternId": 0,
            "MoveDirection": 'Left', 
            "Pattern": []
            }

        with open('%s.json'%self.mainRead, 'w') as f:
            json.dump(readF, f)
        self.mainActionID += 1
    
    def initailTestBoard(self):
        self.testActionID = -1
        readF = {
            "ActionId": self.testActionID,
            "PatternId": 0,
            "MoveDirection": 'Left', 
            "Pattern": []
            }

        with open('%s.json'%self.testRead, 'w') as f:
            json.dump(readF, f)
        self.testActionID += 1

    def readMainBoard(self):
        with open('%s.json'%self.mainWrite) as f:
            writeF = json.load(f)
        return writeF
    
    def writeOnMain(self, PatternId, MoveDirection):
        
        readF = {
            "ActionId": self.mainActionID,
            "PatternId": PatternId,
            "MoveDirection": MoveDirection, 
            "Pattern": []
            }
        with open('%s.json'%self.mainRead, 'w') as f:
            json.dump(readF, f)

        self.mainActionID += 1
        if self.mainActionID == 99999999999:
            self.mainActionID = -1

    def readTestBoard(self):
        with open('%s.json'%self.testWrite) as f:
            writeF = json.load(f)
        return writeF

    def writeOnTest(self, PatternId, MoveDirection, Pattern):
        readF = {
            "ActionId": self.testActionID,
            "PatternId": PatternId,
            "MoveDirection": MoveDirection, 
            "Pattern": Pattern
            }
        print('Test action: ',self.testActionID)
        with open('%s.json'%self.testRead, 'w') as f:
            json.dump(readF, f)

        self.testActionID += 1
        if self.testActionID == 99999999999:
            self.testActionID = -1
        
    def state2array(self, state):
        grid = []
        for icon in state:
            grid.append(icon.split('_')[1])
        grid = np.array(grid).reshape(self.NCOLS ,self.NROWS)
        return grid

    def state2categorical(self, state):
        grid = []
        for icon in state:
            lab = icon.split('_')
            icon = lab[1]
            if icon == 'choco':
                grid.append(33)
            elif len(lab) == 2:
                grid.append(int(icon))
            elif lab[2] == 'extra':
                grid.append(int(icon)+8*3)    
            elif lab[3] == 'horiz':
                grid.append(int(icon)+8)
            elif lab[3] == 'vert':
                grid.append(int(icon)+8*2)

        grid = np.array(grid) - 1
        grid = to_categorical(grid, num_classes=33)
        grid = grid.reshape(grid.shape[0]*grid.shape[1])
        grid = grid.tolist()
        return grid

    def validMoveExists(self, grid):
        for i in range(self.NROWS):
            for j in range(self.NCOLS):
                if self.isValidMove(grid,(i,j),(i,j+1)) or self.isValidMove(grid,(i,j),(i+1,j)):
                    return True
        return False

    def isValidMove(self, grid,coord1,coord2):
        #coord is (x,y)
        if coord1 == coord2:
            return False
        if self.isValidCoord(coord1) and self.isValidCoord(coord2):
            if (abs(coord1[0] - coord2[0]) == 1 and coord1[1] == coord2[1]) or (abs(coord1[1] - coord2[1]) == 1 and coord1[0] == coord2[0]):
                gridCopy = np.copy(grid)
                gridCopy[coord1], gridCopy[coord2] = gridCopy[coord2], gridCopy[coord1]
                colorRowsSet1, colorColsSet1 = self.exploreCoord(gridCopy, coord1[0], coord1[1])
                if len(colorRowsSet1) >= 3 or len(colorColsSet1) >= 3:
                    return True
                colorRowsSet2, colorColsSet2 = self.exploreCoord(gridCopy, coord2[0], coord2[1])
                if len(colorRowsSet2) >= 3 or len(colorColsSet2) >= 3:
                    return True
        return False

    def exploreCoord(self, grid, i, j):
        color = grid[i, j]
        colorRowsSet = set([(i, j)])
        colorColsSet = set([(i, j)])
        #explore above
        for row in range(i):
            if color != grid[i-row-1, j]:
                break
            colorRowsSet.add((i-row-1, j))
        #explore below
        for row in range(i+1, self.NROWS):
            if color != grid[row, j]:
                break
            colorRowsSet.add((row, j))
        #explore left
        for col in range(j):
            if color != grid[i, j-col-1]:
                break
            colorColsSet.add((i, j-col-1))
        #explore right
        for col in range(j+1, self.NCOLS):
            if color != grid[i, col]:
                break
            colorColsSet.add((i, col))
        return colorRowsSet, colorColsSet

if __name__ == "__main__":
    game = BJ_Sarsa()
    game.main()