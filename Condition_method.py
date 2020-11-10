import numpy as np
from random import randrange, randint

#evaluates the points to be won on a board

class Condition:

    def __init__(self, loc):
        self.NUM_GEMS = 7
        self.SAMPLE_SIZE = 500

        self.ROWS, self.COLS = int(loc**.5), int(loc**.5)
        # print(self.ROWS, self.COLS)
        self.pos1 = (0, 0)
        self.pos2 = (0, 0)

    def evalBoard(self):
        total = 0
        for r in range(self.ROWS):
            for c in range(self.COLS):
                gem = self.matrix[r][c]

                #horizontal
                for i in range(1, self.COLS - c):
                    if self.matrix[r][c + i] == gem:
                        if i >= 2:
                            total += 1
                    else:
                        break

                #vertical
                for j in range(1, self.ROWS - r):
                    if self.matrix[r + j][c] == gem:
                        if j >= 2:
                            total += 1
                    else:
                        break
        return total

    #swap two positions in the matrix
    def swapMatrix(self, pos1, pos2):
        self.matrix[pos1[1]][pos1[0]], self.matrix[pos2[1]][pos2[0]] = self.matrix[pos2[1]][pos2[0]], self.matrix[pos1[1]][pos1[0]]

    #save the best swap found
    def rankSwap(self, pos1, pos2):
        #swap positions
        self.swapMatrix(pos1, pos2)
        score = self.evalBoard()
        if score > self.best:

            self.ways = []
            self.best = score
            self.pair = pos1, pos2
            self.ways.append(self.pair)
        if score == self.best:
            try:
                self.pair = pos1, pos2
                self.ways.append(self.pair)
            except:
                pass
        #swap back to original
        self.swapMatrix(pos1, pos2)    

    #run through possible position swaps
    def findBestSwap(self):
        
        self.best, self.pair = 0, []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if row < self.ROWS - 1:            
                    self.rankSwap((col, row), (col, row + 1))
                if col < self.COLS - 1:
                    self.rankSwap((col, row), (col + 1, row))
        if len(self.ways) > 0:
            ind = randint(0,len(self.ways)-1)
            self.pair = self.ways[ind]
            self.ways = []
            return self.pair
        else:
            return None

    
    def action(self, matrix):
        self.matrix = matrix
        self.findBestSwap()

        pos1, pos2 = self.pair[0], self.pair[1]

        if pos2[0] - pos1[0] == 0: # down
            act = (pos1[0] * self.COLS + pos1[1])*2 + 1
        elif pos2[0] - pos1[0] == 1:
            act = (pos1[0] * self.COLS + pos1[1])*2 
        
        temp = np.zeros(((self.ROWS * self.COLS)-1)*2)
        temp[act] = 1
        
        return temp[np.newaxis,:]

