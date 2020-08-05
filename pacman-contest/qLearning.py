from captureAgents import CaptureAgent
import random,util,math
from game import *
from game import Directions, Actions
from capture import AgentRules, GameState
import pickle
import time
import copy
from util import PriorityQueue
from util import Queue

def createTeam(firstIndex, secondIndex, isRed,
               first = 'QLearningAgent', second = 'QLearningAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class QLearningAgent(CaptureAgent):

    targetedFoods = []
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.trainNum = 1
        self.epsilon = 0.1
        self.learningRate = 0.2
        self.discountFactor = 0.95
        self.start = gameState.getAgentPosition(self.index)
        self.weights = util.Counter()
        self.readWeights("weights_file")
        self.ckWeights()

        self.end = {}
        self.deadEnd = {}
        self.endNew = {}
        
        initMap(self, gameState)

        self.lastState = None
        self.isRed = gameState.isOnRedTeam(self.index)
        if self.isRed:
            self.ourIndices = gameState.getRedTeamIndices()
        else:
            self.ourIndices = gameState.getBlueTeamIndices()
        self.enemyIndex = [index for index in self.getOpponents(gameState)]

    def chooseAction(self, gameState):
        self.ckWeights()
        actions = gameState.getLegalActions(self.index)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            simuAction = self.simulate(gameState)
            if simuAction==None:
                action = random.choice(actions)
            else:
                action = simuAction
        if action in actions:   
            return action
        else:
            return random.choice(actions)

    def features(self, state, action):
        # Finds the minimum distance to our home state

        nextState = self.getSuccessor(state, action)
        
        # extract the grid of food and wall locations and get the ghost locations
        foodList = self.getFood(nextState).asList()
        walls = nextState.getWalls()
        #We get the enemy states based on the old state. They shouldn't move unless we ate them
        enemies = [state.getAgentState(i) for i in self.enemyIndex]
        # enemies_new = [nextState.getAgentState(i) for i in self.enemyIndex]
        mapArea = (walls.width * walls.height)
        for agentIndex in self.ourIndices:
            if agentIndex == self.index:
                myState = nextState.getAgentState(agentIndex)
            else:
                allyState = nextState.getAgentState(agentIndex)

        capsules = self.getCapsules(nextState)

        features = util.Counter()
        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        nextPos = nextState.getAgentPosition(self.index)
        next_x, next_y = nextPos

        features["distanceToHome"] = self.distanceToHome(myState,walls, next_x, next_y,mapArea)

        # deal with the enemy
        for enemy in enemies:
            enemyPosition = enemy.getPosition()
            if enemyPosition is None:
                continue

            if nextPos != enemyPosition:
                features["eatFood"] = 1.0

            distanceToEnemy = float(self.getMazeDistance(nextPos, enemyPosition)) 
            if distanceToEnemy == 0:
                distanceToEnemy = 0.1
            enemyIsPacman = enemy.isPacman
            if enemyIsPacman and not myState.isPacman and myState.scaredTimer <= 1:
                features["killpacman"] = 1.0
                if distanceToEnemy < features["invaderDistance"] or features["invaderDistance"] == 0:
                    features["invaderDistance"] = distanceToEnemy

            elif not enemyIsPacman and enemy.scaredTimer > 1:
                features["killpacman"] = 1.0
                if distanceToEnemy < features["invaderDistance"] or features["invaderDistance"] == 0:
                    features["invaderDistance"] = distanceToEnemy
            elif not enemyIsPacman:
                features["#ghosts-one-step"] += (next_x, next_y) in Actions.getLegalNeighbors(enemyPosition, walls)

                if distanceToEnemy < features["ghostDistance"] or features["ghostDistance"] == 0:
                    features["ghostDistance"] = distanceToEnemy

        allyDistance = float(self.getMazeDistance(nextPos, allyState.getPosition())) + 0.1
        features["allyDistance"] = 0.01 / allyDistance / mapArea
        
        # if there is no danger of ghosts then add the food feature
        
        if features["ghostDistance"] > 3 or features["ghostDistance"] == 0: 
            
            if len(foodList) > 2:
                myOldFoodTar = None
                friendFoodTar = None
                for (recordedIndex, foodLocation) in self.targetedFoods:
                    if recordedIndex == self.index:
                        myOldFoodTar = (recordedIndex, foodLocation)
                    else:
                        friendFoodTar = foodLocation
                if not myOldFoodTar is None:
                    self.targetedFoods.remove(myOldFoodTar)
                if not friendFoodTar is None and friendFoodTar in foodList:
                    foodList.remove(friendFoodTar)

                clstFoodDistance, closestFood = min( [ (self.getMazeDistance(nextPos, f), f) for f in foodList])
                if clstFoodDistance == 0:
                    clstFoodDistance = 0.1
                if clstFoodDistance is not None:
                    self.targetedFoods.append((self.index, closestFood))
                    features["closest-food"] = 1 / float(clstFoodDistance) / mapArea  

        else:
            if nextPos in self.deadEnd.keys():
                features['deadEnd'] = 1.0

            if len(capsules) > 0:
                capsuleDistance = min([self.getMazeDistance(nextPos, c) for c in capsules])
                if capsuleDistance == 0:
                    capsuleDistance = 0.1
                if capsuleDistance is not None:
                    features["closest-cap"] = float(capsuleDistance) / mapArea
        
        if features["ghostDistance"]:
            features["ghostDistance"] = 1 / features["ghostDistance"]
        if features["invaderDistance"]:
            features["invaderDistance"] = 1 / features["invaderDistance"]
        
        if action == Directions.STOP: features['stop'] = 1

        features.divideAll(10.0)
        return features

    def distanceToHome(self,myState,walls, next_x, next_y,mapArea):
            width = walls.width
            height = walls.height
            borderX = int(width/2-self.red)
            yRange = list(range(0, height))
            wallList = []
            #store all the walls on the border
            for y in yRange:
              if walls[borderX][y]:
                wallList.append(y)
            # remove recorded walls from yRange and the rest is entrance
            for y in wallList:
              yRange.remove(y)

            minDistance = min(self.getMazeDistance((next_x, next_y), (borderX, y)) for y in yRange)
            
            return (myState.numCarrying) * float(minDistance) / mapArea

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor

    def computeQValue(self, state, action):
        '''
        this function returns a Q(s,a) for each state and action
        '''
        QVal = self.weights * self.features(state,action)
        return QVal

    def readWeights(self, file_name):
        with open(file_name, 'rb') as inputfile:
            self.weights = pickle.load(inputfile)

    def saveWeights(self, file_name):
        with open(file_name, 'wb') as outputfile:
            pickle.dump(self.weights, outputfile,0)

    def final(self, state):
        self.saveWeights("weights_file")
        QLearningAgent.targetedFoods = []

    def ckWeights(self):
        #negative
        if self.weights['closest-cap'] > 0:
            self.weights['closest-cap'] = -self.weights['closest-cap']
        if self.weights['distanceToHome'] > 0:
            self.weights['distanceToHome'] = -self.weights['distanceToHome']
        if self.weights['ghostDistance'] > 0:
            self.weights['ghostDistance'] = -self.weights['ghostDistance']
        if self.weights['#ghosts-one-step'] > 0:
            self.weights['#ghosts-one-step'] = -self.weights['#ghosts-one-step']
        if self.weights['deadEnd'] > 0:
            self.weights['deadEnd'] = -self.weights['deadEnd']
        if self.weights["teamwork"] > 0:
            self.weights['teamwork'] = -self.weights['teamwork']
        if self.weights["stop"] > 0:
            self.weights['stop'] = -self.weights['stop']
        if self.weights["reverse"] > 0:
            self.weights['reverse'] = -self.weights['reverse']
        if self.weights['allyDistance'] > 0:
            self.weights['allyDistance'] = -self.weights['allyDistance']


        #should be postitive
        if self.weights['closest-food'] < 0:
            self.weights['closest-food'] = -self.weights['closest-food']
        if self.weights['invaderDistance'] < 0:
            self.weights['invaderDistance'] = -self.weights['invaderDistance']
        if self.weights['killpacman'] < 0:
            self.weights['killpacman'] = -self.weights['killpacman']
        if self.weights['#-of-pacmen-1-step-away'] < 0:
            self.weights['#-of-pacmen-1-step-away'] = -self.weights['#-of-pacmen-1-step-away']

    def simulate(self, state):
        qVals = []
        queue = Queue()
        positon0 = state.getAgentState(self.index).getPosition()
        x, y = positon0
        positon0 = (int(x), int(y))
        queue.push(([positon0],0,state))

        while (not queue.isEmpty()):
            positonListOld,QvalueOld,gameStateOld= queue.pop()
            positonList = copy.copy(positonListOld)
            actions =gameStateOld.getLegalActions(self.index)
            for action in actions:
                positonListnew = copy.copy(positonList)
                if action != 'Stop':

                    successorState= self.getSuccessor(gameStateOld, action)
                    positon = successorState.getAgentState(self.index).getPosition()
                    x, y = positon
                    positon = (int(x), int(y))
                    if positonListnew.count(positon)<2:

                        positonListnew.append(positon)

                        qvalue = self.computeQValue(gameStateOld, action)

                        if len(positonListnew) < 5:
                            q=QvalueOld + math.pow(0.5, len(positonListnew)) * qvalue
                            queue.push((positonListnew, q,
                                       successorState))
                        if len(positonListnew)==5:

                            qVals.append((copy.copy(positonListnew), QvalueOld + math.pow(0.7, len(positonListnew)) * qvalue))

        history = self.observationHistory
        historyLength=len(history)
        if historyLength>=4:
            direction0 = history[historyLength-1].getAgentState(self.index).getDirection()
            # print direction0
            direction1 = history[historyLength - 2].getAgentState(self.index).getDirection()
            # print direction1
            direction2 = history[historyLength - 3].getAgentState(self.index).getDirection()
            # print direction2

            if direction0==getOppoDir(direction1) and  direction0==direction2:
                if state!= None and  state.getLegalActions(self.index)!=None :
                    if direction2 in state.getLegalActions(self.index):

                        return direction2

                    return None
        return self.getBestAction(qVals,state)

    def getBestAction(self,Qvalue,state):
        path=[]
        maxVal=-10000
        for item in Qvalue:
            curX,curY=item
            if curY>maxVal:
                maxVal=curY
                path=curX
        nextPosition = path[1]
        curPosition = state.getAgentPosition(self.index)
        curX,curY=curPosition
        nextX,nextY=nextPosition
        bestAction= Actions.vectorToDirection((nextX-curX,nextY-curY))
        return bestAction

def initMap(self, gameState):
  walls = gameState.getWalls()
  width = gameState.data.layout.width
  height = gameState.data.layout.height
  if self.red:

    for i in range(int(width/2), width):
      for j in range(1, height):
         findPosWithThreeWalls(self,walls, (i, j))

    self.deadEnd=mergeDicts(self.deadEnd, copy.copy(self.end))

    while len(self.end)!=0:
      for item in self.end:
        checkNeighbourhood(self,walls,item,self.end[item])

      self.deadEnd = mergeDicts(self.deadEnd, copy.copy(self.endNew))
      self.end=copy.copy(self.endNew)
      self.endNew={}

  else:

    for i in range(0, width/2):
      for j in range(1, height):
        findPosWithThreeWalls(self, walls, (i, j))
    self.deadEnd = mergeDicts(self.deadEnd, copy.copy(self.end))

    while len(self.end) != 0:
      for item in self.end:
        checkNeighbourhood(self, walls, item, self.end[item])

      self.deadEnd = mergeDicts(self.deadEnd, copy.copy(self.endNew))
      self.end = copy.copy(self.endNew)
      self.endNew = {}
    

def mergeDicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def findPosWithThreeWalls(self,state, pos):

    i, j = pos
    cnt = 0
    dir=""
    if not state[i][j]:

      if i < state.width - 1:
          if state[i + 1][j]:
            cnt += 1
          else:
            dir='E'
      if i >  0:
          if state[i - 1][j]:
            cnt += 1
          else:
            dir = 'W'
      if j< state.height - 1:
          if state[i][j + 1]:
            cnt += 1
          else:
            dir = 'N'
      if j>0:
          if state[i][j - 1]:
            cnt += 1
          else:
            dir = 'S'

    if cnt == 3:
      self.end[(i, j)]=dir

def isInTunn(state, pos,oldDir):
  i, j = pos
  dir=[]

  if not state[i][j]:
    if i < state.width - 1:
      if not state[i + 1][j]:
        dir.append('E')

    if i > 0:
      if not state[i - 1][j]:
        dir.append('W')

    if j < state.height - 1:
      if not state[i][j + 1]:
        dir.append('N')

    if j > 0:
      if not state[i][j - 1]:
        dir.append('S')

  if len(dir)==2:
    if dir[0]==getOppoDir(dir[1]):
      newDirection=oldDir

    else:
      for item in dir:
        if getOppoDir(oldDir)!=item:
            newDirection=item


    return (True,newDirection)
  else:
    return (False,False)

def checkNeighbourhood(self,walls,pos,dir):
    i, j = pos

    if dir=='N':
      j+=1

    elif dir == 'S':
      j-=1

    elif dir == 'W':
      i-=1

    elif dir == 'E':
      i += 1

    isTunnel,newDir=isInTunn(walls, (i, j), dir)
    if isTunnel:
        self.endNew[(i, j)] = newDir


def getOppoDir(dir):
  if dir=='N':
    return 'S'
  elif dir == 'S':
    return 'N'
  elif dir == 'E':
    return 'W'
  elif dir == 'W':
    return 'E'

  if dir=='North':
    return 'South'
  elif dir == 'South':
    return 'North'
  elif dir == 'East':
    return 'West'
  elif dir == 'West':
    return 'East'
  