
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
from util import nearestPoint
import copy
import game


def createTeam(firstIndex, secondIndex, isRed,
               first = 'AstarAttacker', second = 'Defender'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class BasicAgent(CaptureAgent):


    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        self.start = gameState.getAgentPosition(self.index)
        self.isRed = gameState.isOnRedTeam(self.index)

        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.lastEatenFoodPosition=None

        if (hasattr(self, 'oldfood') == False):
            self.oldfood = self.getFood(gameState).asList()


        self.enemy_index = [index for index in self.getOpponents(gameState)]
        self.foodClass = DistinctFoodType(gameState, self)
        self.alleyAll = self.foodClass.detectAlley(gameState)
        self.ouralley = self.alleyAll[0]
        self.oppoalley = self.alleyAll[1]

        self.safeFoods = self.foodClass.getSafeFoods(gameState)  # a list of tuple contains safe food location
        # self.dangerFoods = self.foodClass.getDangerFoods(self.safeFoods)
        self.dfoods = self.foodClass.Dfood(gameState)
        self.pre_problem = None

        self.justeatcapsule = False

        self.eatcapsuletime=0

        if (hasattr(self, 'problem') == False):
            self.problem="SearchFood"

        CaptureAgent.registerInitialState(self, gameState)


    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)

    def DetectGhost(self, gameState):
        # Computes distance to ghost we can see
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # detect ghost and its scared time
        enemies = [gameState.getAgentState(i) for i in self.enemy_index]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        if len(ghosts) > 0:
            nearest = float('inf')
            for i in ghosts:
                dist = self.getMazeDistance(myPos, i.getPosition())
                if dist < nearest:
                    nearest = dist
                    scared = i.scaredTimer
            return [dist, scared]
        else:
            return None

    def nullHeuristic(self,state, problem=None):
        return 0

    def AttackerHeuristic(self, state_pos, gameState):
        distant = 0
        # print(state_pos)
        # foods = self.getFood(gameState).asList()
        # if foods:
        #     distant = min([self.getMazeDistance(state_pos, f) for f in self.getFood(gameState).asList()])
        # else:
        # distant = 0

        enemies = [gameState.getAgentState(i) for i in self.enemy_index]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        if ghosts != None and len(ghosts) > 0 :
            dis_to_ghost = [self.getMazeDistance(state_pos, a.getPosition()) for a in ghosts]
            # fixed when ghosts is scared, do not avoid it when eatting dangerfood
            if len(ghosts) == 2:
                if min([i.scaredTimer for i in ghosts]) < 4 :
                    s = sum([pow((5-i),5) for i in dis_to_ghost])
                    return s+distant
            else:
                if ghosts[0].scaredTimer < 4:
                    if min(dis_to_ghost) < 2:
                        s = (5 - min(dis_to_ghost)) ** 5
                        return s+distant

        return distant


    def defenHeuristic(self, state, gameState): #search pacman
        h=10
        opponents = [gameState.getAgentState(i) for i in self.enemy_index]
        pacmans = [o for o in opponents if o.isPacman and o.getPosition() != None]
        # make ghosts as wall, so don't need avoid it at all when go home
        # ghosts = [a for a in opponents if not a.isPacman and a.getPosition() != None]
        if pacmans != None and len(pacmans) > 0:
            pacmanPos = [p.getPosition() for p in pacmans]
            h = min([self.getMazeDistance(state, pos) for pos in pacmanPos])
        return h

    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
        PrioQueue = util.PriorityQueue()
        start_state = problem.getStartState()
        closed = []
        best_g = dict()
        h = heuristic(start_state, gameState)
        g = 0
        f = g + h
        PrioQueue.push((problem.getStartState(), [], g), f)

        while not PrioQueue.isEmpty():
            state, path, cost = PrioQueue.pop()
            g_n = cost

            if problem.isGoalState(state):
                return path

            if (state not in closed) or g_n <best_g[state]:
                closed.append(state)
                best_g[state] = g_n

                for n_node in problem.getSuccessors(state):
                    g_n = n_node[2] + cost
                    f_n = g_n + heuristic(n_node[0], gameState)
                    if n_node[0] not in closed:
                        PrioQueue.push((n_node[0], path + [n_node[1]], g_n), f_n)
        return []

    def opponent_scared_time(self, gameState):
        for i in self.enemy_index:
            t = gameState.getAgentState(i).scaredTimer
            if t > 1:
                return t
        return 0

    def _boundary(self, gameState):
        i = (int(self.width/2)) - 1 if self.red else (int(self.width/2)) + 1

        bounds = [(i, j) for j in range(self.height)]
        validPos = [i ]
        for i in bounds:
            if not gameState.hasWall(i[0], i[1]):
                validPos.append(i)
        return validPos

    def locationOfLastEatenFood(self,gameState):

        if len(self.observationHistory) > 1:
            prevState = self.getPreviousObservation()
            prevFoodList = self.getFoodYouAreDefending(prevState).asList()
            currentFoodList = self.getFoodYouAreDefending(gameState).asList()
            if len(prevFoodList) != len(currentFoodList):
                for food in prevFoodList:
                    if food not in currentFoodList:
                        self.lastEatenFoodPosition = food

    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

class PositionSearchProblem:

    "initialize probelm"
    def __init__(self, gameState, agent, agentIndex = 0,costFn = lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        # self.gameState = gameState

        self.startState = gameState.getAgentState(agentIndex).getPosition()
        # print(self.startState)

        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        # self.isRed = gameState.isOnRedTeam(self.index)
        self.agent = agent

        self.lastEatenFood = agent.lastEatenFoodPosition

        self.food = agent.getFood(gameState).asList()

        if (len(agent.oldfood) < len(self.food)):
            # print('1safe', end='')
            # print(agent.safeFoods)
            # retD = list(set(self.food).difference(set(agent.oldfood)))
            # print('2baofood', end='')
            # print(retD)
            extrasafefood = agent.foodClass.getextraSafeFood(gameState, agent.oldfood)
            # print('3extrasafefood', end='')
            # print(extrasafefood)
            agent.safeFoods = list(set(agent.safeFoods).union(set(extrasafefood)))
            # print('4safe', end='')
            # print(agent.safeFoods)

        agent.oldfood = self.food

        self.capsule = agent.getCapsules(gameState)

        if len(self.capsule) == 1 & agent.justeatcapsule == False & agent.eatcapsuletime==0:
            agent.justeatcapsule = True

        if agent.justeatcapsule == True:
            agent.eatcapsuletime+=1
            # print(agent, end='')
            # print(agent.eatcapsuletime)

        if agent.eatcapsuletime==13:
            agent.justeatcapsule = False


        self.safeFoods = agent.safeFoods
        # self.dangerFoods = agent.dangerFoods
        self.dfoods = agent.dfoods
        self.homeBoundary = agent._boundary(gameState)

        self.opponents = [gameState.getAgentState(i) for i in agent.getOpponents(gameState)]
        self.pacmans = [o for o in self.opponents if o.isPacman and o.getPosition() != None]
        self.ghosts = [o for o in self.opponents if not o.isPacman and o.getPosition() != None]
        self.ghostsPos = [p.getPosition() for p in self.ghosts] if len(self.ghosts) > 0 else None
        self.scaretime=agent.opponent_scared_time(gameState)
        # if len(self.pacmans) > 0:
        self.pacmansPos = [p.getPosition() for p in self.pacmans]
        # else:
        #     self.pacmansPos = []

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def Sel_food_can_eat(self, foodlist):
        # The food can be eat need to have enough distance to ghost
        # myPos = self.startState

        maskFood = set()
        canEatFood = set()
        # determine the positions of the opponents

        poss = [s.configuration.pos for s in self.ghosts
            # only agent which is a ghost and not scared will mask the food
            if s.scaredTimer == 0
        ]

        print("foodlist:",foodlist)
        for food in foodlist:

            if any(self.agent.getMazeDistance(food, p) < 6 for p in poss):
                maskFood.add(food)
            else:
                canEatFood.add(food)
        print("mask=",maskFood)
        if len(canEatFood) > 0:
            return canEatFood
        else:
            return foodlist

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            # for x in self.ghostsPos:
            #     print(x)
            if self.ghostsPos:
                if not self.walls[nextx][nexty] and (((nextx,nexty) not in self.ghostsPos and self.scaretime==0)
                                                     or self.scaretime!=0):
                    nextState = (nextx, nexty)
                    cost = self.costFn(nextState)
                    successors.append((nextState, action, cost))
            else:
                if not self.walls[nextx][nexty]:
                    nextState = (nextx, nexty)
                    cost = self.costFn(nextState)
                    successors.append((nextState, action, cost))

        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors


class Defender(BasicAgent):

    def chooseAction(self, gameState):

        self.locationOfLastEatenFood(gameState)
        opponents = [gameState.getAgentState(index) for index in self.enemy_index]
        pacmans = [o for o in opponents if o.isPacman]
        nearpacmans = [o for o in pacmans if o.getPosition()!= None]

        #Down food update
        Dfoods = []
        for i in self.getFood(gameState).asList():
            if i in self.dfoods:
                Dfoods.append(i)

        self.dfoods = copy.deepcopy(Dfoods)

        # if len(nearpacmans) != len(pacmans):
        #     print("len nearpacman")
        if gameState.getAgentPosition(self.index) in self.ouralley:
            print('in')
        # print(gameState.getAgentPosition(self.index))

        myState = gameState.getAgentState(self.index)

        if len(pacmans) == 0 or gameState.getAgentPosition(self.index) == self.lastEatenFoodPosition or len(nearpacmans) > 0:
            self.lastEatenFoodPosition = None

        if  len(pacmans) > 0 and myState.scaredTimer == 0:
            #myState.scaredTimer == 0 and (self.lastEatenFoodPosition!=None or len(nearpacmans)>0):


            if self.problem == "SearchPacmans":
                print('true')
                if myState.getPosition() in self.ouralley:
                    print('stop')
                    return 'Stop'

            if len(nearpacmans)==0 and self.lastEatenFoodPosition!=None:
                problem = SearchLastEatenFood(gameState, self, self.index)
                self.problem == "SearchLastEatenFood"
                p = self.aStarSearch(problem, gameState, self.defenHeuristic)
                if p != None:
                    if len(p) != 0:
                        return p[0]

            elif len(nearpacmans)>0:
                # find enemy's pacman
                # when no lasteatfood and cannot detect enemy's position will cause problem
                problem = SearchPacmans(gameState, self, self.index)
                self.problem = "SearchPacmans"
                p = self.aStarSearch(problem, gameState, self.defenHeuristic)
                print(problem)
                if p!= None:
                    if len(p) != 0:
                        return p[0]

            else:#if cannot detect the enemy position,go home first
                problem = goHome(gameState,self,self.index)
                p = self.aStarSearch(problem, gameState, self.AttackerHeuristic)
                print(problem)
                self.problem = "goHome"
                if p != None:
                    if len(p) != 0:
                        return p[0]

        else:
            left_point = len(self.getFood(gameState).asList()) + len(self.getCapsules(gameState))
            if myState.numCarrying < 10 and left_point > 2:

                if len(self.dfoods) > 0:
                    problem = SearchDownFood(gameState, self, self.index)
                    self.problem="SearchDownFood"
                else:
                    problem = SearchCapFood(gameState, self, self.index)
                    self.problem = "SearchCapFood"

                if self.DetectGhost(gameState) != None and self.DetectGhost(gameState)[0] < 5 and self.DetectGhost(gameState)[1]<4:
                    problem = Escape(gameState, self, self.index)
                    self.problem = "Escape"
                    # print(problem)

                p = self.aStarSearch(problem, gameState, self.AttackerHeuristic)
                if p != None and len(p) != 0:
                    return p[0]
            else:
                problem = goHome(gameState, self, self.index)
                self.problem = "goHome"
                p = self.aStarSearch(problem, gameState, self.AttackerHeuristic)
                print(problem)
                if p != None:
                    if len(p) != 0:
                        return p[0]

        return 'Stop'

class AstarAttacker(BasicAgent):

    def chooseAction(self, gameState):

        currState = gameState.getAgentState(self.index)
        currPos = currState.getPosition()
        SafeFoods = []
        # DangerousFoods = []

        opponents = [gameState.getAgentState(index) for index in self.enemy_index]
        nearpacmans = [o for o in opponents if o.getPosition() != None and o.isPacman]
        if len(nearpacmans) > 0:
            TeammatePos = gameState.getAgentPosition(self.index+2)
            disToEnemyPacman = min([self.getMazeDistance(currPos,pos.getPosition()) for pos in nearpacmans])
            teammateToPac = min([self.getMazeDistance(TeammatePos,pos.getPosition()) for pos in nearpacmans])


        # update safe/dangerous food list
        for i in self.getFood(gameState).asList():
            if i in self.safeFoods:
                SafeFoods.append(i)

        # for i in self.getFood(gameState).asList():
        #     if i in self.dangerFoods:
        #         DangerousFoods.append(i)

        self.safeFoods = copy.deepcopy(SafeFoods)
        # self.dangerFoods = copy.deepcopy(DangerousFoods)

        # print(self.getCurrentObservation())

        if currState.isPacman and self.DetectGhost(gameState)!=None and self.DetectGhost(gameState)[0] < 3 and \
                self.DetectGhost(gameState)[1] < 5:
            problem = Escape(gameState, self, self.index)

        #if a ghost nearby, go and catch it
        elif not currState.isPacman and len(nearpacmans) > 0 and disToEnemyPacman < 4 and teammateToPac > 6:
            problem = SearchPacmans(gameState, self, self.index)
            p = self.aStarSearch(problem, gameState, self.defenHeuristic)
            if p != None:
                if len(p) != 0:
                    return p[0]
        else:

            if len(self.safeFoods) < 2 and len(self.getCapsules(gameState)) != 0 and \
                    self.opponent_scared_time(gameState) < 10 and self.justeatcapsule==False:
                problem = SearchCapsule(gameState, self, self.index)

            # elif len(self.dangerFoods) > 1 and self.opponent_scared_time(gameState) > 10:
            #     problem = SearchDangerFood(gameState,self,self.index)
            elif len(self.safeFoods) >= 3 and self.opponent_scared_time(gameState) < 7:
                problem = SearchSafeFood(gameState, self, self.index)

            else :
                problem = SearchFood(gameState, self, self.index)
                if currState.isPacman and self.DetectGhost(gameState) != None and self.DetectGhost(gameState)[0] < 5 and \
                        self.DetectGhost(gameState)[1] < 6:
                    problem = Escape(gameState, self, self.index)


            # problem = SearchFood(gameState, self, self.index)

            left_point = len(self.getFood(gameState).asList()) + len(self.getCapsules(gameState))
            if self.opponent_scared_time(gameState) < 5:
                if currState.numCarrying > 15  :
                    problem = goHome(gameState, self, self.index)

            if left_point < 3:
                problem = goHome(gameState, self, self.index)

            if self.DetectGhost(gameState)!=None and self.DetectGhost(gameState)[0] < 4 \
                    and self.DetectGhost(gameState)[1] > 5 and self.DetectGhost(gameState)[1] < 10:
                problem = EatCrazyGhost(gameState, self, self.index)

        # if len(self.safeFoods) >= 2:
        #     problem = SearchSafeFood(gameState, self, self.index)

        actions = self.aStarSearch(problem, gameState, self.AttackerHeuristic)
        print("Attacker probelm:", problem)
        if actions != None and len(actions) != 0:

            # print("actions:",actions[0])
            return actions[0]
        else:
            return 'Stop'


class SearchCapsule(PositionSearchProblem):
    """
    search capsule
    """
    def isGoalState(self, state):
        # the goal state is the location of capsule
        return state in self.capsule

class SearchFood(PositionSearchProblem):
    """
     The goal state is to find all the food
    """
    def isGoalState(self, state):
        return state in self.Sel_food_can_eat(self.food)

class SearchCapFood(PositionSearchProblem):
    """
     The goal state is to find all the food
    """
    def isGoalState(self, state):
        return state in self.food or state in self.capsule

class SearchDownFood(PositionSearchProblem):

    def isGoalState(self, state):

        return state in self.dfoods

class SearchSafeFood(PositionSearchProblem):

    def isGoalState(self, state):
        return state in self.safeFoods

# class SearchDangerFood(PositionSearchProblem):
#
#     def isGoalState(self, state):
#         return state in self.dangerFoods
# #
class Escape(PositionSearchProblem):
    """
    Used to escape
    """
    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsulep
        return state in self.homeBoundary or state in self.capsule

class goHome(PositionSearchProblem):
    """
    Used to escape
    """
    def isGoalState(self, state):
        # the goal state is the boudary of home or the positon of capsule
        return state in self.homeBoundary

class SearchLastEatenFood(PositionSearchProblem):

    def isGoalState(self, state):
        return state == self.lastEatenFood

class SearchPacmans(PositionSearchProblem):
    "search opponents' pacman"
    def isGoalState(self, state):
        return state in self.pacmansPos

class EatCrazyGhost(PositionSearchProblem):

    def isGoalState(self, state):
        return state in self.ghostsPos



class DistinctFoodType:
    """
    A Class Below is used for scanning the map to find
    Safe food and dangerousfood

    Note: Safe food is the food that from this food there are at least two ways lead
    home

    food density
    """

    def __init__(self, gameState, agent):
        self.food = agent.getFood(gameState).asList()
        self.walls = gameState.getWalls()
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        self.updownBound = self.height/2
        self.homeBoundary = agent._boundary(gameState)
        self.isRed = gameState.isOnRedTeam(agent.index)

    def detectAlley(self,gameState):
        oppo_alley = []

        if self.isRed:
            x_start = int(self.width/2 + 1)
            x_end = int(self.width-3)
        else:
            x_start = 3
            x_end = int(self.width/2 - 1)
        for x in range(x_start,x_end):
            for y in range(1,self.height):
                if not self.walls[x][y]:
                    #vertical direction
                    self.walls[x][y] = True #for compute whether (x,y) is an alley entry,later this will be reverse
                    if self.walls[x+1][y] and self.walls[x-1][y] and (not self.walls[x][y+1]) and (not self.walls[x][y-1]):
                        a_up = (x,y+1)
                        a_down = (x,y-1)
                        visited = []
                        visited.append(a_up)
                        closed = copy.deepcopy(visited)
                        if not self.BFS(a_up, closed, a_down):
                            oppo_alley.append((x, y))

                    if self.walls[x][y+1] and self.walls[x][y-1] and (not self.walls[x+1][y]) and (not self.walls[x-1][y]):
                        a_left = (x-1,y)
                        a_right = (x+1,y)
                        visited = []
                        visited.append(a_left)
                        closed = copy.deepcopy(visited)
                        if not self.BFS(a_left, closed, a_right):
                            oppo_alley.append((x, y))
                    self.walls[x][y] = False

        # print(oppo_alley)
        our_alley = [(self.width - x-1, self.height - y-1) for (x, y) in oppo_alley]
        print("our:", our_alley)
        alley = [our_alley,oppo_alley]
        return alley

    def Dfood(self,gameState):
        dfoods = []
        for food in self.food:
            if food[1] < self.updownBound:
                dfoods.append(food)
        return dfoods

    def getSafeFoods(self, gameState):
        alley = self.detectAlley(gameState)[1]
        for i in alley:
            self.walls[i[0]][i[1]] = True
        foods = []
        safe_foods = []
        for food in self.food:
            food_fringes = []
            food_valid_fringes = []
            count = 0
            x = food[0]
            y = food[1]
            food_fringes.append((x + 1, y))  # right
            food_fringes.append((x - 1, y))  # left
            food_fringes.append((x, y + 1))  # up
            food_fringes.append((x, y - 1))  # down
            for f_fr in food_fringes:
                if not gameState.hasWall(f_fr[0], f_fr[1]):
                    count = count + 1
                    food_valid_fringes.append(f_fr)
            if count > 1:
                foods.append((food, food_valid_fringes))

        for food in foods:
            if self.getNumOfValidActions(food) > 1:
                safe_foods.append(food[0])

        for i in alley:
            self.walls[i[0]][i[1]] = False
        return safe_foods

    def getextraSafeFood(self, gameState, oldfood):
        # print("getextrasafefood")
        foods = []
        safe_foods = []
        for food in oldfood:
            food_fringes = []
            food_valid_fringes = []
            count = 0
            x = food[0]
            y = food[1]
            food_fringes.append((x + 1, y))  # right
            food_fringes.append((x - 1, y))  # left
            food_fringes.append((x, y + 1))  # up
            food_fringes.append((x, y - 1))  # down
            for food_fringe in food_fringes:
                if not gameState.hasWall(food_fringe[0], food_fringe[1]):
                    count = count + 1
                    food_valid_fringes.append(food_fringe)
            if count > 1:
                foods.append((food, food_valid_fringes))

        for food in foods:
            if self.getNumOfValidActions(food) > 1:
                safe_foods.append(food[0])
        return safe_foods

    # def getDangerFoods(self, safe_foods):
    #     danger_foods = []
    #     for food in self.food:
    #         if food not in safe_foods:
    #             danger_foods.append(food)
    #     return danger_foods

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def isGoalState(self, state,goal=None):
        if goal == None:
            return state in self.homeBoundary
        else:
            return state == goal

    def getNumOfValidActions(self, foods):
        food = foods[0]
        food_fringes = foods[1]
        visited = []
        visited.append(food)
        count = 0
        for food_fringe in food_fringes:
            closed = copy.deepcopy(visited)
            if self.BFS(food_fringe, closed):
                count = count + 1
        return count

    def BFS(self, food_fringe, closed,goal = None):

        fringe = util.Queue()
        fringe.push((food_fringe, []))
        while not fringe.isEmpty():
            state, actions = fringe.pop()
            closed.append(state)
            if self.isGoalState(state,goal):
                return True
            for successor, direction in self.getSuccessors(state):
                if successor not in closed:
                    closed.append(successor)
                    fringe.push((successor, actions + [direction]))
