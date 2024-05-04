# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance

def betterEvaluationFunction(currentGameState):
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghost.scaredTimer for ghost in newGhostStates]

    # Tính điểm từ trạng thái
    score = currentGameState.getScore()

    # Xem xét khoảng cách đến tất cả các chấm thức ăn còn lại
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        score += 1.0 / min(foodDistances)  # Khuyến khích Pacman di chuyển về phía thức ăn

    # Xem xét khoảng cách đến ma và thời gian ma sợ hãi
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0]
    if ghostDistances:
        minGhostDistance = min(ghostDistances)
        if minGhostDistance > 0:
            score -= 2.0 / minGhostDistance  # Ngăn Pacman lại gần ma đang hoạt động

    # Xem xét khoảng cách đến ma đang sợ để ăn chúng
    scaredGhostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer > 0]
    if scaredGhostDistances:
        score += 2.0 / min(scaredGhostDistances)  # Khuyến khích Pacman đuổi theo ma đang sợ

    # Xem xét số lượng capsule còn lại
    remainingCapsules = len(currentGameState.getCapsules())
    if remainingCapsules > 0:
        score -= 20.0 * remainingCapsules  # Khuyến khích Pacman ăn capsule

    return score



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Maximize for Pacman
                return max_value(agentIndex, depth, state)
            else:  # Minimize for ghosts
                return min_value(agentIndex, depth, state)

        def max_value(agentIndex, depth, state):
            maxEval = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                evaluation, _ = minimax((depth + 1) % state.getNumAgents(), depth + 1, successor)
                if evaluation > maxEval:
                    maxEval, bestAction = evaluation, action
            return maxEval, bestAction

        def min_value(agentIndex, depth, state):
            minEval = float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                evaluation, _ = minimax((depth + 1) % state.getNumAgents(), depth + 1, successor)
                if evaluation < minEval:
                    minEval, bestAction = evaluation, action
            return minEval, bestAction

        _, action = minimax(0, 0, gameState)
        if action == None:  # Nếu không có hành động hợp lệ
            return Directions.STOP  # Chọn STOP làm hành động mặc định
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(state, agentIndex, depth, alpha, beta):
            # If it's the last agent, wrap around to Pacman and increase depth
            if agentIndex == state.getNumAgents():
                agentIndex = 0
                depth += 1

            # If terminal state or maximum depth reached, return evaluated score
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # Initialize variables
            result = None
            optimalAction = None
            
            if agentIndex == 0:  # Maximizing for Pacman
                result = float("-inf")
                actions = state.getLegalActions(agentIndex)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, agentIndex + 1, depth, alpha, beta)
                    if value > result:
                        result = value
                        optimalAction = action
                    alpha = max(alpha, result)
                    if beta <= alpha:
                        break
                return result, optimalAction
            else:  # Minimizing for ghosts
                result = float("inf")
                actions = state.getLegalActions(agentIndex)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = alphaBeta(successor, agentIndex + 1, depth, alpha, beta)
                    if value < result:
                        result = value
                        optimalAction = action
                    beta = min(beta, result)
                    if beta <= alpha:
                        break
                return result, optimalAction

        # Initial call to alphaBeta function from the root node
        _, action = alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
        if action == None:  # Nếu không có hành động hợp lệ
            return Directions.STOP  # Chọn STOP làm hành động mặc định
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, agentIndex, depth):
            # If it's the last agent, wrap around to Pacman and increase depth
            if agentIndex == state.getNumAgents():
                agentIndex = 0
                depth += 1

            # If terminal state or maximum depth reached, return evaluated score
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            actions = state.getLegalActions(agentIndex)
            
            if len(actions) == 0:
                return self.evaluationFunction(state), None

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                max_value = float("-inf")
                best_action = None
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, agentIndex + 1, depth)
                    if value > max_value:
                        max_value = value
                        best_action = action
                return max_value, best_action
            else:  # Ghosts' turn (expectimax averaging)
                total_value = 0
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(successor, agentIndex + 1, depth)
                    total_value += value
                average_value = total_value / len(actions)
                return average_value, None

        # Initial call to expectimax from the root node
        _, action = expectimax(gameState, 0, 0)
        if action == None:  # Nếu không có hành động hợp lệ
            return Directions.STOP  # Chọn STOP làm hành động mặc định
        return action


# Abbreviation
better = betterEvaluationFunction
