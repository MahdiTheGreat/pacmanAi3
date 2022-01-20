# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):

    def __init__(self, mdp, discount = 0.9, iterations = 100):

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        print()

    def runValueIteration(self):

        for times in range(self.iterations):

            updatedValues = util.Counter()
            states = self.mdp.getStates()
            print()

            for state in states:
                qValues=[self.getQValue(state,action)for action in self.mdp.getPossibleActions(state)]
                updatedValues[state] = max(qValues) if len(qValues) else 0

            self.values=updatedValues


    def getValue(self, state):

        return self.values[state]


    def computeQValueFromValues(self, state, action):

        qValue=0
        for nextState, t in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue +=  t * (self.discount *self.values[nextState]+self.mdp.getReward(state, action, nextState))
        return qValue


    def computeActionFromValues(self, state):

        actions=self.mdp.getPossibleActions(state)
        if len(actions)==0:return None
        qValues=[self.getQValue(state,action)for action in actions]
        return actions[qValues.index(max(qValues))]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):

        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):

    def __init__(self, mdp, discount = 0.9, iterations = 1000):

        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        states = self.mdp.getStates()
        i=0
        for times in range(self.iterations):
            state=states[i%len(states)]
            qValues=[self.computeQValueFromValues(state,action)for action in self.mdp.getPossibleActions(state)]
            self.values[state] = max(qValues) if len(qValues) else 0
            i+=1



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):

    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):

        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getPredecessors(self, currentState):

        predecessors = set()

        if self.mdp.isTerminal(currentState):return predecessors

        states = self.mdp.getStates()
        actions = ('north', 'west', 'south', 'east')
        for state in states:
            if self.mdp.isTerminal(state):states.remove(state)

        for state in states:
            for action in actions:
                if (action in self.mdp.getPossibleActions(state))and currentState in self.mdp.getTransitionStatesAndProbs(state,action)[0] :
                    predecessors.add(state)

        return list(predecessors)


    def maxQvalue(self,state):
        qValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
        return max(qValues) if len(qValues) else 0

    def runValueIteration(self):

        priorityQueue = util.PriorityQueue()
        states = self.mdp.getStates()

        for state in states:
          if not self.mdp.isTerminal(state):
            maxQvalue= self.maxQvalue(state)
            diff = abs(self.values[state] - maxQvalue)
            priorityQueue.push(state, -diff)


        for i in range(self.iterations):

          if priorityQueue.isEmpty():break

          state = priorityQueue.pop()
          self.values[state] = self.maxQvalue(state)

          for p in self.getPredecessors(state):
            maxQvalue = self.maxQvalue(p)
            diff = abs(self.values[p] - maxQvalue)

            if diff > self.theta:
              priorityQueue.update(p, -diff)

