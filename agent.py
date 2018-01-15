import random
import itertools
import ast
import copy

class QLearningAgent:
    def __init__(self, saved_strategy=None):
        self.bird = None
        self.pipes = None
        self.pp = None
        self.current_state = None
        # time
        #self.t = 0
        self.t = 1000
        if saved_strategy is not None:
            self.t = 100
        self.q_data = {}
        if saved_strategy is None:
            s_a_pairs = self.generateStates()
            for elem in s_a_pairs:
                key = str(elem)
                self.q_data[key] = 0.0
        else:
            with open(saved_strategy, 'r') as f:
                data_string = f.read()
                data_dict = ast.literal_eval(data_string)
                self.q_data = data_dict
        self.gamma = 0.7
        if saved_strategy is None:
            self.alpha = lambda n: 1./(1+n)
        else:
            self.alpha = lambda n: 1./(10000+n)

        self.n_data = copy.deepcopy(self.q_data)

        if saved_strategy is None:
             for k in self.n_data.iterkeys():
                self.n_data[k] = 1.0
        else:
            for k in self.n_data.iterkeys():
                self.n_data[k] = 10.0

        self.path = []
        return

    def generateStates(self):
        nodes = list(itertools.product( (0,1,2,3,4), repeat=2))
        STATE_ACTION_PAIRS = state_action_pairs = list(itertools.product(nodes, ('S', 'J')))
        return state_action_pairs


    def observeState(self, bird, pipes, pp):
        bird_height = bird.y
        pipe_bottom = 500 - pp.bottom_height_px
        pipe_dist = pp.x
        height_category = 0
        dist_to_pipe_bottom = pipe_bottom - bird.y
        if dist_to_pipe_bottom < 8:
            height_category = 0
        elif dist_to_pipe_bottom < 20:
            height_category = 1
        elif dist_to_pipe_bottom < 125:
            height_category = 2
        elif dist_to_pipe_bottom < 250:
            height_category = 3
        else:
            height_category = 4
        dist_category = 0
        dist_to_pipe_horz = pp.x - bird.x
        if dist_to_pipe_horz < 8:
            dist_category = 0
        elif dist_to_pipe_horz < 20:
            dist_category = 1
        elif dist_to_pipe_horz < 125:
            dist_category = 2
        elif dist_to_pipe_horz < 250:
            dist_category = 3
        else:
            dist_category = 4

        pipe_collision = any(p.collides_with(bird) for p in pipes)
        collision = pipe_collision
        state = (height_category, dist_category, collision)
        return state

    def performAction(self, state):
        actions = self.getActions(state)
        if not self.explore():
            best_action = self.findMaxReward(actions)
        else:
            best_action = self.exploreDecision()
        return best_action, state

    def updateTime(self):
        self.t += 1
        self.alpha(self.t)

    def collectReward(self, state, collision):
        reward = 1
        if collision:
            reward = -1000
        return reward

    def updateQArray(self, prev_state, action, reward):
        key = ((prev_state[0], prev_state[1]), action)
        q_sample = reward + self.gamma * self.q_data[str(key)]
        q_old = self.q_data[str(key)]
        q_new = q_old + 1/self.n_data[str(key)] * q_sample
        self.q_data[str(key)] = q_new
        self.n_data[str(key)] += 1.0
        return
    def updateState(self):
        self.current_state = self.observeState(self.bird, self.pipes, self.pp)
        return

    def saveStrategy(self, fname):
        f = open(fname, "w")
        f.write(str(self.q_data))
        f.close()
    def newEpisode(self, bird, pipes):
        self.bird = bird
        self.pipes = pipes

    def newIteration(self):
        self.pp = self.pipes[0]

    def stepAndMakeChoice(self):
        self.current_state = self.observeState(self.bird, self.pipes, self.pp)
        action,state = self.performAction(self.current_state)
        return action,state

    def learnFromChoice(self, action, prev_state, collision):
        self.updateTime()
        self.updateState()
        reward = self.collectReward(prev_state, collision)
        if reward < 1:
            print str(reward)
        self.updateQArray(prev_state, action, reward)


    def findMaxReward(self, actions):
        decision = 'S'
        reward_jump = self.q_data[actions[0]]
        reward_stay = self.q_data[actions[1]]
        print "REWARD JUMP="+str(reward_jump)
        print "REWARD STAY="+str(reward_stay)
        if reward_jump > reward_stay:
            decision = 'J'

        return decision

    def getActions(self, state):
        t = (state[0], state[1])
        jump = str((t, 'J'))
        stay = str((t, 'S'))
        return [jump, stay]

    def explore(self):
        r = random.random()
        if self.alpha(self.t) > r:
            print "EXPLORE!"
            return True
        else:
            return False

    def exploreDecision(self):
        action = 'S'
        index = random.randint(0,1)
        if index == 1:
            action = 'J'
        return action

    def trackPath(self, prev_state, action):
        key = str(((prev_state[0], prev_state[1]), action))
        if len(self.path) < 10:
            self.path.append(key)
        else:
            self.path.pop(0)
            self.path.append(key)

    def updatePathValues(self, reward):
        i = 0
        for key in self.path:
            i += 1
            weight = lambda i: 1./(1+i)
            q_sample = reward + self.gamma * self.q_data[str(key)]
            q_old = self.q_data[str(key)]
            q_new = q_old + self.alpha(self.t) * q_sample
            self.q_data[str(key)] = q_new * weight(self.t)


    def transition_probs(self, new_state, action, prev_state):
        possible_fst = []
        if prev_state[0] == 0:
            y0 = prev_state[0]
            y1 = prev_state[0] + 1
            possible_fst.append(y0)
            possible_fst.append(y1)
        elif prev_state[0] == 2:
            y1 = prev_state[0] - 1
            y2 = prev_state[0]
            possible_fst.append(y1)
            possible_fst.append(y2)
        else:
            y0 = prev_state[0]
            y1 = prev_state[0] + 1
            y2 = prev_state[0] - 1
            possible_fst.append(y0)
            possible_fst.append(y1)
            possible_fst.append(y2)

        possible_snd = []
        if prev_state[1] == 0:
            x0 = prev_state[1]
            x1 = prev_state[1] + 1
            possible_snd.append(x0)
            possible_snd.append(x1)
        elif prev_state[1] == 2:
            x1 = prev_state[1] - 1
            x2 = prev_state[1]
            possible_fst.append(x1)
            possible_fst.append(x2)
        else:
            x0 = prev_state[0]
            x1 = prev_state[0] + 1
            x2 = prev_state[0] - 1
            possible_snd.append(x0)
            possible_snd.append(x1)
            possible_snd.append(x2)
        possible_states = list(itertools.product(possible_fst, possible_snd))
        state_action_pairs  = list(itertools.product(possible_states, ['J','S']))

        qk_sp_ap = {}
        for elem in state_action_pairs:
            qk_sp_ap[str(elem)] = self.q_data[str(elem)]

        max_qk_sp_ap = max(qk_sp_ap.iterkeys(), key=(lambda key: qk_sp_ap[key]))

        discounted_max = self.gamma * max_qk_sp_ap

        key = ((prev_state[0], prev_state[1]), action)
        key = str(key)

        reward = self.q_data[key]