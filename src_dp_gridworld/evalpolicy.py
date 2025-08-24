from collections import defaultdict


class EvalPolicy:
    def __init__(self):
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

    def argmax(self, dict):
        max_index = 0
        max_value = max(dict.values())
        for k, v in dict.items():
            if max_value == v:
                max_index = k
        return max_index

    def eval_onestep(self, pi, values, env, gamma=0.9):
        for state in env.getStates():
            # 如果是目标位置，得分为0
            if state == env.goal_state:
                values[state] = 0
                continue

            action_probs = pi[state]
            new_values = 0

            for action, action_prob in action_probs.items():
                next_state = env.to_next_position(state, action)
                r = env.reward(state, action, next_state)

                new_values += action_prob * (r + gamma * values[next_state])
            values[state] = new_values
        return values

    # 随机策略
    def policy_eval(self, pi, values, env, gamma, threshold=0.001):
        while True:
            old_values = values.copy()
            values = self.eval_onestep(pi, values, env, gamma)

            delta = 0
            for state in values.keys():
                t = abs(values[state] - old_values[state])
                if delta < t:
                    delta = t

            if delta < threshold:
                break
        return values

    # 贪婪策略
    def policy_greedy(self, pi, values, env, gamma, threshold=0.001):
        while True:
            values = self.eval_onestep(pi, values, env, gamma)
            new_pi = self.getGreedyPolicy(values, env, gamma)

            if new_pi == pi:
                break
            env.render_env(3, 4, values, self.getSpeActions(pi))
            pi = new_pi
        return values

    # 贪婪策略
    def getGreedyPolicy(self, values, env, gamma):
        pi = {}

        for state in env.getStates():
            action_values = {}

            for action in env.action_space:
                next_state = env.to_next_position(state, action)
                r = env.reward(state, action, next_state)
                value = r + gamma * values[next_state]
                action_values[action] = value
            max_action = self.argmax(action_values)
            action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
            action_probs[max_action] = 1
            pi[state] = action_probs

        return pi

    def getSpeActions(self, ori_actions):
        actions = {}
        for k,v in ori_actions.items():
            direction = self.argmax(v)
            actions[k] = direction
        return actions
