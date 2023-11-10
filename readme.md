# Reinforcement Learning
Teaching a software agent how to behave in an environment to maximize a reward

# Four Parts
1. **Game**: The *environment* where your AI snake operates.

2. **Agent**: Your AI *snake*, which learns to maximize rewards.

3. **Training**: The process of teaching your snake to make the *best moves*.

4. **Model**: In this case, a PyTorch neural network that learns the *Q-values* for actions.

# Reward
1. Eat Food : +10
2. Game Over : -10
3. else : 0

# Action
We use only three actions
1. Straight [1, 0, 0]
2. Right [0, 1, 0]
3. Left [0, 0, 1]

These are relative directions and depend on the current state. No 180 degree turn else the entity will die. Also easy for model

# States (11 values)
The model needs to know about environment
[
    danger straight, danger right, danger left,
    direction left, direction right, direction up, direction down, 
    food left, food right, food up, food down
]
All are boolean states. **For Example** a snake is going *right* and food is on *left* and there is *no* danger.

states = 
[
    0,0,0,
    0,1,0,0,
    1,0,0,0
]

# Model

## (Deep) Q Learning
State (11 values) --> Deep Q Learning --> Actions
Uses deep neural network to teach the action

### Q = Quality of Action

0. Init Q Value (=INIT MODEL) random parameters
1. Choose action (model.predict(state)) or (random move) in the beginning
2. Perform Action
3. Measure reward
4. Update Q value (+train model)

4 to 1 training loop

# Bellman Equation (Foundation of Reinforcement Learning)


NewQ(s,a) = Q(s,a) + α[R(s,a) + γ * maxQ(s′,a′) - Q(s,a)]

**s** -> state
**a** -> action
**α** -> learning rate
**R** -> reward
**γ** -> discount factor
**maxQ(s', a')** -> maximum Q-value for next state and all possible actions, maximum expected future reward

## Q Update Rule Simplified
Q(s,a) = R + γ maxQ(s′,a′)

## Loss Function 
loss = (NewQ - Q**2) Mean Squared Error