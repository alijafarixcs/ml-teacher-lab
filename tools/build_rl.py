"""Build all reinforcement-learning notebooks with compact custom environments."""

from __future__ import annotations

from course_factory import LessonSpec, write_lesson


RL_SETUP = r"""
def plot_rl_metrics(rewards, exploration_rates, episode_lengths, title):
    rewards = np.asarray(rewards, dtype=float)
    window = min(20, len(rewards))
    moving = np.convolve(rewards, np.ones(window)/window, mode="valid")
    episodes = np.arange(1, len(rewards)+1)
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(episodes, rewards, alpha=.35, color=COURSE_COLORS["blue"], label="Episode reward")
    axes[0].plot(episodes[window-1:], moving, color=COURSE_COLORS["orange"], label=f"{window}-episode average")
    axes[0].set(title=title, ylabel="Reward"); axes[0].legend()
    axes[1].plot(episodes, exploration_rates, color=COURSE_COLORS["olive"])
    axes[1].set(ylabel="Exploration rate")
    axes[2].plot(episodes, episode_lengths, color=COURSE_COLORS["pink"])
    axes[2].set(xlabel="Episode", ylabel="Episode length")
    figure.tight_layout()
    plt.show()
"""


LINE_WORLD = r"""
class LineWorld:
    def __init__(self, length=6, max_steps=20):
        self.length=length
        self.max_steps=max_steps
    def reset(self):
        self.state=0
        self.steps=0
        return self.state
    def step(self, action):
        self.steps+=1
        self.state=int(np.clip(self.state + (-1 if action==0 else 1),0,self.length-1))
        terminated=self.state==self.length-1
        truncated=self.steps>=self.max_steps
        reward=1.0 if terminated else -0.02
        return self.state,reward,terminated,truncated
"""


def base_spec(**overrides) -> LessonSpec:
    spec = {
        "title": "Reinforcement Learning",
        "objectives": ["Define the core reinforcement-learning objects", "Implement the update from scratch", "Visualize reward, exploration, and episode length"],
        "prerequisites": "Python loops, NumPy arrays, probability, expected value, and basic supervised-learning vocabulary.",
        "problem": "Learn a policy for sequential decisions from rewards produced through interaction.",
        "use_cases": ["Adaptive control", "Resource allocation", "Sequential recommendation experiments"],
        "intuition": "An agent tries actions, observes consequences, and changes its behavior so long-term reward improves.",
        "equation": r"$$G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}.$$",
        "symbols": [r"$G_t$ is return from time t.", r"$R_{t+k+1}$ is a future reward.", r"$\gamma\in[0,1]$ is discount factor.", r"$T$ is episode end.", r"$k$ counts how far a reward is in the future."],
        "why_equation": "A discounted return turns a sequence of delayed rewards into one target for comparing actions and policies.",
        "numerical_example": "With rewards [1,2,3] and gamma=.5, return is 1+.5(2)+.25(3)=2.75.",
        "python_connection": "sum((gamma**k)*reward for k,reward in enumerate(rewards)) computes the return.",
        "steps": ["Reset the environment", "Observe state", "Choose an action from the policy", "Receive next state and reward", "Update values or policy", "Repeat until termination and across episodes"],
        "scratch_description": "A tiny custom environment keeps states, actions, rewards, and transitions visible.",
        "library_description": "NumPy handles tabular methods; PyTorch is introduced only for value or policy function approximation.",
        "dataset_description": "Reinforcement learning generates experience through environment interaction rather than loading labeled rows.",
        "preprocessing_description": "Discrete states are indexed directly; neural agents convert states to normalized float tensors.",
        "setup_code": RL_SETUP,
        "scratch_code": "def discounted_return(rewards, gamma):\n    return float(sum((gamma**k)*reward for k,reward in enumerate(rewards)))\n\nprint('Example return:', discounted_return([1,2,3],.5))",
        "experiment_code": "rewards=[0,1,0,1,1]*20\nexploration=np.linspace(1,.1,len(rewards))\nlengths=[1]*len(rewards)\nplot_rl_metrics(rewards,exploration,lengths,'Illustrative agent interaction')",
        "interpretation": "The moving average reveals learning more clearly than noisy individual rewards; exploration and length explain behavioral changes.",
        "hyperparameter_text": "Discount factor controls how strongly distant rewards influence current decisions.",
        "hyperparameter_code": "gammas=np.linspace(0,1,11)\nreturns=[discounted_return([0,0,1],value) for value in gammas]\nfigure,axis=plt.subplots(); axis.plot(gammas,returns,marker='o',color=COURSE_COLORS['blue'],label='Return')\naxis.set(title='Discounting changes the value of delayed reward',xlabel='Discount factor gamma',ylabel='Return'); axis.legend(); plt.show()",
        "mistakes": ["Treating reward as a supervised label", "Evaluating while exploration is still active", "Comparing agents with different random seeds or episode limits", "Ignoring truncation versus true termination"],
        "advantages": ["Directly optimizes sequential behavior", "Can learn from interaction without action labels"],
        "disadvantages": ["Experience can be expensive or unsafe", "Results are noisy and sensitive to reward design"],
        "use_when": "actions influence future states and the objective is naturally expressed through cumulative reward.",
        "avoid_when": "a static labeled prediction problem is sufficient or unsafe exploration cannot be controlled.",
        "exercises": ["Why is RL not supervised learning?", "Calculate return for rewards [2,0,4] with gamma=.5.", "Change one hyperparameter and compare learning across at least three seeds."],
        "solutions": ["The agent is not given correct actions; it receives consequences and must handle delayed credit.", "2+.5(0)+.25(4)=3.", "Keep all other settings fixed, plot each seed plus aggregate mean, and avoid selecting only the best run."],
        "takeaways": ["The policy maps states to action choices", "Return assigns credit across time", "Exploration gathers information while exploitation uses current knowledge", "Learning curves require repeated, controlled evaluation"],
        "further_reading": ["[Gymnasium documentation](https://gymnasium.farama.org/)", "[Sutton and Barto: Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)"],
    }
    spec.update(overrides)
    return spec


def overview_spec() -> LessonSpec:
    scratch = r"""
def discounted_return(rewards,gamma):
    running=0.0
    for reward in reversed(rewards):
        running=reward+gamma*running
    return running

trajectory=[-0.02,-0.02,1.0]
print("Trajectory return:",round(discounted_return(trajectory,.9),3))
"""
    experiment = LINE_WORLD + r"""
environment=LineWorld()
Q=np.zeros((environment.length,2))
rewards=[]; exploration_rates=[]; lengths=[]
snapshots={0:Q.copy()}
for episode in range(160):
    state=environment.reset(); total=0.0; epsilon=max(.05,.9*(.97**episode))
    for step in range(environment.max_steps):
        action=int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(Q[state]))
        next_state,reward,terminated,truncated=environment.step(action)
        Q[state,action]+=.3*(reward+.95*np.max(Q[next_state])-Q[state,action])
        state=next_state; total+=reward
        if terminated or truncated: break
    rewards.append(total); exploration_rates.append(epsilon); lengths.append(step+1)
    if episode in [39,159]: snapshots[episode+1]=Q.copy()
print("Q-table before:\n",snapshots[0])
print("Q-table after 40 episodes:\n",snapshots[40].round(2))
print("Q-table after training:\n",snapshots[160].round(2))
plot_rl_metrics(rewards,exploration_rates,lengths,"Q-learning overview in LineWorld")
"""
    hyper = r"""
gammas=[0,.5,.9,.99]
delayed_values=[discounted_return([0,0,0,1],value) for value in gammas]
figure,axis=plt.subplots(); axis.bar([str(value) for value in gammas],delayed_values,color=COURSE_COLORS["blue"])
axis.set(title="Discount factor changes delayed-goal value",xlabel="Gamma",ylabel="Return at episode start"); plt.show()
"""
    return base_spec(
        title="Reinforcement Learning Overview",
        objectives=["Contrast supervised learning and reinforcement learning", "Define agent, environment, state, action, reward, policy, episode, and return", "Inspect a Q-table before, during, and after learning"],
        problem="Choose actions whose consequences may be delayed and whose outcomes change future choices.",
        intuition="Unlike a labeled worksheet, the environment does not reveal the correct action; it returns consequences and the agent must explore.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="A larger discount factor values the delayed terminal reward more strongly.",
        hyperparameter_code=hyper,
        interpretation="As epsilon falls, the moving reward rises and episode length falls because rightward actions acquire larger Q-values.",
    )


def bandit_spec() -> LessonSpec:
    scratch = r"""
def incremental_mean(old_mean,reward,count):
    return old_mean+(reward-old_mean)/count

mean=0.0
for count,reward in enumerate([1.,3.,2.],start=1):
    mean=incremental_mean(mean,reward,count)
print("Incremental/sample mean:",mean,np.mean([1.,3.,2.]))
"""
    experiment = r"""
true_means=np.array([.2,.5,.8,1.0])
estimates=np.zeros(4); counts=np.zeros(4,dtype=int)
rewards=[]; exploration_rates=[]; lengths=[]
for episode in range(400):
    epsilon=max(.02,.3*(.995**episode))
    action=int(rng.integers(4)) if rng.random()<epsilon else int(np.argmax(estimates))
    reward=float(rng.normal(true_means[action],.2))
    counts[action]+=1
    estimates[action]=incremental_mean(estimates[action],reward,counts[action])
    rewards.append(reward); exploration_rates.append(epsilon); lengths.append(1)
print("True means:",true_means,"estimated:",estimates.round(3),"pull counts:",counts)
plot_rl_metrics(rewards,exploration_rates,lengths,"Epsilon-greedy four-armed bandit")
"""
    hyper = r"""
epsilon_values=[0,.05,.1,.3,.8]
expected_random_fraction=epsilon_values
figure,axis=plt.subplots(); axis.plot(epsilon_values,expected_random_fraction,marker="o",color=COURSE_COLORS["blue"],label="Random-action fraction")
axis.set(title="Epsilon directly controls random exploration",xlabel="Epsilon",ylabel="Expected random-action share"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Multi-Armed Bandits",
        objectives=["Estimate action values incrementally", "Implement epsilon-greedy exploration", "Read regret-relevant reward curves"],
        problem="Repeatedly select one action with unknown reward distribution while balancing learning and earning.",
        intuition="Try several slot-machine arms, remember their average rewards, and usually choose the best-looking arm while occasionally checking alternatives.",
        equation=r"$$Q_{n+1}(a)=Q_n(a)+\frac{1}{N_n(a)}[R_n-Q_n(a)].$$",
        symbols=[r"$Q_n(a)$ is action a's estimate.", r"$N_n(a)$ is its selection count.", r"$R_n$ is the new reward.", r"$1/N_n(a)$ is the sample-average step size."],
        why_equation="It updates the mean without storing every prior reward.",
        numerical_example="Old mean 2 after two pulls and new reward 5 gives 2+(5-2)/3=3.",
        python_connection="old_mean + (reward-old_mean)/count is the exact incremental update.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Epsilon controls exploration frequency; too little can lock onto a bad early estimate and too much wastes reward.",
        hyperparameter_code=hyper,
        interpretation="Pull counts concentrate on the highest-mean arm while nonzero epsilon continues gathering information.",
    )


def mdp_spec() -> LessonSpec:
    scratch = r"""
states=["start","safe","terminal"]
transition={
    ("start","safe"):[("safe",1.0,0.0)],
    ("start","risky"):[("terminal",.5,2.0),("safe",.5,-1.0)],
    ("safe","finish"):[("terminal",1.0,1.0)],
}
def expected_backup(outcomes,values,gamma):
    return sum(probability*(reward+gamma*values[next_state]) for next_state,probability,reward in outcomes)
values={"start":0.0,"safe":1.0,"terminal":0.0}
print("Safe/risky action values:",expected_backup(transition[("start","safe")],values,.9),expected_backup(transition[("start","risky")],values,.9))
"""
    experiment = r"""
policy_action="risky"
rewards=[]; exploration_rates=[]; lengths=[]
for episode in range(150):
    if policy_action=="risky":
        success=rng.random()<.5
        trajectory_reward=2.0 if success else 0.0
        length=1 if success else 2
    else:
        trajectory_reward=1.0
        length=2
    rewards.append(trajectory_reward); exploration_rates.append(0.0); lengths.append(length)
print("Empirical mean return:",round(np.mean(rewards),3),"theoretical risky return:",1.0)
plot_rl_metrics(rewards,exploration_rates,lengths,"Fixed-policy MDP rollouts")
"""
    hyper = r"""
gammas=np.linspace(0,1,21)
safe_values=[0+.0+gamma*1 for gamma in gammas]
risky_values=[.5*2+.5*(gamma*1-1) for gamma in gammas]
figure,axis=plt.subplots(); axis.plot(gammas,safe_values,label="Safe",color=COURSE_COLORS["blue"]); axis.plot(gammas,risky_values,label="Risky",color=COURSE_COLORS["orange"])
axis.set(title="Discount factor can change action preference",xlabel="Gamma",ylabel="Expected action value"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Markov Decision Processes",
        objectives=["Specify states, actions, rewards, and transitions", "Explain the Markov property", "Compute an expected Bellman backup"],
        problem="Model sequential decisions where actions probabilistically change state and reward.",
        intuition="An MDP is a map of possible situations plus rules describing what each action may cause next.",
        equation=r"$$Q(s,a)=\sum_{s'}P(s'|s,a)[R(s,a,s')+\gamma V(s')].$$",
        symbols=[r"$s$ is current state.", r"$a$ is action.", r"$s'$ is next state.", r"$P$ is transition probability.", r"$R$ is immediate reward.", r"$V$ is future state value."],
        why_equation="This Bellman expectation equation combines every possible immediate consequence with discounted future value.",
        numerical_example="Two equally likely outcomes worth 2 and 0 have expected value .5(2)+.5(0)=1.",
        python_connection="The scratch sum multiplies each outcome's probability by reward plus discounted next value.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Discounting can alter which action has higher expected long-term value.",
        hyperparameter_code=hyper,
        interpretation="Individual trajectories vary even under a fixed policy; their average approaches expected return as episodes accumulate.",
    )


def dynamic_programming_spec() -> LessonSpec:
    scratch = r"""
GRID=4
terminal=GRID*GRID-1
def transitions(state,action):
    row,column=divmod(state,GRID)
    moves=[(-1,0),(0,1),(1,0),(0,-1)]
    dr,dc=moves[action]
    next_row=int(np.clip(row+dr,0,GRID-1)); next_column=int(np.clip(column+dc,0,GRID-1))
    next_state=next_row*GRID+next_column
    return next_state,(0.0 if next_state==terminal else -1.0)

def value_iteration(gamma=.95,tolerance=1e-8):
    values=np.zeros(GRID*GRID)
    for _ in range(500):
        updated=values.copy()
        for state in range(terminal):
            updated[state]=max(reward+gamma*values[next_state] for action in range(4) for next_state,reward in [transitions(state,action)])
        if np.max(np.abs(updated-values))<tolerance: break
        values=updated
    policy=np.zeros(terminal,dtype=int)
    for state in range(terminal):
        policy[state]=np.argmax([reward+gamma*values[next_state] for action in range(4) for next_state,reward in [transitions(state,action)]])
    return values,policy

values,policy=value_iteration()
print("Optimal values:\n",values.reshape(GRID,GRID).round(1))
"""
    experiment = r"""
rewards=[]; exploration_rates=[]; lengths=[]
for episode in range(100):
    state=0; total=0.0
    for step in range(30):
        action=int(policy[state]); state,reward=transitions(state,action); total+=reward
        if state==terminal: break
    rewards.append(total); exploration_rates.append(0.0); lengths.append(step+1)
print("Policy actions 0=up,1=right,2=down,3=left:\n",np.append(policy,-1).reshape(GRID,GRID))
plot_rl_metrics(rewards,exploration_rates,lengths,"Value-iteration policy evaluation")
"""
    hyper = r"""
gammas=[0,.5,.9,.99]
start_values=[value_iteration(gamma=value)[0][0] for value in gammas]
figure,axis=plt.subplots(); axis.plot(gammas,start_values,marker="o",color=COURSE_COLORS["blue"],label="Start-state value")
axis.set(title="Discount factor changes planned value",xlabel="Gamma",ylabel="V(start)"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Dynamic Programming: Policy and Value Iteration",
        objectives=["Perform Bellman optimality backups", "Implement value iteration", "Extract and evaluate a greedy policy"],
        problem="Plan an optimal policy when the complete transition and reward model is known.",
        intuition="Repeatedly improve every state's estimate using one-step lookahead until further sweeps barely change anything.",
        equation=r"$$V_{k+1}(s)=\max_a\sum_{s'}P(s'|s,a)[R+\gamma V_k(s')].$$",
        symbols=[r"$V_k(s)$ is state value after sweep k.", r"$a$ is a candidate action.", r"$P$ is transition probability.", r"$R$ is immediate reward.", r"$\gamma$ discounts future value."],
        why_equation="The best one-step lookahead becomes exact after repeated contraction-style updates in a finite discounted MDP.",
        numerical_example="If two actions yield backed-up values 2 and 3.5, the new state value is max(2,3.5)=3.5.",
        python_connection="The nested max and transition loop in value_iteration is the Bellman optimality equation.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Gamma changes how strongly step costs and distant goals affect planned value.",
        hyperparameter_code=hyper,
        interpretation="The deterministic policy reaches the goal in a constant short path, so moving reward and episode length are flat after planning.",
    )


def monte_carlo_spec() -> LessonSpec:
    scratch = r"""
def returns_from_rewards(rewards,gamma):
    returns=np.zeros(len(rewards)); running=0.0
    for index in range(len(rewards)-1,-1,-1):
        running=rewards[index]+gamma*running
        returns[index]=running
    return returns

print("Returns-to-go:",returns_from_rewards([0,0,1],.9).round(3))
"""
    experiment = r"""
state_count=7
values=np.zeros(state_count)
counts=np.zeros(state_count)
rewards_history=[]; exploration_rates=[]; lengths=[]
for episode in range(500):
    state=3; visited=[]; rewards=[]
    for step in range(30):
        visited.append(state)
        state+=-1 if rng.random()<.5 else 1
        reward=1.0 if state==state_count-1 else 0.0
        rewards.append(reward)
        if state in [0,state_count-1]: break
    returns=returns_from_rewards(rewards,1.0)
    seen=set()
    for visited_state,ret in zip(visited,returns):
        if visited_state not in seen:
            counts[visited_state]+=1
            values[visited_state]+=(ret-values[visited_state])/counts[visited_state]
            seen.add(visited_state)
    rewards_history.append(sum(rewards)); exploration_rates.append(1.0); lengths.append(step+1)
print("Estimated nonterminal values:",values[1:-1].round(3))
print("Analytic values:",np.arange(1,state_count-1)/(state_count-1))
plot_rl_metrics(rewards_history,exploration_rates,lengths,"First-visit Monte Carlo random walk")
"""
    hyper = r"""
episodes=np.arange(1,len(rewards_history)+1)
running_success=np.cumsum(rewards_history)/episodes
figure,axis=plt.subplots(); axis.plot(episodes,running_success,color=COURSE_COLORS["blue"],label="Running right-terminal rate")
axis.axhline(.5,color=COURSE_COLORS["orange"],linestyle="--",label="Symmetric expectation")
axis.set(title="Monte Carlo estimate stabilizes with episodes",xlabel="Episode",ylabel="Running success rate"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Monte Carlo Methods",
        objectives=["Calculate returns-to-go", "Implement first-visit value estimation", "Explain why complete episodes are required"],
        problem="Estimate values directly from sampled complete returns without a transition model.",
        intuition="Finish an episode, look backward at what each visited state eventually earned, and average those outcomes.",
        equation=r"$$V(s)\leftarrow V(s)+\frac{1}{N(s)}[G-V(s)].$$",
        symbols=[r"$V(s)$ is estimated state value.", r"$N(s)$ is visit count.", r"$G$ is sampled return from the visit."],
        why_equation="It incrementally averages unbiased sampled returns for a fixed policy.",
        numerical_example="Old value .4 after four returns and new return 1 gives .4+(1-.4)/5=.52.",
        python_connection="values[state] += (return-value)/count is the incremental mean update.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="More episodes reduce sampling noise; the running success rate makes convergence visible.",
        hyperparameter_code=hyper,
        interpretation="Values rise toward the rewarding right terminal. Individual episode rewards stay binary and noisy, so averaging is essential.",
    )


def tabular_control_spec(algorithm: str) -> LessonSpec:
    on_policy = algorithm == "SARSA"
    update = (
        "next_action=int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(Q[next_state]))\n"
        "        target=reward+(0 if terminated else gamma*Q[next_state,next_action])"
        if on_policy else
        "next_action=None\n"
        "        target=reward+(0 if terminated else gamma*np.max(Q[next_state]))"
    )
    experiment = LINE_WORLD + f'''
environment=LineWorld()
Q=np.zeros((environment.length,2))
alpha=.3; gamma=.95
rewards=[]; exploration_rates=[]; lengths=[]
snapshots={{0:Q.copy()}}
for episode in range(220):
    state=environment.reset(); epsilon=max(.03,.9*(.975**episode))
    action=int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(Q[state]))
    total=0.0
    for step in range(environment.max_steps):
        next_state,reward,terminated,truncated=environment.step(action)
        {update}
        Q[state,action]+=alpha*(target-Q[state,action])
        state=next_state
        action=next_action if next_action is not None else (int(rng.integers(2)) if rng.random()<epsilon else int(np.argmax(Q[state])))
        total+=reward
        if terminated or truncated: break
    rewards.append(total); exploration_rates.append(epsilon); lengths.append(step+1)
    if episode in [49,219]: snapshots[episode+1]=Q.copy()
print("Q-table before:\\n",snapshots[0])
print("Q-table after 50 episodes:\\n",snapshots[50].round(3))
print("Q-table after training:\\n",snapshots[220].round(3))
plot_rl_metrics(rewards,exploration_rates,lengths,"{algorithm} in LineWorld")
'''
    hyper = r"""
rates=[.05,.1,.3,.7]
update_sizes=[rate*1.0 for rate in rates]
figure,axis=plt.subplots(); axis.plot(rates,update_sizes,marker="o",color=COURSE_COLORS["blue"],label="Update for unit TD error")
axis.set(title="Learning rate scales each TD correction",xlabel="Alpha",ylabel="Q-value change"); axis.legend(); plt.show()
"""
    if on_policy:
        equation = r"$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)].$$"
        intuition = "Update toward the value of the next action the exploratory policy actually selects."
        comparison = "Because SARSA learns from the behavior action, exploration risk is represented in its target."
    else:
        equation = r"$$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)].$$"
        intuition = "Act with exploration but update toward the best estimated next action."
        comparison = "Q-learning is off-policy: its target is greedy even while behavior explores."
    return base_spec(
        title=f"{algorithm}: Tabular Temporal-Difference Control",
        objectives=[f"Derive the {algorithm} update", "Inspect Q-tables before, during, and after training", "Relate exploration decay to reward and episode length"],
        problem="Learn action values online without knowing transition probabilities.",
        intuition=intuition,
        equation=equation,
        symbols=[r"$Q(S_t,A_t)$ is current action value.", r"$\alpha$ is learning rate.", r"$R_{t+1}$ is reward.", r"$\gamma$ is discount factor.", r"The bracketed term is temporal-difference error."],
        why_equation="It bootstraps from a one-step reward plus an estimate of remaining value.",
        numerical_example="If Q=.2, alpha=.5, reward=0, gamma=.9, and next estimate=.8, new Q=.2+.5(.72-.2)=.46.",
        python_connection="Q[state, action] += alpha * (target - Q[state, action]).",
        scratch_code="Q_demo=np.zeros((3,2))\nold=Q_demo[0,1]; target=.72; Q_demo[0,1]+=.5*(target-old)\nprint('One updated Q-value:',Q_demo[0,1])",
        experiment_code=experiment,
        hyperparameter_text="Alpha controls how quickly new experience overwrites old estimates.",
        hyperparameter_code=hyper,
        interpretation=f"{comparison} Both reward and episode length improve as rightward values propagate backward.",
    )


def dqn_spec() -> LessonSpec:
    scratch = r"""
def dqn_target(reward,terminated,next_q,gamma):
    return reward if terminated else reward+gamma*np.max(next_q)

print("Terminal/nonterminal targets:",dqn_target(1,True,[.2,.8],.9),dqn_target(0,False,[.2,.8],.9))
"""
    experiment = LINE_WORLD + r"""
import torch
from torch import nn
torch.set_num_threads(1)
environment=LineWorld()
model=nn.Sequential(nn.Linear(environment.length,16),nn.ReLU(),nn.Linear(16,2))
target_model=nn.Sequential(nn.Linear(environment.length,16),nn.ReLU(),nn.Linear(16,2))
target_model.load_state_dict(model.state_dict())
optimizer=torch.optim.Adam(model.parameters(),lr=.01); loss_fn=nn.SmoothL1Loss()
replay=[]; rewards=[]; exploration_rates=[]; lengths=[]

def encode(state):
    result=np.zeros(environment.length,dtype="float32"); result[state]=1; return result

for episode in range(180):
    state=environment.reset(); epsilon=max(.03,.9*(.975**episode)); total=0.0
    for step in range(environment.max_steps):
        with torch.no_grad(): q_values=model(torch.tensor(encode(state)))
        action=int(rng.integers(2)) if rng.random()<epsilon else int(torch.argmax(q_values))
        next_state,reward,terminated,truncated=environment.step(action)
        replay.append((state,action,reward,next_state,terminated))
        replay=replay[-500:]
        if len(replay)>=24:
            indices=rng.choice(len(replay),24,replace=False); batch=[replay[index] for index in indices]
            states=torch.tensor(np.stack([encode(item[0]) for item in batch]))
            actions=torch.tensor([item[1] for item in batch],dtype=torch.long)
            batch_rewards=torch.tensor([item[2] for item in batch],dtype=torch.float32)
            next_states=torch.tensor(np.stack([encode(item[3]) for item in batch]))
            terminals=torch.tensor([item[4] for item in batch],dtype=torch.float32)
            chosen=model(states).gather(1,actions[:,None]).squeeze()
            with torch.no_grad(): targets=batch_rewards+.95*(1-terminals)*target_model(next_states).max(1).values
            loss=loss_fn(chosen,targets); optimizer.zero_grad(); loss.backward(); optimizer.step()
        state=next_state; total+=reward
        if terminated or truncated: break
    if episode%15==0: target_model.load_state_dict(model.state_dict())
    rewards.append(total); exploration_rates.append(epsilon); lengths.append(step+1)
with torch.no_grad(): learned_q=model(torch.eye(environment.length)).numpy()
print("Input/hidden/output:",(environment.length,environment.length),(environment.length,16),learned_q.shape)
print("Parameters:",sum(p.numel() for p in model.parameters()),"final Q-values:\n",learned_q.round(2))
plot_rl_metrics(rewards,exploration_rates,lengths,"Deep Q-Network in LineWorld")
"""
    hyper = r"""
buffer_sizes=[24,100,500,2000]
memory_items=np.array(buffer_sizes)*5
figure,axis=plt.subplots(); axis.plot(buffer_sizes,memory_items,marker="o",color=COURSE_COLORS["blue"],label="Stored transition fields")
axis.set(title="Replay capacity trades memory for experience diversity",xlabel="Replay transitions",ylabel="Scalar fields stored"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Deep Q-Networks",
        objectives=["Replace a Q-table with a neural approximator", "Use replay and a target network", "Inspect neural Q-values and learning curves"],
        problem="Estimate action values when a table is impractical because states are numerous or continuous.",
        intuition="A network predicts every action value from state features; replay breaks short-term correlations and a delayed target stabilizes the moving objective.",
        equation=r"$$L(\theta)=\left[r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a')-Q_\theta(s,a)\right]^2.$$",
        symbols=[r"$\theta$ are online-network parameters.", r"$\theta^-$ are target-network parameters.", r"$r$ is reward.", r"$d$ indicates termination.", r"$s,a,s'$ are transition state, action, next state."],
        why_equation="It turns a temporal-difference target into a differentiable regression loss.",
        numerical_example="For r=0, gamma=.9, terminal=0, max next Q=.8, and current Q=.2, target=.72 and TD error=.52.",
        python_connection="gather selects Q(s,a); a no-gradient target network constructs the regression target.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Replay capacity changes experience diversity and memory usage; large buffers can also retain obsolete behavior.",
        hyperparameter_code=hyper,
        interpretation="The network assigns increasingly large values to rightward actions near the goal; learning curves show improved reward and shorter paths.",
    )


def policy_gradient_spec() -> LessonSpec:
    scratch = r"""
def softmax_np(logits):
    values=np.exp(logits-np.max(logits))
    return values/values.sum()

logits=np.array([0.,0.])
probabilities=softmax_np(logits)
gradient_log_policy=np.array([-probabilities[0],1-probabilities[1]])
print("Action probabilities:",probabilities,"example score-function gradient:",gradient_log_policy)
"""
    experiment = r"""
import torch
from torch import nn
torch.set_num_threads(1)
true_means=torch.tensor([.1,.9])
logits=nn.Parameter(torch.zeros(2))
optimizer=torch.optim.Adam([logits],lr=.08)
rewards=[]; exploration_rates=[]; lengths=[]
for episode in range(250):
    distribution=torch.distributions.Categorical(logits=logits)
    action=distribution.sample()
    reward=float(rng.normal(float(true_means[action]),.25))
    loss=-distribution.log_prob(action)*reward
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    rewards.append(reward)
    probabilities=torch.softmax(logits.detach(),dim=0).numpy()
    exploration_rates.append(float(1-probabilities.max()))
    lengths.append(1)
print("Learned action probabilities:",torch.softmax(logits.detach(),dim=0).numpy().round(3))
print("Trainable parameters:",logits.numel())
plot_rl_metrics(rewards,exploration_rates,lengths,"REINFORCE on a two-action bandit")
"""
    hyper = r"""
scales=[.25,.5,1,2]
gradient_scale=scales
figure,axis=plt.subplots(); axis.plot(scales,gradient_scale,marker="o",color=COURSE_COLORS["blue"],label="Relative policy-gradient magnitude")
axis.set(title="Reward scale directly affects REINFORCE updates",xlabel="Reward multiplier",ylabel="Relative gradient magnitude"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Policy Gradients",
        objectives=["Parameterize a stochastic policy", "Derive the REINFORCE loss", "Train a policy directly from sampled rewards"],
        problem="Optimize action probabilities directly instead of deriving a policy from action values.",
        intuition="Increase the log-probability of actions followed by high return and decrease it for actions with poor relative return.",
        equation=r"$$\nabla_\theta J(\theta)=\mathbb{E}[G_t\nabla_\theta\log\pi_\theta(A_t|S_t)].$$",
        symbols=[r"$J$ is expected return.", r"$\theta$ are policy parameters.", r"$G_t$ is sampled return.", r"$\pi_\theta$ is action probability.", r"$A_t,S_t$ are action and state."],
        why_equation="The log-derivative trick makes a sampled discrete action produce a differentiable policy update.",
        numerical_example="If return is 2 and gradient log-probability is .3, the sampled gradient contribution is .6.",
        python_connection="-distribution.log_prob(action) * reward is minimized to perform gradient ascent on expected reward.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="Reward scale changes gradient magnitude and therefore interacts with learning rate and variance.",
        hyperparameter_code=hyper,
        interpretation="Probability concentrates on the higher-reward action while stochastic sampling remains explicit.",
    )


def actor_critic_spec() -> LessonSpec:
    scratch = r"""
reward=.8
value=.5
next_value=.6
gamma=.9
td_error=reward+gamma*next_value-value
print("One-step advantage estimate:",td_error)
"""
    experiment = LINE_WORLD + r"""
import torch
from torch import nn
torch.set_num_threads(1)
environment=LineWorld()
actor=nn.Sequential(nn.Linear(environment.length,16),nn.Tanh(),nn.Linear(16,2))
critic=nn.Sequential(nn.Linear(environment.length,16),nn.Tanh(),nn.Linear(16,1))
optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=.01)
rewards=[]; exploration_rates=[]; lengths=[]
def encode(state):
    result=torch.zeros(environment.length); result[state]=1; return result
for episode in range(220):
    state=environment.reset(); total=0.0; entropies=[]
    for step in range(environment.max_steps):
        state_tensor=encode(state)
        distribution=torch.distributions.Categorical(logits=actor(state_tensor))
        action=distribution.sample()
        next_state,reward,terminated,truncated=environment.step(int(action))
        value=critic(state_tensor).squeeze()
        with torch.no_grad(): next_value=torch.tensor(0.) if terminated else critic(encode(next_state)).squeeze()
        advantage=torch.tensor(reward)+.95*next_value-value
        actor_loss=-distribution.log_prob(action)*advantage.detach()
        critic_loss=advantage.pow(2)
        loss=actor_loss+.5*critic_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        entropies.append(float(distribution.entropy().detach()))
        state=next_state; total+=reward
        if terminated or truncated: break
    rewards.append(total); exploration_rates.append(float(np.mean(entropies))); lengths.append(step+1)
print("Actor/critic parameters:",sum(p.numel() for p in actor.parameters()),sum(p.numel() for p in critic.parameters()))
print("Actor output shape:",tuple(actor(torch.eye(environment.length)).shape),"critic output shape:",tuple(critic(torch.eye(environment.length)).shape))
plot_rl_metrics(rewards,exploration_rates,lengths,"Actor-Critic in LineWorld (exploration proxy = entropy)")
"""
    hyper = r"""
coefficients=[0,.01,.05,.1,.2]
figure,axis=plt.subplots(); axis.plot(coefficients,coefficients,marker="o",color=COURSE_COLORS["blue"],label="Entropy bonus weight")
axis.set(title="Entropy coefficient controls exploration pressure",xlabel="Entropy coefficient",ylabel="Bonus per entropy unit"); axis.legend(); plt.show()
"""
    return base_spec(
        title="Actor-Critic Fundamentals",
        objectives=["Separate policy and value roles", "Use TD error as an advantage estimate", "Train actor and critic together"],
        problem="Reduce policy-gradient variance by learning a baseline that evaluates states.",
        intuition="The actor chooses; the critic judges whether the outcome was better or worse than expected; that surprise trains both.",
        equation=r"$$\delta_t=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t).$$",
        symbols=[r"$\delta_t$ is TD error and advantage estimate.", r"$R_{t+1}$ is reward.", r"$\gamma$ discounts.", r"$V_\phi$ is critic value with parameters phi.", r"$S_t,S_{t+1}$ are states."],
        why_equation="Subtracting expected value centers policy updates and trains the critic toward a one-step target.",
        numerical_example="For reward .8, gamma=.9, next value .6, current value .5, delta=.8+.54-.5=.84.",
        python_connection="advantage = reward + gamma*next_value - value drives actor log-probability and critic squared-error losses.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="An entropy bonus can preserve exploration, but too much prevents the policy from becoming decisive.",
        hyperparameter_code=hyper,
        interpretation="Reward increases and episode length falls as the actor learns rightward actions; entropy decreases as the policy becomes more certain.",
    )


def capstone_spec() -> LessonSpec:
    scratch = r"""
def q_update(old,reward,next_value,alpha=.3,gamma=.95):
    return old+alpha*(reward+gamma*next_value-old)
print("Example Q update:",round(q_update(.2,0,.8),3))
"""
    experiment = LINE_WORLD + r"""
def train(method,seed,episodes=240):
    local_rng=np.random.default_rng(seed); environment=LineWorld(length=8,max_steps=30); Q=np.zeros((environment.length,2))
    rewards=[]; epsilons=[]; lengths=[]; snapshots={0:Q.copy()}
    for episode in range(episodes):
        state=environment.reset(); epsilon=max(.03,.9*(.98**episode)); action=int(local_rng.integers(2)) if local_rng.random()<epsilon else int(np.argmax(Q[state])); total=0
        for step in range(environment.max_steps):
            next_state,reward,terminated,truncated=environment.step(action)
            next_action=int(local_rng.integers(2)) if local_rng.random()<epsilon else int(np.argmax(Q[next_state]))
            next_value=Q[next_state,next_action] if method=="SARSA" else np.max(Q[next_state])
            target=reward+(0 if terminated else .95*next_value)
            Q[state,action]+=.3*(target-Q[state,action])
            state=next_state; action=next_action; total+=reward
            if terminated or truncated: break
        rewards.append(total); epsilons.append(epsilon); lengths.append(step+1)
        if episode+1==episodes//2: snapshots[episode+1]=Q.copy()
    snapshots[episodes]=Q.copy()
    return Q,np.array(rewards),epsilons,lengths,snapshots

results={}
for method in ["Q-learning","SARSA"]:
    results[method]=train(method,RANDOM_SEED)
    print(method,"final moving reward:",round(results[method][1][-20:].mean(),3))
best_method=max(results,key=lambda name:results[name][1][-20:].mean())
Q,rewards,epsilons,lengths,snapshots=results[best_method]
print("Selected Q-table before training:\n",snapshots[0].round(2))
print("Selected Q-table during training:\n",snapshots[120].round(2))
print("Selected Q-table after training:\n",snapshots[240].round(2))
plot_rl_metrics(rewards,epsilons,lengths,f"Capstone selected agent: {best_method}")
"""
    hyper = r"""
summary=[]
for method in ["Q-learning","SARSA"]:
    for seed in [7,21,42]:
        _,seed_rewards,_,seed_lengths,_=train(method,seed)
        summary.append({"method":method,"seed":seed,"final_reward":seed_rewards[-20:].mean(),"final_length":np.mean(seed_lengths[-20:])})
summary=pd.DataFrame(summary)
display(summary.round(3))
aggregate=summary.groupby("method")[["final_reward","final_length"]].agg(["mean","std"])
display(aggregate.round(3))
figure,axis=plt.subplots()
means=summary.groupby("method")["final_reward"].mean(); stds=summary.groupby("method")["final_reward"].std()
axis.bar(means.index,means.values,yerr=stds.values,color=COURSE_COLORS["blue"],capsize=5)
axis.set(title="Three-seed agent comparison",xlabel="Algorithm",ylabel="Final 20-episode mean reward")
plt.show()
"""
    return base_spec(
        title="Reinforcement Learning Capstone: LineWorld Agent",
        objectives=["Define success before training", "Compare Q-learning and SARSA across seeds", "Select, evaluate, and critique a final agent"],
        problem="Reach a delayed goal quickly in a custom environment; success is high final moving reward and short episodes across seeds.",
        use_cases=["Safe algorithm comparison", "Reproducible sequential-decision prototyping", "Training-curve and policy error analysis"],
        intuition="Build the smallest meaningful environment, establish random exploration, compare on-policy and off-policy control, and require repeatability.",
        scratch_description="A scalar TD update is checked first, then both complete tabular algorithms are implemented with NumPy.",
        library_description="No high-level RL algorithm is hidden here; NumPy implements learning and Matplotlib supports auditable curves.",
        dataset_description="Experience is generated from a documented eight-state LineWorld with two actions, step cost, goal reward, and episode cap.",
        preprocessing_description="Discrete states index Q-table rows directly; seeds and episode budgets are held fixed across candidates.",
        scratch_code=scratch,
        experiment_code=experiment,
        hyperparameter_text="The final comparison uses three seeds rather than selecting a lucky run. Mean and standard deviation quantify variability.",
        hyperparameter_code=hyper,
        interpretation="The selected agent has the stronger final moving reward for the primary seed; the multi-seed table determines whether that advantage is repeatable.",
        mistakes=["Selecting the best seed", "Changing episode budgets between candidates", "Reporting reward without episode length", "Ignoring environment simplifications"],
        advantages=["Transparent reproducible environment", "Baseline and multiple-algorithm comparison", "Multi-seed uncertainty check"],
        disadvantages=["Tiny discrete state space", "Reward design is simplified", "No safety or off-distribution evaluation"],
        exercises=["Why compare the same seeds?", "For old Q=.4, reward=1, terminal=True, alpha=.5, calculate new Q.", "Add a small stochastic action-slip probability and rerun the comparison."],
        solutions=["Matched seeds reduce variation from different random experiences and make comparison fairer.", "Terminal target is 1, so new Q=.4+.5(1-.4)=.7.", "With probability p replace the chosen action by the other action, keep p fixed for both methods, and compare multi-seed reward and length."],
        takeaways=["A real RL result includes environment and reward definitions", "Q-tables reveal what the agent learned", "Reward, moving reward, exploration, and length tell complementary stories", "Multiple seeds and limitations are required for a credible conclusion"],
    )


def main() -> None:
    lessons = {
        "04_reinforcement_learning/00_reinforcement_learning_overview.ipynb": overview_spec(),
        "04_reinforcement_learning/01_multi_armed_bandits.ipynb": bandit_spec(),
        "04_reinforcement_learning/02_markov_decision_processes.ipynb": mdp_spec(),
        "04_reinforcement_learning/03_dynamic_programming.ipynb": dynamic_programming_spec(),
        "04_reinforcement_learning/04_monte_carlo_methods.ipynb": monte_carlo_spec(),
        "04_reinforcement_learning/05_q_learning.ipynb": tabular_control_spec("Q-learning"),
        "04_reinforcement_learning/06_sarsa.ipynb": tabular_control_spec("SARSA"),
        "04_reinforcement_learning/07_deep_q_network.ipynb": dqn_spec(),
        "04_reinforcement_learning/08_policy_gradients.ipynb": policy_gradient_spec(),
        "04_reinforcement_learning/09_actor_critic.ipynb": actor_critic_spec(),
        "04_reinforcement_learning/10_reinforcement_learning_capstone.ipynb": capstone_spec(),
    }
    for path, spec in lessons.items():
        write_lesson(path, spec)
        print("Built", path)


if __name__ == "__main__":
    main()
