import minari
import torch.nn as nn

############################################
#           User, Server 정의부              #
############################################

class UserFedRL:
    def __init__(self, actor, critic, model, dataset):
        self.actor = actor
        self.critic = critic
        self.model = model
        self.dataset = dataset


class Server:
    def __init__(self, actor, critic, model):
        self.actor = actor
        self.critic = critic
        self.model = model


############################################
#            Actor, Critic 정의부            #
############################################

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1



############################################
#                Model 정의부                #
############################################
class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1)  # next state + reward
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        output = self.net(x)
        next_state = output[:, :-1]
        reward = output[:, -1]
        return next_state, reward



def getDatasetInfo(datasetName):
    dataset = minari.load_dataset(datasetName)
    state_dim = dataset.observation_space.shape[0] ## state 차원
    action_dim = dataset.action_space.shape[0] ## action 차원
    
    ## Actor에서 action의 범위는 tanh함수를 사용하여 (-1, 1)인데, 데이터셋에서는 범위가 일반적으로 (a, b).
    ## Mujoco 환경에서의 범위는 (-a, a) 형태이므로 a값을 Actor에 넣어주어야 함.
    ## Minari에서는 이 때의 a값을 max_action으로 정의함
    max_action = dataset.action_space.high[0]
    
    return dataset, state_dim, action_dim, max_action




if __name__ == '__main__':

    ## 데이터셋 이름들 명시하기
    datasetNames = [
        "mujoco/inverteddoublependulum/expert-v0", 
        "mujoco/inverteddoublependulum/medium-v0"
    ]

    ## 공통 정보 추출 (서버/클라이언트 공유용)
    _, state_dim, action_dim, max_action = getDatasetInfo(datasetNames[0])

    ## Server Initialization ##
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)
    model = Model(state_dim, action_dim)

    server = Server(actor, critic, model)

    ## Users Initialization ##
    users = []
    for datasetName in datasetNames:
        dataset, _, _, _ = getDatasetInfo(datasetName)
        actor = Actor(state_dim, action_dim, max_action)
        critic = Critic(state_dim, action_dim)
        model = Model(state_dim, action_dim)

        user = UserFedRL(actor, critic, model, dataset)
        users.append(user)


    print("여기까지 옴")