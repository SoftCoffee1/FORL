import minari
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy

############################################
#           Client, Server 정의부            #
############################################


class UserFedRL:
    def __init__(self, actor, critic, model, dataset):
        self.actor = actor
        self.critic = critic
        self.model = model
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            SequenceTransitionDataset(dataset, h_step=3), batch_size=32, shuffle=True
        )

    def train(self, num_epochs, actor, critic):
        self.train_actor()
        self.train_critic()
        self.train_model()

    def train_actor(self):
        pass

    def train_critic(self):
        pass

    def train_model(self, h_step=3):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        loss_list = []

        for i in range(len(self.dataset)):  # Epoch loop
            total_loss = 0.0

            for j in range(len(self.dataset[i]) - h_step):
                states = self.dataset[i].observations[j : j + h_step]
                actions = self.dataset[i].actions[j : j + h_step]

                # numpy → torch 변환
                states = torch.from_numpy(states).float().to(device)
                actions = torch.from_numpy(actions).float().to(device)

                loss = 0.0
                cur_s_hat = states[0]

                for step in range(h_step - 1):
                    next_s_hat = model(states[step], actions[step])
                    loss += loss_fn(
                        next_s_hat - cur_s_hat, states[step + 1] - states[step]
                    )
                    cur_s_hat = next_s_hat

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {i+1}] Loss: {total_loss:.4f}")
            loss_list.append(total_loss)

        # 시각화 및 저장
        plt.figure(figsize=(8, 4))
        plt.plot(loss_list, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)

        plt.savefig("training_loss.png")  # PNG로 저장

        # # 학습 루프
        # for epoch in range(num_epochs):
        #     total_loss = 0.0

        #     for i in range(0, len(inputs), batch_size):
        #         batch_inputs = inputs[i:i+batch_size]
        #         batch_targets = targets[i:i+batch_size]

        #         # 텐서 변환
        #         s1_batch = torch.stack([item[0] for item in batch_inputs]).to(device)
        #         a1_batch = torch.stack([item[1] for item in batch_inputs]).to(device)
        #         a2_batch = torch.stack([item[2] for item in batch_inputs]).to(device)
        #         s2_batch = torch.stack([item[0] for item in batch_targets]).to(device)
        #         s3_batch = torch.stack([item[1] for item in batch_targets]).to(device)

        #         optimizer.zero_grad()

        #         # state2 예측
        #         pred_s2 = model(s1_batch, a1_batch)
        #         # state3 예측: 예측한 state2를 입력으로 다시 사용
        #         pred_s3 = model(pred_s2.detach(), a2_batch)  # detach()로 역전파 경로 분리

        #         loss_s2 = loss_fn(pred_s2, s2_batch)
        #         loss_s3 = loss_fn(pred_s3, s3_batch)
        #         loss = loss_s2 + loss_s3

        #         loss.backward()
        #         optimizer.step()

        #         total_loss += loss.item()

        #     print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")


class Server:
    def __init__(self, actor, critic, model, datasetNames):
        self.actor = actor
        self.critic = critic
        self.model = model
        self.users = []

        # user들 초기화하기
        for datasetName in datasetNames:
            dataset, _, _, _ = getDatasetInfo(datasetName)
            actor = Actor(state_dim, action_dim, max_action)
            critic = Critic(state_dim, action_dim)
            model = Model(state_dim, action_dim)
            user = UserFedRL(actor, critic, model, dataset)
            self.users.append(user)

    def train(self):
        print("train start...")

        self.send_parameters_actor()
        self.send_parameters_critic()

        for i, user in enumerate(self.users):
            print(f"Training user {i}...")
            user_actor = copy.deepcopy(self.actor)
            user_critic = copy.deepcopy(self.critic)

            user.train(10, user_actor, user_critic)

        print("aggregate start...")
        self.aggregate_parameters_actor()
        self.aggregate_parameters_critic()
        self.aggregate_parameters_model()
        self.human_feedback()
        print("train end...")

    def send_parameters_actor(self):
        for user in self.users:
            user.actor.load_state_dict(self.actor.state_dict())

    def send_parameters_critic(self):
        for user in self.users:
            user.critic.load_state_dict(self.critic.state_dict())

    def aggregate_parameters_actor(self):
        return 0

    def aggregate_parameters_critic(self):
        return 0

    def aggregate_parameters_model(self):
        return 0

    def human_feedback(self):
        return 0


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
    def __init__(self, state_dim, action_dim, hidden_dim=256, hasRewardOutput=False):
        super(Model, self).__init__()
        self.hasRewardOutput = hasRewardOutput

        layers = [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ]

        if self.hasRewardOutput:
            layers.append(nn.Linear(hidden_dim, state_dim + 1))  # next state + reward
        else:
            layers.append(nn.Linear(hidden_dim, state_dim))  # next state only

        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        output = self.net(x)

        if self.hasRewardOutput:
            next_state = output[:, :-1]
            reward = output[:, -1]
            return next_state, reward
        else:
            return output


############################################
#              DataLoader 정의부             #
############################################
class SequenceTransitionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, h_step):
        self.dataset = dataset
        self.h_step = h_step
        self.samples = []

        for episode_id in range(len(dataset)):
            episode_length = len(dataset[episode_id].observations)
            # index를 설정: 마지막 h_step+1개는 예측 불가능하므로 제외
            for idx in range(episode_length - h_step):
                self.samples.append((episode_id, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_id, index = self.samples[idx]
        state_t = self.dataset[episode_id].observations[index]

        action_seq = [
            self.dataset[episode_id].actions[index + i] for i in range(self.h_step)
        ]
        next_state_seq = [
            self.dataset[episode_id].observations[index + i + 1]
            for i in range(self.h_step)
        ]

        action_seq = torch.cat(action_seq, dim=-1)  # (h * action_dim,)
        next_state_seq = torch.cat(next_state_seq, dim=-1)  # (h * state_dim,)

        return torch.tensor(state_t, dtype=torch.float32), action_seq, next_state_seq


def getDatasetInfo(datasetName):
    dataset = minari.load_dataset(datasetName)
    state_dim = dataset.observation_space.shape[0]  ## state 차원
    action_dim = dataset.action_space.shape[0]  ## action 차원

    ## Actor에서 action의 범위는 tanh함수를 사용하여 (-1, 1)인데, 데이터셋에서는 범위가 일반적으로 (a, b).
    ## Mujoco 환경에서의 범위는 (-a, a) 형태이므로 a값을 Actor에 넣어주어야 함.
    ## Minari에서는 이 때의 a값을 max_action으로 정의함
    max_action = dataset.action_space.high[0]

    return dataset, state_dim, action_dim, max_action


if __name__ == "__main__":

    ## 데이터셋 이름들 명시하기
    datasetNames = [
        "mujoco/inverteddoublependulum/expert-v0",
        "mujoco/inverteddoublependulum/medium-v0",
    ]

    ## 공통 정보 추출 (서버/클라이언트 공유용)
    _, state_dim, action_dim, max_action = getDatasetInfo(datasetNames[0])

    ## Server Initialization ##
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)
    model = Model(state_dim, action_dim)
    server = Server(actor, critic, model, datasetNames)
    server.train()
    print("여기까지 옴")
