import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import minari
from torch.utils.data import DataLoader

############################################
#           Client, Server 정의부            #
############################################


class UserFedRL:
    def __init__(self, actor, critic, model, dataset):
        self.actor = actor
        self.critic = critic
        self.model = model
        self.dataset = dataset
        self.pol_val = 0

        # 데이터셋을 직접 생성하여 차원 확인
        self.seq_dataset = SequenceTransitionDataset(dataset, h_step=3)
        print(f"Sequence Dataset created with {len(self.seq_dataset)} samples")

        self.stateAction_dataset = StateActionDataset(dataset)
        print(f"State Action Dataset created with {len(self.seq_dataset)} samples")

        # 데이터로더 생성
        self.dataloader = torch.utils.data.DataLoader(
            self.seq_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,  # 배치 크기에 맞지 않는 마지막 배치 무시
        )

        # 데이터로더 생성
        self.stateActionDataloader = torch.utils.data.DataLoader(
            self.stateAction_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True,  # 배치 크기에 맞지 않는 마지막 배치 무시
        )

    def train(
        self,
        server_critic,
        h_step=3,
        num_epochs=10,
        lr=1e-3,
        device="cpu",
    ):
        print("Training local user components...")
        self.train_actor()
        self.train_critic(server_critic, num_epochs=num_epochs, lr=lr, device=device)
        self.train_model(h_step=h_step, num_epochs=num_epochs, lr=lr, device=device)

    def train_actor(self):
        print("Training actor... (Not implemented yet)")
        pass

    def train_critic(self, server_critic, num_epochs=10, lr=1e-3, device="cpu"):

        self.critic.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        print(f"length : {len(self.stateActionDataloader)}")

        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0
            for state, action in self.stateActionDataloader:
                print(f"batch_count : {batch_count}")
                batch_count += 1

                # 현재 Q값 추정
                current_Q1, current_Q2 = self.critic(state, action)

                with torch.no_grad():
                    fed_Q1, fed_Q2 = server_critic(state, action)

                # 여기서 loss 계산 및 역전파 등의 로직 추가
                loss = F.mse_loss(current_Q1, fed_Q1) + F.mse_loss(current_Q2, fed_Q2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"[User Critic Epoch {epoch+1}] Loss: {total_loss:.4f}, Batches: {batch_count}"
            )

        print("Training user critic completed.")

    def train_model(self, h_step=3, num_epochs=10, lr=1e-3, device="cpu"):
        print("Training local model...")

        # 모델 차원 정보 확인
        action_dim = self.model.action_dim
        state_dim = self.model.state_dim
        print(f"Model dimensions - State: {state_dim}, Action: {action_dim}")

        # 데이터 샘플 확인 -> 이거 안되면 훈련 중단 !
        try:
            sample_batch = next(iter(self.dataloader))
            s0, a_seq, s_seq = sample_batch
            print(
                f"Sample batch shapes - s0: {s0.shape}, a_seq: {a_seq.shape}, s_seq: {s_seq.shape}"
            )
        except Exception as e:
            print(f"Error inspecting sample batch: {e}")
            # 데이터로더 문제가 있으면 훈련 중단
            return

        # 요거는 이전이랑 똑같음.
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            batch_count = 0

            print(f"length of dataloaer : {len(self.dataloader)}")

            for s0, a_seq, s_seq in self.dataloader:
                batch_count += 1
                print(f"batch_count : {batch_count}")
                s0 = s0.to(device)  # (batch_size, state_dim)
                a_seq = a_seq.to(device)  # (batch_size, h * action_dim)
                s_seq = s_seq.to(device)  # (batch_size, h * state_dim)

                optimizer.zero_grad()

                pred_s_seq = []
                s_current = s0

                # 반복적으로 h-step 예측
                for h in range(h_step):
                    # 각 시점의 action 추출
                    a_t = a_seq[:, h * action_dim : (h + 1) * action_dim]
                    # 이거 수정했는데 맞나? hmmm

                    # 입력 차원 확인
                    if h == 0 and epoch == 0 and batch_count == 1:
                        print(
                            f"Step {h} - s_current: {s_current.shape}, a_t: {a_t.shape}"
                        )

                    # 다음 state 예측
                    s_next, _ = self.model(
                        s_current, a_t
                    )  # s_next랑 reward 둘다 output이니까.

                    pred_s_seq.append(s_next)
                    s_current = s_next.detach()  # 다음 입력을 위해 detach

                pred_s_seq = torch.cat(pred_s_seq, dim=1)  # (batch_size, h * state_dim)

                # 출력 차원 확인
                if epoch == 0 and batch_count == 1:
                    print(f"pred_s_seq: {pred_s_seq.shape}, s_seq: {s_seq.shape}")

                ## difference의 difference를 계산하는 방식(https://arxiv.org/pdf/2109.05549)
                loss = 0.0
                for i in range(1, h_step):
                    pred_s_diff = pred_s_seq[:, i] - pred_s_seq[:, i - 1]
                    s_diff = s_seq[:, i] - s_seq[:, i - 1]
                    loss += loss_fn(pred_s_diff, s_diff)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"[User Model Epoch {epoch+1}] Loss: {total_loss:.4f}, Batches: {batch_count}"
            )

        print("Training user model completed.")


class Server:
    def __init__(self, actor, critic, model, datasetNames):
        self.actor = actor
        self.critic = critic
        self.model = model
        self.users = []

        print(f"Initializing server with {len(datasetNames)} datasets")

        # user들 초기화하기
        for i, datasetName in enumerate(datasetNames):
            try:
                print(f"Loading dataset {i}: {datasetName}")
                dataset, state_dim, action_dim, max_action = getDatasetInfo(datasetName)

                print(
                    f"Creating models for user {i} - State dim: {state_dim}, Action dim: {action_dim}"
                )
                user_actor = Actor(state_dim, action_dim, max_action)
                user_critic = Critic(state_dim, action_dim)
                user_model = Model(state_dim, action_dim)

                # 모델 파라미터 초기화 상태 확인
                user_params = sum(p.numel() for p in user_model.parameters())
                print(f"User {i} model has {user_params} parameters")

                user = UserFedRL(user_actor, user_critic, user_model, dataset)
                self.users.append(user)
            except Exception as e:
                print(f"Error initializing user {i}: {e}")

    def train(self):
        print("train start...")

        if not self.users:
            print("No users initialized. Training aborted.")
            return

        try:
            self.distribute_parameters()

            for i, user in enumerate(self.users):
                print(f"Training user {i}...")
                user.train(
                    server_critic=self.critic,
                    h_step=3,
                    num_epochs=10,
                    lr=1e-3,
                    device="cpu",
                )

            print("aggregate start...")
            self.aggregate_parameters()
            self.human_feedback()
            print("train end...")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            traceback.print_exc()

    def distribute_parameters(self):
        """서버 모델 파라미터를 각 유저에게 배포"""
        for user in self.users:
            # Actor 모델 파라미터 복사
            for user_param, server_param in zip(
                user.actor.parameters(), self.actor.parameters()
            ):
                user_param.data.copy_(server_param.data)

            # Critic 모델 파라미터 복사
            for user_param, server_param in zip(
                user.critic.parameters(), self.critic.parameters()
            ):
                user_param.data.copy_(server_param.data)

            # Model 파라미터 복사 (필요한 경우)
            for user_param, server_param in zip(
                user.model.parameters(), self.model.parameters()
            ):
                user_param.data.copy_(server_param.data)

    def aggregate_parameters(self):
        """각 유저 모델의 파라미터를 가중치를 적용하여 서버 모델에 통합"""
        # 가중치 계산
        total_val = sum(np.exp(user.pol_val) for user in self.users)
        weights = [np.exp(user.pol_val) / total_val for user in self.users]

        # 서버 모델 파라미터 초기화
        for param in self.actor.parameters():
            param.data.zero_()
        for param in self.critic.parameters():
            param.data.zero_()

        # 가중 평균으로 파라미터 집계
        for user_idx, user in enumerate(self.users):
            # Actor 파라미터 집계
            for server_param, user_param in zip(
                self.actor.parameters(), user.actor.parameters()
            ):
                server_param.data.add_(user_param.data * weights[user_idx])

            # Critic 파라미터 집계
            for server_param, user_param in zip(
                self.critic.parameters(), user.critic.parameters()
            ):
                server_param.data.add_(user_param.data * weights[user_idx])

    def human_feedback(self):
        """인간 피드백을 통해 사용자 정책 평가값 업데이트"""
        # 실제 구현에서는 인간 피드백을 받아 각 사용자의 pol_val 값을 업데이트
        print("Human feedback simulation: Updating policy values...")
        # 예시로 임의 값 할당
        for i, user in enumerate(self.users):
            user.pol_val = np.random.random() * 2 - 1  # -1과 1 사이의 값

    def extract_trajectories(self):
        """각 사용자 모델에서 궤적 추출"""
        trajectories = []
        # 실제 구현에서는 각 사용자 모델로 환경과 상호작용하여 궤적 생성
        print("Extracting trajectories from user models...")
        return trajectories


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
        self.state_dim = state_dim
        self.action_dim = action_dim  # 모델에 action_dim 저장

        print(f"Creating Model with state_dim={state_dim}, action_dim={action_dim}")

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # next state + reward
        )

    def forward(self, state, action):
        try:
            # 입력 타입과 차원 디버깅
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32)

            # 차원 확인
            if len(state.shape) == 1:
                state = state.unsqueeze(0)  # 단일 상태를 배치 형태로 변환
            if len(action.shape) == 1:
                action = action.unsqueeze(0)  # 단일 액션을 배치 형태로 변환

            # 디버그 출력
            # print(f"Forward - state: {state.shape}, action: {action.shape}")

            x = torch.cat([state, action], dim=-1)
            output = self.net(x)
            next_state = output[:, : self.state_dim]  # 명시적으로 state_dim 사용
            reward = output[:, -1]
            return next_state, reward

        except Exception as e:
            print(f"Error in Model.forward: {e}")
            print(
                f"state: {type(state)}, shape: {state.shape if hasattr(state, 'shape') else 'unknown'}"
            )
            print(
                f"action: {type(action)}, shape: {action.shape if hasattr(action, 'shape') else 'unknown'}"
            )
            raise


class StateActionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = []

        # 구조 파악
        if hasattr(dataset[0], "observations"):
            self.has_observations = True
        elif hasattr(dataset[0], "observation"):
            self.has_observations = False
        else:
            raise ValueError(
                "Dataset must have 'observations' or 'observation' attribute"
            )

        for episode_id in range(len(dataset)):
            if self.has_observations:
                episode_len = len(dataset[episode_id].observations)
            else:
                episode_len = len(dataset[episode_id].observation)

            for idx in range(episode_len - 1):  # 마지막은 다음 상태가 없음
                self.samples.append((episode_id, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_id, index = self.samples[idx]

        if self.has_observations:
            state = self.dataset[episode_id].observations[index]
        else:
            state = self.dataset[episode_id].observation[index]

        action = self.dataset[episode_id].actions[index]

        # numpy → torch 변환
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)

        return state, action


class SequenceTransitionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, h_step):
        self.dataset = dataset
        self.h_step = h_step
        self.samples = []

        # 데이터셋 구조 파악
        if hasattr(dataset[0], "observations"):
            self.has_observations = True
            print("Dataset has 'observations' attribute")
        elif hasattr(dataset[0], "observation"):
            self.has_observations = False
            print("Dataset has 'observation' attribute")
        else:
            raise ValueError(
                "Dataset format not recognized. Need either 'observations' or 'observation' attribute"
            )

        # 샘플 인덱스 생성
        for episode_id in range(len(dataset)):
            if self.has_observations:
                episode_length = len(dataset[episode_id].observations)
            else:
                episode_length = len(dataset[episode_id].observation)

            # 디버그 출력
            if episode_id == 0:
                print(f"Episode {episode_id} length: {episode_length}")

            # index를 설정: 마지막 h_step+1개는 예측 불가능하므로 제외
            for idx in range(episode_length - h_step):
                self.samples.append((episode_id, idx))

        print(
            f"Created {len(self.samples)} transition samples from {len(dataset)} episodes"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        episode_id, index = self.samples[idx]

        # 데이터셋 구조에 따라 적절히 데이터 추출
        if self.has_observations:
            state_t = self.dataset[episode_id].observations[index]
        else:
            state_t = self.dataset[episode_id].observation[index]

        action_seq = []
        next_state_seq = []

        for i in range(self.h_step):
            action = self.dataset[episode_id].actions[index + i]

            if self.has_observations:
                next_state = self.dataset[episode_id].observations[index + i + 1]
            else:
                next_state = self.dataset[episode_id].observation[index + i + 1]

            # 텐서 변환 시 차원 유지를 위한 처리
            action_tensor = torch.tensor(action, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            action_seq.append(action_tensor)
            next_state_seq.append(next_state_tensor)

        # 데이터의 형태 확인 및 디버깅
        if idx == 0:
            print(
                f"First sample - state shape: {state_t.shape if hasattr(state_t, 'shape') else 'unknown'}"
            )
            print(f"First action tensor shape: {action_seq[0].shape}")

        # 액션과 상태 시퀀스 결합
        try:
            action_seq = torch.cat(action_seq)
            next_state_seq = torch.cat(next_state_seq)
        except RuntimeError as e:
            print(f"Error concatenating sequences: {e}")
            print(f"Action seq shapes: {[a.shape for a in action_seq]}")
            print(f"State seq shapes: {[s.shape for s in next_state_seq]}")
            raise

        return torch.tensor(state_t, dtype=torch.float32), action_seq, next_state_seq


def getDatasetInfo(datasetName):
    # 데이터셋 로드 시 예외 처리
    try:
        dataset = minari.load_dataset(datasetName)
        print(f"Dataset '{datasetName}' loaded successfully")

        # 데이터셋 기본 정보 출력
        print(f"Dataset contains {len(dataset)} episodes")

        # 첫 번째 에피소드 정보 확인
        if len(dataset) > 0:
            first_episode = dataset[0]
            has_observations = hasattr(first_episode, "observations")
            obs_attr = "observations" if has_observations else "observation"

            if hasattr(first_episode, obs_attr):
                first_obs = getattr(first_episode, obs_attr)
                print(f"First episode has {len(first_obs)} observations")
                print(f"Observation shape: {first_obs[0].shape}")

                if hasattr(first_episode, "actions"):
                    print(f"Action shape: {first_episode.actions[0].shape}")

        # 상태와 행동 공간 차원 추출
        if hasattr(dataset, "observation_space") and hasattr(
            dataset.observation_space, "shape"
        ):
            state_dim = dataset.observation_space.shape[0]  # state 차원
        else:
            # 관측 공간 정보가 없다면 첫 번째 에피소드에서 추출
            first_obs = getattr(dataset[0], obs_attr)[0]
            state_dim = first_obs.shape[0]

        print(f"State dimension: {state_dim}")

        if hasattr(dataset, "action_space") and hasattr(dataset.action_space, "shape"):
            action_dim = dataset.action_space.shape[0]  # action 차원
        else:
            # 행동 공간 정보가 없다면 첫 번째 에피소드에서 추출
            action_dim = dataset[0].actions[0].shape[0]

        print(f"Action dimension: {action_dim}")

        # 행동 공간의 범위 확인
        if hasattr(dataset, "action_space") and hasattr(dataset.action_space, "high"):
            max_action = float(dataset.action_space.high[0])
        else:
            # 기본값 설정
            max_action = 1.0

        print(f"Max action value: {max_action}")

        return dataset, state_dim, action_dim, max_action

    except Exception as e:
        print(f"Error loading dataset '{datasetName}': {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    ## 데이터셋 이름들 명시하기
    datasetNames = [
        "mujoco/inverteddoublependulum/expert-v0",
        "mujoco/inverteddoublependulum/medium-v0",
    ]

    ## 공통 정보 추출 (서버/클라이언트 공유용)
    dataset_init, state_dim, action_dim, max_action = getDatasetInfo(datasetNames[0])

    ## Server Initialization ##
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)
    model = Model(state_dim, action_dim)

    # 서버 초기화 및 학습 진행
    server = Server(actor, critic, model, datasetNames)
    server.train()
    print("학습 완료!")
