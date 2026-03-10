import flwr as fl
import torch
from model import Net


ALPHA = 0.000001
BETA = 0.0000005


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader):

        self.model = Net()
        self.trainloader = trainloader

        self.compute_energy = 0
        self.comm_energy = 0

    def get_parameters(self, config):

        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        training_steps = 0

        for data, target in self.trainloader:

            optimizer.zero_grad()

            output = self.model(data)

            loss = criterion(output, target)

            loss.backward()

            optimizer.step()

            training_steps += 1

        # compute energy
        cpu_cycles = training_steps * 1000
        compute_energy = cpu_cycles * ALPHA

        # communication energy
        model_size = sum(p.numel() for p in self.model.parameters())
        comm_energy = model_size * BETA

        total_energy = compute_energy + comm_energy

        metrics = {
            "compute_energy": compute_energy,
            "comm_energy": comm_energy,
            "total_energy": total_energy,
        }

        return self.get_parameters({}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        return 0.0, len(self.trainloader.dataset), {}