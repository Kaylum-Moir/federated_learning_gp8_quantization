import flwr as fl
import torch
import matplotlib.pyplot as plt

from client import FlowerClient
from dataset import load_datasets
from model import Net


NUM_CLIENTS = 50
NUM_ROUNDS = 100


trainloaders, testloader = load_datasets(NUM_CLIENTS)


accuracy_history = []
energy_history = []


def test_model(parameters):

    model = Net()

    params_dict = zip(model.state_dict().keys(), parameters)

    state_dict = {k: torch.tensor(v) for k, v in params_dict}

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for data, target in testloader:

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)

            correct += (predicted == target).sum().item()

    accuracy = correct / total

    return accuracy


def evaluate(server_round, parameters, config):

    accuracy = test_model(parameters)

    print(f"Round {server_round} Accuracy: {accuracy}")

    accuracy_history.append(accuracy)

    return 0.0, {"accuracy": accuracy}


def client_fn(cid: str):

    cid = int(cid)

    return FlowerClient(trainloaders[cid])


class EnergyStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        total_energy = 0

        for _, fit_res in results:

            if "total_energy" in fit_res.metrics:

                total_energy += fit_res.metrics["total_energy"]

        energy_history.append(total_energy)

        print(f"Round {server_round} Energy: {total_energy}")

        return aggregated_parameters, aggregated_metrics


strategy = EnergyStrategy(
    fraction_fit=0.2,
    min_fit_clients=10,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate,
)


fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)


# Plot results

plt.figure()

plt.plot(accuracy_history)

plt.title("Accuracy vs Rounds")

plt.xlabel("Round")

plt.ylabel("Accuracy")

plt.show()


plt.figure()

plt.plot(energy_history)

plt.title("Energy Consumption vs Rounds")

plt.xlabel("Round")

plt.ylabel("Energy")

plt.show()