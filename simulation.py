import flwr as fl
import torch
import matplotlib.pyplot as plt
import random

from client import FlowerClient
from dataset import load_datasets
from model import Net


NUM_CLIENTS = 50
NUM_ROUNDS = 100


compute_energy_log = []
communication_energy_log = []
total_energy_log = []
accuracy_history = []


trainloaders, testloader = load_datasets(NUM_CLIENTS)


# ------------------------------------------------
# Heterogeneous IoT Device Simulation
# ------------------------------------------------

device_types = ["sensor", "mobile", "edge"]

client_profiles = {}

for cid in range(NUM_CLIENTS):

    device = random.choice(device_types)

    if device == "sensor":
        profile = {
            "cpu_factor": 0.5,
            "compression": 0.4
        }

    elif device == "mobile":
        profile = {
            "cpu_factor": 0.8,
            "compression": 0.7
        }

    else:
        profile = {
            "cpu_factor": 1.2,
            "compression": 1.0
        }

    client_profiles[cid] = profile


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

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

    return correct / total


def evaluate(server_round, parameters, config):

    accuracy = test_model(parameters)

    accuracy_history.append(accuracy)

    print(f"Round {server_round} Accuracy: {accuracy}")

    return 0.0, {"accuracy": accuracy}


# ------------------------------------------------
# Client creation
# ------------------------------------------------

def client_fn(cid: str):

    cid = int(cid)

    return FlowerClient(trainloaders[cid], client_profiles[cid])


# ------------------------------------------------
# Strategy (ONLY ENERGY TRACKING)
# ------------------------------------------------

class EnergyStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(self, server_round, results, failures):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        round_compute = 0
        round_comm = 0
        round_total = 0

        for _, fit_res in results:

            round_compute += fit_res.metrics.get("compute_energy", 0)
            round_comm += fit_res.metrics.get("communication_energy", 0)
            round_total += fit_res.metrics.get("total_energy", 0)

        compute_energy_log.append(round_compute)
        communication_energy_log.append(round_comm)
        total_energy_log.append(round_total)

        print(f"Round {server_round}")
        print(f"Compute Energy: {round_compute}")
        print(f"Communication Energy: {round_comm}")
        print(f"Total Energy: {round_total}")

        return aggregated_parameters, aggregated_metrics


strategy = EnergyStrategy(
    fraction_fit=0.2,
    min_fit_clients=10,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=evaluate
)


# ------------------------------------------------
# Start Simulation
# ------------------------------------------------

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)


# ------------------------------------------------
# Plots
# ------------------------------------------------

plt.figure()
plt.plot(accuracy_history)
plt.title("Accuracy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.show()


plt.figure()
plt.plot(compute_energy_log)
plt.title("Computation Energy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Energy")
plt.show()


plt.figure()
plt.plot(communication_energy_log)
plt.title("Communication Energy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Energy")
plt.show()


plt.figure()
plt.plot(total_energy_log)
plt.title("Total Energy vs Rounds")
plt.xlabel("Round")
plt.ylabel("Energy")
plt.show()