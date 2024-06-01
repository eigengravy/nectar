from flwr.client import ClientApp, NumPyClient

from task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


class FlowerClient(NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self.net = Net().to(DEVICE)
        self.trainloader, self.testloader = load_data()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net, self.trainloader, self.testloader, epochs=1, device=DEVICE
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
