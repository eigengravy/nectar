import copy
import time
import flwr as fl

from hydra.utils import instantiate
from nectar.utils.params import get_params, set_params


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        trainloader,
        valloader,
        model,
        optimizer,
        criterion,
        distiller,
        device,
        train_fn,
        mi_fn,
        test_fn,
    ):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.distiller = distiller
        self.device = device
        self.train_fn = train_fn
        self.mi_fn = mi_fn
        self.test_fn = test_fn
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        epochs = config["epochs"]
        teacher = copy.deepcopy(self.model)
        start_time = time.time()
        results = instantiate(
            self.train_fn,
            self.model,
            teacher,
            self.optimizer,
            self.criterion,
            self.distiller,
            self.trainloader,
            epochs,
            self.device,
        )
        end_time = time.time()

        mi_type = config["mi_type"]
        mi = instantiate(
            self.mi_fn, self.model, teacher, self.valloader, mi_type, device=self.device
        )
        results["mi"] = mi
        results["t_diff"] = end_time - start_time

        if (
            config["opti_mifl"] == 1
            and mi > config["lower_bound"]
            and mi < config["upper_bound"]
        ):
            return get_params(self.model), len(self.trainloader.dataset), results
        elif config["opti_mifl"] == 0:
            return get_params(self.model), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        loss, accuracy = instantiate(
            self.test_fn, self.model, self.valloader, device=self.device
        )
        return (
            float(loss),
            len(self.valloader.dataset),
            {"accuracy": float(accuracy), "loss": float(loss)},
        )
