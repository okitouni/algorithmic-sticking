import os
import shutil
import torch

# Logger class to handle logging of metrics and model saving.
class Logger:
    def __init__(self, name):
        # Create directories for log files and model files
        root = os.path.join("log", name)
        models = os.path.join(root, "models")
        os.makedirs(root, exist_ok=True)
        if os.path.exists(models):
            shutil.rmtree(models)
        os.makedirs(models, exist_ok=True)

        # Create a metrics file with column headers
        self.metrics_file = open(os.path.join(root, "metrics.csv"), "w")
        self.metrics_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        self.model_file = os.path.join(models, "{}.pt")

    # Log method for the Logger class that logs the metrics and saves the model if provided.
    def log(self, model=None, **metrics):
        epoch = metrics.get("epoch", -1)
        train_loss = metrics.get("train_loss", -1)
        train_acc = metrics.get("train_acc", -1)
        val_loss = metrics.get("val_loss", -1)
        val_acc = metrics.get("val_acc", -1)

        self.metrics_file.write("{},{},{},{},{}\n".format(epoch, train_loss, train_acc, val_loss, val_acc))

        if model is not None:
            torch.save(model.state_dict(), self.model_file.format(epoch))

