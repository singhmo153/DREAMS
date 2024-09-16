import pdb
from torch import nn
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights=None, class_weights=None):
        super(MultiTaskLoss, self).__init__()
        # If task_weights is not provided, assign equal weight to all tasks
        if task_weights is None:
            self.task_weights = [1.0, 1.0]
        else:
            self.task_weights = task_weights

        if class_weights is None:
        # Define the loss functions for each task
            self.loss_taskA = nn.CrossEntropyLoss()
            self.loss_taskB = nn.CrossEntropyLoss()
        else:
            self.loss_taskA = nn.CrossEntropyLoss(weight=class_weights[0].cuda())
            self.loss_taskB = nn.CrossEntropyLoss(weight=class_weights[1].cuda())

    def forward(self, outputs, targets_taskA, targets_taskB):
        # Split the outputs into individual predictions for each task
        predictions_taskA, predictions_taskB = outputs

        # Compute the individual losses for each task
        loss_taskA = self.loss_taskA(predictions_taskA, targets_taskA)
        loss_taskB = self.loss_taskB(predictions_taskB, targets_taskB)

        # Combine the losses using the provided task weights
        combined_loss = self.task_weights[0] * loss_taskA + self.task_weights[1] * loss_taskB

        return combined_loss