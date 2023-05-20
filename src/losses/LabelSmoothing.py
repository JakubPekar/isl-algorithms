import torch
import torch.nn



class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super(LabelSmoothing, self).__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing



    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(prediction, dim=-1)
        nll_loss = -logprobs.gather(-1, target.to(torch.long))
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
