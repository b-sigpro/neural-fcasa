import torch


class ApproxBernoulli(torch.distributions.RelaxedBernoulli):
    def rsample(self, sample_shape=torch.Size()):  # noqa
        x = super().rsample(sample_shape)
        return x - x.detach() + (x > 0.5).to(x.dtype)
