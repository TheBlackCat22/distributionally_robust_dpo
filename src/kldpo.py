import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch import distributed as dist
from openrlhf.trainer import DPOTrainer


class KLDPOLoss(nn.Module):
    """
    KLDPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False, kldpo_tau : float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.kldpo_tau = kldpo_tau

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        
        #############################################################################################
        all_losses = [torch.zeros_like(losses) for _ in range(dist.get_world_size())]
        dist.all_gather(all_losses, losses.detach())
        all_losses = torch.cat(all_losses)

        ps = torch.exp((losses - all_losses.mean()) / self.kldpo_tau).clamp(max=10).detach()
        loss = (ps * losses).mean()
        #############################################################################################
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class KLDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = KLDPOLoss(self.beta, self.args.label_smoothing, self.args.ipo, self.args.kldpo_tau)
