import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch import distributed as dist
from openrlhf.trainer import DPOTrainer


class DRDPOLoss(nn.Module):
    """
    DRDPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False, drdpo_beta : float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.drdpo_beta = drdpo_beta

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
        loss = - self.drdpo_beta * torch.log(torch.mean(torch.exp( - losses / self.drdpo_beta)))
        #############################################################################################
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class DRDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = DRDPOLoss(self.beta, self.args.label_smoothing, self.args.ipo, self.args.drdpo_beta)
