import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
import torch.nn.functional as F
from openrlhf.trainer import DPOTrainer
from openrlhf.utils.distributed_sampler import DistributedSampler


class WDPOLoss(nn.Module):
    """
    WDPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False, wdpo_rho: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.wdpo_rho = wdpo_rho

    #############################################################################################
    @torch.enable_grad()
    #############################################################################################
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        model_embeddings: torch.Tensor,
        train_eval: bool
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
        grads = torch.autograd.grad(
            outputs=losses,
            inputs=model_embeddings,
            grad_outputs=torch.ones_like(losses),
            create_graph=train_eval=='train'
        )[0]
        grads = grads.pow(2).sum(dim=(-1, -2))
        grads = grads[: len(grads) // 2] + grads[len(grads) // 2 : ]
        losses = losses + self.wdpo_rho*grads
        #############################################################################################

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class WDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = WDPOLoss(self.beta, self.args.label_smoothing, self.args.ipo, self.args.wdpo_rho)
    
    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.ref_model.eval()
            # train
            for data in self.train_dataloader:
                if not self.packing_samples:
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    #####################################################################################################
                    chosen_logps, rejected_logps, aux_loss, nll_loss, model_embeddings = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    #####################################################################################################
                    with torch.no_grad():
                        #####################################################################################################
                        reference_chosen_logps, reference_rejected_logps, _, _, _ = self.concatenated_forward(
                            self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                        )
                        #####################################################################################################
                else:
                    packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                    packed_input_ids, packed_attention_masks = packed_input_ids.to(
                        torch.cuda.current_device()
                    ), packed_attention_masks.to(torch.cuda.current_device())
                    #####################################################################################################
                    chosen_logps, rejected_logps, aux_loss, nll_loss, model_embeddings = self.packed_samples_forward(
                        self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    #####################################################################################################
                    with torch.no_grad():
                        #####################################################################################################
                        reference_chosen_logps, reference_rejected_logps, _, _, _ = self.packed_samples_forward(
                            self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                        )
                        #####################################################################################################

                # loss function
                #####################################################################################################
                preference_loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, model_embeddings, 'train'
                )
                #####################################################################################################
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef + nll_loss * self.args.nll_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # dpo logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of global_step %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
        loss_sum = 0
        times = 0
        for data in eval_dataloader:
            if not self.packing_samples:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                #############################################################################################
                chosen_logps, rejected_logps, aux_loss, _, model_embeddings = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                #############################################################################################
                with torch.no_grad():
                    #############################################################################################
                    reference_chosen_logps, reference_rejected_logps, _, _, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                    )
                    #############################################################################################
            else:
                packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
                packed_input_ids, packed_attention_masks = packed_input_ids.to(
                    torch.cuda.current_device()
                ), packed_attention_masks.to(torch.cuda.current_device())
                #############################################################################################
                chosen_logps, rejected_logps, aux_loss, _, model_embeddings = self.packed_samples_forward(
                    self.model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                )
                #############################################################################################
                with torch.no_grad():
                    #############################################################################################
                    reference_chosen_logps, reference_rejected_logps, _, _, _ = self.packed_samples_forward(
                        self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens
                    )
                    #############################################################################################

            #############################################################################################
            loss, chosen_reward, reject_reward = self.loss_fn(
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, model_embeddings, 'test'
            )
            #############################################################################################
            acc_sum += (chosen_reward > reject_reward).float().mean().item()
            loss_sum += loss.item()
            times += 1
            step_bar.update()

        logs = {
            "eval_loss": loss_sum / times,
            "acc_mean": acc_sum / times,
        }
        logs = self.strategy.all_reduce(logs)
        step_bar.set_postfix(logs)

        if self.strategy.is_rank_0():
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        #####################################################################################################
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean(), output["embeddings"]
        #####################################################################################################
    
    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        chosen_logps = all_logps_sum[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps_sum[len(packed_seq_lens) // 2 :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        #####################################################################################################
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: len(packed_seq_lens) // 2].mean(), output["embeddings"]
        #####################################################################################################
