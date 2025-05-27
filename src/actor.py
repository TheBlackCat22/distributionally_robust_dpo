import torch
import torch.distributed as dist
from openrlhf.models import Actor
from typing import Optional, Union
from torch.nn import functional as F
from openrlhf.models.ring_attn_utils import convert_ring_attn_params
from openrlhf.models.utils import log_probs_from_logits, reset_position_ids

class Actor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        logps_allgather=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            # convert attention_mask to position_ids
            if ring_attn_group is not None:
                labels = sequences
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None
        
        #####################################################################################################
        inputs_embeds = self.model.get_input_embeddings()(sequences)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids)
        output['embeddings'] = inputs_embeds
        #####################################################################################################
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if num_actions is None:
            assert return_output
            return output

        if not self.packing_samples:
            log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
            action_log_probs = log_probs[:, -num_actions:]
        else:
            if ring_attn_group is not None and logps_allgather:
                rank = dist.get_rank(ring_attn_group)
                ring_attn_size = dist.get_world_size(ring_attn_group)
                total_seq_len = labels.numel()
                local_seq_len = total_seq_len // ring_attn_size
                local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
                local_label = labels[:, local_slice]
                if rank == ring_attn_size - 1:
                    # add a dummy label to the last logit
                    local_label = F.pad(local_label, (0, 1), value=0)
                local_per_token_logps = torch.gather(
                    output["logits"].log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
                ).squeeze(2)
                per_token_logps = all_gather(local_per_token_logps, ring_attn_group).reshape((1, -1))
                log_probs = per_token_logps[:, :-1]
            else:
                log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs
    