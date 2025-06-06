import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F

def compute_rl_sft_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask,  
    rl_mask,
    sft_mask, 
    sft_weight = 1.0,
    do_rl_clip = False,
    rl_clip_upper_bound = None,
    rl_clip_low_bound = None,
    do_sft_clip = False,
    sft_clip_upper_bound = None,
    sft_clip_low_bound = None,
    on_policy_reshape="no_reshape", 
):
    #breakpoint()
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask * rl_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    # rl 
    on_pg_losses = -advantages * ratio
    if do_rl_clip:
        on_pg_losses2 = -advantages * torch.clamp(ratio, rl_clip_low_bound, rl_clip_upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, rl_mask * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, rl_mask * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)

    if on_pg_loss.isnan().any().item():
        on_pg_loss = torch.tensor(0.0)

    # sft
    sft_losses = -1 * log_prob
    if do_sft_clip:
        sft_losses2 = -1 * torch.clamp(log_prob, sft_clip_low_bound, sft_clip_upper_bound)
        sft_clipfrac = verl_F.masked_mean(torch.gt(sft_losses2, sft_losses).float(), eos_mask)
        sft_losses = torch.max(sft_losses, sft_losses2)
        sft_loss = verl_F.masked_mean(sft_losses, sft_mask * eos_mask)
    else:
        sft_loss = verl_F.masked_mean(sft_losses, sft_mask * eos_mask)
        sft_clipfrac = torch.tensor(0.0)
    
    if sft_loss.isnan().any().item():
        sft_loss = torch.tensor(0.0)

    losses = rl_mask * on_pg_losses + sft_mask * sft_losses
    loss = (losses * eos_mask).sum() / eos_mask.shape[-1]

    return {
        "loss": loss,
        "sft_loss": sft_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": sft_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl
    }

    

def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False,
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability

    #breakpoint()
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)
    if upper_bound == clip_upper_bound:
        print('clip upper bound is used: ', clip_upper_bound)

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        elif off_policy_reshape == "p_div_p_0.01":
            off_ratio = off_ratio / (off_ratio + 0.01)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }

