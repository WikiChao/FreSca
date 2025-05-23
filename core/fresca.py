import torch
from torch import fft
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

def Fourier_filter(
    x: torch.Tensor, 
    scale_low: float = 1.0, 
    scale_high: float = 1.5, 
    freq_cutoff: int = 20
) -> torch.Tensor:
    """
    Apply frequency-dependent scaling to a tensor using Fourier transforms.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        scale_low: Scaling factor for low-frequency components
        scale_high: Scaling factor for high-frequency components
        freq_cutoff: Number of frequency indices around center to consider as low-frequency
    
    Returns:
        Filtered tensor with frequency-specific scaling applied
    """
    # Validate inputs
    if x.dim() < 4:
        raise ValueError(f"Expected 4D input tensor (B,C,H,W), got shape {x.shape}")
    
    # Preserve input properties
    dtype, device = x.dtype, x.device
    
    # Use torch.autocast for mixed precision when beneficial
    with torch.autocast(device_type=device.type, enabled=False):
        x = x.to(torch.float32)
        
        # 1) FFT → shift to center
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))

        B, C, H, W = x_freq.shape
        crow, ccol = H // 2, W // 2
        
        # Ensure freq_cutoff is within bounds
        freq_cutoff = min(freq_cutoff, min(crow, ccol))

        # 2) Build scaling mask (pre-allocate on device)
        mask = torch.full((B, C, H, W), scale_high, device=device, dtype=torch.float32)
        
        # 3) Apply low-frequency scaling (in-place operation)
        mask[..., crow - freq_cutoff:crow + freq_cutoff,
                ccol - freq_cutoff:ccol + freq_cutoff] = scale_low

        # 4) Apply mask and convert back to spatial domain
        x_freq.mul_(mask)  # in-place multiplication
        x_filtered = fft.ifftn(fft.ifftshift(x_freq, dim=(-2, -1)), dim=(-2, -1)).real

    # Restore original dtype
    return x_filtered.to(dtype)


def frequency_filter(
    tensor: torch.Tensor,
    threshold: float = 0.2,
    scale_low: float = 1.0,
    scale_high: float = 1.5,
    max_cutoff: Optional[int] = None
) -> torch.Tensor:
    """
    Dynamic frequency-domain filter with adaptive cutoff selection.
    
    Args:
        tensor: Input tensor of shape (B, C, H, W)
        threshold: Energy threshold for determining frequency cutoff (0.0-1.0)
        scale_low: Scaling factor for low-frequency components
        scale_high: Scaling factor for high-frequency components
        max_cutoff: Optional maximum cutoff frequency (defaults to min(H,W)/4)
    
    Returns:
        Filtered tensor with adaptive frequency cutoff
    """
    if not (0.0 <= threshold <= 1.0):
        logger.warning(f"Threshold {threshold} outside [0,1] range, clipping.")
        threshold = max(0.0, min(threshold, 1.0))
        
    # 1) Compute shifted FFT
    x_freq = fft.fftn(tensor, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2
    
    # Maximum allowed cutoff value
    if max_cutoff is None:
        max_cutoff = min(H, W) // 4
    
    # 2) Calculate 1D magnitude spectrum along central row & column
    # Average of horizontal and vertical directions for robustness
    mag_h = x_freq.abs().mean(dim=(0, 1))[crow, :]  # shape [W]
    mag_v = x_freq.abs().mean(dim=(0, 1))[:, ccol]  # shape [H]
    
    # 3) Compute average cutoff from both dimensions
    cutoff_h = _get_energy_cutoff(mag_h, threshold, max_cutoff)
    cutoff_v = _get_energy_cutoff(mag_v, threshold, max_cutoff)
    
    # Average the cutoffs from both directions
    freq_cutoff = max(3, (cutoff_h + cutoff_v) // 2)  # Ensure minimum cutoff of 3
    
    # 4) Apply Fourier filter with the dynamic cutoff
    logger.debug(f"Using frequency cutoff: {freq_cutoff} (threshold={threshold})")
    return Fourier_filter(
        tensor,
        scale_low=scale_low,
        scale_high=scale_high,
        freq_cutoff=freq_cutoff
    )


def _get_energy_cutoff(magnitude: torch.Tensor, threshold: float, max_cutoff: int) -> int:
    """Helper to find energy-based cutoff point in a 1D magnitude spectrum"""
    energy = magnitude
    cum_energy = torch.cumsum(energy, dim=0)
    total_energy = cum_energy[-1]
    
    # Find where cumulative energy exceeds the threshold
    cutoff_idxs = torch.nonzero(cum_energy >= threshold * total_energy, as_tuple=False)
    
    if cutoff_idxs.numel() > 0:
        cutoff = int(cutoff_idxs[0, 0].item())
        return min(cutoff, max_cutoff)
    else:
        # Fallback to max_cutoff if threshold not met
        return max_cutoff


def apply_fresca(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    guidance_scale: float = 7.5,
    energy_threshold: float = 0.2,
    scale_low: float = 1.0, 
    scale_high: float = 1.5
) -> torch.Tensor:
    """
    Apply FreSca to classifier-free guidance in diffusion models.
    
    Args:
        cond_pred: Conditional prediction from UNet
        uncond_pred: Unconditional prediction from UNet
        guidance_scale: CFG scale factor
        energy_threshold: Energy threshold for frequency cutoff
        scale_low: Scaling factor for low-frequency components
        scale_high: Scaling factor for high-frequency components
        
    Returns:
        Combined noise prediction with FreSca applied
    """
    # Calculate guidance difference
    noise_diff = cond_pred - uncond_pred
    
    # Apply FreSca with dynamic frequency cutoff
    scaled_diff = frequency_filter(
        noise_diff,
        threshold=energy_threshold,
        scale_low=scale_low,
        scale_high=scale_high
    )
    
    # Combine predictions with guidance scale
    return uncond_pred + guidance_scale * scaled_diff


# Example usage:
"""
# Basic integration example
def model_forward_with_fresca(x_t, t, prompt, guidance_scale=7.5):
    # Get conditional and unconditional predictions
    with torch.no_grad():
        # Conditional (with prompt)
        cond_out = model(x_t, t, prompt)
        
        # Unconditional (empty prompt)
        uncond_out = model(x_t, t, "")
    
    # Apply FreSca
    return apply_fresca(
        cond_out,
        uncond_out,
        guidance_scale=guidance_scale,
        energy_threshold=0.2,  # Adjust based on your model
        scale_low=1.0,
        scale_high=1.5
    )
"""
