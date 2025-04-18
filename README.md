<h1 align="center">FreSca: Unveiling the Scaling Space in Diffusion Models</h1>
<h5 align="center" style="color:gray">
  <a href="https://wikichao.github.io/" target="_blank" rel="noopener noreferrer">Chao Huang</a>, 
  <a href="https://liangsusan-git.github.io/" target="_blank" rel="noopener noreferrer">Susan Liang</a>, 
  <a href="https://yunlong10.github.io/" target="_blank" rel="noopener noreferrer">Yunlong Tang</a>, 
  <a href="https://limacv.github.io/homepage/" target="_blank" rel="noopener noreferrer">Li Ma</a>, 
  <a href="https://www.yapengtian.com/" target="_blank" rel="noopener noreferrer">Yapeng Tian</a>, 
  <a href="https://www.cs.rochester.edu/~cxu22/" target="_blank" rel="noopener noreferrer">Chenliang Xu</a><br>
</h5>
<h5 align="center" style="color:gray">
  University of Rochester, Netflix Eyeline Studios, The University of Texas at Dallas
</h5>
<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. </h5>

<h5 align="center">
<a href="https://wikichao.github.io/FreSca/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>  <a href="https://arxiv.org/pdf/2504.02154"><img src="https://img.shields.io/badge/arXiv-FreSca-b31b1b.svg" height=20.5></a> <a href="https://huggingface.co/papers/2504.02154"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Paper&color=green" height=20.5></a> 

</h5>

<p align="center">
  <img src="assets/teaser.png" alt="FreSca teaser figure" width="100%">
</p>

## 📰 News

- **[2025-04]** 🔥🔥 Exciting discovery! **FreSca** boosts video diffusion models' performance without training! Preliminary [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter) results added. Suggestions for new tasks are welcome—gallery updates coming soon 🚀🚀

* **[2025.04]** 🔥🔥 Released example implemenation for **FreSca**.

## 🎬 Gallery

### Video Generation Results

<table>
<tr>
    <td colspan="3" align="center"><b>VideoCrafter2 (Original)</b></td>
</tr>
<tr>
    <td><a href="demo/VideoCrafter2/Origin/0001.mp4"><img src="demo/VideoCrafter2/Origin/0001.gif" width="100%"></a></td>
    <td><a href="demo/VideoCrafter2/Origin/0004.mp4"><img src="demo/VideoCrafter2/Origin/0004.gif" width="100%"></a></td>
    <td><a href="demo/VideoCrafter2/Origin/0005.mp4"><img src="demo/VideoCrafter2/Origin/0005.gif" width="100%"></a></td>
</tr>
<tr>
    <td colspan="3" align="center"><b>VideoCrafter2 + FreSca</b></td>
</tr>
<tr>
    <td><a href="demo/VideoCrafter2/Fresca/0001.mp4"><img src="demo/VideoCrafter2/Fresca/0001.gif" width="100%"></a></td>
    <td><a href="demo/VideoCrafter2/Fresca/0004.mp4"><img src="demo/VideoCrafter2/Fresca/0004.gif" width="100%"></a></td>
    <td><a href="demo/VideoCrafter2/Fresca/0005.mp4"><img src="demo/VideoCrafter2/Fresca/0005.gif" width="100%"></a></td>
</tr>
<tr>
    <td align="center"><i>"A tiger walks in the forest, photorealistic, 4k, high definition"</i></td>
    <td align="center"><i>"A child excitedly swings on a rusty swing set, laughter filling the air"</i></td>
    <td align="center"><i>"A young woman with glasses is jogging in the park wearing a pink headband"</i></td>
</tr>
</table>

<p align="left"><i>Click on any GIF to view the full MP4 video.</i></p>

## 🖥️ Demo

* You can try the demo applications in ``demo/*``.

## 🤖 FreSca Implementation
```python
import torch
import torch.fft as fft

def Fourier_filter(x, scale_low=1.0, scale_high=1.5, freq_cutoff=20):
    """
    Apply frequency-dependent scaling to an image tensor using Fourier transforms.
    
    Parameters:
        x:           Input tensor of shape (B, C, H, W)
        scale_low:   Scaling factor for low-frequency components (default: 1.0)
        scale_high:  Scaling factor for high-frequency components (default: 1.5)
        freq_cutoff: Number of frequency indices around center to consider as low-frequency (default: 20)
    
    Returns:
        x_filtered: Filtered version of x in spatial domain with frequency-specific scaling applied.
    """
    # Preserve input dtype and device
    dtype, device = x.dtype, x.device
    
    # Convert to float32 for FFT computations
    x = x.to(torch.float32)
    
    # 1) Apply FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # 2) Create a mask to scale frequencies differently
    B, C, H, W = x_freq.shape
    crow, ccol = H // 2, W // 2
    
    # Initialize mask with high-frequency scaling factor
    mask = torch.ones((B, C, H, W), device=device) * scale_high
    
    # Apply low-frequency scaling factor to center region
    mask[..., crow - freq_cutoff : crow + freq_cutoff,
         ccol - freq_cutoff : ccol + freq_cutoff] = scale_low

    # 3) Apply frequency-specific scaling
    x_freq = x_freq * mask
    
    # 4) Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    # 5) Restore original dtype
    x_filtered = x_filtered.to(dtype)
    
    return x_filtered
```

## 🚀 Integration with Diffusion Models
To integrate FreSca into your diffusion model pipeline, apply the Fourier filter to the noise prediction during the denoising process:

```python
class DiffusionModel:
    def denoise_step(self, x_t, t, guidance_scale=7.5):
        # Standard diffusion model denoising step
        
        # For classifier-free guidance
        if guidance_scale > 1.0:
            # Get conditional and unconditional noise predictions
            cond_noise_pred = self.unet(x_t, t, context)
            uncond_noise_pred = self.unet(x_t, t, null_context)
            
            # Apply FreSca to the guidance component
            noise_diff = cond_noise_pred - uncond_noise_pred
            noise_diff = Fourier_filter(
                noise_diff, 
                scale_low=1.0,   # Preserve low frequencies
                scale_high=1.5,  # Boost high frequencies
                freq_cutoff=20
            )
            
            # Combine predictions with guidance scale
            noise_pred = uncond_noise_pred + guidance_scale * noise_diff
        else:
            # Single forward pass
            noise_pred = self.unet(x_t, t, context)
            noise_pred = Fourier_filter(noise_pred)
        
        # Continue with the rest of the denoising step...
```

## ✅ ToDo
- [x] Add implementation for the core algorithm
- [x] Update demo for image editing (e.g., LEdits++)
- [x] Update demo for diffusion-based image depth estimation (e.g., Marigold)
- [x] Add gallery section with VideoCrafter2 video generation results

- [ ] Include more applications...

## 📑 Citation
If you use this code for your research, please cite our work:
```
@article{huang2025fresca,
        title={FreSca: Unveiling the Scaling Space in Diffusion Models},
        author={Huang, Chao and Liang, Susan and Tang, Yunlong and Ma, Li and Tian, Yapeng and Xu, Chenliang},
        journal={arXiv preprint arXiv:2504.02154},
        year={2025}
}
```
