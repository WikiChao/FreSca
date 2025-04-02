# FreSca + Marigold: Enhanced Depth Estimation

This demo showcases how FreSca can be integrated with Marigold to improve depth estimation quality through frequency-domain scaling.

## 🚀 Quick Start

### Option 1: Demo

We've integrated FreSca into the Marigold diffusers pipeline for easy experimentation:

```bash
# Run the demo with FreSca enabled
python marigold_diffuser.py
```

### 🤖 Usage

**We offer a simple way to start the demo with Marigold**:

Thanks to the [Marigold Pipelines into diffusers 🧨](https://huggingface.co/docs/diffusers/api/pipelines/marigold), we integrated our method into the Marigold code. To toggle FreSca on/off, modify line 761 in marigold_diffuser.py:
```
# Enable FreSca frequency scaling
frequency_scaling=True  # Set to False to disable
```
For additional Marigold usage examples, see the [diffusers tutorial](https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage). Try inserting our method or not to the Marigold and explore the difference.

### Option 2: Quantitative Evaluation
To reproduce our benchmark results:


1. Follow the setup instructions in the [official repo](https://github.com/prs-eth/Marigold/blob/main/README.md) to prepare the environment and datasets.
2. Run the Marigold evaluation scripts.


### 📊 Results Comparison

We evaluate FreSca integration with Marigold across three standard depth estimation benchmarks. The results demonstrate consistent improvements in both absolute relative error (AbsRel) and accuracy (δ1).

| Method | DIODE |  | KITTI |  | ETH3D |  |
|:-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** |
| Marigold (w/o ensemble) | 31.0 | 77.2 | 10.5 | 90.4 | 7.1 | 95.1 |
| Marigold (w/ ensemble) | 30.8 | 77.3 | 9.9 | 91.6 | 6.5 | **96.0** |
| **Marigold w/ FreSca** | **30.2** | **77.8** | **9.8** | **91.7** | **6.4** | 95.9 |

**FreSca Configuration:**
- **DIODE dataset**: high-freq scale=1.5, low-freq scale=1.0
- **KITTI dataset**: high-freq scale=1.2, low-freq scale=1.0
- **ETH3D dataset**: high-freq scale=1.1, low-freq scale=1.0

*Lower AbsRel is better (↓), higher δ1 is better (↑). Best results are in bold.*

## 👍 Acknowledgements

We thank [Marigold](https://github.com/prs-eth/Marigold/tree/main) team for their excellent codebase.