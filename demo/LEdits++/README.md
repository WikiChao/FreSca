# FreSca + LEdits++: Enhanced Image Editing

This demo showcases how FreSca can be integrated with LEdits++ to improve image editing through frequency-domain scaling, enhancing detail preservation and editing quality.


## 🚀 Quick Start

We've integrated FreSca into the LEdits++ diffusers pipeline for easy experimentation:

```bash
# Run the demo with FreSca enabled
python LEdits++_FreSca.py
```

### 🔧 Configuration

To toggle FreSca on/off, modify line 1122 in LEdits++_FreSca.py:
```
# Enable FreSca frequency scaling
frequency_scaling=True  # Set to False to disable
```

## 👍 Acknowledgements

This implementation builds upon [LEdits++](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/ledits_pp). We thank the authors for their excellent work.