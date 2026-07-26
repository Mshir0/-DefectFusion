# Example data

Generate tiny synthetic images (no model download is needed for generation):

```bash
python examples/generate_data.py
```

The layout is `normal/` for defect-free reference images and
`prototypes/<defect_type>/` for one or more few-shot exemplars. Replace these
images with your own data while keeping the directory layout.
