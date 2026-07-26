# Example data

Generate tiny synthetic images (no model download is needed for generation):

```bash
python examples/generate_data.py
```

The layout is `normal/` for defect-free reference images and `test.png` for an
anomalous query. The optional `prototypes/<defect_type>/` directories exercise
the auxiliary typing head; they are not required for anomaly detection.
