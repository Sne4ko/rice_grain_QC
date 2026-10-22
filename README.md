# rice_grain_QC

Automated rice grain quality control using image processing.

Run the preprocessing pipeline with the sample image:

```powershell
python main.py
```

To process a different image, change `IMAGE_NAME` in `main.py`. The image path
is built automatically from the project `data` folder and that filename.

The pipeline normalizes lighting, thresholds the image, removes noise, and
separates grains with watershed segmentation. It displays all stages together
in one figure and returns the watershed marker labels.

Processing and visualization parameters are centralized in the workspace-level
`config.py`.
