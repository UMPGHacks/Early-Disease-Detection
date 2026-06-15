# EDD Multi Disease Detection

This folder contains a unified Streamlit web app for early disease screening using the datasets available in `D:\ml project\dataset`.

## Modules
- Symptom-based disease prediction
- Heart failure risk screening
- Liver disease detection
- Lung X-ray placeholder for a future image model

## Run

```bash
cd "D:\ml project\EDD"
python model_training.py
streamlit run app.py
```

## Notes
- Models are saved inside `EDD/models`.
- The lung X-ray tab is ready for integration, but it does not include a trained image model yet.
- This project is for educational use and should not replace a doctor.

## Lung Model

To train the lung X-ray model:

```bash
cd "D:\ml project\EDD"
python train_lung_model.py
```
