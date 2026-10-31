# GridGuard AI: Transformer Load and Thermal Forecasting

A PowerGuard AI module. It forecasts the hourly load of every substation transformer up to 7 days ahead with an AI model, turns
the forecast into top-oil and hot-spot temperatures and insulation ageing (IEEE C57.91), and gives each transformer a failure risk,
an FMEA priority and an action (including the load transfer needed at the peak).

## Pages
- **Substation Overview**: results from the analysis pages, or the demo substation.
- **Load & Thermal Forecast**: one transformer: data, load history (demo or CSV), weather forecast (normal climate of a Gulf city,
  an uploaded forecast or the demo forecast) and Analyze. PDF report and hourly CSV.
- **Thermal & Ageing Check**: IEEE C57.91 check of one daily load cycle and the highest load the unit can carry at that ambient.
- **AI Model & Accuracy**: accuracy on data the model never saw, forecast against actual, and training on the client's own files.
- **Combined Report**: PDF, text, table and FMEA CSV.
- **Historical Study**: replays every day, forecasting 7 days ahead, and compares the forecast hot-spots with what happened.

## Files
- `gridguard_core.py`: engine (calendar, Gulf climate, features, forecast, IEEE C57.91 thermal model, risk, replay, demo data).
- `train_gridguard_model.py`: trains and tests the model (`python train_gridguard_model.py` for the demo, or with client CSVs).
- `gridguard_model.pkl`: trained model (demo substation, synthetic data, trained until 2025-05-01).
- `report_pdf.py`: PDF reports. `GridGuard_AI.app.py`: Streamlit app.

## Data formats (CSV)
- Transformer register: `Transformer ID, Bay, Load Type, Rating (MVA), Cooling, Top-Oil Rise (K), Hot-Spot Gradient (K)`
- Load history: `Timestamp, Transformer ID, Load (MVA)` (hourly), or straight from the DCS: MW and MVAR per side
  (`HV MW, HV MVAR, MV MW, MV MVAR, LV MW, LV MVAR`, or `MW, MVAR`). MVA = sqrt(MW^2 + MVAR^2); the transformer load is the HV side
  (or MV + LV when the HV side is not metered); the power factor and the loading of every side are shown.
- Ambient temperature / weather forecast: `Timestamp, Ambient (C)` (hourly)

Login credentials are read from the Streamlit secrets `APP_USERNAME` and `APP_PASSWORD`. The demo data is synthetic; the model
must be re-trained on the client's own load and weather history before any accuracy figure is quoted for a site.

© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. Confidential and proprietary.
