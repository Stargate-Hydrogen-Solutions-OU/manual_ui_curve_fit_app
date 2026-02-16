# manual_ui_curve_fit_app
Simple web app to simulate influence of fitting parameters on the shape of raw UI curve.

The app does not save any data. It overlays provided data and a curve calculated from values inserted via sliders.
Fitting can be initiated by clicking a button. Multiple parameters can be fixed before fitting.

## Runtime options

This repository now supports two runtime modes:

1. **Server mode (Streamlit server, Python backend)**
   - Best for full-featured deployments where the app runs as a normal Streamlit service.
2. **Serverless mode (stlite + Pyodide in browser)**
   - Best for lightweight/static hosting where Python executes in the browser.

## Repository structure

- `manual_fit_app.py` - shared Streamlit application source.
- `docker/Dockerfile.server` - container image for server mode.
- `docker/Dockerfile.stlite` - container image serving static stlite assets.
- `stlite/index.html` - stlite bootstrap page that loads and runs `manual_fit_app.py` in the browser.
- `stlite/requirements.txt` - Pyodide-side dependencies used by stlite.
- `Dockerfile` - legacy server-mode Dockerfile kept for backward compatibility.

## Build and run with Docker

> **Working directory:** run all commands below from the repository root directory (the folder containing `manual_fit_app.py`, `docker/`, and `stlite/`).
>
> Example:
>
> ```bash
> cd /path/to/manual_ui_curve_fit_app
> ```

### 1) Server mode

```bash
docker build -f docker/Dockerfile.server -t manual-ui-curve-fit:server .
docker run --rm -p 8501:8501 manual-ui-curve-fit:server
```

Open: `http://localhost:8501`

### 2) Serverless stlite mode

```bash
docker build -f docker/Dockerfile.stlite -t manual-ui-curve-fit:stlite .
docker run --rm -p 8080:80 manual-ui-curve-fit:stlite
```

Open: `http://localhost:8080`

## Local stlite testing without rebuilding Docker images

This workflow is useful while iterating on UI/app code because you can refresh the browser instead of rebuilding an image.

1. Open a terminal and navigate to the repository root:

   ```bash
   cd /path/to/manual_ui_curve_fit_app
   ```

2. Start a local static Python server from the repository root:

   ```bash
   python -m http.server 8000
   ```

3. Open the app in your browser:

   ```text
   http://localhost:8000/stlite/
   ```

   > Why not run the server inside `stlite/`?
   > `manual_fit_app.py` lives in the repository root and is fetched by the stlite bootstrap. If you serve only `stlite/`, the app source returns 404.

4. Edit `manual_fit_app.py`, `stlite/index.html`, or `stlite/requirements.txt` as needed, then refresh the browser.

5. Stop the local server with `Ctrl+C`.


### Troubleshooting local browser run

- If the page shows `Failed to load app: stlite is not defined` or `Unable to load stlite runtime`, your browser likely cannot download `@stlite/browser` from the public CDN.
- The app now tries **both** jsDelivr and unpkg. If both are blocked (for example by a corporate firewall), allow access to one of them or host `@stlite/browser` locally and point `stlite/index.html` to that local script.

## Suggested docker-compose layout

Use two services so you can run either target with one Compose file:

```yaml
services:
  curve-fit-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.server
    ports:
      - "8501:8501"

  curve-fit-stlite:
    build:
      context: .
      dockerfile: docker/Dockerfile.stlite
    ports:
      - "8080:80"
```

This keeps deployment concerns separated while preserving one shared app source.
