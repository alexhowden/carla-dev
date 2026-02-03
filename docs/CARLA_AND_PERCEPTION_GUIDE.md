# CARLA Setup & Off-Road Perception Guide

This document covers: running CARLA, project scripts, camera and map options, and how to prototype with pre-trained segmentation models for off-road perception.

---

## 1. CARLA setup (quick reference)

- **CARLA install path:** `C:\CARLA_0.9.16` (override with `CARLA_ROOT` env var).
- **Python:** 3.12 required for the CARLA 0.9.16 Windows wheel.
- **Start server:** `python scripts/start_carla.py` (or run `CarlaUE4.exe`). Wait for the Unreal window and a map to load.
- **Install client:**
  `pip install "C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"`
  Then: `pip install -r requirements.txt`
- **Test connection:** With CARLA running, `python scripts/run_carla_connection.py` then `python scripts/run_carla_connection.py --demo`.

---

## 2. Project scripts

| Script | Purpose |
|--------|--------|
| `scripts/start_carla.py` | Start CARLA server (CarlaUE4.exe). Blocks until you close CARLA. |
| `scripts/run_carla_connection.py` | Test connection; `--demo` spawns vehicle + camera for a few seconds. |
| `scripts/run_autopilot_camera.py` | Ego RGB camera while car drives on autopilot. Defaults: 120° FOV, ~5 MP. Supports `--map`, `--fov`, `--width`, `--height`. |
| `scripts/run_carla_manual_control.py` | CARLA’s manual control (HUD, keybinds, camera angles, reverse). Defaults to sedan (`vehicle.audi.a2`). |
| `scripts/run_carla_generate_traffic.py` | Spawn NPC vehicles and pedestrians. |
| `scripts/run_carla_automatic_control.py` | Autopilot agent + camera + HUD. Uses `--loop` by default so it keeps driving to new targets. |
| `scripts/list_carla_maps.py` | Print maps available on the running CARLA server (run CARLA first). |

**Order of use:** Start CARLA with `start_carla.py` (or the exe), then in another terminal run any of the other scripts.

---

## 3. Camera parameters (autopilot camera)

The ego camera in `run_autopilot_camera.py` is set to match a real-world spec: **120° FOV**, **~5 MP RGB**.

- **FOV:** `--fov` (degrees). Default: 120.
- **Resolution:** `--width`, `--height`. Default: 2560×1920 (~5 MP). Override for different resolution or aspect ratio.
- **Other options (in CARLA):** On the camera blueprint you can also set `sensor_tick`, `gamma`, and camera transform (position/rotation).

Example:

```powershell
python scripts/run_autopilot_camera.py --map Town03 --fov 120 --width 2560 --height 1920
```

---

## 4. Maps

### Loading a map from the client

- From Python, after connecting: `client.load_world("Town02")` then `world = client.get_world()`.
- From the autopilot camera script: `python scripts/run_autopilot_camera.py --map Town03`.
- Map load can take **30–90 seconds** on first load; the script uses a 90 s timeout for `load_world()`.

### Which maps are available?

- **Base install (typical):** Town01, Town02, Town03, Town04, Town05. Some builds also include Town06.
- **Extra maps (optional):** Town06, Town07, Town10, etc. require the **additional maps package** from CARLA (download from CARLA releases, import into the `Import` folder, run ImportAssets). If you did not install that package, **Town07 (and sometimes Town06) will not exist** and you get: `RuntimeError: Map 'Town07' not found`.

**If you get “Map 'Town07' not found”:**
Your CARLA build doesn’t include that map. Use a map that is in the base install, e.g.:

```powershell
python scripts/run_autopilot_camera.py --map Town03
```

To see exactly which maps your server has, run (with CARLA already running):

```powershell
python scripts/list_carla_maps.py
```

### Most “off-road-y” maps (when available)

- **Town07** – Country road, most rural (requires extra maps package).
- **Town03** – Mix of town and open/hilly; good fallback.
- **Town06** – Long highway with rural sections (may require extra maps).

If Town07 is not installed, use **Town03** as the most varied / least urban option in the base set.

### How to get Town06 and Town07 (additional maps)

Town06 and Town07 are not in the base CARLA package; you have to download and import the **Additional Maps** asset.

1. **Download**
   Go to [CARLA releases](https://github.com/carla-simulator/carla/releases) and find the release that matches your version (e.g. **0.9.16**). In the assets list, download the **Additional Maps** package (often named like `AdditionalMaps_0.9.16.zip` or similar for your version).

2. **Import into CARLA**
   - Unzip the downloaded file (or use the zip as-is if the docs say so).
   - Copy the package into the **`Import`** folder of your CARLA install:
     `C:\CARLA_0.9.16\Import\`
   - Run CARLA’s import script so the editor loads the new maps. For **pre-built Windows builds** this is often:
     - A batch file under the CARLA root, e.g. `ImportAssets.bat` or similar, or
     - Instructions in CARLA’s docs under “Additional Maps” or “Import” for your exact build (0.9.16).
   If there is no batch file, check the [CARLA 0.9.16 docs](https://carla.readthedocs.io/en/0.9.16/) or the release notes for “import additional maps” / “ImportAssets”.

3. **Restart CARLA**
   After importing, close and restart CARLA (and run `python scripts/list_carla_maps.py` again). You should see **Town06** and **Town07** in the list.

**Note:** Some Windows builds ship as a single executable and may not support adding maps via Import; in that case you may need a build that includes Town06/07 or a build from source. If your CARLA came from a specific installer, check its documentation for “additional maps” or “DLC”.

---

## 5. Pre-trained segmentation models for off-road (dirt roads)

### Suggested models

- **General outdoor/driving (easy start):**
  **SegFormer** (B2/B5) or **DeepLabV3+** pretrained on **Cityscapes** or **ADE20K**. Classes like road, vegetation, terrain, obstacles; not dirt-specific but usable for “road vs trees vs obstacles”.

- **Better for off-road / dirt:**
  Models trained or fine-tuned on **RUGD** or **RELLIS-3D** (off-road/trail datasets). Search for “RUGD semantic segmentation” or “off-road segmentation” (e.g. GitHub, papers); many use SegFormer or DeepLab and release weights. These give dirt-road vs vegetation vs obstacles more directly.

- **Other options:** **BiSeNet**, **PIDNet** (real-time driving). **Mapillary Vistas** or **Semantic KITTI** pretrained models for road/terrain/vegetation.

**Summary:** Start with **SegFormer or DeepLabV3+** (Cityscapes/ADE20K); for dirt-road behavior, prefer models trained on **RUGD/RELLIS** or similar off-road datasets.

---

## 6. Hooking segmentation models up to CARLA

- **Input:** Use the **ego RGB camera** from your CARLA client (e.g. the same feed as `run_autopilot_camera.py`). Each frame from that camera is the image you run through the model.

- **Where:** In the same Python process as the CARLA client. In the camera callback (or a loop that reads the “latest” frame): convert CARLA image to numpy (e.g. RGB), resize/normalize to the model’s input size, run the model, get a per-pixel segmentation map (H×W of class IDs).

- **Display:**
  - **Overlay:** Map class IDs to a color palette (e.g. Cityscapes or the model’s dataset), resize the mask to camera resolution, blend with the camera image (e.g. 0.5×image + 0.5×colored_mask), show in the same OpenCV/Pygame window.
  - **Side-by-side:** One panel = camera image, one = colored segmentation mask.

- **Pipeline:** Extend (or copy) the autopilot-camera script: (1) Keep the part that receives camera frames. (2) Load the pretrained segmentation model once at startup. (3) For each new (or latest) frame: preprocess → model forward → colorize mask → show overlay or side-by-side.

- **Performance:** Use GPU if available; choose a fast model (e.g. SegFormer-B2, BiSeNet) or lower input resolution for near real-time. Sync vs async in CARLA only affects how often new frames arrive; the “run model on latest frame and display” logic is the same.

---

## 7. Telemetry and HUD

- The on-screen panel in CARLA’s manual control that shows speed, throttle, brake, etc. is the **HUD** (heads-up display).
- The **data** (speed, throttle, brake, steer, etc.) is **telemetry**.

---

## 8. Troubleshooting

| Issue | What to do |
|-------|------------|
| Connection failed | Ensure CARLA is running and a map is loaded. Check ports 2000 and 2001. |
| `Map 'Town07' not found` | Use a base map (e.g. `--map Town03`) or install CARLA’s additional maps package. Run `python scripts/list_carla_maps.py` to see available maps. |
| Map load timeout | First load can take 30–90 s. The autopilot script uses a 90 s timeout; if it still times out, try a different map or restart CARLA. |
| `ImportError: No module named 'carla'` | Install the CARLA `.whl` from your CARLA install path (see section 1). |
| Wrong Python | CARLA 0.9.16 wheel is for Python 3.12. Use `python --version` and the correct venv. |
