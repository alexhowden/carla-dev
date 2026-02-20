# CARLA Setup & Off-Road Perception Guide

This document covers: running CARLA, project scripts, camera and map options, and how to prototype with pre-trained segmentation models for off-road perception.

---

## 1. CARLA setup (quick reference)

- **CARLA install path:** `C:\CARLA_0.9.16` (override with `CARLA_ROOT` env var).
- **Python:** 3.12 required for the CARLA 0.9.16 Windows wheel.
- **Start server:** `python scripts/start_carla.py` (or run `CarlaUE4.exe`). Wait for the Unreal window and a map to load. To run **without the spectator window** (saves resources; cameras still work): `python scripts/start_carla.py --headless`.
- **Install client:**
  `pip install "C:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl"`
  Then: `pip install -r requirements.txt`
- **Test connection:** With CARLA running, `python scripts/connection.py` then `python scripts/connection.py --demo`.

---

## 2. Project scripts

| Script | Purpose |
|--------|--------|
| `scripts/start_carla.py` | Start CARLA server (CarlaUE4.exe). Use `--headless` to run without the spectator window (cameras still work). |
| `scripts/connection.py` | Test connection; `--demo` spawns vehicle + camera (default vehicle: Audi TT). Use `--vehicle <id>` or `--vehicle random`. |
| `scripts/autopilot.py` | Ego RGB camera while car drives on autopilot. Defaults: 120° FOV, ~5 MP, sporty car (`vehicle.audi.tt`). Supports `--map`, `--vehicle`, `--fov`, `--width`, `--height`. |
| `scripts/autopilot_segformer.py` | Run SegFormer on the live ego camera; shows camera + segmentation side-by-side. Works with ADE20K and fine-tuned models. Use `--infer-every 5 --width 320 --height 240` for lower-spec machines. |
| `scripts/autopilot_deeplabv3.py` | Run DeepLabV3+ (binary road detection) on the live ego camera; green road overlay. |
| `scripts/autopilot_ground_projection.py` | SegFormer + IPM bird's-eye view ground projection. Shows camera feed and BEV segmentation side-by-side with meter grid. Use `--bev-x`, `--bev-ymin`, `--bev-ymax` to tune range. |
| `scripts/manual_control_ground_projection.py` | Manual drive (WASD) + SegFormer + IPM bird's-eye view. Same BEV output as autopilot version. |
| `scripts/manual_control_segformer.py` | Manual drive (WASD) + SegFormer. Works with ADE20K and fine-tuned models. Supports hand brake (SPACE), reverse (Q). |
| `scripts/manual_control.py` | CARLA’s manual control (HUD, keybinds, camera angles, reverse). Defaults to sedan (`vehicle.audi.a2`). |
| `scripts/generate_traffic.py` | Spawn NPC vehicles and pedestrians. |
| `scripts/automatic_control.py` | Autopilot agent + camera + HUD. Uses `--loop` by default so it keeps driving to new targets. |
| `scripts/list_maps.py` | Print maps available on the running CARLA server (run CARLA first). |
| `scripts/load_map.py` | Load a map and exit (e.g. `python scripts/load_map.py --map Town07`). Use before running other scripts so they see the desired map. |

**Order of use:** Start CARLA with `start_carla.py` (or the exe), then in another terminal run any of the other scripts.

### Vehicle selection

- **Default:** `vehicle.dodge.charger_2020` (ground projection scripts) or `vehicle.audi.tt` (other scripts).
- **Override:** `--vehicle <blueprint_id>` (e.g. `--vehicle vehicle.ford.mustang`) or `--vehicle random`.
- Full catalogue: [CARLA Vehicles](https://carla.readthedocs.io/en/latest/catalogue_vehicles/).

---

## 3. Camera parameters (autopilot camera)

The ego camera in `autopilot.py` is set to match a real-world spec: **120° FOV**, **~5 MP RGB**.

- **FOV:** `--fov` (degrees). Default: 120.
- **Resolution:** `--width`, `--height`. Default: 2560×1920 (~5 MP). Override for different resolution or aspect ratio.
- **Other options (in CARLA):** On the camera blueprint you can also set `sensor_tick`, `gamma`, and camera transform (position/rotation).

Example:

```powershell
python scripts/autopilot.py --map Town03 --fov 120 --width 2560 --height 1920
```

---

## 4. Maps

### Loading a map from the client

- **Standalone (recommended):** Run once, then use any script on the new map:
  ```powershell
  python scripts/load_map.py --map Town07
  ```
  Then run `autopilot.py`, `manual_control.py`, etc. without `--map`; they will use the current map.

- From Python, after connecting: `client.load_world("Town02")` then `world = client.get_world()`.
- From the autopilot camera script: `python scripts/autopilot.py --map Town03` (still supported).
- Map load can take **30–90 seconds** on first load; `load_map.py` uses a 90 s timeout for `load_world()`.

### Which maps are available?

- **Base install (typical):** Town01, Town02, Town03, Town04, Town05. Some builds also include Town06.
- **Extra maps (optional):** Town06, Town07, Town10, etc. require the **additional maps package** from CARLA (download from CARLA releases, import into the `Import` folder, run ImportAssets). If you did not install that package, **Town07 (and sometimes Town06) will not exist** and you get: `RuntimeError: Map 'Town07' not found`.

**If you get “Map 'Town07' not found”:**
Your CARLA build doesn’t include that map. Use a map that is in the base install, e.g.:

```powershell
python scripts/autopilot.py --map Town03
```

To see exactly which maps your server has, run (with CARLA already running):

```powershell
python scripts/list_maps.py
```

### Most “off-road-y” maps (when available)

- **Town07** – Country road, most rural (requires extra maps package).
- **Town03** – Mix of town and open/hilly; good fallback.
- **Town06** – Long highway with rural sections (may require extra maps).

If Town07 is not installed, use **Town03** as the most varied / least urban option in the base set.

### How to get Town06 and Town07 (additional maps)

1. Download **AdditionalMaps** from [CARLA GitHub releases](https://github.com/carla-simulator/carla) for your version.
2. Extract and merge into your CARLA install — copy `AdditionalMaps\CarlaUE4\Content\` into `C:\CARLA_0.9.16\CarlaUE4\Content\` (and same for `Plugins\`).
3. Restart CARLA, then verify with `python scripts/list_maps.py`.

---

## 5. Troubleshooting

| Issue | What to do |
|-------|------------|
| Connection failed | Ensure CARLA is running and a map is loaded. Check ports 2000 and 2001. |
| `Map 'Town07' not found` | Use a base map (e.g. `--map Town03`) or install CARLA’s additional maps package. Run `python scripts/list_maps.py` to see available maps. |
| Map load timeout | First load can take 30–90 s. The autopilot script uses a 90 s timeout; if it still times out, try a different map or restart CARLA. |
| `ImportError: No module named 'carla'` | Install the CARLA `.whl` from your CARLA install path (see section 1). |
| Wrong Python | CARLA 0.9.16 wheel is for Python 3.12. Use `python --version` and the correct venv. |
