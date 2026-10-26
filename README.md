# Pokémon Emerald AI Agent (CachyOS Native Setup)

A lightweight, sandboxed Reinforcement Learning environment running Stable-Baselines3, Gymnasium, and compiled `mGBA` Python bindings optimized for your AMD Ryzen 5 5600G.

## 🚀 Quick Start Commands

### 1. Setup Environment
```bash
cd ~/pokemon_ai_sandbox
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


### 2. Run Training Mode

Ensure PokemonEmerald.gba is in this folder, verify MODE = "train" in main.py, then run:
```bash
main.py
```
    Automatic Features:

        Spawns 4 parallel CPU environments (NUM_CPU = 4) to fully saturate your Ryzen 5 5600G cores.

        Launches TensorBoard in the background at http://localhost:6006.

        Records your training session to train_journey.mp4.


### 3. Run Watch / Evaluation Mode

Open main.py, change configuration to:
```bash
MODE = "watch"
```

Then run:
```bash
python main.py
```

    Automatic Features:

        Automatically drops to a single clean window (NUM_CPU = 1).

        Loads your trained brain checkpoint (emerald_ppo_checkpoint.zip).

        Records clean playback to watch_playback.mp4.


### 4. Safe Shutdown

Press Ctrl + C anytime in the terminal. The script cleanly saves your model weights, releases video files, and terminates background server processes.


---

#### 2. `requirements.txt`
```text
gymnasium==0.29.1
stable-baselines3==2.2.1
opencv-python==4.9.0.80
tensorboard==2.15.1
numpy==1.26.4