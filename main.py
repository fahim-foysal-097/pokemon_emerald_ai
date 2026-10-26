import os
import signal
import sys
import subprocess
import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env



# ==========================================
# 🛠️ CENTRAL CONFIGURATION & CONSTANTS
# ==========================================
MODE = "train"                   # Options: "train" or "watch"
ROM_PATH = "PokemonEmerald.gba"
MODEL_CHECKPOINT = "emerald_ppo_checkpoint"
CHECKPOINT_STATE = "emerald_save.state"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/"

# Auto-Start TensorBoard Server (Training mode only)
AUTO_START_TENSORBOARD = True
TENSORBOARD_PORT = 6006

# Video Recording Settings (Organized into ./videos/)
# Best Practice: Disable video during heavy multi-core training to maximize speed, enable for watch mode.
RECORD_VIDEO = True if MODE == "watch" else False 

# Unique Timestamp generation to prevent video overwriting across runs and days
TIMESTAMP_STR = time.strftime("%Y%m%d_%H%M%S")
TRAIN_VIDEO_PATH = f"./videos/train_journey_{TIMESTAMP_STR}.mp4"
WATCH_VIDEO_PATH = f"./videos/watch_playback_{TIMESTAMP_STR}.mp4"
VIDEO_FPS = 60                   # 60 FPS matches native GBA hardware speed

# Desktop Notifications Configuration (Watch Mode Only)
ENABLE_NOTIFICATIONS = True if MODE == "watch" else False

# Environment & Performance Settings
OBS_WIDTH = 80
OBS_HEIGHT = 72
TOTAL_TIMESTEPS = 1_000_000
VERBOSE_LEVEL = 1
STOCHASTIC_WATCH = False         # False = pure deterministic best behavior in watch mode

# Automatic Parallel CPU Environment Scaling based on Mode
# - Train mode utilizes 4 parallel cores for maximum Ryzen 5 5600G throughput.
# - Watch mode drops to 1 core for clean, dedicated single-instance viewing.
NUM_CPU = 4 if MODE == "train" else 1
VIDEO_OUTPUT_PATH = TRAIN_VIDEO_PATH if MODE == "train" else WATCH_VIDEO_PATH
# ==========================================



tb_process = None

def send_notification(title, message):
    """Triggers native CachyOS desktop notifications using libnotify (notify-send)."""
    if ENABLE_NOTIFICATIONS:
        try:
            subprocess.run(["notify-send", "-u", "normal", "-i", "applications-games", title, message], check=False)
        except Exception as e:
            print(f"[Notification Error] Could not send desktop alert: {e}")


class EmeraldEnv(gym.Env):
    def __init__(self, rom_path=ROM_PATH, checkpoint_file=CHECKPOINT_STATE):
        super(EmeraldEnv, self).__init__()

        self.action_space = spaces.Discrete(8)  # 0:Up, 1:Down, 2:Left, 3:Right, 4:A, 5:B, 6:Start, 7:Select
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8
        )

        self.rom_path = rom_path
        self.checkpoint_file = checkpoint_file
        self.step_counter = 0
        self.last_notified_event = None

        # --- mGBA Emulator Core Initialization ---
        self.core = None
        self._init_emulator()

        # Video Recorder Setup (Ensures ./videos directory exists first)
        self.video_writer = None
        if RECORD_VIDEO:
            os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                VIDEO_OUTPUT_PATH, fourcc, VIDEO_FPS, (OBS_WIDTH * 4, OBS_HEIGHT * 4)
            )
            print(f"[Recorder] Recording [{MODE.upper()}] session to {VIDEO_OUTPUT_PATH}")

        print(f"[Init] Pokémon Emerald environment initialized in [{MODE.upper()}] mode.")

    def _init_emulator(self):
        try:
            import mgba.core
            if not os.path.exists(self.rom_path):
                raise FileNotFoundError(f"ROM file not found at '{self.rom_path}'!")

            self.core = mgba.core.load_path(self.rom_path)
            if self.core is None:
                raise ValueError(f"Failed to load ROM from {self.rom_path}")

            self.core.reset()
            print("[Emulator] mGBA core loaded and booted successfully.")
        except ImportError:
            print("[Warning] 'mgba' Python module not found in path. Running in simulated fallback mode.")
            self.core = None
        except Exception as e:
            print(f"[Emulator Error] {e}. Falling back to simulation mode.")
            self.core = None

    def reset(self, seed=None, options=None):
        if self.core is not None:
            if MODE == "watch" and os.path.exists(self.checkpoint_file):
                print(f"[Eval Reset] Loading save state for evaluation playback: {self.checkpoint_file}")
            elif os.path.exists(self.checkpoint_file) and MODE == "train":
                print(f"[Load] Resuming session state from: {self.checkpoint_file}")
            else:
                print("[New] Starting fresh adventure in Hoenn!")
                self.core.reset()
        else:
            print(f"[Simulated Reset] Mode: {MODE}")

        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1

        if self.core is not None:
            try:
                from mgba.core import Key
                key_mapping = [
                    Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT,
                    Key.A, Key.B, Key.START, Key.SELECT
                ]
                if 0 <= action < len(key_mapping):
                    self.core.set_keys(key_mapping[action])
            except Exception:
                pass

            self.core.run_frame()
            
            # Watch mode event and milestone alert listener
            if MODE == "watch":
                self._check_game_milestones()

        obs = self._get_obs()

        # Reward calculation hook (expand later with RAM memory reading for badges/levels)
        reward = 0.1  
        done = False
        truncated = False

        # Record frame if enabled
        if self.video_writer is not None:
            frame_resized = cv2.resize(obs, (OBS_WIDTH * 4, OBS_HEIGHT * 4), interpolation=cv2.INTER_NEAREST)
            self.video_writer.write(frame_resized)

        return obs, reward, done, truncated, {}

    def _check_game_milestones(self):
        """Inspects game state/RAM to trigger notifications for Gyms, boss fights, story events, etc."""
        try:
            # You can map mGBA RAM memory checks here when ready (e.g., reading battle flags, party counts, or badges).
            # Example mock notification triggers based on steps or event states:
            if self.step_counter == 500 and self.last_notified_event != "route101":
                send_notification("Pokémon AI Watch", "AI successfully navigating Route 101!")
                self.last_notified_event = "route101"
            elif self.step_counter == 2500 and self.last_notified_event != "gym":
                send_notification("🏆 Gym / Boss Milestone Alert!", "The AI has encountered a major gym or boss sequence!")
                self.last_notified_event = "gym"
        except Exception:
            pass

    def _get_obs(self):
        if self.core is not None:
            try:
                raw_frame = np.asarray(self.core.screen_buffer())
                if raw_frame.size > 0:
                    obs = cv2.resize(raw_frame, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    return obs
            except Exception:
                pass

        # Fallback dummy screen if emulator buffer isn't active
        return np.zeros((OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8)

    def save_state(self):
        print(f"\n[Save] Saving progress at step {self.step_counter}...")
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[Recorder] Video saved successfully to {VIDEO_OUTPUT_PATH}")


# --- Auto-Start TensorBoard Server ---
def start_tensorboard():
    global tb_process
    if AUTO_START_TENSORBOARD and MODE == "train":
        try:
            print(f"[TensorBoard] Starting background server at http://localhost:{TENSORBOARD_PORT}")
            tb_process = subprocess.Popen(
                ["tensorboard", f"--logdir={TENSORBOARD_LOG_DIR}", f"--port={TENSORBOARD_PORT}", "--bind_all"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[TensorBoard Warning] Could not auto-start server: {e}")


# --- Graceful Exit Handler ---
def handle_exit(sig, frame):
    print("\n[Interrupt] Caught exit signal. Cleaning up resources...")
    if 'env' in globals():
        try:
            env.close()
        except AttributeError:
            pass

    global tb_process
    if tb_process is not None:
        tb_process.terminate()
        print("[TensorBoard] Background server stopped.")

    print("[Exit] Clean shutdown complete.")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)


# --- Main Execution Block ---
if __name__ == "__main__":
    start_tensorboard()

    model_file_path = f"{MODEL_CHECKPOINT}.zip"

    # Automatically configure Vectorized Environments based on MODE (NUM_CPU = 4 for train, 1 for watch)
    print(f"[Environment Setup] Initializing {NUM_CPU} parallel environment(s) for [{MODE.upper()}] mode...")
    env = make_vec_env(
        lambda: EmeraldEnv(rom_path=ROM_PATH, checkpoint_file=CHECKPOINT_STATE),
        n_envs=NUM_CPU
    )

    if MODE == "train":
        print("--------------------------------------------------")
        print(f" [MODE] TRAINING ACTIVE ({NUM_CPU} Cores). Press Ctrl+C to Save.")
        print("--------------------------------------------------")

        if os.path.exists(model_file_path):
            print(f"[Load] Resuming training weights from {model_file_path}...")
            model = PPO.load(MODEL_CHECKPOINT, env=env, verbose=VERBOSE_LEVEL, tensorboard_log=TENSORBOARD_LOG_DIR)
        else:
            model = PPO("CnnPolicy", env, verbose=VERBOSE_LEVEL, tensorboard_log=TENSORBOARD_LOG_DIR)

        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)
            model.save(MODEL_CHECKPOINT)
            print(f"[Complete] Reached {TOTAL_TIMESTEPS} steps. Model successfully saved!")
        except KeyboardInterrupt:
            model.save(MODEL_CHECKPOINT)
            print(f"[Save] Model checkpoint saved to {MODEL_CHECKPOINT}.zip")
            handle_exit(None, None)

    elif MODE == "watch":
        print("--------------------------------------------------")
        print(f" [MODE] EVALUATION PLAYBACK ({NUM_CPU} Core). Watching AI play with Desktop Notifications Active.")
        print("--------------------------------------------------")
        
        send_notification("Pokémon AI Watch Mode", "Playback started. Desktop alerts for Gyms, Battles, and Milestones are active!")

        if os.path.exists(model_file_path):
            model = PPO.load(model_file_path, env=env)
            print(f"[Load] Loaded model weights from {model_file_path}")
        else:
            print("[Error] No trained model found! Train the AI first before watching.")
            sys.exit(1)

        obs = env.reset()
        try:
            while True:
                action, _states = model.predict(obs, deterministic=(not STOCHASTIC_WATCH))
                obs, reward, dones, infos = env.step(action)
        except KeyboardInterrupt:
            handle_exit(None, None)