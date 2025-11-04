"""
config_manager.py
Dynamic configuration system for CAM modules.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.json")

DEFAULT_CONFIG = {
    "retrieval": {
        "max_distance": 0.6
    },
    "usefulness_filter": {
        "min_word_count": 3,
        "min_char_count": 15,
        "blacklist_phrases": [
            "what did i say",
            "when did i tell you",
            "do you remember",
            "remember that",
            "ok",
            "okay",
            "thanks",
            "thank you",
            "hmm",
            "yes",
            "no",
            "haha",
            "lol",
            "idk"
        ]
    },
    "context_decider": {
        "continuity_base": 0.45,
        "continuity_std_factor": 0.15
    }
}


def load_config():
    """Loads configuration, falling back to defaults if file missing."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG


def save_config(config):
    """Writes configuration back to file."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
