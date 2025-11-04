import importlib
from modules import config_manager

def load_llm_client():
    cfg = config_manager.load_config()
    provider = cfg.get("provider", "openai").lower()
    module_name = f"proxy_api.clients.{provider}_client"
    module = importlib.import_module(module_name)
    return module.Client()