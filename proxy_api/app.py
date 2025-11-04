"""
proxy_api/app.py
FastAPI entry point for the Context-Augmented Memory Proxy.
"""

# proxy_api/app.py
from fastapi import FastAPI
from proxy_api.router import router

app = FastAPI(title="Context Augmented Memory Proxy API")
app.include_router(router)

