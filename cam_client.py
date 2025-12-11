"""
Context Augmented Memory (CAM) - Simple Python Client
Drop-in replacement for OpenAI with automatic memory augmentation
"""

import requests
import json
from typing import List, Dict, Optional, Any


class CAMClient:
    """
    Simple client for Context Augmented Memory API.
    Drop-in replacement for OpenAI with automatic memory.
    """

    def __init__(self, api_key: str, base_url: str = "http://localhost:8080/v1", model: str = "gpt-4o-mini"):
        """
        Initialize CAM client.

        Args:
            api_key: Your OpenAI API key
            base_url: CAM proxy server URL (default: http://localhost:8080/v1)
            model: Model to use (default: gpt-4o-mini)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/v1')
        self.model = model

        # Check if CAM service is running
        self._check_service()

    def _check_service(self):
        """Check if CAM service is running."""
        try:
            response = requests.get(f"{self.base_url}/v1/memory/debug", timeout=5)
            if response.status_code != 200:
                print("⚠️ CAM service may not be running. Run './start_cam.sh' first.")
        except requests.exceptions.RequestException:
            print("⚠️ CAM service not accessible. Run './start_cam.sh' first.")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send chat completion request with automatic memory augmentation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional OpenAI parameters

        Returns:
            Response from OpenAI API with memory-augmented context
        """
        # Prepare request payload
        payload = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "api_key": self.api_key,
            **{k: v for k, v in kwargs.items() if k != "model"}
        }

        try:
            # Send request to CAM proxy
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ CAM request failed: {e}")
            return {"error": str(e)}

    def chat_with_memory(self, user_message: str, **kwargs) -> str:
        """
        Simple chat method for single messages.

        Args:
            user_message: User's message
            **kwargs: Additional OpenAI parameters

        Returns:
            AI response text
        """
        messages = [{"role": "user", "content": user_message}]
        response = self.chat(messages, **kwargs)

        if "error" in response:
            return f"Error: {response['error']}"

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "Error: Unexpected response format"

    def clear_memory(self):
        """Clear stored memory (requires service restart)."""
        print("💡 To clear memory, restart the CAM service with './start_cam.sh'")
        print("   Or manually delete the './CAM_project/chroma_db' directory")


class OpenAIWithMemory:
    """
    OpenAI-compatible interface that automatically adds memory.
    Use exactly like you would use OpenAI's client.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize with automatic memory if using CAM.

        Args:
            api_key: OpenAI API key
            base_url: If None, uses normal OpenAI. If provided, uses CAM.
            **kwargs: Additional OpenAI parameters
        """
        if base_url is None:
            # Use regular OpenAI
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, **kwargs)
            self.uses_cam = False
        else:
            # Use CAM
            self.client = CAMClient(api_key=api_key, base_url=base_url, **kwargs)
            self.uses_cam = True

    def chat(self, **kwargs):
        """Send chat completion request."""
        if self.uses_cam:
            return self.client.chat(**kwargs)
        else:
            return self.client.chat.completions.create(**kwargs)

    def simple_chat(self, message: str, **kwargs) -> str:
        """Simple chat method for single messages."""
        if self.uses_cam:
            return self.client.chat_with_memory(message, **kwargs)
        else:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                **kwargs
            )
            return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    # Example 1: Direct CAM client
    print("=== Example 1: Direct CAM Client ===")
    cam = CAMClient(api_key="your-api-key-here")
    response = cam.chat_with_memory("Hello, my name is Alex. Remember that!")
    print(f"AI: {response}")

    # Example 2: OpenAI-compatible interface
    print("\n=== Example 2: OpenAI-Compatible ===")

    # With memory (CAM)
    client_with_memory = OpenAIWithMemory(
        api_key="your-api-key-here",
        base_url="http://localhost:8080/v1"
    )

    # Without memory (regular OpenAI)
    from openai import OpenAI
    client_without_memory = OpenAI(api_key="your-api-key-here")

    print("Both clients work the same, but one has memory!")