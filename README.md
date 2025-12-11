# Context Augmented Memory (CAM)

> **Give your AI long-term memory for conversations**

## ⚡ Quick Start

### Option 1: Try It Now (1 minute)
```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start chatting
python main.py
```

### Option 2: Use in Your Apps (2 minutes)
```bash
# Start the memory service
./start_cam.sh

# Use like OpenAI (just change the URL)
client = OpenAI(api_key="sk-...", base_url="http://localhost:8080/v1")
```

## 🧠 What It Does

CAM makes AI conversations feel **continuous** and **personal** by automatically:
- ✅ Remembering what you said before
- ✅ Understanding when to use old memories
- ✅ Filtering out meaningless chat ("ok", "thanks")
- ✅ Working with OpenAI, Claude, Mistral, and Gemini

### Example

```
You: "My name is Alex"
AI: "Nice to meet you, Alex!"

[30 minutes later...]

You: "What's my name?"
AI: "Your name is Alex!" ← CAM remembered!
```

## 🎯 Use Cases

- **Personal AI Assistant** - Remembers your preferences
- **Customer Support** - Maintains conversation context
- **Learning Companion** - Builds on previous explanations
- **Productivity Tool** - Remembers project decisions

## 📚 Documentation

See `documentation.md` for detailed explanations and technical details.

## 🛠️ How It Works

1. **Detects** if you're stating facts or asking questions
2. **Finds** relevant past conversations when needed
3. **Injects** context to help the AI remember
4. **Stores** new information for future use

## 🔧 Configuration

Edit `config.json` to customize how CAM works:
- Memory retrieval sensitivity
- Content filtering rules
- Context decision thresholds

## 🚀 What Makes CAM Special

**Traditional RAG**: Search documents you uploaded
**CAM**: Remembers conversations you had

- **No documents needed** - learns from your conversations
- **Always relevant** - only uses memories that help right now
- **Personal** - builds knowledge about YOU specifically
- **Continuous** - works across days, weeks, months

## 📁 Project Structure

```
CAM/
├── main.py                 # CLI chat interface
├── start_cam.sh           # Start CAM as a service
├── config.json            # Configuration settings
├── documentation.md       # Detailed technical docs
├── .env.example           # Environment template
├── cam_client.py          # Simple Python client
├── requirements.txt       # Python dependencies
├── modules/               # Core CAM functionality
└── proxy_api/            # API for external apps
```

## 🎉 That's It!

Your AI now has a perfect memory that gets smarter with every conversation.

---

Made with ❤️ for better AI conversations