# Context Augmented Memory (CAM)

> **Give your AI long-term memory for conversations**

## 🧠 What This Does

CAM makes AI conversations feel **continuous** and **personal** by:

- ✅ **Remembering** what you said before
- ✅ **Understanding** when to use old memories
- ✅ **Filtering** out meaningless chat ("ok", "thanks")
- ✅ **Working** with OpenAI, Claude, Mistral, and Gemini
- ✅ **Dropping in** to existing apps with 1 line change

## 🚀 How It Works (Simple)

```
You: "My name is Alex"
AI: "Nice to meet you, Alex!"

[30 minutes later...]

You: "What's my name?"
AI: "Your name is Alex!" ← CAM remembered!
```

CAM automatically:
1. **Detects** if you're stating facts or asking questions
2. **Finds** relevant past conversations when needed
3. **Injects** context to help the AI remember
4. **Stores** new information for future use

## 📂 Core Files Explained

### **Main Files You'll Use**

#### `main.py` - Chat with CAM directly
```bash
python main.py
```
**What it does**: Interactive chat where CAM automatically remembers things
**Why use it**: Test CAM, have personal conversations, explore features

#### `start_cam.sh` - Start CAM as a service
```bash
./start_cam.sh
```
**What it does**: Starts CAM as a background service
**Why use it**: Use CAM in your own applications via API

#### `config.json` - Settings
**What it does**: Controls how "smart" CAM is (when to remember, what to filter)
**Why use it**: Fine-tune CAM for your specific needs

### **How It Works Behind the Scenes**

#### `modules/embedding.py` - Text Understanding
**What it does**: Converts text to numbers so CAM can find similar conversations
**Why it matters**: Lets CAM find relevant memories even when you use different words

#### `modules/memory.py` - Memory Storage
**What it does**: Saves conversations in a smart database
**Why it matters**: Fast searching and reliable storage of your memories

#### `modules/retrieval.py` - Smart Context Finding
**What it does**: Finds the most relevant past conversations
**Why it matters**: Only brings up relevant context, not everything you've ever said

#### `modules/intent_classifier.py` - Purpose Detection
**What it does**: Figures out if you're stating facts, asking questions, or just chatting
**Why it matters**: Different handling for different types of conversation

#### `proxy_api/` - API for Your Apps
**What it does**: Makes CAM available as a web service
**Why it matters**: Drop CAM into any app that uses OpenAI

## 🔧 Quick Setup

### **Option 1: Try It Now (CLI)**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI key to .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Start chatting
python main.py
```

### **Option 2: Use in Your Apps (API)**
```bash
# 1. Start the service
./start_cam.sh

# 2. Use like OpenAI (just change the URL)
# OLD:
client = OpenAI(api_key="sk-...")

# NEW:
client = OpenAI(api_key="sk-...", base_url="http://localhost:8080/v1")
```

## 💡 Smart Features

### **Context Intelligence**
- **Pronoun Smarts**: When you say "it", CAM knows what "it" refers to
- **Topic Detection**: Knows when you're continuing a conversation
- **Relevance Scoring**: Only brings up memories that actually help

### **Quality Control**
- **Filter**: Ignores "ok", "thanks", "lol" etc.
- **Auto-Tag**: Categories like "refund", "complaint", "review"
- **Session Tracking**: Keeps conversations organized

### **Multi-LLM Support**
- **OpenAI**: GPT-4, GPT-3.5
- **Anthropic**: Claude 3.5, Claude 3
- **Mistral**: Mistral Large, Mixtral
- **Google**: Gemini Pro

## 🎯 Use Cases

### **Personal AI Assistant**
```python
# CAM remembers your preferences over time
user: "I prefer morning meetings"
assistant: "Got it, you prefer morning meetings"

[Weeks later]
user: "Schedule a meeting with Sarah"
assistant: "I'll schedule a morning meeting with Sarah"
```

### **Customer Support**
```python
# CAM remembers customer context
customer: "My order #12345 is delayed"
agent_bot: "Let me check on order #12345"

[Later in same conversation]
customer: "What's the status?"
agent_bot: "Your delayed order #12345 is now shipped"
```

### **Learning Companion**
```python
# CAM builds on previous explanations
student: "Explain photosynthesis again"
teacher_bot: "Building on what we discussed about chlorophyll..."

[Next day]
student: "What about plants in winter?"
teacher_bot: "Remember how photosynthesis needs sunlight? In winter..."
```

## 🛠️ Configuration

Edit `config.json` to customize CAM:

```json
{
  "retrieval": {
    "max_distance": 0.8  // How strict to be about finding similar content
  },
  "usefulness_filter": {
    "min_word_count": 3,  // Ignore very short messages
    "blacklist_phrases": ["ok", "thanks", "lol"]  // Ignore these
  }
}
```

## 🔄 Data Flow (Super Simple)

```
Your message → CAM thinks about it → Finds relevant memories →
Adds context to your message → AI gets smarter response →
CAM stores the new conversation → Ready for next time
```

## 🚀 What Makes CAM Special

**Traditional RAG**: Search documents you uploaded
**CAM**: Remembers conversations you had

- **No documents needed** - learns from your conversations
- **Always relevant** - only uses memories that help right now
- **Personal** - builds knowledge about YOU specifically
- **Continuous** - works across days, weeks, months

## 🎉 Get Started

1. **Clone and install**: `pip install -r requirements.txt`
2. **Add API key**: Put OpenAI key in `.env`
3. **Try CLI**: `python main.py`
4. **Use in apps**: `./start_cam.sh` then change one line in your code

That's it! 🎉 Your AI now has a perfect memory.