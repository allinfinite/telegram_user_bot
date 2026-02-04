# Telegram User Bot

An AI-powered Telegram bot that participates in group chats using large language models. The bot maintains conversation context, learns user behavior over time, and responds naturally with a customizable personality.

## Features

- **AI-Powered Responses**: Uses Venice API or local Ollama for generating responses
- **Conversation Memory**: Maintains in-memory context and persistent database history
- **User Profiling**: Tracks user behavior patterns, heated messages, and conversation history
- **Grudge System**: Remembers users who have disrespected the bot for future roasts
- **Image Understanding**: Describes and reacts to images shared in chat
- **URL Previews**: Fetches Open Graph metadata from shared links
- **Multiple Response Modes**:
  - `direct` - Responds when mentioned or replied to
  - `defend` - Stands up for users being bullied
  - `roast` - Fires back when attacked
  - `deescalate` - Cools down heated conversations
  - `chime` - Jumps into interesting conversations
  - `advice` - Shares wisdom when relevant
- **Owner Commands**: DM the bot to update personality, change models, or trigger posts

## Requirements

- Python 3.11+
- Telegram API credentials (api_id, api_hash)
- Venice API key (optional, for cloud LLM) OR Ollama running locally (for private LLM)

## Installation

```bash
# Clone the repository
git clone https://github.com/allinfinite/telegram_user_bot.git
cd telegram_user_bot

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## Configuration

Edit `.env` with your credentials:

```env
# Telegram credentials
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+1234567890

# Bot identity
BOT_NAME=MyBot
SELF_USERNAME=mybot

# Group chat to participate in
GROUP_CHAT_ID=-100123456789

# Owner usernames (for DM commands)
OWNER_USERNAMES=username1,username2

# Venice AI (cloud LLM)
VENICE_API_KEY=your_venice_key
VENICE_MODEL=llama-3.3-70b
VENICE_VISION_MODEL=mistral-31-24b

# OR Ollama (local LLM)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:14b
OLLAMA_VISION_MODEL=llava:7b

# Choose backend: "ollama" or "venice"
TEXT_BACKEND=venice

# Tuning
BUFFER_SIZE=50
MIN_MSG_GAP=15
RANDOM_CHIME_CHANCE=0.08
```

## Running the Bot

### Development

```bash
python3 bot.py
```

### Production (macOS launchd)

```bash
# Copy the launchd plist
cp com.telegrambot.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.telegrambot.bot.plist

# Start the bot
launchctl start com.telegrambot.bot

# View logs
tail -f bot.log
```

## Dashboard

A Flask web dashboard is included for managing the bot:

```bash
python3 dashboard.py
```

Access at `http://localhost:5555` (default password: `admin`)

### Dashboard Features

- **Overview**: Message stats, service status, recent logs
- **Personality**: Add/edit personality override instructions
- **Messages**: Search and filter stored messages
- **Grudges**: View and manage recorded grudges
- **Configuration**: Adjust models, tuning parameters, restart bot

## Owner Commands (DM the bot)

Send these commands in a private message to the bot:

### Personality Management

```
personality: You are now sarcastic and dry
show personality      # List active overrides
clear personality #1  # Clear specific override
clear personality     # Clear all overrides
```

### Backend Control

```
backend ollama        # Switch to local Ollama
backend venice        # Switch to Venice API
model llama-3.3-70b   # Change text model
vision model mistral-31-24b  # Change vision model
status                # Show current configuration
```

### Trigger Posts

```
chime in about [topic]    # Post about a specific topic
post in group             # Post something about current conversation
```

## Database

The bot uses SQLite (`bot.db`) to store:

- **messages**: All chat messages with timestamps, user info, and flags
- **personality_overrides**: Custom instructions that modify bot behavior
- **grudges**: Record of users who have disrespected the bot

## File Structure

```
telegram_bot/
├── bot.py           # Main bot application
├── dashboard.py     # Flask web dashboard
├── send.py          # One-off message sender
├── bot.db           # SQLite database (created on first run)
├── bot_session.session  # Telegram session file
├── bot.log          # Application logs
├── requirements.txt # Python dependencies
└── .env.example     # Configuration template
```

## Response Modes

The bot uses an LLM classifier to decide how to respond:

| Mode | Trigger | Behavior |
|------|---------|----------|
| direct | @mention or reply | Direct response to user |
| defend | Someone being bullied | Stand up for the victim |
| roast | Someone attacking the bot | Use their history against them |
| grudge_snipe | Known enemy posting | Take a shot at them |
| deescalate | Heated argument | Cool things down |
| chime | Interesting topic | Jump into conversation |
| advice | Relevant moment | Share wisdom |

## Customization

### System Prompt

Edit `SYSTEM_PROMPT` in `bot.py` to define your bot's base personality:

```python
SYSTEM_PROMPT = """You are {BOT_NAME}, a participant in a group chat.
- Customize this section with your bot's personality
- Define how they talk, what they care about
- Set communication style and boundaries
"""
```

### Heated Message Detection

The `HEATED_WORDS` regex pattern identifies aggressive language for tracking and response adjustment.

## License

MIT