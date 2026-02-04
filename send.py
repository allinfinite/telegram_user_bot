"""Send a one-off message without disrupting the running bot.

Usage:
    python3 send.py <chat_id> "message text"

Briefly borrows the session (bot will auto-reconnect via launchd).
"""

import sys
import os
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
SESSION = os.getenv("SESSION_NAME", os.path.join(os.path.dirname(__file__), "bot_session"))


async def main():
    if len(sys.argv) < 3:
        print("Usage: python3 send.py <chat_id> \"message\"")
        sys.exit(1)

    chat_id = int(sys.argv[1])
    text = sys.argv[2]

    client = TelegramClient(SESSION, API_ID, API_HASH)
    await client.connect()
    await client.send_message(chat_id, text)
    print(f"Sent to {chat_id}")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
