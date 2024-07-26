from openai import AsyncOpenAI
import chainlit as cl

API_KEY = "sk-N0bqh035SgYk0qcowyzfLIGSi214nsjj6ENFbZf0LpOzDBRA"
BASE_URL = "https://api.deepbricks.ai/v1/"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

settings = {
    "model": "llama-3.1-405b",
    "temperature": 0.7,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant. You reply in the same language that the user uses."}],
    )


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
