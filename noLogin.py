from openai import AsyncOpenAI
import chainlit as cl

API_KEY = ""
BASE_URL = ""

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
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant. Answer every dialogue in Chinese."}],
    )
    welcome_message = "我是基于llama-3.1-405b的API接口的聊天机器人，请随时向我提问 :)"
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    try:
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

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await cl.Message(content=error_message).send()

