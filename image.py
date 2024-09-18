import chainlit as cl


@cl.on_chat_start
async def start():
    image = cl.Image(path="./cat.jpg", name="image1", display="inline")

    # Attach the image to the message
    await cl.Message(
        content="This message has an image!",
        elements=[image],
    ).send()
