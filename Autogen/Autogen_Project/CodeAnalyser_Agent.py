from autogen_agentchat.agents import CodeExecutorAgent
import asyncio
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


# Autogen has a code excecutor to execute the codes
# here wew are using code executor and it works on the contain space which is docker where it has a space
async def main():

    docker = DockerCommandLineCodeExecutor(work_dir="/tmp", timeout=120)

    code_executor_agent = CodeExecutorAgent(
        name="CodeExecutorAgent",
        code_executor=docker,
    )

    task = TextMessage(
        content="""Here is the code 
```python
print("Hello, World!")
```
    """,
        source="user",
    )

    await docker.start()

    try:
        result = await code_executor_agent.on_messages(
            messages=[task], cancellation_token=CancellationToken()
        )
        print("The result is", result)
    finally:
        await docker.stop()  # 👈 Ensures clean shutdown


if __name__ == "__main__":
    asyncio.run(main())
