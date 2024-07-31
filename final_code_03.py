# Import necessary libraries and modules
from typing import Callable, List
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentType, initialize_agent, load_tools

# Ensure you have your OpenAI API key set up in your environment
os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

# Define the DialogueAgent class
class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chat model to the message history and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

# Define the DialogueSimulator class
class DialogueSimulator:
    def __init__(self, agents: List[DialogueAgent], selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

# Define the DialogueAgentWithTools class
class DialogueAgentWithTools(DialogueAgent):
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI, tool_names: List[str], **tool_kwargs) -> None:
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)

    def send(self) -> str:
        """
        Applies the chat model to the message history and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content

# Agent Descriptions
agent_descriptions = {
    "Alex": "Alex is a strong advocate for remote work, emphasizing its flexibility and productivity benefits.",
    "Jordan": "Jordan is skeptical about remote work, focusing on potential downsides like reduced team interaction."
}

# Generate System Messages
def generate_system_message(name, description):
    return f"""Your name is {name}.
          Your description is as follows: {description}

          Your goal is to persuade your conversation partner of your point of view.

          DO look up information with your tool to refute your partner's claims.
          DO cite your sources.

          DO NOT fabricate fake citations.
          DO NOT cite any source that you did not look up.

          Do not add anything else.

          Stop speaking the moment you finish speaking from your perspective.
          """

agent_system_messages = {name: generate_system_message(name, description) for name, description in agent_descriptions.items()}

# Topic Specification
specified_topic = "The Impact of Remote Work on Employee Productivity"

# Agent Setup
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4", temperature=0.2),
        tool_names= ["arxiv", "ddg-search", "wikipedia"],
        top_k_results=2,
    ) for name, system_message in agent_system_messages.items()
]

# Speaker Selection Function
def select_next_speaker(step, agents):
    return step % len(agents)

# Running the Simulation
max_iters = 4
simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)
simulator.reset()
simulator.inject("Moderator", specified_topic)

for _ in range(max_iters):
    name, message = simulator.step()
    print(f"({name}): {message}")
