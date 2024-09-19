from opendevin.core.exceptions import (
    InvalidSummaryResponseError,
    LLMMalformedActionError,
    LLMResponseError,
)
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.utils import json
from opendevin.events.action.agent import AgentSummarizeAction
from opendevin.events.event import EventSource
from opendevin.events.serialization.action import action_from_dict

WORD_LIMIT = 200
MESSAGE_SUMMARY_WARNING_FRAC = 0.75

# SUMMARY_PROMPT_SYSTEM = """
# Your job is to summarize a history of previous messages in a conversation between an AI persona and a human. The conversation you are given is a from a fixed context window and may not be complete. Keep your summary less than {WORD_LIMIT} words, do NOT exceed this word limit. Include all the critical information in conversation till now in the summary.
# Only output the summary, do NOT include anything else in your output.
# Given the following actions and observations, create a JSON response with:
#     - "action": "summarize"
#     - args:
#       - "summarized_actions": A precise sentence summarizing all the provided actions, written in the first person.
#       - "summarized_observations": A few precise sentences summarizing all the provided observations, written in the third person.
# Example:
# {
#     "action": "summarize",
#     "args": {
#         "summarized_actions": "I located the UML specification PDF, parsed its content, and searched for information about sequence diagrams.",
#         "summarized_observations": "The agent encountered a UnicodeDecodeError when initially searching the PDF text, but was able to resolve this by installing the PyPDF2 library and successfully extracting relevant information about sequence diagrams."
#     }
# }
# Make sure to include in observations any relevant information that the agent needs to remember.
# %(events)s
# """

SUMMARY_PROMPT_SYSTEM = """
Your job is to summarize a history of previous messages in a conversation between an AI persona and a human.
The conversation you are given is a from a fixed context window and may not be complete.
Messages sent by the AI are marked with the 'assistant' role.
Messages the user sends are in the 'user' role.
The 'user' role is also used for important system events, such as login events and heartbeat events (heartbeats run the AI's program without user action, allowing the AI to act without prompting from the user sending them a message).
Summarize what happened in the conversation from the perspective of the AI (use the first person).
Include all the critical information in conversation till now in the summary.
Make sure that your response is in the following format. Given the following observations, create a JSON response with:
    - "action": "summarize"
    - args:
      - "summary": A precise sentence summarizing all the provided actions, and all the provided observations, written in the third person.

Here's an example of what the response should look like. Example:
{
    "action": "summarize",
    "args": {
        "summary": "The agent located the UML specification PDF, parsed its content, and searched for information about sequence diagrams. The agent encountered a UnicodeDecodeError when initially searching the PDF text, but was able to resolve this by installing the PyPDF2 library and successfully extracting relevant information about sequence diagrams.",
    }
}
Keep your summary less than {WORD_LIMIT} words, do NOT exceed this word limit.
Only output the summary, do NOT include anything else in your output.
%(events)s
"""


def parse_summary_response(response: str) -> AgentSummarizeAction:
    """
    Parses a JSON summary of events.

    Parameters:
    - response: The response string to be parsed
    Returns:
    - The summary action output by the model
    """
    try:
        action_dict = json.loads(response)
        action = action_from_dict(action_dict)
        if action is None or not isinstance(action, AgentSummarizeAction):
            error_message = f'Expected a summarize action, but the response got {str(type(action)) if action else None}'
            logger.error(error_message)
            raise InvalidSummaryResponseError(error_message)
        action._source = EventSource.AGENT  # type: ignore
    except (LLMResponseError, LLMMalformedActionError) as e:
        logger.error(f'Failed to parse summary response: {str(e)}')
        raise InvalidSummaryResponseError(
            f'Failed to parse the response: {str(e)}'
        ) from e
    return action
