from function_schema import get_function_schema

import json, datetime
from typing import Annotated, Any

from .llm_client import LLMClient


# >>>>> exmple function >>>>>


def create_calendar_event(
    summary: Annotated[str, "Title of the event to be added (default: 'New Event')"],
    start_time: Annotated[
        str, "Start date and time of the event (format: 'yyyy-MM-dd HH:mm')"
    ],
    end_time: Annotated[
        str, "End date and time of the event (format: 'yyyy-MM-dd HH:mm')"
    ],
) -> None:
    """Creates a new calendar event."""
    # 내부 구현은 생략 (pass)
    # 실제 구현 시에는 캘린더 API를 호출하여 이벤트를 생성합니다.
    # 예: google_calendar.create_event(summary=summary, start=start_time, end=end_time)
    pass


def fetch_calendar_events(
    start_date: Annotated[str, "Start date of the search range (format: yyyy-MM-dd)"],
    end_date: Annotated[str, "End date of the search range (format: yyyy-MM-dd)"],
) -> str:
    """
    Retrieves calendar events within a specified date range.
    Requires authorization first. If not authorized, should call authorize_calendar_access.
    Returns a JSON string representing the events or an error message.
    """
    # 내부 구현은 생략 (pass)
    # 실제 구현 시에는 인증 상태 확인 후 캘린더 API를 호출합니다.
    # is_authorized = check_auth()
    # if not is_authorized:
    #     return json.dumps({"message": "You need to authorize the assistant to access your calendar."})
    # try:
    #     events = calendar_api.fetch_events(start_date, end_date)
    #     return json.dumps(events)
    # except Exception as e:
    #     return json.dumps({"message": f"Error fetching calendar events: {e}"})
    pass


def web_search(query: Annotated[str, "The query to search for on the web."]) -> str:
    """
    Searches the web (DuckDuckGo) for the given query.
    Returns a JSON string containing search results.
    """
    # 내부 구현은 생략 (pass)
    # 실제 구현 시에는 웹 검색 라이브러리나 API를 호출합니다.
    # results = duckduckgo_search(query)
    # return json.dumps(results)
    pass


# <<<<< example function <<<<<


class ToolManager:
    def __init__(self, llm: LLMClient):
        tools = [
            create_calendar_event,
            fetch_calendar_events,
            web_search,
        ]
        tool_schemas = [get_function_schema(tool) for tool in tools]
        self.tool_schemas = tool_schemas
        self.llm = llm

    def get_tools(self):
        return self.tool_schemas

    def invoke_tool(self, tool_name: str, params: dict) -> Any:
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                return self.__virtual_tool_executor(tool, params, schema=tool)

        available_tools = [tool["name"] for tool in self.tool_schemas]
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        )

    def __virtual_tool_executor(
        self, tool_name: str, params: dict, schema: dict
    ) -> Any:

        prompt = f"""You are an expert function simulator. Based on the following function description and the provided arguments, simulate the execution of this function call.

        Function Name: {tool_name}

        Function Description: {schema['description']}

        Function Schema:
        {json.dumps(schema, indent=2)}

        Arguments Provided:
        {json.dumps(params, indent=2)}

        Current Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Assume this is the time of execution)

        Task:
        Generate a plausible JSON response string that represents what the function '{tool_name}' would return if it were actually executed with the given arguments.
        - Consider the function's description (e.g., does it fetch data, create something, authorize, search?).
        - Consider the argument values (e.g., dates, search terms).
        - If the function description mentions potential errors (like needing authorization for 'fetch_calendar_events'), sometimes simulate those error responses.
        - If the function returns nothing on success (like 'create_calendar_event' or 'authorize_calendar_access'), return a JSON indicating success, like '{{"status": "success"}}' or an empty JSON object '{{}}'.
        - For functions returning data (like 'fetch_calendar_events' or 'web_search'), generate realistic-looking example data formatted as a JSON string.
        - Ensure your entire output is ONLY the JSON string, without any introductory text, explanations, or markdown formatting like ```json ... ```. Just the raw JSON string.
        """

        response, _ = self.llm.json_output(
            system_prompt="You are an expert function simulator outputting only JSON strings.",
            prompt=prompt,
            reasoning=True,
        )
        return response


if __name__ == "__main__":
    tool_manager = ToolManager(llm=LLMClient())

    # Get available tools
    tools = tool_manager.get_tools()
    print("Available tools:", tools)

    # Invoke a specific tool with parameters
    try:
        result = tool_manager.invoke_tool(
            "fetch_calendar_events",
            {
                "start_date": "2023-10-01",
                "end_date": "2023-10-31",
            },
        )
        print("Tool invocation result:", result)
    except ValueError as e:
        print(e)
