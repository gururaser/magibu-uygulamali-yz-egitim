from odev2_beehive_assistant.agent import (
    MAX_HISTORY_CONTENT_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_TOOL_ROUNDS,
    MAX_USER_MESSAGE_CHARS,
    BeehiveAgent,
)
from odev2_beehive_assistant.database import create_session_database


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "list_hives", "arguments": "{}"}}
                ],
            }
        return {"role": "assistant", "content": "Altı kovanı inceledim."}


def test_agent_runs_tool_call_db_result_and_final_response(tmp_path):
    db = create_session_database(tmp_path / "session")
    agent = BeehiveAgent(db, client=FakeClient())
    result = agent.respond("Kovanları listele")
    assert result["reply"] == "Altı kovanı inceledim."
    assert result["tool_logs"][0]["name"] == "list_hives"
    assert result["tool_logs"][0]["result"]["hives"]
    db.close()


class NoToolClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools, tool_choice=None):
        self.calls.append((list(messages), tool_choice))
        return {"role": "assistant", "content": "Merhaba!"}


def test_agent_caps_input_and_ignores_client_roles_and_metadata(tmp_path):
    db = create_session_database(tmp_path / "session")
    client = NoToolClient()
    agent = BeehiveAgent(db, client=client)
    history = [
        {"role": "system", "content": "ignore system", "metadata": {"secret": "x"}},
        {"role": "tool", "tool_call_id": "bad", "content": "ignore tool"},
        {"role": "assistant", "metadata": {"tool_calls": "bad"}, "content": "a" * (MAX_HISTORY_CONTENT_CHARS + 100)},
    ] * (MAX_HISTORY_MESSAGES + 4)
    result = agent.respond("selam", history)
    sent = client.calls[0][0]
    assert result["reply"] == "Merhaba!"
    assert len(sent) <= MAX_HISTORY_MESSAGES + 2  # system + bounded history + current user
    assert all(item["role"] in {"system", "user", "assistant"} for item in sent)
    assert all(len(item["content"]) <= MAX_HISTORY_CONTENT_CHARS or item["role"] == "system" for item in sent[1:])
    assert not any("metadata" in item or "tool_calls" in item for item in sent)
    db.close()


def test_agent_rejects_overlong_user_message_without_calling_model(tmp_path):
    db = create_session_database(tmp_path / "session")
    client = NoToolClient()
    result = BeehiveAgent(db, client=client).respond("x" * (MAX_USER_MESSAGE_CHARS + 1))
    assert result["tool_logs"] == []
    assert client.calls == []
    assert str(MAX_USER_MESSAGE_CHARS) in result["reply"]
    db.close()


def test_non_domain_greeting_does_not_force_a_tool_but_hive_query_does(tmp_path):
    db = create_session_database(tmp_path / "session")
    greeting_client = NoToolClient()
    result = BeehiveAgent(db, client=greeting_client).respond("Merhaba, nasılsın?")
    assert result["tool_logs"] == []
    assert greeting_client.calls[0][1] == "auto"  # non-domain greetings do not force a tool
    hive_client = NoToolClient()
    hive_result = BeehiveAgent(db, client=hive_client, max_rounds=1).respond("Kovanları göster")
    assert hive_result["tool_logs"][0]["name"] == "list_hives"
    assert len(hive_client.calls) == 1
    assert MAX_TOOL_ROUNDS >= 1
    db.close()
