import operator
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph.checkpoint.sqlite")

from langgraph.graph import END, START, StateGraph  # noqa: E402

from genesis_l0.adapters.langgraph import GenesisL0CheckpointSaver  # noqa: E402


class State(TypedDict):
    steps: Annotated[list, operator.add]


def _graph(saver):
    builder = StateGraph(State)
    builder.add_node("a", lambda s: {"steps": ["a"]})
    builder.add_node("b", lambda s: {"steps": ["b"]})
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    return builder.compile(checkpointer=saver)


def test_checkpoints_persist_across_savers(db_path):
    cfg = {"configurable": {"thread_id": "t1"}}
    saver = GenesisL0CheckpointSaver.from_path(db_path)
    assert saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    graph = _graph(saver)
    graph.invoke({"steps": []}, cfg)
    assert graph.invoke({"steps": []}, cfg)["steps"] == ["a", "b", "a", "b"]
    assert len(list(graph.get_state_history(cfg))) >= 2
    saver.close()

    reopened = GenesisL0CheckpointSaver.from_path(db_path)
    assert _graph(reopened).get_state(cfg).values["steps"] == ["a", "b", "a", "b"]
    assert reopened.get_tuple({"configurable": {"thread_id": "missing"}}) is None
    reopened.close()
