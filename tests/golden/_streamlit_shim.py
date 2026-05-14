"""Shim that lets ``streamlit``-decorated functions run inside a plain Python
script. Import this module BEFORE any ``src.modules.*`` import.

It does two things:
  1. Sets up a minimal session-state replacement so code that touches
     ``st.session_state`` outside ``streamlit run`` does not raise.
  2. Silences the noisy "missing ScriptRunContext" warnings.
"""

from __future__ import annotations

import logging
import warnings


def install() -> None:
    # Silence the warnings Streamlit emits when called outside ``streamlit run``.
    for name in (
        "streamlit",
        "streamlit.runtime.scriptrunner.script_run_context",
        "streamlit.runtime.state.session_state_proxy",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="streamlit")

    import streamlit as st

    # Replace session_state with a dict-with-attribute-access so writes succeed.
    class _SS(dict):
        def __getattr__(self, k):
            if k in self:
                return self[k]
            raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

        def __delattr__(self, k):
            if k in self:
                del self[k]
            else:
                raise AttributeError(k)

    # Streamlit's ``session_state`` is a property that raises without context.
    # Patch the module attribute to return a real dict.
    try:
        del st.__class__.session_state  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    st.session_state = _SS()  # type: ignore[assignment]


install()
