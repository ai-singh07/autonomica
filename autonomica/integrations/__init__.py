"""Integration exports."""
from autonomica.integrations.langchain import wrap_langchain_tools

def wrap_crewai_tools(*args, **kwargs):
    from autonomica.integrations.crewai import wrap_crewai_tools as _wrap
    return _wrap(*args, **kwargs)

__all__ = ["wrap_langchain_tools", "wrap_crewai_tools"]
