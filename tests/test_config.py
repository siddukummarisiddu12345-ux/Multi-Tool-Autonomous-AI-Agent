from src.config import DEFAULT_MODEL, Settings


def test_defaults_without_keys():
    s = Settings.from_env({})
    assert not s.has_llm and not s.has_search and s.model == DEFAULT_MODEL and s.secrets == ()


def test_reads_env_and_alias():
    s = Settings.from_env({"GEMINI_API_KEY": " k123456 ", "GEMINI_MODEL": "m-x", "TAVILY_API_KEY": "t123456",
                           "MAX_STEPS": "99", "MAX_RETRIES": "abc"})
    assert s.google_api_key == "k123456" and s.model == "m-x" and s.has_search
    assert s.max_steps == 12 and s.max_retries == 2


def test_repr_hides_secrets():
    assert "k123456" not in repr(Settings(google_api_key="k123456"))
