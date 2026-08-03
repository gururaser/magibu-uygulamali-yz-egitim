import warnings
import inspect


def test_gradio_app_imports_without_starting_server_or_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from odev2_beehive_assistant.app import build_demo

        demo = build_demo()
    assert demo is not None
    assert not [warning for warning in caught if "moved from the Blocks constructor" in str(warning.message)]


def test_ui_has_one_explicit_send_button_and_enter_submission():
    from odev2_beehive_assistant import app

    source = inspect.getsource(app.build_demo)
    assert "submit_btn=" not in source
    assert "send.click(" in source
    assert "message.submit(" in source
