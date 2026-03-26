"""
Launch the Gradio app without opening a browser.

Useful for local testing from an existing virtual environment:
    python gui/launch_local.py
"""

import os

from app import create_interface


def main():
    host = os.environ.get("MRAF_HOST", "127.0.0.1")
    port = int(os.environ.get("MRAF_PORT", "7860"))
    open_browser = os.environ.get("MRAF_INBROWSER", "0") == "1"

    demo = create_interface()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        inbrowser=open_browser,
    )


if __name__ == "__main__":
    main()
