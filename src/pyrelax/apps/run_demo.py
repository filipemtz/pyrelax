import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main():
    app_path = str(Path(__file__).parent / "streamlit_demo.py")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
