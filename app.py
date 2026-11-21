"""Run the standalone CineComic web application."""

from src.web_server import app


if __name__ == "__main__":
    import os

    app.run(
        host=os.environ.get("CINECOMIC_HOST", "127.0.0.1"),
        port=int(os.environ.get("CINECOMIC_PORT", "8501")),
        debug=False,
        threaded=True,
    )
