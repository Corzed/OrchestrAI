import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Single console instance
console = Console()

def log_message(tag: str, message: str, level: str = "INFO") -> None:
    """Log message with timestamp and tag."""
    timestamp = time.strftime("%H:%M:%S")
    console.print(f"[{timestamp}] {level:<5} - {tag:<15} | {message}")

@contextmanager
def spinner(message: str):
    """Display spinner during operations."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
    )
    with progress:
        task = progress.add_task(message, start=True)
        try:
            yield
        except Exception as e:
            log_message("Error", str(e), level="ERROR")
            raise
        finally:
            progress.remove_task(task)
