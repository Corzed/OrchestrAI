import time
from contextlib import contextmanager

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Create a Rich Console instance for pretty logging.
console = Console()

def log_message(tag: str, message: str, level: str = "INFO") -> None:
    """
    Logs a message with a timestamp, severity level, and a tag.
    """
    timestamp = time.strftime("%H:%M:%S")
    console.print(f"[{timestamp}] {level.upper():<5} - {tag:<15} | {message}")

@contextmanager
def spinner(message: str):
    """
    Context manager that displays a spinner with a message during long operations.
    Uses the Rich library's progress spinner.
    """
    progress = Progress(
        SpinnerColumn(style=""),
        TextColumn("{task.description}", style=""),
        transient=True,  # Remove the spinner after completion.
    )
    with progress:
        task = progress.add_task(message, start=True)
        try:
            yield  # Execute the code within the context.
        except Exception as e:
            log_message("Spinner", f"Error: {e}", level="ERROR")
            raise
        finally:
            progress.remove_task(task)
            time.sleep(0.2)  # Pause briefly for user visibility.
