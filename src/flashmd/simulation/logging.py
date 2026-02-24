"""Logging configuration for MLCG simulation using loguru."""

import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

# Remove default handler to avoid duplicate logs
logger.remove()

# Module-level flag to track if file logging is configured
_file_handler_id = None


def setup_simulation_logging(
    filename: str = None,
    output_dir: str = "./outputs",
    level: str = "INFO",
    add_timestamp: bool = True,
):
    """
    Configure loguru for simulation logging.

    Sets up both console and file logging. The file log includes a timestamp
    in the filename for easy identification of runs.

    Parameters
    ----------
    filename : str, optional
        Base name or full path for the log file. If None, only console logging
        is enabled. If filename contains a directory path, output_dir is ignored.
    output_dir : str, optional
        Directory for log files. Default is "./outputs". Ignored if filename
        already contains a directory path.
    level : str, optional
        Logging level. Default is "INFO".
    add_timestamp : bool, optional
        Whether to add timestamp to log filename. Default is True.

    Returns
    -------
    str or None
        Path to the log file if file logging is enabled, None otherwise.
    """
    global _file_handler_id

    # Add console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level=level,
        colorize=True,
    )

    log_file_path = None

    if filename is not None:
        filename_path = Path(filename)

        # Check if filename already contains a directory path
        if filename_path.parent != Path("."):
            # filename includes directory - use it directly
            output_path = filename_path.parent
            base_filename = filename_path.name
        else:
            # filename is just a name - use output_dir
            output_path = Path(output_dir)
            base_filename = filename

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Build log filename with optional timestamp
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{base_filename}_{timestamp}.log"
        else:
            log_filename = f"{base_filename}.log"

        log_file_path = output_path / log_filename

        # Remove previous file handler if exists
        if _file_handler_id is not None:
            logger.remove(_file_handler_id)

        # Add file handler
        _file_handler_id = logger.add(
            log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
            level=level,
            rotation="100 MB",
            retention="7 days",
        )

        logger.info(f"Log file: {log_file_path}")

    return str(log_file_path) if log_file_path else None


# Export logger for use in other modules
__all__ = ["logger", "setup_simulation_logging"]
