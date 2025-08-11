"""
Centralized logging configuration using Loguru.

This module sets up structured logging for the RAG system with:
- Console logging with colors
- File logging with rotation
- JSON formatting for structured logs
- Context binding for request tracing
"""

import os
import sys
from pathlib import Path

from config import config
from loguru import logger

# Remove default logger
logger.remove()


def setup_logging():
    """Configure Loguru with console and file logging based on config settings."""

    # Create logs directory if it doesn't exist
    log_dir = Path(config.LOG_FILE_PATH)
    log_dir.mkdir(exist_ok=True)

    # Console logging setup
    if config.LOG_TO_CONSOLE:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=config.LOG_LEVEL,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File logging setup
    if config.LOG_TO_FILE:
        # General application logs
        logger.add(
            log_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level=config.LOG_LEVEL,
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        # Debug-level logs for development
        logger.add(
            log_dir / "debug.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        # Error-only logs for monitoring
        logger.add(
            log_dir / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
            level="ERROR",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

        # API-specific logs with structured JSON format
        logger.add(
            log_dir / "api.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | API | {extra} | {message}",
            level="INFO",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            compression="zip",
            filter=lambda record: "api" in record["extra"],
        )


def get_logger(name: str = None):
    """
    Get a logger instance with optional name binding.

    Args:
        name: Optional logger name for identification

    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(logger_name=name)
    return logger


def get_api_logger(name: str = None):
    """
    Get a logger instance specifically for API logging.

    Args:
        name: Optional logger name for identification

    Returns:
        Logger instance configured for API logging
    """
    if name:
        return logger.bind(api=True, logger_name=name)
    return logger.bind(api=True)


def log_execution_time(func_name: str = None):
    """
    Decorator to log function execution time.

    Args:
        func_name: Optional custom function name for logging
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            start_time = time.time()

            name = func_name or f"{func.__module__}.{func.__name__}"
            bound_logger = logger.bind(function=name)

            try:
                bound_logger.debug(f"Starting execution of {name}")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                bound_logger.info(f"Completed {name} in {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                bound_logger.error(f"Error in {name} after {execution_time:.3f}s: {e}")
                raise

        return wrapper

    return decorator


# Initialize logging when module is imported
setup_logging()

# Create module-level logger instance
log = get_logger("rag_system")
