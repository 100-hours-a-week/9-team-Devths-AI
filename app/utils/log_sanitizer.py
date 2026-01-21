"""
Log sanitization utilities to prevent log injection attacks.
"""


def sanitize_for_log(value: str, max_length: int = 100) -> str:
    """
    Sanitize user input for safe logging to prevent log injection attacks.

    Args:
        value: The string value to sanitize
        max_length: Maximum length to truncate to

    Returns:
        Sanitized string safe for logging
    """
    if not value:
        return ""

    # Remove all ASCII control characters (code points < 32 and = 127)
    # Only allow printable ASCII characters (32-126)
    sanitized = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in value).strip()

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized
