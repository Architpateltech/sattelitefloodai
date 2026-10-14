def get_severity(area):
    """
    Returns a severity level string based on the affected area in km².

    Levels:
      Level 1 - Low       : area < 10 km²
      Level 2 - Moderate  : 10 km² <= area < 100 km²
      Level 3 - Emergency : area >= 100 km²
    """

    if area < 10:
        return "Level 1 - Low Severity"

    elif area < 100:
        return "Level 2 - Moderate (Action Required)"

    else:
        return "Level 3 - Emergency (Instant Action Required)"
