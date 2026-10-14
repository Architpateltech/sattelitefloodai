def get_severity(area):


    if area < 10:

        return "Level 1 - Action Required"


    elif area < 100:

        return "Level 2 - Instant Action Required"


    else:

        return "Level 3 - Emergency"