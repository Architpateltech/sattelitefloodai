import logging

logger = logging.getLogger(__name__)


class Analytics:

    def __init__(self, database):
        self.database = database

    def get_statistics(self) -> dict:
        """
        Aggregate statistics from the analysis history.

        Returns:
            dict with keys:
                'cases'    – total number of analyses
                'area'     – total affected area in km²
                'disasters'– dict mapping disaster type to count
        """
        history = self.database.get_history()

        total_cases = len(history)
        total_area = 0.0
        disasters: dict = {}

        for row in history:
            # Row layout: (id, date, disaster, confidence, area, severity)
            disaster = row[2]
            area = row[4]

            try:
                total_area += float(area)
            except (ValueError, TypeError):
                logger.warning("Could not parse area value '%s' in row %s", area, row[0])

            disasters[disaster] = disasters.get(disaster, 0) + 1

        return {
            "cases": total_cases,
            "area": round(total_area, 2),
            "disasters": disasters,
        }