import numpy as np


class AreaCalculator:


    def __init__(self, pixel_size=10):

        # Satellite resolution
        # Example Sentinel-2 = 10 meters

        self.pixel_size = pixel_size



    def calculate(self, mask):


        affected_pixels = np.sum(
            mask == 1
        )


        pixel_area = (
            self.pixel_size *
            self.pixel_size
        )


        area_m2 = (
            affected_pixels *
            pixel_area
        )


        area_km2 = (
            area_m2 / 1000000
        )


        return round(
            area_km2,
            2
        )