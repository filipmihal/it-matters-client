class Feature:
    def __init__(self, positive_regions, negative_regions):
        # arrays of regions
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
    
    def compute(self, integral):
        return sum(integral.get_region(positive) for positive in self.positive_regions) - sum(integral.get_region(negative) for negative in self.negative_regions)
