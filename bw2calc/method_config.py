from typing import Iterable, Optional
from pydantic import BaseModel, model_validator


class MethodConfig(BaseModel):
    impact_categories: Iterable[tuple[str, ...]]
    normalizations: Optional[dict[tuple[str, ...], tuple[str, ...]]] = None
    weightings: Optional[dict[tuple[str, ...], tuple[str, ...]]] = None

    @model_validator(mode='after')
    def normalizations_reference_impact_categories(self):
        if not self.normalizations:
            return self
        difference = set(self.normalizations).difference(set(self.impact_categories))
        if difference:
            raise ValueError(f"Impact categories in `normalizations` not present in `impact_categories`: {difference}")
        return self

    @model_validator(mode='after')
    def unique_normalizations(self):
        if self.normalizations:
            overlap = set(self.normalizations.values()).intersection(set(self.impact_categories))
            if overlap:
                raise ValueError(f"Normalization identifiers overlap impact category identifiers: {overlap}")
        return self

    @model_validator(mode='after')
    def weightings_reference_impact_categories_or_normalizations(self):
        if not self.weightings:
            return self
        possibles = set(self.impact_categories)
        if self.normalizations:
            possibles = possibles.union(set(self.normalizations.values()))
        difference = set(self.weightings).difference(possibles)
        if difference:
            raise ValueError(f"`weightings` refers to missing impact categories or normalizations: {difference}")
        return self

    @model_validator(mode='after')
    def unique_weightings_to_impact_categories(self):
        if not self.weightings:
            return self
        overlap = set(self.weightings.values()).intersection(set(self.impact_categories))
        if overlap:
            raise ValueError(f"Weighting identifiers overlap impact category identifiers: {overlap}")
        return self

    @model_validator(mode='after')
    def unique_weightings_to_normalizations(self):
        if not self.weightings:
            return self
        if self.normalizations:
            overlap = set(self.weightings.values()).intersection(set(self.normalizations))
            if overlap:
                raise ValueError(f"Weighting identifiers overlap normalization identifiers: {overlap}")
        return self
