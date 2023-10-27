from typing import Optional, Sequence

from pydantic import BaseModel, model_validator


class MethodConfig(BaseModel):
    impact_categories: Sequence[tuple[str, ...]]
    normalizations: Optional[dict[tuple[str, ...], list[tuple[str, ...]]]] = None
    weightings: Optional[dict[tuple[str, ...], list[tuple[str, ...]]]] = None

    @model_validator(mode="after")
    def normalizations_reference_impact_categories(self):
        if not self.normalizations:
            return self
        references = set.union(*[set(lst) for lst in self.normalizations.values()])
        difference = references.difference(set(self.impact_categories))
        if difference:
            raise ValueError(
                f"Impact categories in `normalizations` not present in `impact_categories`: {difference}"
            )
        return self

    @model_validator(mode="after")
    def normalizations_unique_from_impact_categories(self):
        if not self.normalizations:
            return self

        references = set.union(*[set(lst) for lst in self.normalizations.values()])
        overlap = set(self.normalizations).intersection(references)
        if overlap:
            raise ValueError(
                f"Normalization identifiers overlap impact category identifiers: {overlap}"
            )
        return self

    @model_validator(mode="after")
    def weightings_reference_impact_categories_or_normalizations(self):
        if not self.weightings:
            return self

        if self.normalizations:
            possibles = set(self.normalizations).union(set(self.impact_categories))
        else:
            possibles = set(self.impact_categories)

        references = set.union(*[set(lst) for lst in self.weightings.values()])
        difference = set(references).difference(possibles)
        if difference:
            raise ValueError(
                f"`weightings` refers to missing impact categories or normalizations: {difference}"
            )
        return self

    @model_validator(mode="after")
    def weightings_unique_from_impact_categories(self):
        if not self.weightings:
            return self
        overlap = set(self.weightings).intersection(set(self.impact_categories))
        if overlap:
            raise ValueError(
                f"Weighting identifiers overlap impact category identifiers: {overlap}"
            )
        return self

    @model_validator(mode="after")
    def weightings_unique_from_normalizations(self):
        if not self.weightings:
            return self
        if self.normalizations:
            overlap = set(self.weightings).intersection(set(self.normalizations))
            if overlap:
                raise ValueError(
                    f"Weighting identifiers overlap normalization identifiers: {overlap}"
                )
        return self
