from typing import Optional, Sequence

from pydantic import BaseModel, model_validator

from .errors import InconsistentLCIA


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
                (
                    "Impact categories in `normalizations` not present in `impact_categories`: "
                    + f"{difference}"
                )
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
    def normalizations_cover_all_impact_categories(self):
        if not self.normalizations:
            return self
        missing = set(self.impact_categories).difference(
            set(ic for lst in self.normalizations.values() for ic in lst)
        )
        if missing:
            raise InconsistentLCIA(
                f"Normalization not provided for all impact categories; missing {missing}"
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
        if not self.weightings or not self.normalizations:
            return self
        overlap = set(self.weightings).intersection(set(self.normalizations))
        if overlap:
            raise ValueError(f"Weighting identifiers overlap normalization identifiers: {overlap}")
        return self

    @model_validator(mode="after")
    def weightings_cant_have_mixed_references(self):
        if not self.weightings or not self.normalizations:
            return self
        normalization_references = set(
            nor for lst in self.weightings.values() for nor in lst
        ).intersection(set(self.normalizations))
        ic_references = set(nor for lst in self.weightings.values() for nor in lst).intersection(
            set(self.impact_categories)
        )
        if normalization_references and ic_references:
            raise InconsistentLCIA(
                "Weightings must reference impact categories or normalizations, not both"
            )
        return self

    @model_validator(mode="after")
    def weightings_cover_all_impact_categories(self):
        if not self.weightings:
            return self
        references = set(nor for lst in self.weightings.values() for nor in lst)
        missing = set(self.impact_categories).difference(references)
        if references.intersection(self.impact_categories) and missing:
            raise InconsistentLCIA(
                f"Weighting not provided for all impact categories; missing {missing}"
            )
        return self

    @model_validator(mode="after")
    def weightings_cover_all_normalizations(self):
        if not self.weightings or not self.normalizations:
            return self
        references = set(nor for lst in self.weightings.values() for nor in lst)
        missing = set(self.normalizations).difference(references)
        if references.intersection(self.normalizations) and missing:
            raise InconsistentLCIA(
                f"Weighting not provided for all normalizations; missing {missing}"
            )
        return self
