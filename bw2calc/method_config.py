from typing import Optional, Sequence

from pydantic import BaseModel, model_validator

from .errors import InconsistentLCIA


class MethodConfig(BaseModel):
    """
    A class that stores the logical relationships between impact categories, normalization, and
    weighting.

    The basic object in all three categories is an identifying tuple, i.e. tuples of strings. These
    tuples have no length restrictions.

    `impact_categories` is a list of tuples which identify each impact category (`bw2data.Method`).

    `normalizations` link normalization factors to impact categories. They are optional. If
    provided, they take the form of a dictionary, with keys of tuples which identify each
    normalization (`bw2data.Normalization`), and values of *lists* of impact categories tuples.

    If `normalizations` is defined, **all** impact categories must have a normalization.

    `weightings` link weighting factors to either normalizations *or* impact categories. They are
    optional. If provided, they take the form of a dictionary, with keys of tuples which identify
    each weighting (`bw2data.Weighting`), and values of *lists* of normalizations or impact
    categories tuples. They keys identify the weighting data, and the values refer to either
    impact categories or normalizations - mixing impact categories and normalizations is not
    allowed.

    If `normalizations` is defined, **all** impact categories or normalizations must have a
    weighting.

    The identifying tuples for `impact_categories`, `normalizations`, and `weightings` must all be
    unique.

    Example
    -------

    ```python
    {
        "impact_categories": [
            ("climate change", "100 years"),
            ("climate change", "20 years"),
            ("eutrophication",),
        ],
        "normalizations": {
            ("climate change", "global normalization"): [
                ("climate change", "100 years"),
                ("climate change", "20 years"),
            ],
            ("eut european reference", "1990"): [
                ("eutrophication",),
            ]
        },
        "weightings": {
            ("climate change", "bad"): [
                ("how bad?", "dead", "people")
            ],
            ("eutrophication", "also bad"): [
                ("how bad?", "dead", "fish")
            ]
        }
    }
    ```

    """

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
