from utils import load
from utils import fit, MODEL


class dataConfig:
    def __init__(
        self,
        group_attrs,
        other_attrs,
        actionable_attrs,
        agg_attr,
        fit_attrs,
        path,
        action_constraints=[],
        instance_constraints=[],
        appro_instance_constraints=[],
        conditional_constraints=[],
        appro_conditional_constraints=[],
        name="none",
    ):
        """
        Generate dataset configurations
        Args:
            group_attrs: List
                Group attributes
            other_attrs: List
                Other attributes
            actionable_attrs: List
                Actionable attributes
            agg_attr: List
                Aggregation attribute
            fit_attrs: List
                Attributes to fit the model
            path: String
                path to dataset
            action_constraints, instance_constraints,
            appro_instance_constraints, conditional_constraints,
            appro_conditional_constraints: List
                Input constraints
            name: String
                Dataset name
        """

        self.group_attrs = group_attrs
        self.other_attrs = other_attrs
        self.actionable_attrs = actionable_attrs
        self.agg_attr = agg_attr
        self.fit_attrs = fit_attrs
        self.path = path
        self.action_constraints = action_constraints
        self.instance_constraints = instance_constraints
        self.appro_instance_constraints = appro_instance_constraints
        self.conditional_constraints = conditional_constraints
        self.appro_conditional_constraints = appro_conditional_constraints
        self.name = name
        self.data = self.load_data()
        self.X, self.y, self.model = fit(
            self.data, self.fit_attrs, self.agg_attr, MODEL
        )

    def load_data(self):
        attrs = (
            self.group_attrs + self.other_attrs + self.actionable_attrs + self.agg_attr
        )
        self.data = load(self.path, attrs)
        return self.data
