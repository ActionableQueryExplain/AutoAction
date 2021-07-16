from dataConfig import dataConfig


class dataLoader:
    """
    Load dataset configurations
    """

    def __init__(self):
        return

    # Load default dataset configs
    def loader(self, dataset, constraint_setting="default"):

        group_attrs = []
        other_attrs = []
        actionable_attrs = []
        agg_attr = []
        fit_attrs = []
        action_constraints = []
        instance_constraints = []
        appro_instance_constraints = []
        conditional_constraints = []
        appro_conditional_constraints = []
        path = ""

        if dataset == "adult":

            group_attrs = ["workclass", "occupation"]
            other_attrs = [
                "age",
                "fnlwgt",
                "education",
                "marital-status",
                "relationship",
                "race",
                "sex",
                "capital-gain",
                "capital-loss",
                "native-country",
            ]
            actionable_attrs = ["hours-per-week", "education-num"]
            agg_attr = ["Salary"]
            fit_attrs = [
                "age",
                "fnlwgt",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "education-num",
            ]
            path = "datasets/adult.csv"

            if constraint_setting != "none":
                if constraint_setting == "one":
                    action_constraints.append([0, 0, 1, 1, 2])
                elif constraint_setting == "two":
                    action_constraints.append([0, 0, 1, 1, 2])
                    action_constraints.append([-1, 2, -1, 2, -1])
                elif constraint_setting == "default":
                    action_constraints.append([0, 0, 1, 1, 2])
                    action_constraints.append([-1, 2, -1, 2, -1])
                    appro_conditional_constraints.append([1, -1, 0, 1, 12, -75])
                else:
                    raise ValueError("The constraint setting is a invalid value")

            print("training prediction model...")

        elif dataset == "student":

            group_attrs = ["StageID", "SectionID"]
            other_attrs = ["AnnouncementsView", "Discussion"]
            actionable_attrs = ["raisedhands", "VisITedResources"]
            agg_attr = ["Class"]

            fit_attrs = [
                "AnnouncementsView",
                "Discussion",
                "raisedhands",
                "VisITedResources",
            ]
            path = "datasets/student.csv"

            if constraint_setting != "none":
                if constraint_setting == "one":
                    action_constraints.append([-1, 2, -1, 2, -5])
                elif constraint_setting == "two":
                    action_constraints.append([-1, 2, -1, 2, -5])
                    appro_conditional_constraints.append([1, 1, 0, -1, -10, 10])
                elif constraint_setting == "default":
                    action_constraints.append([-1, 2, -1, 2, -5])
                    action_constraints.append([1, 2, 1, 2, 20])
                    appro_conditional_constraints.append([1, 1, 0, -1, -10, 10])
                else:
                    raise ValueError("The constraint setting is a invalid value")

        elif dataset == "credit100k" or dataset == "credit1m":

            group_attrs = ["g1", "g2"]
            other_attrs = ["u1", "u2"]
            actionable_attrs = ["x1", "x2"]
            agg_attr = ["y"]

            fit_attrs = ["g1", "g2", "u1", "u2", "x1", "x2"]
            if dataset == "credit100k":
                path = "datasets/credit100K.csv"
            else:
                path = "datasets/credit1M.csv"

            if constraint_setting != "none":
                if constraint_setting == "one":
                    action_constraints.append([0, 0, 1, 1, 1])
                elif constraint_setting == "two":
                    action_constraints.append([0, 0, 1, 1, 1])
                    action_constraints.append([-1, 2, -1, 2, -0.5])
                elif constraint_setting == "default":
                    action_constraints.append([0, 0, 1, 1, 1])
                    action_constraints.append([-1, 2, -1, 2, -0.5])
                    appro_conditional_constraints.append([0, 1, 1, 1, 15, 10])
                else:
                    raise ValueError("The constraint setting is a invalid value")

        else:
            raise ValueError("The dataset setting is a invalid value")

        dc = dataConfig(
            group_attrs=group_attrs,
            other_attrs=other_attrs,
            actionable_attrs=actionable_attrs,
            agg_attr=agg_attr,
            fit_attrs=fit_attrs,
            path=path,
            action_constraints=action_constraints,
            instance_constraints=instance_constraints,
            appro_instance_constraints=appro_instance_constraints,
            conditional_constraints=conditional_constraints,
            appro_conditional_constraints=appro_conditional_constraints,
            name=dataset,
        )

        return dc

    # Load customized dataset configs
    def customized_loader(
        self,
        group_attrs,
        other_attrs,
        actionable_attrs,
        agg_attr,
        fit_attrs,
        path,
        action_constraints,
        instance_constraints,
        appro_instance_constraints,
        conditional_constraints,
        appro_conditional_constraints,
        name,
    ):

        return dataConfig(
            group_attrs=group_attrs,
            other_attrs=other_attrs,
            actionable_attrs=actionable_attrs,
            agg_attr=agg_attr,
            fit_attr=fit_attrs,
            path=path,
            action_constraints=action_constraints,
            instance_constraints=instance_constraints,
            appro_instance_constraints=appro_instance_constraints,
            conditional_constraints=conditional_constraints,
            appro_conditional_constraints=appro_conditional_constraints,
            name=name,
        )
