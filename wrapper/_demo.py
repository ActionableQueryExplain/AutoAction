from autoaction import autoAction
from dataConfig import dataLoader


class autoaction_demo:
    """
    The demo for autoaction.
    Args:
        ds_config: String
            Dataset used
        time: Integer
            Time budget(s)
    """

    def __init__(self, ds_config, time):

        self.ds_config = ds_config
        self.time = time

    def run(self):
        dl = dataLoader()
        dataset = dl.loader(dataset=self.ds_config)

        autoAction(dataset=dataset, t=self.time, demo=True)
