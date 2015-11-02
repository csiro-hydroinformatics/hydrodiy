
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



class Simplot(object):

    def __init__(self, \
        fig, \
        obs, \
        sim, \
        nfloods=4):

        # Figure to draw on
        self.fig = fig

        # Number of flood events
        self.nfloods = nfloods

        # Grid spec
        fig_ncols = 1 + nfloods/2
        fig_nrows = 3
        self.gs = gridspec.GridSpec(fig_nrows, fig_ncols,
                width_ratios=[1] * fig_ncols,
                height_ratios=[1] * fig_nrows)

        # data
        self.idx = pd.notnull(obs) & (obs >= 0)
        self.data = pd.DataFrame(obs)
        self.data = self.data.join(sim)

    def draw_fdc(self):
        pass

    def draw_floods(self):
        pass

    def draw_annual(self):
        pass

    def draw_balance(self):
        pass



