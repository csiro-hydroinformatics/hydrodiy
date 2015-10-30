
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



#nval = 100
#
#for i, j in itertools.product(range(fig_nrows),
#                            range(fig_ncols)):
#
#    ax = fig.add_subplot(gs[i, j])
#
#    xx = np.random.uniform(size=(nval, 2))
#    x = xx[:,0]
#    y = xx[:,1]
#
#    # Scatter plot
#    ax.plot(x, y, 'o',
#        markersize=10,
#        mec='black',
#        mfc='pink',
#        alpha=0.5,
#        label='points')
#
#    # Decoration
#    ax.legend(frameon=True,
#        shadow=True,
#        fancybox=True,
#        framealpha=0.7,
#        numpoints=1)
#
#    ax.set_title('Title')
#    ax.set_xlabel('X label')
#    ax.set_xlabel('Y label')
#
#fig.suptitle('Overall title')
#
#
#
