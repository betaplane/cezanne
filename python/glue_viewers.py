from glue.viewers.custom.qt import CustomViewer

class DailyCycle(CustomViewer):
    name = 'Daily Cycle'
    x = 'att(e)'

    def plot_data(self, axes, x):
        axes.plot(x.T, color='grey')

    def plot_subset(self, axes, x, style):
        axes.plot(x.T, color=style.color)
