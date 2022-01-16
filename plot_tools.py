import matplotlib

colors = {
    'dark grey':[0.5]*3+[1.0]#'#FF9d9d9d'
}
style_colors = {
    'spines': colors['dark grey'],
    'ticks': colors['dark grey']
}
def ax_style(ax,spine_lims=[None]*4,keep=['left','bottom']):
    """
    Removes spines except those in 'keep'.
    Colors them dark grey.
    
    """
    all_sides = ['top','right','left','bottom']

    for side in list(set(all_sides)-set(keep)):
        ax.spines[side].set_visible(False)#.set_color('none'
    for side in keep:
        ax.spines[side].set_color(style_colors['spines'])
        ax.spines[side].set_linewidth(1.3)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which=u'both', color=style_colors['spines'])
    for side,spine_lim in zip(all_sides,spine_lims):
        if spine_lim is not None:
            ax.spines[side].set_bounds(*spine_lim)

def fig_labels(fig,axes):
    alphabets = list(string.ascii_uppercase)
    for ax,alpbt in zip(axes,alphabets):
        ax.set_title(alpbt,va='bottom',ha='right',loc='left',fontweight='bold')