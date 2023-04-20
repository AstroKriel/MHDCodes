from matplotlib import rcParams

## uncomment to see all options
# rcParams.keys()

## enable latex font
rcParams["text.usetex"]         = True
# rcParams["text.latex.preamble"] = r"\usepackage{bm, amsmath, mathrsfs, amssymb, url, xfrac}"
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

## set default font characteristics
rcParams["font.family"] = "DejaVu Sans"

SMALL_SIZE  = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

rcParams["font.size"]        = BIGGER_SIZE # controls default text sizes
rcParams["axes.titlesize"]   = BIGGER_SIZE # fontsize of the axes title
rcParams["axes.labelsize"]   = BIGGER_SIZE # fontsize of the x and y labels
rcParams["xtick.labelsize"]  = MEDIUM_SIZE # fontsize of the tick labels
rcParams["ytick.labelsize"]  = MEDIUM_SIZE # fontsize of the tick labels
rcParams["legend.fontsize"]  = MEDIUM_SIZE # legend fontsize
rcParams["figure.titlesize"] = SMALL_SIZE  # fontsize of the figure title

## set default lines
rcParams["lines.linewidth"] = 1.2
rcParams["axes.linewidth"]  = 0.8

## change x-axis characteristics
rcParams["xtick.top"]           = True
rcParams["xtick.direction"]     = "in"
rcParams["xtick.minor.visible"] = True
rcParams["xtick.major.size"]    = 6
rcParams["xtick.minor.size"]    = 3
rcParams["xtick.major.width"]   = 0.75
rcParams["xtick.minor.width"]   = 0.75
rcParams["xtick.major.pad"]     = 5
rcParams["xtick.minor.pad"]     = 5

## change y-axis characteristics
rcParams["ytick.right"]         = True
rcParams["ytick.direction"]     = "in"
rcParams["ytick.minor.visible"] = True
rcParams["ytick.major.size"]    = 6
rcParams["ytick.minor.size"]    = 3
rcParams["ytick.major.width"]   = 0.75
rcParams["ytick.minor.width"]   = 0.75
rcParams["ytick.major.pad"]     = 5
rcParams["ytick.minor.pad"]     = 5

## set default legend characteristics
rcParams["legend.fontsize"]     = 12
rcParams["legend.labelspacing"] = 0.2
rcParams["legend.loc"]          = "upper left"
rcParams["legend.frameon"]      = False

## set figure size/resolution
rcParams["figure.figsize"] = (8.0, 5.0)
# rcParams["figure.dpi"]     = 200

## set figure saving size/resolution
rcParams["savefig.dpi"]  = 200
rcParams["savefig.bbox"] = "tight"

## END OF STYLE SHEET