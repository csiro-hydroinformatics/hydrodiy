library(geometry)

ch <- 'D:\\gis\\AUSTRALIA\\australia_oz'
ch <- '/home/magpie/Dropbox/code/src/mytools/mygis'
setwd(ch)

options(warn=0)
library(oz)
options(warn=2)

getdata <- function(sec, nm){
	xy <- NULL
    n <- length(sec)
	for(i in 1:n){
		# Append data
        kk <- sec[i]
	    bnd <- ozRegion(sections=kk)$lines
        dd <- data.frame(item=nm, id=kk, x=bnd[[1]]$x, y=bnd[[1]]$y)
		xy <- rbind(xy, dd)
    }
	return(xy)
}

sections <- list()
sections$QLD <- c(3, 13, 11, 12)
sections$TAS <- 6
sections$NSW <- c(4, 15, 14, 13)
sections$SA <- c(7, 8, 10, 12, 14, 16)
sections$NT <- c(2, 9, 10, 11)
sections$WA <- c(1, 9, 8)
sections$VIC <- c(5, 15, 16)
sections$coast_mainland <- c(1:5, 7)

# Get states boundaries
xy <- NULL
for(i in 1:length(sections)){
	tmp <- getdata(sections[[i]], names(sections)[i])
	xy <- rbind(xy, tmp)
}

# draw mask boundaries for mainland
tmp <- xy[xy$item=='coast_mainland',]
kk <- which(tmp$id == 1)
x0 <- tmp$x[1]
y0 <- tmp$y[1]
xy_mask1 <- NULL 
for(i in unique(tmp$id)) xy_mask1 <- rbind(xy_mask1,tmp[tmp$id==i,])
xy_mask1$item <- 'mask_mainland'
xy_mask1 <- rbind(xy_mask1, 
                 data.frame(item='mask_mainland', id=991, 
                            x=c(x0, x0, 170, 170, 100, 100, x0, x0), 
                            y=c(y0, -40.1, -40.1, 0, 0, -40.1, -40.1, y0)))

tmp <- xy[xy$item=='TAS',]
uu <- which(tmp$y == max(tmp$y))
x0 <- tmp$x[uu]
y0 <- tmp$y[uu]
xy_mask2 <- rbind(tmp[uu:nrow(tmp),], 
                    tmp[1:(uu-1),], 
                    data.frame(item='mask_tas', id=991, 
                            x=c(x0, x0, 100, 100, 170, 170, x0, x0), 
                            y=c(y0, -39.9, -39.9, -50, -50, -39.9, -39.9, y0)))
xy_mask2$item <- 'mask_tas'

xy <- rbind(xy, xy_mask1, xy_mask2)

#plot(xy$x, xy$y, pch="")
#
#for(st in unique(xy$item)[1:7]){
#    kk <- xy$item==st
#    for(id in unique(xy$id[kk])){
#        ii <- kk & xy$id==id
#        lines(xy$x[ii], xy$y[ii], type="l")
#        nv <- length(which(ii))
#    }
#}
#ii <- xy$item == 'mask_mainland'
#polygon(xy$x[ii], xy$y[ii], col='pink', border='grey')
#ii <- xy$item == 'mask_tas'
#polygon(xy$x[ii], xy$y[ii], col='skyblue', border='grey')
#
#points(x0,y0)

fxy <- "data/oz_data.csv"
write.csv(xy, fxy, quote=FALSE, row.names=FALSE)


# Print py code
tab <- '    '
item <- c(sprintf("%s%s%s[", tab,tab,tab),
		sprintf("%s%s%s%s'%s',", tab,tab,tab,tab,xy$item),
		sprintf("%s%s%s],", tab,tab,tab))
id <- c(sprintf("%s%s%s[", tab,tab,tab),
		sprintf("%s%s%s%s%d,", tab,tab,tab,tab,xy$id),
		sprintf("%s%s%s],", tab,tab,tab))
x <- c(sprintf("%s%s%s[", tab,tab,tab),
		sprintf("%s%s%s%s%0.5f,", tab,tab,tab,tab,xy$x),
		sprintf("%s%s%s],", tab,tab,tab))
y <- c(sprintf("%s%s%s[", tab,tab,tab),
		sprintf("%s%s%s%s%0.5f,", tab,tab,tab,tab,xy$y),
		sprintf("%s%s%s]", tab,tab,tab))

xy_py <- c("import pandas as pd",
	"import numpy as np",
	"from matplotlib import pyplot as plt",
	"from matplotlib.path import Path",
	"import matplotlib.patches as patches",
	"",
	"class oz_map:",
	sprintf("%s''' retrieve Australia coast lines and state boundaries '''",tab),
	"",
	sprintf("%sdef __init__(self):",tab),
	sprintf("%s%sdata={'item':",tab, tab))
xy_py <- c(xy_py,
	item,
	sprintf("%s%s'id':",tab,tab),id,
	sprintf("%s%s'x':",tab,tab),x,
	sprintf("%s%s'y':",tab,tab),y,
	sprintf("%s%s}",tab,tab),
	sprintf("%s%sself.xy = pd.DataFrame(data)",tab,tab),
	"",
	sprintf("%sdef get_xy(self, item):",tab),
	sprintf("%s%sidx = self.xy.item==item",tab,tab),
	sprintf("%s%sreturn self.xy.ix[idx]",tab,tab),
	"",
	sprintf("%sdef get_range(self, item):",tab),
	sprintf("%s%s%s", tab,tab,"''' Get x/y range for an item '''"),
	sprintf("%s%sxy = self.get_xy(item)",tab,tab),
	sprintf("%s%smini = xy.min()",tab,tab),
	sprintf("%s%smaxi = xy.max()",tab,tab),
	sprintf("%s%sreturn (mini['x'], maxi['x'], mini['y'], maxi['y'])",tab,tab),
	"",
	sprintf("%sdef get_range_nice(self):",tab),
	sprintf("%s%s%s", tab,tab,"''' Get nice x/y range for Australian coastline '''"),
	sprintf("%s%sreturn (113.0, 154.0, -45.26, -8.73)",tab,tab),
	"",
	sprintf("%sdef plot_items(self, ax, items=['coast_mainland','TAS']):",tab),
	sprintf("%s%s%s", tab,tab,"''' plot a list of items '''"),
	sprintf("%s%sif type(items) is not list:", tab,tab),
	sprintf("%s%s%sitems = [items]", tab,tab,tab),
	sprintf("%s%sfor it in items:", tab,tab),
	sprintf("%s%s%sxy = self.get_xy(it)", tab,tab,tab),
	sprintf("%s%s%sids = np.unique(xy.id)", tab,tab,tab),
	sprintf("%s%s%sfor id in ids:", tab,tab,tab),
	sprintf("%s%s%s%sidx = xy.id==id", tab, tab,tab, tab),
	sprintf("%s%s%s%sax.plot(xy.x[idx], xy.y[idx], 'k-')", tab, tab,tab, tab),
	"",
	sprintf("%sdef plot_coast(self, ax):",tab),
	sprintf("%s%sself.plot_items(ax, ['TAS', 'coast_mainland'])",tab, tab),
	"",
	sprintf("%sdef plot_states(self, ax):",tab),
	sprintf("%s%sstates = ['NSW', 'QLD', 'NT', 'VIC', 'WA', 'TAS', 'SA']",tab, tab),
	sprintf("%s%sself.plot_items(ax, states)",tab, tab),
	"",
	sprintf("%sdef plot_mask(self, ax):",tab),
	sprintf("%s%s%s", tab,tab,"''' plot the coastline mask '''"),
	sprintf("%s%sfor m in ['mask_mainland','mask_tas']:", tab,tab),	
	sprintf("%s%s%sxy = self.get_xy(m)", tab, tab, tab),	
	sprintf("%s%s%snval = len(xy)", tab,tab, tab),	
	sprintf("%s%s%sverts = zip(xy.x, xy.y)", tab,tab, tab),	
	sprintf("%s%s%scodes = [Path.LINETO]*nval", tab,tab, tab),	
	sprintf("%s%s%scodes[0] = Path.MOVETO", tab,tab, tab),	
	sprintf("%s%s%scodes[-1] = Path.CLOSEPOLY", tab,tab, tab),	
	sprintf("%s%s%spath = Path(verts, codes)", tab,tab, tab),	
	sprintf("%s%s%spatch = patches.PathPatch(path, facecolor='white', edgecolor='none')", tab,tab, tab),	
	sprintf("%s%s%sax.add_patch(patch)", tab, tab, tab)
	)

fxy <- "oz.py"
writeLines(xy_py, fxy)






