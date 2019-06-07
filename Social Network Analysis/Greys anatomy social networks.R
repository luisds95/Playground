# December 16, 2018.
# Luis Da Silva.
library(ergm)

# Start by reading in the adjancecy matrix showing relationships between pairs of characters
ga.mat<-as.matrix(read.table("Grey's Anatomy - sociomat.tsv", sep="\t",
                             header=T, row.names=1, quote="\""))
# check it:
ga.mat

# Next import the attribute file
ga.atts <- read.table("Grey's Anatomy - attributes.tsv", sep="\t",
                    header=T, quote="\"",
                    stringsAsFactors=F, strip.white=T, as.is=T)
# check it  and familiarise yourself with the attributes available:
ga.atts
ga.atts$sex = as.logical(model.matrix(~ sex, data=ga.atts)[,-1])

# create the network object to use for the coursework tasks
ga.net <- network(ga.mat, vertex.attr=ga.atts,
                vertex.attrnames=colnames(ga.atts),
                directed=F, hyper=F, loops=F, multiple=F, bipartite=F)
# check it:
ga.net

# Visualise the network, colour nodes based gender and include labels (names of characters)
plot(ga.net, vertex.col=c("blue","red")[1+(get.vertex.attribute(ga.net, "sex")==0)],  
     label=get.vertex.attribute(ga.net, "name"), label.cex=.7)   # label.cex determines the label size

# ERGM
model1 <- ergm(ga.net~edges+nodematch("sex")+nodematch("position")+degree(c(1))) 
summary(model1)

# simulation----
model1.sim <- simulate(model1,nsim=10)
class(model1.sim)
summary(model1.sim)
plot(model1.sim[[3]], vertex.col=c("blue","red")[1+(get.vertex.attribute(ga.net, "sex")==0)])

# Goodness of Fit----
model1.gof <- gof(model1~degree)
model1.gof
par(mfrow=c(2,1))   # Separate the plot window into a 2 by 1 orientation
plot(model1.gof)
dev.off()
