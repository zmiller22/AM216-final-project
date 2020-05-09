library(nat.nblast)
library(nat)

# Read in the reflected nrns as a neuronlist
nrn.dir = "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/Zbrain_neurons_flipped_right"
nrns <- read.neurons(nrn.dir, format="swc")

### Get class labels ###

# Look at the neurons to make sure they look ok
open3d()
plot3d(nrns, soma=TRUE)

# Convert to dotprops
nrns.dps <- dotprops(nrns, k=5, resample=1, OmitFailures=TRUE)

# Get metadata of dotprops for later
lbl_df <- as.data.frame(nrns.dps)

# Calculate NBLAST scores and cluster
nrn.scores <- nblast_allbyall(nrns.dps)
nrn.clust <- nhclust(scoremat=nrn.scores, normalisation="mean", labels=FALSE)

# Look at the Dendrogram with 10 cluster labels
d.nrns <- colour_clusters(nrn.clust, k=10)
plot(d.nrns, main="Heiarchical Clustering of Neurons split into 10 classes", leaflab="none")

# Get the labels for the neurons based on NBLAST
lbls <-cutree(nrn.clust, k=10)
lbl_df$nblast_cluster <- lbls

### Get graph representations ###
graph.dir <- "/home/zack/Desktop/Lab_Work/Data/neuron_morphologies/Zebrafish/aligned_040120/Zbrain_neurons_graphs"

# Iterate over all neurons and save it as a gml graph in gml format into graph.dir folder
# with the file name as the name of the neuron
for(nrn in nrns) {
  nrn.name <- nrn$NeuronName
  nrn.graph <- as.ngraph(nrn)
  graph.path <- paste(graph.dir, "/", nrn.name, sep="")
  igraph::write_graph(test, graph.path, format="gml")
}


