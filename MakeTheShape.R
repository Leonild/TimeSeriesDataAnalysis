library(dplyr)
library(tidyr)
library(sp)
library(raster)
library(rgeos)
library(rgbif)
library(viridis)
library(gridExtra)
library(rasterVis)

if (!require("parallel")) install.packages("parallel")
if (!require("sp")) install.packages("sp")
if (!require("rgdal")) install.packages("rgdal")
if (!require("maptools")) install.packages("maptools")
gpclibPermit()
gpclibPermitStatus()
if (!require("dplyr")) install.packages("dplyr")
if (!require("rgeos")) install.packages("rgeos")
if (!require("reshape2")) install.packages("reshape2")
if (!require("labeling")) install.packages("labeling")
if (!require("spdep")) install.packages("spdep")
if (!require("data.table")) install.packages("data.table")
if (!require("bit")) install.packages("bit")


args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) < 2) {
  stop("At least two argument must be supplied: (1) the hexagono dimension side! AND (2) The shapefile diretory path", call.=FALSE)
} else if (length(args)>2) {
  # default output file
  stop("Only two argumente must be supplied: (1) the hexagono dimension side! AND (2) The shapefile diretory path", call.=FALSE)
}



# read shape file
#urbanmetro_mal <- readOGR(dsn="/home/leonildo/Dropbox/ICMC-USP/Doutorado/codigos/Granularity/shapes/California", layer="grid")
#urbanmetro_mal <- readOGR(dsn="./California", layer="grid")
urbanmetro_mal <- readOGR(dsn=args[2])

# Assign projection in UTM
myprojection_utm <- CRS("+proj=utm +zone=33 +ellps=GRS80 +units=m +no_defs")
urbanmetro_mal <- spTransform(urbanmetro_mal, myprojection_utm)


###### 2 Function to create hexagonal grid -----------

HexGrid <- function(mycellsize, originlpolygon, clip = FALSE) { 
  
  # Define size of hexagon bins in meters to create points
  HexPts <- spsample(originlpolygon, type="hexagonal", offset=c(0,0), cellsize=mycellsize)
  
  # Create Grid - transform into spatial polygons
  HexPols <- HexPoints2SpatialPolygons(HexPts)
  
  # convert to spatial polygon data frame
  df <- data.frame(idhex = getSpPPolygonsIDSlots(HexPols))
  row.names(df) <- getSpPPolygonsIDSlots(HexPols)
  hexgrid <- SpatialPolygonsDataFrame(HexPols, data =df)
  
  # clip to boundary of study area
  if (clip) {
    hexgrid <- gIntersection(hexgrid, originlpolygon, byid = TRUE)
  } else {
    hexgrid <- hexgrid[x, ]
  }
  # clean up feature IDs
  row.names(hexgrid) <- as.character(1:length(hexgrid))
  
  return(hexgrid)
}

###### 3: Create Hexagonal grid -------------------

# parameter: meters side
hex_got <- HexGrid(as.numeric(args[1]), urbanmetro_mal, clip = TRUE)
#hex_got <- HexGrid(500000, urbanmetro_mal, clip = TRUE)

#plot(urbanmetro_mal, col = "grey50", bg = "light blue", axes = FALSE)
#plot(hex_got, border = "orange", add = TRUE)
#box()
path_grid <- paste0(args[2], "/", args[1])
dir.create(path_grid) #creating the directory path
hex_got <- spTransform(hex_got, CRS("+proj=longlat +datum=WGS84"))
#raster::shapefile(hex_got, "/home/leonildo/Dropbox/ICMC-USP/Doutorado/codigos/Granularity/shapes/hex-grid.shp", overwrite=FALSE)
raster::shapefile(hex_got, paste0(path_grid,"/", "hex-grid.shp"), overwrite=TRUE)