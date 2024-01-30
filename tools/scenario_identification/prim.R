# Vehicle analysis
# Jawad Mroueh 29May18
# Output data from .mat file
# ANALYSIS SCRIPT
setwd("~/Documents/Michigan/Research/Driving/matlab_data/")
rm(list = ls())

# Load all required libraries
library(dplyr)
library(tidyr)
library(plotly)
library(scatterplot3d)
library(plot3D)
library(plot3Drgl)
library(jpeg) 
library(png)
library(grid)
library(colorRamps)
library(scatterplot3d)
library(reshape2)
library(gridExtra)
library(dtw)
#############################################################################################
#############################################################################################

# Put in filename here to be read (confirm correct format)
df <- read.csv('small_data.csv')         # 882MB csv file / 65MB Matlab file   (Largest file)
# df <- read.csv('data38028.csv')       # 225MB csv file / 17MB Matlab file   (Middle)
# df <- read.csv('data40543.csv')       # 72MB csv file / 5MB Matlab file     (Smallest file)

df$Time <- df$Time / 1000000   # Time in seconds

# Option for scientific notation ## getOption("scipen") # 0  ### options(scipen=999)

# Latitude and Longitude degree to meter
latd <- 111321   # meters / degree latitude
latitude <- 42.28  # average latitude of dataset
# latitude <-   mean(df$LA_1) 
beta <- atan(0.99664719 * tan(latitude * pi/180))
lond <- pi/180 * 6378137 * cos(beta)  # meters / degree longitude

#Normalize distance
rotate_t <- function(df) {
  for (i in 1:nrow(df)) {
    df$x1
  }
}

#Check on distance
angle <- function(x,y) {
  rotation = atan(abs(y / x))
  if (x>= 0 & y>=0) {
    return(pi - rotation)
  }
  if (x <0 & y >= 0) {
    return(rotation)
  }
  if (x >= 0 & y < 0) {
    return(pi + rotation)
  }
  return(2 * pi - rotation)
}

get_rel_distance <- function(encounter) {
  plotdf <- df %>% filter(ID == encounter) %>% dplyr::arrange(Time)
  
  # Add useful columns to plotdf (make part of function later)
  plotdf$t <- plotdf$Time - min(plotdf$Time)
  plotdf$x1 <- (plotdf$LO_1 - plotdf$LO_1[1]) * lond
  plotdf$y1 <- (plotdf$LA_1 - plotdf$LA_1[1]) * latd
  plotdf$x2 <- (plotdf$LO_2 - plotdf$LO_1[1]) * lond
  plotdf$y2 <- (plotdf$LA_2 - plotdf$LA_1[1]) * latd
  
  # Encounter plot
  #  scale_alpha_continuous()
  plotdf$relx <- plotdf$x2 - plotdf$x1                                           # x2 - x1 in meters
  plotdf$rely <- plotdf$y2 - plotdf$y1
  plotdf$reldist <- sqrt(plotdf$relx^2 + plotdf$rely^2)
  plotdf <- plotdf[plotdf$reldist < 100,]
  if (nrow(plotdf) == 0) {
    return(NULL)
  }
  
  plotdf <- plotdf[!duplicated(plotdf$t),]
  plotdf$norm_t <- (plotdf$t - min(plotdf$t)) / max(plotdf$t - min(plotdf$t))

  rotation <- angle(plotdf$relx[1], plotdf$rely[1])
  rotation_matrix <- diag(rep(cos(rotation), 2))
  rotation_matrix[1,2] <- -sin(rotation)
  rotation_matrix[2,1] <- sin(rotation)
  plotdf$r_x1 <- NA
  plotdf$r_y1 <- NA
  plotdf$r_x2 <- NA
  plotdf$r_y2 <- NA
  
  plotdf[, c("r_x1", "r_y1")] <- as.matrix(plotdf[, c("x1", "y1")]) %*% t(rotation_matrix)
  plotdf[, c("r_x2", "r_y2")] <- as.matrix(plotdf[, c("x2", "y2")]) %*% t(rotation_matrix)
  plotdf$r_relx <- plotdf$r_x2 - plotdf$r_x1 
  plotdf$r_rely <- plotdf$r_y2 - plotdf$r_y1 
  plotdf$r_reldist <- sqrt(plotdf$r_relx^2 + plotdf$r_rely^2) 
  
  return(plotdf)
}

get_interpolated_data <- function(df, var, encounter_list, normalize = F) {
  if (normalize) {
    return(lapply(encounter_list, function(encounter) {
      approx(df$norm_t[df$encounter == encounter],
             df[df$encounter == encounter, var] / 
             max(df[df$encounter == encounter, "reldist"]),
             xout = seq(0, 1, by = 0.01))}))
  } else {
    return(lapply(encounter_list, function(encounter) {
      approx(df$norm_t[df$encounter == encounter],
             df[df$encounter == encounter, var],
             xout = seq(0, 1, by = 0.01))}))
  }
}

get_dist_m <- function(interpolated_data) {
  dist_m <- matrix(0, nrow = length(interpolated_data), ncol = length(interpolated_data))
  for (i in 1:(length(interpolated_data) - 1)) {
    for (j in (i + 1):length(interpolated_data)) {
      max_diff = max(abs(interpolated_data[[i]]$y - interpolated_data[[j]]$y), na.rm = T)
      if (max_diff)
        dist_m[i, j] = max_diff
      dist_m[j, i] = dist_m[i,j]
    }
  }
  dist_m
}

cv_for_fold <- function(data, fold, k, ...) {
  kmean_results <- kmeans(data[-fold,], centers = k, ...)
  fold_error <- apply(data[fold,], 1, function(row) {
    min(apply(kmean_results$centers, 1, function(center) {
      norm(row - center, type = "2")}))
  })
  mean(fold_error)
} 

run_cv <- function(p_dist, num_groups = 30, norm = F) {
  if (norm) {
    p_dist <- p_dist / max(p_dist)
  }
  p_dist_mds <- cmdscale(p_dist, k = 10)
  cv_folds <- split(sample(nrow(p_dist_mds)), 1:10)
  mclapply(2:num_groups, function(k) {
    print(k);
    mean(sapply(cv_folds, function(fold) {
    cv_for_fold(as.matrix(p_dist_mds), fold, k, iter.max=30)
  }))}, mc.cores = 3)
}

#Calculates the Procrustes distance with reflection between two encounters
#Assumes x1_m, x2_m are the trajectories from the first encounter
#Assumes y1_m, y2_m are the trajectories from the second encounter
calc_p_dist <- function(x1_v, x2_v, y1_v, y2_v, normalize = F) {
  x1_centered <- x1_v - mean(c(x1_v, x2_v))
  x2_centered <- x2_v - mean(c(x1_v, x2_v))
  y1_centered <- y1_v - mean(c(y1_v, y2_v))
  y2_centered <- y2_v - mean(c(y1_v, y2_v))
  if (normalize) {
    if (all(c(x1_v, x2_v) != 0)) {
      x1_centered <- x1_centered / max(abs(c(x1_v, x2_v)))
      x2_centered <- x2_centered / max(abs(c(x1_v, x2_v)))
    }
    if (all(c(y1_v, y2_v) != 0)) {
      y1_centered <- y1_centered / max(abs(c(y1_v, y2_v)))
      y2_centered <- y2_centered / max(abs(c(y1_v, y2_v)))
    }
  }
  2* min(-sum(diag(tcrossprod(c(y1_centered, y2_centered), c(x1_centered, x2_centered)))),
         -sum(diag(tcrossprod(c(y1_centered, y2_centered), c(x2_centered, x1_centered))))) +
    norm(c(x1_centered, x2_centered), type = "2")^2 +
    norm(c(y1_centered, y2_centered), type = "2")^2
}

#Calculates the Procrustes distance without reflection between two encounters
#Assumes x1_m, x2_m are the trajectories from the first encounter
#Assumes y1_m, y2_m are the trajectories from the second encounter
calc_p_dist_no_reflection <- function(x1_m, x2_m, y1_m, y2_m, normalize = F) {
  x1_m <- as.matrix(x1_m)
  x2_m <- as.matrix(x2_m)
  y1_m <- as.matrix(y1_m)
  y2_m <- as.matrix(y2_m)
  x1_centered <- sweep(x1_m, 2, colMeans(rbind(x1_m, x2_m)), "-")
  x2_centered <- sweep(x2_m, 2, colMeans(rbind(x1_m, x2_m)), "-")
  y1_centered <- sweep(y1_m, 2, colMeans(rbind(y1_m, y2_m)), "-")
  y2_centered <- sweep(y2_m, 2, colMeans(rbind(y1_m, y2_m)), "-")
  if (normalize) {
    if (all(c(x1_m, x2_m) != 0)) {
      x1_centered <- x1_centered / max(abs(c(x1_m, x2_m)))
      x2_centered <- x2_centered / max(abs(c(x1_m, x2_m)))
    }
    if (all(c(y1_m, y2_m) != 0)) {
      y1_centered <- y1_centered / max(abs(c(y1_m, y2_m)))
      y2_centered <- y2_centered / max(abs(c(y1_m, y2_m)))
    }
  }
  svd_decomp_1 <- svd(crossprod(y1_centered, x1_centered) + crossprod(y2_centered, x2_centered))
  svd_decomp_2 <- svd(crossprod(y1_centered, x2_centered) + crossprod(y2_centered, x1_centered))
  2 * min(-(svd_decomp_1$d[1] + svd_decomp_1$d[2] * 
            det(crossprod(svd_decomp_1$v, svd_decomp_1$u))),
          -(svd_decomp_2$d[1] + svd_decomp_2$d[2] * 
            det(crossprod(svd_decomp_2$v, svd_decomp_2$u)))) +
    sum(x1_centered^2) + sum(x2_centered^2) + 
    sum(y1_centered^2) + sum(y2_centered^2)
}

#Calculates the Procrustes distance matrix
get_p_dist_m <- function(chg_pt_data, normalize = F, speed = F, traj = F) {
  dist_m <- matrix(0, nrow = length(chg_pt_data), ncol = length(chg_pt_data))
  for (i in 1:(nrow(dist_m) - 1)) {
    for (j in (i + 1):nrow(dist_m)) {
      cat(i, j, "\n")
      if (traj) {
        dist_m[i, j] = calc_p_dist(chg_pt_data[[i]][, "x1"], 
                                   chg_pt_data[[i]][, "y1"], 
                                   chg_pt_data[[j]][, "x1"],
                                   chg_pt_data[[j]][, "y1"], normalize) +
          calc_p_dist(chg_pt_data[[i]][, "x2"], 
                      chg_pt_data[[i]][, "y2"], 
                      chg_pt_data[[j]][, "x2"],
                      chg_pt_data[[j]][, "y2"], normalize)
      } else {
        dist_m[i, j] = calc_p_dist(chg_pt_data[[i]][, "x1"], 
                                   chg_pt_data[[i]][, "x2"], 
                                   chg_pt_data[[j]][, "x1"],
                                   chg_pt_data[[j]][, "x2"], normalize) +
          calc_p_dist(chg_pt_data[[i]][, "y1"], 
                      chg_pt_data[[i]][, "y2"], 
                      chg_pt_data[[j]][, "y1"],
                      chg_pt_data[[j]][, "y2"], normalize)
      }
      if (speed) {
        dist_m[i, j] = dist_m[i, j] + 
          norm(chg_pt_data[[i]][, "speed_1"] - chg_pt_data[[j]][, "speed_1"], type = "2")^2 +
          norm(chg_pt_data[[i]][, "speed_2"] - chg_pt_data[[j]][, "speed_2"], type = "2")^2
      }
      dist_m[j, i] = dist_m[i,j]
    }
  }
  dist_m
}

#Calculates the Procrustes distance matrix using the chg_pt_data
get_entire_p_dist_m <- function(chg_pt_data, reflection = T) {
  #dist_m <- matrix(0, nrow = length(chg_pt_data), ncol = length(chg_pt_data))
  dist_m <- do.call(rbind, mclapply(2:length(chg_pt_data) - 1, function(i) {
    dist_m_row <- rep(0, length(chg_pt_data))
    for (j in (i + 1):length(chg_pt_data)) {
      cat(i, j, "\n")
      if (reflection) {
        dist_m_row[j] = calc_p_dist(c(chg_pt_data[[i]][, "x1"],
                                      chg_pt_data[[i]][, "y1"]),
                                    c(chg_pt_data[[i]][, "x2"], 
                                      chg_pt_data[[i]][, "y2"]),
                                    c(chg_pt_data[[j]][, "x1"],
                                      chg_pt_data[[j]][, "y1"]),
                                    c(chg_pt_data[[j]][, "x2"], 
                                      chg_pt_data[[j]][, "y2"]))
      } else {
        dist_m_row[j] = calc_p_dist_no_reflection(
          chg_pt_data[[i]][, c("x1", "y1")],
          chg_pt_data[[i]][, c("x2", "y2")],
          chg_pt_data[[j]][, c("x1", "y1")],
          chg_pt_data[[j]][, c("x2", "y2")])
      }
    }
    dist_m_row
  }, mc.cores = 4))
  dist_m <- rbind(dist_m, 0)
  dist_m = t(dist_m) + dist_m
}

make_group_plots <- function(data_m, assignment, id, sample_encounters = NULL,
                             cluster_max = max(assignment)) {
  #assignment_m <- as.data.frame(cbind(id, assignment))
  assignment_m <- data.frame("id" = as.character(id), 
                             "assignment" = assignment, 
                             stringsAsFactors = F)
  if (is.null(sample_encounters)) {
    random_rep = unlist(sapply(1:max(assignment_m$assignment), function(i) {
      sample(assignment_m$id[assignment_m$assignment == i], 
             min(5, sum(assignment_m$assignment == i)))}))
    sample_encounters = random_rep
    # sample_encounters = c(random_rep, 
    #                       sample(setdiff(assignment_m$id, random_rep), 
    #                              100 - length(random_rep)))
  }
  data_m$cluster <- unlist(lapply(unique(data_m$encounter), 
                                  function(id) {rep(assignment_m$assignment[which(assignment_m$id == id)], 
                                                    sum(data_m$encounter == id))}))
  do.call(grid.arrange, c(lapply(1:cluster_max, function(i) {
    ggplot(data_m[data_m$encounter %in% sample_encounters &
                    data_m$cluster == i,]) + 
      geom_path(aes(x = x1, y = y1, color = encounter, group = encounter), size = 2) +
      geom_path(aes(x = x2, y = y2, color = encounter, group = encounter), size = 2) +
      geom_point(data = data_m[data_m$encounter %in% sample_encounters &
                                 data_m$cluster == i & 
                                 data_m$norm_t == 0,],
                 aes(x = x1, y = y1, color = encounter), size = 4) +
      geom_point(data = data_m[data_m$encounter %in% sample_encounters &
                                 data_m$cluster == i & 
                                 data_m$norm_t == 0,],
                 aes(x = x2, y = y2, color = encounter), size = 4) +
      xlab("") + ylab("") + 
      ggtitle(paste("Cluster = ", i)) +
      theme(legend.text = element_text(size = 20), 
            legend.title=element_blank(), 
            legend.position="none")}), "ncol" = ceiling(sqrt(length(unique(assignment)))))) 
}

rel_dist <- lapply(unique(df$ID), function(encounter) {cbind(get_rel_distance(encounter), encounter)})
rel_dist_m <- as.data.frame(do.call(rbind, rel_dist))
rel_dist_m$encounter <- as.factor(rel_dist_m$encounter)

encounter_list <- levels(rel_dist_m$encounter)
gap_encounters <- sapply(unique(rel_dist_m$encounter), function(encounter) {
  times <- rel_dist_m$t[rel_dist_m$encounter == encounter];
  times <- sort(times);
  if (any((times[-1] - times[-length(times)]) > .15)) {
    return(as.numeric(as.character(encounter)))
  }
  return(NULL)
})
gap_encounters <- unlist(gap_encounters)
clean_encounters <- setdiff(unique(rel_dist_m$encounter), gap_encounters)
clean_rel_dist_m <- rel_dist_m[rel_dist_m$encounter %in% clean_encounters,]
clean_rel_dist_m$encounter <- droplevels(clean_rel_dist_m$encounter)
clean_rel_dist_m$centered_t <- clean_rel_dist_m$t - min(clean_rel_dist_m$t)

ggplot(rel_dist_m[!(rel_dist_m$encounter %in% gap_encounters),], aes(x = norm_t, y = reldist, group = encounter)) + geom_path()
sample_encounters <- sample(clean_encounters, 100)
ggplot(rel_dist_m[rel_dist_m$encounter %in% sample_encounters,], aes(x = norm_t, y = r_reldist, group = encounter)) + geom_path()


sample_encounters <- sample(encounter_list, 100)
ggplot(rel_dist_m[rel_dist_m$encounter %in% sample_encounters,], aes(x = t, y = reldist, group = encounter)) + geom_path()

ggplot(rel_dist_m[rel_dist_m$encounter %in% sample_encounters,], aes(x = norm_t, y = reldist, group = encounter)) + geom_path()


#build_interpolated_data <- function(encounter_data) {
#  new_times <- seq(0.01, 0.99, by = 0.01)
#  interpolated_encounter <- sapply(new_times, function(i) {
#    .01
#  })
#}

interpolated_data <- lapply(clean_encounters, 
                            function(encounter) {
                              approx(rel_dist_m$norm_t[rel_dist_m$encounter == encounter],
                                     rel_dist_m$reldist[rel_dist_m$encounter == encounter],
                                     xout = seq(0, 1, by = 0.01))
                            })

interpolated_data_m <- lapply(interpolated_data, function(data) {do.call(cbind, data)})
interpolated_data_m <- lapply(1:length(interpolated_data_m), function(i) {as.data.frame(cbind(
  interpolated_data_m[[i]], clean_encounters[i]
))})
interpolated_data_m <- do.call(rbind, interpolated_data_m)
colnames(interpolated_data_m) <- c("V1", "V2", "id")
interpolated_data_m$id <- as.factor(interpolated_data_m$id)
interpolated_data_m[, c("V1", "V2")] <- sapply(interpolated_data_m[, c("V1", "V2")], function(col) {as.numeric(as.character(col))})
plot_sample <- sample(unique(interpolated_data_m$id), 100)
ggplot(interpolated_data_m[interpolated_data_m$id %in% plot_sample,], aes(x = V1, y = V2, group = id)) + geom_path()



dist_m <- matrix(0, nrow = length(interpolated_data), ncol = length(interpolated_data))
for (i in 1:(length(interpolated_data) - 1)) {
  for (j in (i + 1):length(interpolated_data)) {
    max_diff = max(abs(interpolated_data[[i]]$y - interpolated_data[[j]]$y), na.rm = T)
    if (max_diff)
    dist_m[i, j] = max_diff
    dist_m[j, i] = dist_m[i,j]
  }
}

mds_dist <- cmdscale(dist_m, k = 10)
qplot(mds_dist[,1], mds_dist[,2])
mds_dist <- as.data.frame(mds_dist)
plot_ly(mds_dist, x = ~V1, y =~ V2, z =~ V3)



cv_folds <- split(sample(nrow(mds_dist)), 1:15)
sapply(2:10, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(mds_dist), fold, k)
}))})

kmeans_results <- kmeans(mds_dist, centers = 8)
mds_dist$cluster <- kmeans_results$cluster
ggplot(mds_dist, aes(x = V1, y = V2, color = factor(cluster))) + geom_point()

interpolated_data_m$cluster <- unlist(lapply(1:length(unique(interpolated_data_m$id)), 
       function(i) {rep(kmeans_results$cluster[i], 
                        sum(interpolated_data_m$id == 
                              unique(interpolated_data_m$id)[i]))}))
setwd("~/Documents/Michigan/Research/Driving/plots/rel_dist_1d/")
png("interpolated_1d_clusters.png", width = 800, height = 600)
ggplot(interpolated_data_m[interpolated_data_m$id %in% plot_sample,], aes(x = V1, y = V2, group = id, color = factor(cluster))) + geom_path()
dev.off()


plot_ly(rel_dist_m[rel_dist_m$encounter %in% plot_sample,], 
        x =~ norm_t, y =~relx, z =~rely, color =~ encounter, type = 'scatter3d', mode = 'lines')
start_end_x_y_pts <- cbind(
  rel_dist_m$r_relx[rel_dist_m$norm_t == 0],
  rel_dist_m$r_relx[rel_dist_m$norm_t == 1],
  rel_dist_m$r_rely[rel_dist_m$norm_t == 0],
  rel_dist_m$r_rely[rel_dist_m$norm_t == 1]
)
start_end_x_y_pts <- as.data.frame(start_end_x_y_pts)
colnames(start_end_x_y_pts) <- c("start_x", "end_x", "start_y", "end_y")
start_end_x_y_pts$encounter <- unique(rel_dist_m$encounter)

sample_encounters <- sample(encounter_list, 100)
interested_encounters <- which(encounter_list %in% sample_encounters)
ggplot() + 
  geom_path(data = rel_dist_m[rel_dist_m$encounter %in% sample_encounters,], aes(x = r_relx, y = r_rely, group = encounter)) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = start_x, y = start_y, color = "Start")) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = end_x, y = end_y, color = "End"))

weird_encounters <- which(sapply(unique(rel_dist_m$encounter), function(id) {tmp <- rel_dist_m[rel_dist_m$encounter == id,]; sum(tmp$norm_t %in% c(0,1)) != 2}))

#What if we use clustering from earlier
clean_rel_dist_m <- rel_dist_m[rel_dist_m$encounter %in% clean_encounters,]
assignment_m <- cbind(kmeans_results$cluster, clean_encounters)
assignment_m <- apply(assignment_m, 2, as.numeric)
clean_rel_dist_m$cluster <- unlist(lapply(unique(clean_rel_dist_m$encounter), 
                                             function(id) {rep(unname(assignment_m[which(assignment_m[,2] == id), 1]), 
                                                              sum(clean_rel_dist_m$encounter == id))}))

sample_encounters <- sample(unique(clean_rel_dist_m$encounter), 100)
ggplot() + 
  geom_path(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters,], aes(x = r_relx, y = r_rely, group = encounter, color = factor(cluster))) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = start_x, y = start_y, color = "Start")) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = end_x, y = end_y, color = "End"))
ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters,]) + 
  geom_point(aes(x = r_x1, y = r_y1, color = "1", group = encounter)) +
  geom_point(aes(x = r_x2, y = r_y2, color = "2", group = encounter))

png("sample_cluster_rotate_encounters.png")
do.call(grid.arrange, c(lapply(1:8, function(i) {
            ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                      clean_rel_dist_m$cluster == i,]) + 
              geom_path(aes(x = r_x1, y = r_y1, color = encounter, group = encounter)) +
              geom_path(aes(x = r_x2, y = r_y2, color = encounter, group = encounter)) +
              theme(legend.position="none")}), "ncol" = 3))
dev.off()

png("sample_cluster_encounters.png")
do.call(grid.arrange, c(lapply(1:8, function(i) {
  ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                            clean_rel_dist_m$cluster == i,]) + 
    geom_path(aes(x = x1, y = y1, color = encounter, group = encounter)) +
    geom_path(aes(x = x2, y = y2, color = encounter, group = encounter)) +
    theme(legend.position="none")}), "ncol" = 3))
dev.off()

interpolated_x_data <- get_interpolated_data(clean_rel_dist_m, "r_relx", clean_encounters)
x_dist_m <- get_dist_m(interpolated_x_data)
interpolated_y_data <- get_interpolated_data(clean_rel_dist_m, "r_rely", clean_encounters)
y_dist_m <- get_dist_m(interpolated_y_data)
dist_m <- cbind(cmdscale(x_dist_m, k = 3), cmdscale(y_dist_m, k = 3))

cv_folds <- split(sample(nrow(dist_m)), 1:10)
cv_errors <- sapply(2:30, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(dist_m), fold, k)
}))})
#18?
kmeans_results <- kmeans(dist_m, centers = 18)
dist_m <- as.data.frame(dist_m)
dist_m$encounter <- clean_encounters
dist_m$cluster <- kmeans_results$cluster
plot_ly(as.data.frame(dist_m), x =~ V1, y =~ V2, z =~ V3, color =~ factor(cluster))

clean_rel_dist_m$cluster2 <- unlist(lapply(unique(clean_rel_dist_m$encounter), 
                                          function(id) {rep(unname(dist_m$cluster[which(dist_m$encounter == id)]), 
                                                            sum(clean_rel_dist_m$encounter == id))}))
sample_encounters <- sample(unique(clean_rel_dist_m$encounter), 200)
ggplot() + 
  geom_path(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters,], aes(x = r_relx, y = r_rely, group = encounter, color = factor(cluster2))) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = start_x, y = start_y, color = "Start")) +
  geom_point(data = start_end_x_y_pts[start_end_x_y_pts$encounter %in% sample_encounters,], aes(x = end_x, y = end_y, color = "End"))

sample_encounters <- sample(unique(clean_rel_dist_m$encounter), 200)
png("interpolated_2d_clusters.png", width = 1200, height = 1200)
do.call(grid.arrange, c(lapply(1:18, function(i) {
  ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                            clean_rel_dist_m$cluster2 == i,]) + 
    geom_path(aes(x = x1, y = y1, color = encounter, group = encounter)) +
    geom_path(aes(x = x2, y = y2, color = encounter, group = encounter)) +
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster2 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = x1, y = y1, color = encounter)) + 
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster2 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = x2, y = y2, color = encounter)) + 
    theme(legend.position="none")}), "ncol" = 4))
dev.off()

png("interpolated_2d_clusters_rotated.png", width = 1200, height = 1200)
do.call(grid.arrange, c(lapply(1:18, function(i) {
  ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                            clean_rel_dist_m$cluster2 == i,]) + 
    geom_path(aes(x = r_x1, y = r_y1, color = encounter, group = encounter)) +
    geom_path(aes(x = r_x2, y = r_y2, color = encounter, group = encounter)) +
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster2 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = r_x1, y = r_y1, color = encounter)) + 
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster2 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = r_x2, y = r_y2, color = encounter)) + 
    theme(legend.position="none")}), "ncol" = 4))
dev.off()

#Pretend that they are time series
encounter_curves_m <- dcast(interpolated_data_m, id ~ V1, value.var = "V2")

cv_folds <- split(sample(nrow(encounter_curves_m)), 1:10)
cv_results <- sapply(2:30, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(encounter_curves_m[, -1]), fold, k)
}))})

#Pick 19
ec_assignment_m <- as.data.frame(cbind(as.numeric(as.character(encounter_curves_m$id)), 
                                       kmeans(encounter_curves_m[, -1], 19)$cluster))
ec_assignment_m

dist_curve <- dist(encounter_curves_m)
sil_avg_width <- sapply(2:50, function(k) {
  cat(k, "\n");
  summary(silhouette(kmeans(encounter_curves_m[, -1], k)$cluster, dist_curve))$avg.width})
 

clean_rel_dist_m$cluster3 <- unlist(lapply(unique(clean_rel_dist_m$encounter), 
                                           function(id) {rep(unname(ec_assignment_m$V2[which(ec_assignment_m$V1 == id)]), 
                                                             sum(clean_rel_dist_m$encounter == id))}))
png("interpolated_time_series_cluster.png", width = 1200, height = 1200)
do.call(grid.arrange, c(lapply(1:18, function(i) {
  ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                            clean_rel_dist_m$cluster3 == i,]) + 
    geom_path(aes(x = x1, y = y1, color = encounter, group = encounter)) +
    geom_path(aes(x = x2, y = y2, color = encounter, group = encounter)) +
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster3 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = x1, y = y1, color = encounter)) + 
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster3 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = x2, y = y2, color = encounter)) + 
    theme(legend.position="none")}), "ncol" = 4))
dev.off()

png("interpolated_rotated_time_series_cluster.png", width = 1200, height = 1200)
do.call(grid.arrange, c(lapply(1:18, function(i) {
  ggplot(clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                            clean_rel_dist_m$cluster3 == i,]) + 
    geom_path(aes(x = r_x1, y = r_y1, color = encounter, group = encounter)) +
    geom_path(aes(x = r_x2, y = r_y2, color = encounter, group = encounter)) +
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster3 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = r_x1, y = r_y1, color = encounter)) + 
    geom_point(data = clean_rel_dist_m[clean_rel_dist_m$encounter %in% sample_encounters &
                                         clean_rel_dist_m$cluster3 == i & 
                                         clean_rel_dist_m$norm_t == 0,],
               aes(x = r_x2, y = r_y2, color = encounter)) + 
    theme(legend.position="none")}), "ncol" = 4))
dev.off()

#Try with more info
interpolated_x_data <- get_interpolated_data(clean_rel_dist_m, "r_relx", clean_encounters)
interpolated_y_data <- get_interpolated_data(clean_rel_dist_m, "r_rely", clean_encounters)
interpolated_heading_1_data <- get_interpolated_data(clean_rel_dist_m, "Heading_1", clean_encounters)
interpolated_heading_2_data <- get_interpolated_data(clean_rel_dist_m, "Heading_2", clean_encounters)

interpolated_x_data_m <- do.call(rbind, lapply(interpolated_x_data, function(data) {data$y / max(data$y)}))
interpolated_y_data_m <- do.call(rbind, lapply(interpolated_y_data, function(data) {data$y / max(data$y)}))
interpolated_heading_1_data_m <- do.call(rbind, lapply(interpolated_heading_1_data, function(data) {data$y / max(data$y)}))
interpolated_heading_2_data_m <- do.call(rbind, lapply(interpolated_heading_2_data, function(data) {data$y / max(data$y)}))

full_encounter_matrix <- cbind(interpolated_x_data_m, interpolated_y_data_m, 
                               interpolated_heading_1_data_m, interpolated_heading_2_data_m)
cv_folds <- split(sample(nrow(full_encounter_matrix)), 1:10)
cv_results <- sapply(2:30, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(as.matrix(full_encounter_matrix), fold, k)
  }))})


#Try spectral clustering
sim_x <- 1 - x_dist_m / max(x_dist_m)
sim_y <- 1 - y_dist_m / max(y_dist_m)

return_laplacian <- function(sim_m) {
  D <- diag(rowSums(sim_m))
  D - sim_m
}

return_normal_laplacian <- function(sim_m) {
  D <- diag(rowSums(sim_m))
  L <- D - sim_m
  diag(rowSums(sim_m)^(-1/2)) %*% L %*% diag(rowSums(sim_m)^(-1/2))
}

lx <- return_laplacian(sim_x)
ly <- return_laplacian(sim_y)

norm_lx <- return_normal_laplacian(sim_x)
norm_ly <- return_normal_laplacian(sim_y)

eigen_lx <- eigen(lx, symmetric = T)
eigen_ly <- eigen(ly, symmetric = T)

eigen_norm_lx <- eigen(norm_lx, symmetric = T)
eigen_norm_ly <- eigen(norm_ly, symmetric = T)


spectral_m <- cbind(eigen_lx$vectors[, ncol(eigen_lx$vectors)], eigen_ly$vectors[, ncol(eigen_lx$vectors)])

cv_folds <- split(sample(nrow(spectral_m)), 1:10)
cv_results <- sapply(2:10, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(as.matrix(spectral_m), fold, k)
  }))})

#Try 6
s_results <- kmeans(spectral_m, centers = 6)

norm_spectral_m <- cbind(eigen_lx$vectors[, (ncol(eigen_lx$vectors) - 1):ncol(eigen_lx$vectors)], 
                         eigen_ly$vectors[, (ncol(eigen_ly$vectors) - 1):ncol(eigen_ly$vectors)])
cv_folds <- split(sample(nrow(norm_spectral_m)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(as.matrix(norm_spectral_m), fold, k, algorithm="MacQueen", iter.max=100)
  }))})

#Try 5
s_results <- kmeans(norm_spectral_m, centers = 5, algorithm="MacQueen", iter.max=100)

clean_rel_dist_m$cluster4 <- unlist(lapply(unique(clean_rel_dist_m$encounter), 
                                           function(id) {rep(unname(ec_assignment_m$V2[which(ec_assignment_m$V1 == id)]), 
                                                             sum(clean_rel_dist_m$encounter == id))}))

png("interpolated_spectral_clustering_results.png")
p_plots(clean_rel_dist_m, s_results$cluster, as.numeric(clean_encounters))
dev.off()


# Actual starting point here?

clean_rel_dist_m <- rel_dist_m[rel_dist_m$encounter %in% clean_encounters,]
interpolated_x1_data <- get_interpolated_data(clean_rel_dist_m, "x1", clean_encounters)
interpolated_x2_data <- get_interpolated_data(clean_rel_dist_m, "x2", clean_encounters)
interpolated_y1_data <- get_interpolated_data(clean_rel_dist_m, "y1", clean_encounters)
interpolated_y2_data <- get_interpolated_data(clean_rel_dist_m, "y2", clean_encounters)
interpolated_data_m_list <- lapply(1:length(interpolated_x1_data), function(i) {
  tmp <- cbind("x1" = interpolated_x1_data[[i]]$y,
               "x2" = interpolated_x2_data[[i]]$y,
               "y1" = interpolated_y1_data[[i]]$y,
               "y2" = interpolated_y2_data[[i]]$y,
               "t" = interpolated_x1_data[[i]]$x)
})

# normalize_interpolated_data <- function(data) {
#   if (max(abs(data$y)) == 0) {
#     return(data$y)
#   }
#   data$y / max(abs(data$y))
# }
# norm_interpolated_x1_data_m <- do.call(rbind, lapply(interpolated_x1_data, normalize_interpolated_data))
# norm_interpolated_x2_data_m <- do.call(rbind, lapply(interpolated_x2_data, normalize_interpolated_data))
# norm_interpolated_y1_data_m <- do.call(rbind, lapply(interpolated_y1_data, normalize_interpolated_data))
# norm_interpolated_y2_data_m <- do.call(rbind, lapply(interpolated_y2_data, normalize_interpolated_data))
norm_interpolated_x1_data <- get_interpolated_data(clean_rel_dist_m, "x1", clean_encounters, normalize = T)
norm_interpolated_x2_data <- get_interpolated_data(clean_rel_dist_m, "x2", clean_encounters, normalize = T)
norm_interpolated_y1_data <- get_interpolated_data(clean_rel_dist_m, "y1", clean_encounters, normalize = T)
norm_interpolated_y2_data <- get_interpolated_data(clean_rel_dist_m, "y2", clean_encounters, normalize = T)
norm_interpolated_data_m_list <- lapply(1:length(interpolated_x1_data), function(i) {
  tmp <- cbind("x1" = norm_interpolated_x1_data[[i]]$y,
               "x2" = norm_interpolated_x2_data[[i]]$y,
               "y1" = norm_interpolated_y1_data[[i]]$y,
               "y2" = norm_interpolated_y2_data[[i]]$y,
               "t" = norm_interpolated_x1_data[[i]]$x)
})


#p_dist_m <- get_p_dist_m(norm_interpolated_x1_data_m, norm_interpolated_x2_data_m,
#                         norm_interpolated_y1_data_m, norm_interpolated_y2_data_m)
p_dist_m <- get_p_dist_m(interpolated_data_m_list)
norm_p_dist_m <- get_p_dist_m(interpolated_data_m_list)
save(p_dist_m, norm_p_dist_m, file = "p_dist_m.Rdata")

load("p_dist_m.Rdata")
p_sim_m <- 1 - p_dist_m / max(p_dist_m)
#p_sim_m[p_sim_m <= 0.8] = 0
lp <- return_laplacian(p_sim_m)
norm_lp <- return_normal_laplacian(p_sim_m)

eigen_lp <- eigen(lp, symmetric = T)
eigen_norm_lp <- eigen(norm_lp, symmetric = T)
plot(rev(eigen_lp$values))
plot(rev(eigen_norm_lp$values))

spectral_m <- cbind(eigen_lx$vectors[, ncol(eigen_lx$vectors)], eigen_ly$vectors[, ncol(eigen_lx$vectors)])

cv_folds <- split(sample(nrow(spectral_m)), 1:10)
cv_results <- sapply(2:10, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(as.matrix(spectral_m), fold, k)
  }))})

p_mds <- cmdscale(norm_p_dist_m, k = 10)
cv_folds <- split(sample(nrow(p_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(p_mds, fold, k)
  }))})

p_mds_cv <- kmeans(p_mds, 20)
png("procrustes_dist_cluster.png")
p_plots(clean_rel_dist_m, p_mds_cv$cluster, as.numeric(clean_encounters))
dev.off()

png("p_dist_mds_plots.png")
par(mfrow = c(2, 2))
plot(p_mds)
plot(p_mds[apply(p_mds[,1:2], 1, function(row) {all(abs(row) < 1000)}),])
plot(p_mds[apply(p_mds[,1:2], 1, function(row) {all(abs(row) < 100)}),])
plot(p_mds[apply(p_mds[,1:2], 1, function(row) {all(abs(row) < 10)}),])
dev.off()

#Within group distance
avg_within_dist <- lapply(1:length(unique(p_mds_cv$cluster)), function(i) {
  mean(apply(p_dist_m[p_mds_cv$cluster == i,
                      p_mds_cv$cluster == i], 1, 
             function(row) {mean(row[row != 0])}))
})



#Without group distance
avg_without_dist <- lapply(1:14, function(i) {
  mean(sapply((1:14)[-i], function(j) {
    mean(apply(p_dist_m[p_mds_cv$cluster == i,], 1, function(row) {mean(row[p_mds_cv$cluster == j])}))
  }))
})

group_diff <- as.data.frame(cbind(unlist(avg_within_dist), unlist(avg_without_dist)))
colnames(group_diff) <- c("within", "without")
plot_group_diff <- melt(group_diff)
plot_group_diff$n <- 1:length(unique(p_mds_cv$cluster))
ggplot(plot_group_diff, aes(x = n, y = value, fill = factor(variable))) + geom_bar(stat = "identity", position = "dodge")


#Plot example
png("rotation_plot_ex.png")
grid.arrange(ggplot(data = plot_df) + geom_path(aes(x = x1, y = y1, color = "Curve 1")) + 
               geom_path(aes(x = x2, y = y2, color = "Curve 2")) + 
               geom_point(data = plot_df[plot_df$t %in% seq(0, 2, by = 0.05),], aes(x = x1, y = y1, color = "Curve 1")) + 
               geom_point(data = plot_df[plot_df$t %in% seq(0, 2, by = 0.05),], aes(x = x2, y = y2, color = "Curve 2")), 
             ggplot(data = plot_df) + geom_path(aes(x = r_x1, y = r_y1, color = "Curve 1")) + 
               geom_path(aes(x = r_x2, y = r_y2, color = "Curve 2")) + 
               geom_point(data = plot_df[plot_df$t %in% seq(0, 2, by = 0.1),], aes(x = r_x1, y = r_y1, color = "Curve 1")) + 
               geom_point(data = plot_df[plot_df$t %in% seq(0, 2, by = 0.1),], aes(x = r_x2, y = r_y2, color = "Curve 2")), 
             ncol = 2)
dev.off()

#Try cubic spline fitting
#Fit cubic polynomial to y based on x
fit_cubic_poly <- function(x, y) {
  return(lm(y ~ x + I(x^2) + I(x^3)))
}

# The first part to start with here for doing two-spline
#Recursively try to add a change point 
#to the cubic polynomial fitted to y based on x
find_poly_split <- function(x, y, start = 1, end = length(x),
                                       best_choice = length(x)) {
  old_fit = 0
  if (best_choice == length(x)) {
    old_fit = sum(lm(y ~ x + I(x^2) + I(x^3))$residuals^2)
  } else {
    old_split1 = start:best_choice
    old_split2 = best_choice:end
    old_fit = sum(fit_cubic_poly(x[old_split1], y[old_split1])$residuals^2,
                  (fit_cubic_poly(x[old_split2], y[old_split2])$residuals^2)[-1])
  }
  if (old_fit  < 1e-16) {
    return(best_choice)
  }
  new_choice = floor((start + end)/2)
  # new_split1 = start:new_choice
  # new_split2 = new_choice:end
  new_split1 = 1:new_choice
  new_split2 = new_choice:length(x)
  new_fit1 = sum(fit_cubic_poly(x[new_split1], y[new_split1])$residuals^2)
  new_fit2 = sum((fit_cubic_poly(x[new_split2], y[new_split2])$residuals^2)[-1])
  new_fit = new_fit1 + new_fit2
  if (new_fit < old_fit) {
    return(new_choice)
  }
  #Give up if there are only 12 observations because 
  #it's hard to fit if there are only 6
  if (((length(x) - start) <= 12) || ((end - 1) <= 12)) {
    return(best_choice)
  }
  if (abs(start - end) <= 1) {
    return(best_choice)
  }
  if (new_fit1 < new_fit2) {
    return(find_poly_split(x, y, new_choice, end, best_choice))
  } else {
    return(find_poly_split(x, y, start, new_choice, best_choice))
  }
}

#Try to sequentially find all change points recursively
find_all_poly_splits <- function(x, y, start = 1, end = length(x)) {
  if (length(x) <= 6) {
    return(NULL)
  }
  changepoint = find_poly_split(x, y)
  if (changepoint == length(x)) {
    return(NULL)
  } 
  if (changepoint <= 4 || (changepoint >= (length(x) - 4))) {
    return((start:end)[changepoint])
  }
  print(changepoint)
  return(c(find_all_poly_splits(x[1:changepoint], y[1:changepoint], 
                                start, (start:end)[changepoint]),
           (start:end)[changepoint],
           find_all_poly_splits(x[changepoint:length(x)],
                                y[changepoint:length(x)], 
                                (start:end)[changepoint], end)))
}

# Here is where we do the two-step spline primitive splitting and such

#Remove unnecessary change points based on a tolerance
find_poly_changepoints <- function(x, y, tol = 1e-16, chg_pts_cand = NULL) {
  if (is.null(chg_pts_cand)) {
    chg_pts_cand <- c(1, sort(find_all_poly_splits(x, y)), length(x))
  }
  test_ind = 2
  while (chg_pts_cand[test_ind] != length(x)) {
    if ((chg_pts_cand[test_ind] - chg_pts_cand[(test_ind - 1)]) <= 4) {
      chg_pts_cand <- chg_pts_cand[-test_ind]
      next
    }
    test_interval <- chg_pts_cand[(test_ind - 1)]:chg_pts_cand[(test_ind + 1)]
    if (sum(fit_cubic_poly(x[test_interval], y[test_interval])$residuals^2) < tol) {
      chg_pts_cand <- chg_pts_cand[-test_ind]
    } else {
      test_ind = test_ind + 1
    }
  }
  if ((chg_pts_cand[test_ind] - chg_pts_cand[(test_ind - 1)]) <= 4) {
    chg_pts_cand <- chg_pts_cand[-(test_ind - 1)]
  }
  return(chg_pts_cand)
}

#Remove unnecessary change points for splines based on a joint tolerance
find_poly_changepoints_jointly <- function(x, y_list, tol = 1e-16, 
                                           chg_pts_cand_list = NULL) {
  if (is.null(chg_pts_cand_list)) {
    chg_pts_cand_list <- lapply(y_list, function(y_v) {
      c(1, sort(find_all_poly_splits(x, y_v)), length(x))
    })
  }
  chg_pts_cand <- sort(unique(unlist(chg_pts_cand_list)))
  if (length(chg_pts_cand) == 2) {
    return(chg_pts_cand)
  }
  test_ind = 2
  while (chg_pts_cand[test_ind] != length(x)) {
    if ((chg_pts_cand[test_ind] - chg_pts_cand[(test_ind - 1)]) <= 4) {
      chg_pts_cand <- chg_pts_cand[-test_ind]
      next
    }
    test_interval <- chg_pts_cand[(test_ind - 1)]:chg_pts_cand[(test_ind + 1)]
    if (sum(sapply(y_list, function(y) {
      fit_cubic_poly(x[test_interval], y[test_interval])$residuals^2
    })) < tol) {
      chg_pts_cand <- chg_pts_cand[-test_ind]
    } else {
      test_ind = test_ind + 1
    }
  }
  if ((chg_pts_cand[test_ind] - chg_pts_cand[(test_ind - 1)]) <= 4) {
    chg_pts_cand <- chg_pts_cand[-(test_ind - 1)]
  }
  return(chg_pts_cand)
}

#Fit cubic polynomial based on change points
fit_for_chg_pts <- function(x, y, chg_pts) {
  fitted_vals <- fit_cubic_poly(x[chg_pts[1]:chg_pts[2]], 
                                y[chg_pts[1]:chg_pts[2]])$fitted.values
  if (length(chg_pts) == 2) {
    return(fitted_vals)
  }
  for (i in 2:(length(chg_pts) - 1)) {
    fitted_vals <- c(fitted_vals,
                     fit_cubic_poly((x[chg_pts[i]:chg_pts[(i + 1)]])[-1], 
                                    (y[chg_pts[i]:chg_pts[(i + 1)]])[-1])$fitted.values)
  }
  fitted_vals
}

#Get optimal number of change points by
#tuning for the tolerances in test_tol_lest based
#on the sum of the squared error and the number of chg points + 2
get_opt_number_chg_pts <- function(x, y, test_tol_list = 
                                     c(1e-16, 1e-9, 1e-3, 1e-1, .1, 1:10)) {
  chg_pts_cand <- c(1, sort(find_all_poly_splits(x, y)), length(x))
  if (length(chg_pts_cand) == 2) {
    return(chg_pts_cand)
  }
  opt_chg_pts <- c()
  test_error = Inf
  test_error_list <- c()
  for (tol in test_tol_list) {
    tol_chg_pts <- find_poly_changepoints(x, y, tol, chg_pts_cand);
    poly_fit <- fit_for_chg_pts(x, y, tol_chg_pts)
    cand_error = sum((y - poly_fit)^2) + length(tol_chg_pts)
    test_error_list <- c(test_error_list, cand_error)
    if (cand_error < test_error) {
      test_error = cand_error
      opt_chg_pts <- tol_chg_pts
    }
  }
  #print(plot(test_tol_list, test_error_list))
  return(opt_chg_pts)
}

#Get optimal number of change points for 2d spline by
#tuning for the tolerances in test_tol_lest based
#on the sum of the squared error and the number of chg points + 2
get_opt_number_chg_pts_jointly <- function(x, y_list, 
                                           test_tol_list = 
                                             c(1e-16, 1e-9, 1e-3, 1e-1, .1, 1:(10 * length(y_list)))) {
  chg_pts_cand_list <- lapply(y_list, function(y_v) {
    c(1, sort(find_all_poly_splits(x, y_v)), length(x))
  })
  chg_pts_cand <- unique(unlist(chg_pts_cand_list))
  if (length(chg_pts_cand) == 2) {
    return(chg_pts_cand)
  }
  opt_chg_pts <- c()
  test_error = Inf
  test_error_list <- c()
  for (tol in test_tol_list) {
    tol_chg_pts <- find_poly_changepoints_jointly(x, y_list, tol, 
                                                  chg_pts_cand_list);
    cand_error = sum(sapply(y_list, function(y) {
                    (y - fit_for_chg_pts(x, y, tol_chg_pts))^2})) + 
                 length(tol_chg_pts)
    test_error_list <- c(test_error_list, cand_error)
    if (cand_error < test_error) {
      test_error = cand_error
      opt_chg_pts <- tol_chg_pts
    }
  }
  #print(plot(test_tol_list, test_error_list))
  return(opt_chg_pts)
}

merge_chg_pts <- function(x_chg_pts, y_chg_pts) {
  combined_chg_pts <- sort(unique(c(x_chg_pts, y_chg_pts)))
  i = 2
  while ((i + 1) < (length(combined_chg_pts))) {
    if ((combined_chg_pts[(i + 1)] - combined_chg_pts[i]) <= 6) {
      combined_chg_pts[i] = floor(mean(c(combined_chg_pts[i], combined_chg_pts[(i + 1)])))
      combined_chg_pts <- combined_chg_pts[-(i + 1)]
    } else {
      i = i + 1
    }
  }
  return(combined_chg_pts)
}

get_lm_for_chg_pts <- function(x, y, chg_pts) {
  fitted_lm <- list(fit_cubic_poly(x[chg_pts[1]:chg_pts[2]], 
                                   y[chg_pts[1]:chg_pts[2]]))
  if (length(chg_pts) == 2) {
    return(fitted_lm)
  }
  for (i in 2:(length(chg_pts) - 1)) {
    fitted_lm <- c(fitted_lm,
                   list(fit_cubic_poly(x[chg_pts[i]:chg_pts[(i + 1)]], 
                                       y[chg_pts[i]:chg_pts[(i + 1)]])))
  }           
  fitted_lm
}

split_by_time_pts <- function(x, time_pts) {
  if (length(time_pts) == 2) {
    return(list(x))
  }
  time_split <- list(x[x <= time_pts[[2]]])
  for (i in 2:(length(time_pts) - 1)) {
    time_split <- c(time_split, list(
      x[x <= time_pts[[(i + 1)]] & 
          x > time_pts[[i]]]
    ))
  }
  return(time_split)
}

chg_pt_interpolate_data <- function(x, y, chg_pts) {
  new_x <- seq(0, 1, by = 0.01)
  lm_list <- get_lm_for_chg_pts(x, y, chg_pts)
  new_x_time_split <- split_by_time_pts(new_x, x[chg_pts])
  unname(unlist(sapply(1:length(new_x_time_split), function(i) {
    predict(lm_list[[i]], data.frame("x" = new_x_time_split[[i]]), type = "response")
  })))
}

chg_pt_interpolate_prims <- function(x, y_list, chg_pts, encounter = NULL) {
  new_x <- seq(0, 1, by = 0.01)
  lm_list <- lapply(y_list, function(y_v) {get_lm_for_chg_pts(x, y_v, chg_pts)})
  prim_list <- list()
  for (i in 1:(length(chg_pts) - 1)) {
    tmp <- cbind(new_x, sapply(lm_list, function(lm_for_y) {
                  predict(lm_for_y[[i]], 
                          data.frame("x" = new_x * x[chg_pts[i + 1]] + 
                                            (1 - new_x) * x[chg_pts[i]]), 
                          type = "response")}))
    colnames(tmp) <- c("norm_t", names(lm_list))
    if (!is.null(encounter)) {
      tmp <- as.data.frame(tmp)
      tmp$prim_name <- paste(encounter, i, sep = "_") 
    }
    prim_list <- c(prim_list, list(tmp))
  }
  return(prim_list)
}

calc_p_dist_with_chg_pts <- function(x1_v, x2_v, y1_v, y2_v,
                                     x1_time_pts, x2_time_pts,
                                     y1_time_pts, y2_time_pts, 
                                     normalize = F, avg = F) {
  merged_time_pts <- sort(unique(c(x1_time_pts, x2_time_pts,
                                   y1_time_pts, y2_time_pts)))
  interpolated_time <- seq(0, 1, by = 0.01)
  split_time_list <- split_by_time_pts(interpolated_time, merged_time_pts)
  p_dist_list <- sapply(split_time_list, function(split_time) {
    if (length(split_time) == 0) {
      return(NA)
    }
    calc_p_dist(x1_v[interpolated_time %in% split_time], 
                x2_v[interpolated_time %in% split_time],
                y1_v[interpolated_time %in% split_time],
                y2_v[interpolated_time %in% split_time], 
                normalize)
  })
  if (avg) {
    return(mean(p_dist_list, na.rm = T))
  } else {
    return(sum(p_dist_list, na.rm = T))    
  }
}

calc_entire_p_dist_with_chg_pts <- function(x_x1_v, x_y1_v,
                                            x_x2_v, x_y2_v,
                                            y_x1_v, y_y1_v,
                                            y_x2_v, y_y2_v,
                                            x1_time_pts, x2_time_pts,
                                            y1_time_pts, y2_time_pts, 
                                            normalize = F, avg = F) {
  merged_time_pts <- sort(unique(c(x1_time_pts, x2_time_pts,
                                   y1_time_pts, y2_time_pts)))
  interpolated_time <- seq(0, 1, by = 0.01)
  split_time_list <- split_by_time_pts(interpolated_time, merged_time_pts)
  p_dist_list <- sapply(split_time_list, function(split_time) {
    if (length(split_time) == 0) {
      return(NA)
    }
    calc_p_dist(c(x_x1_v[interpolated_time %in% split_time],
                  x_y1_v[interpolated_time %in% split_time]),
                c(x_x2_v[interpolated_time %in% split_time],
                  x_y2_v[interpolated_time %in% split_time]),
                c(y_x1_v[interpolated_time %in% split_time],
                  y_y1_v[interpolated_time %in% split_time]),
                c(y_x2_v[interpolated_time %in% split_time],
                  y_y2_v[interpolated_time %in% split_time]), 
                normalize)
  })
  if (avg) {
    return(mean(p_dist_list, na.rm = T))
  } else {
    return(sum(p_dist_list, na.rm = T))    
  }
}

get_p_dist_with_chg_pts_m <- function(chg_pt_data, time_pt_list, normalize = F, avg = F, traj = F) {
  dist_m <- matrix(0, nrow = length(chg_pt_data), ncol = length(chg_pt_data))
  for (i in 1:(length(chg_pt_data) - 1)) {
    for (j in (i + 1):length(chg_pt_data)) {
      cat(i, j, "\n")
      if (traj) {
        dist_m[i, j] = 
          calc_p_dist_with_chg_pts(chg_pt_data[[i]][[1]], chg_pt_data[[i]][[2]], 
                                   chg_pt_data[[j]][[1]], chg_pt_data[[j]][[2]],
                                   time_pt_list[[i]][[1]], time_pt_list[[i]][[2]],
                                   time_pt_list[[j]][[1]], time_pt_list[[j]][[2]],
                                   normalize, avg) +
          calc_p_dist_with_chg_pts(chg_pt_data[[i]][[3]], chg_pt_data[[i]][[4]], 
                                   chg_pt_data[[j]][[3]], chg_pt_data[[j]][[4]],
                                   time_pt_list[[i]][[1]], time_pt_list[[i]][[2]],
                                   time_pt_list[[j]][[1]], time_pt_list[[j]][[2]],
                                   normalize, avg)
        
      } else {
        dist_m[i, j] = 
          calc_p_dist_with_chg_pts(chg_pt_data[[i]][[1]], chg_pt_data[[i]][[3]], 
                                   chg_pt_data[[j]][[1]], chg_pt_data[[j]][[3]],
                                   time_pt_list[[i]][[1]], time_pt_list[[i]][[2]],
                                   time_pt_list[[j]][[1]], time_pt_list[[j]][[2]],
                                   normalize, avg) +
          calc_p_dist_with_chg_pts(chg_pt_data[[i]][[2]], chg_pt_data[[i]][[4]], 
                                   chg_pt_data[[j]][[2]], chg_pt_data[[j]][[4]],
                                   time_pt_list[[i]][[1]], time_pt_list[[i]][[2]],
                                   time_pt_list[[j]][[1]], time_pt_list[[j]][[2]],
                                   normalize, avg)
        
      }
      dist_m[j, i] = dist_m[i,j]
    }
  }
  dist_m
}

get_entire_p_dist_with_chg_pts_m <- function(chg_pt_data, time_pt_list, normalize = F, avg = F) {
  dist_m <- matrix(0, nrow = length(chg_pt_data), ncol = length(chg_pt_data))
  for (i in 1:(length(chg_pt_data) - 1)) {
    for (j in (i + 1):length(chg_pt_data)) {
      cat(i, j, "\n")
      dist_m[i, j] = 
        calc_entire_p_dist_with_chg_pts(chg_pt_data[[i]][[1]], chg_pt_data[[i]][[2]],
                                        chg_pt_data[[i]][[3]], chg_pt_data[[i]][[4]],
                                        chg_pt_data[[j]][[1]], chg_pt_data[[j]][[2]],
                                        chg_pt_data[[j]][[3]], chg_pt_data[[j]][[4]],
                                        time_pt_list[[i]][[1]], time_pt_list[[i]][[2]],
                                        time_pt_list[[j]][[1]], time_pt_list[[j]][[2]],
                                        normalize, avg)
      dist_m[j, i] = dist_m[i,j]
    }
  }
  dist_m
}

interpolate_to_shorter_prim <- function(short_interpolated_list, long_p_info_list, 
                                        short_length, num_interpolate = 5) {
  #new_x <- seq(0, 1, by = 0.01)
  new_x <- seq(0, 1, length.out = 21)
  long_info <- lapply(1:num_interpolate, function(i) {
    start_time = runif(1, long_p_info_list$start, 
                       long_p_info_list$end - short_length);
    lapply(long_p_info_list[c("x1", "x2", "y1", "y2")], function(lm_for_y) {
      predict(lm_for_y, 
              data.frame("x" = new_x * (start_time + short_length) + 
                           (1 - new_x) * start_time), 
              type = "response")})
  })
  return(list("short" = as.list(short_interpolated_list),
              "long" = long_info))
}


get_p_dist_for_p_info_list <- function(p_info_list, interpolated_prim_list, sum_fct = max, normalize = F,
                                       num_interpolate = 10) {
  dist_m <- matrix(0, nrow = length(p_info_list), ncol = length(p_info_list))
  interpolated_m_list <- lapply(1:num_interpolate, function(i) {dist_m})
  for (i in 1:(length(p_info_list) - 1)) {
    for (j in (i + 1):length(p_info_list)) {
      cat(i, j, "\n")
      interpolate_list = list()
      if (p_info_list[[i]]$length != p_info_list[[j]]$length) {
        if (p_info_list[[i]]$length < p_info_list[[j]]$length) {
          interpolate_list <- interpolate_to_shorter_prim(interpolated_prim_list[[i]], 
                                                          p_info_list[[j]], 
                                                          p_info_list[[i]]$length,
                                                          num_interpolate)
        } else {
          interpolate_list <- interpolate_to_shorter_prim(interpolated_prim_list[[j]], 
                                                          p_info_list[[i]],
                                                          p_info_list[[j]]$length,
                                                          num_interpolate)
        }
        for (k in 1:length(interpolate_list)) {
          interpolated_m_list[[k]][i, j] = 
            calc_p_dist(interpolate_list[["short"]][["x1"]], 
                        interpolate_list[["short"]][["x2"]],
                        interpolate_list[["long"]][[k]][["x1"]], 
                        interpolate_list[["long"]][[k]][["x2"]], normalize) +
              calc_p_dist(interpolate_list[["short"]][["y1"]], 
                          interpolate_list[["short"]][["y2"]],
                          interpolate_list[["long"]][[k]][["y1"]], 
                          interpolate_list[["long"]][[k]][["y2"]], normalize)   
        }
      } else {
        for (k in 1:num_interpolate) {
          interpolated_m_list[[k]][i, j] = 
            calc_p_dist(interpolated_prim_list[[i]][, "x1"],
                        interpolated_prim_list[[i]][, "x2"],
                        interpolated_prim_list[[j]][, "x1"],
                        interpolated_prim_list[[j]][, "x2"]) +
            calc_p_dist(interpolated_prim_list[[i]][, "y1"],
                        interpolated_prim_list[[i]][, "y2"],
                        interpolated_prim_list[[j]][, "y1"],
                        interpolated_prim_list[[j]][, "y2"])
        }
      }
      for (k in 1:num_interpolate) {
        interpolated_m_list[[k]][j, i] = interpolated_m_list[[k]][i, j]
      }
      save(interpolated_m_list, p_info_list, interpolated_prim_list, file = "tmp.Rdata")
    }
  }
  return(interpolated_m_list)
}

get_entire_p_dist_for_p_info_list <- function(p_info_list, interpolated_prim_list, sum_fct = max, normalize = F,
                                           num_interpolate = 10) {
  dist_m <- matrix(0, nrow = length(p_info_list), ncol = length(p_info_list))
  interpolated_m_list <- lapply(1:num_interpolate, function(i) {dist_m})
  for (i in 1:(length(p_info_list) - 1)) {
    for (j in (i + 1):length(p_info_list)) {
      cat(i, j, "\n")
      interpolate_list = list()
      if (p_info_list[[i]]$length != p_info_list[[j]]$length) {
        if (p_info_list[[i]]$length < p_info_list[[j]]$length) {
          interpolate_list <- interpolate_to_shorter_prim(interpolated_prim_list[[i]], 
                                                          p_info_list[[j]], 
                                                          p_info_list[[i]]$length,
                                                          num_interpolate)
        } else {
          interpolate_list <- interpolate_to_shorter_prim(interpolated_prim_list[[j]], 
                                                          p_info_list[[i]],
                                                          p_info_list[[j]]$length,
                                                          num_interpolate)
        }
        for (k in 1:num_interpolate) {
          interpolated_m_list[[k]][i, j] = 
            calc_p_dist(c(interpolate_list[["short"]][["x1"]], 
                          interpolate_list[["short"]][["y1"]]),
                        c(interpolate_list[["short"]][["x2"]], 
                          interpolate_list[["short"]][["y2"]]),
                        c(interpolate_list[["long"]][[k]][["x1"]],
                          interpolate_list[["long"]][[k]][["y1"]]),
                        c(interpolate_list[["long"]][[k]][["x2"]],
                          interpolate_list[["long"]][[k]][["y2"]]), normalize)
        }
      } else {
        all_p_dist = calc_p_dist(c(interpolated_prim_list[[i]][, "x1"],
                                   interpolated_prim_list[[i]][, "y1"]),
                                 c(interpolated_prim_list[[i]][, "x2"],
                                   interpolated_prim_list[[i]][, "y2"]),
                                 c(interpolated_prim_list[[j]][, "x1"],
                                   interpolated_prim_list[[j]][, "y1"]),
                                 c(interpolated_prim_list[[j]][, "x2"],
                                   interpolated_prim_list[[j]][, "y2"]),
                                 normalize)
        for (k in 1:num_interpolate) {
          interpolated_m_list[[k]][i, j] = all_p_dist
        }
      }
      for (k in 1:num_interpolate) {
        interpolated_m_list[[k]][j, i] = interpolated_m_list[[k]][i, j]
      }
      save(interpolated_m_list, p_info_list, interpolated_prim_list, file = "tmp.Rdata")
    }
  }
  return(interpolated_m_list)
}

test_x <- seq(0, 1, by = 0.01)
test_y <- 2 * test_x
test_y[test_x >= 0.25] <- .75 - 4 * test_x[test_x >= 0.25]^2
test_y[test_x >= 0.75] <- -3 + 2 * test_x[test_x >= 0.75]

tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[1],]
tmp_x_chg_pts <- find_poly_changepoints(tmp$norm_t, tmp$x1)
ggplot(tmp) + geom_path(aes(x = norm_t, y = x1)) + geom_point(data = tmp[tmp_x_chg_pts,], 
                                                              aes(x = norm_t, y = x1, color = "chg_pts")) 
#Try with higher tolerance
tmp_x_chg_pts <- find_poly_changepoints(tmp$norm_t, tmp$x1, 3)
png("curve_with_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = norm_t, y = x1)) + geom_point(data = tmp[tmp_x_chg_pts,], 
                                                              aes(x = norm_t, y = x1, color = "chg_pts")) 
dev.off()
tmp$fitted_x1 <- fit_for_chg_pts(tmp$norm_t, tmp$x1, tmp_x_chg_pts)
png("curve_with_chg_pts_and_interpolation.png")
grid.arrange(ggplot(tmp) + geom_path(aes(x = norm_t, y = x1, color = "original")) +
              geom_point(data = tmp[tmp_x_chg_pts,],  aes(x = norm_t, y = x1, color = "chg_pts")),
             ggplot(tmp) + geom_path(aes(x = norm_t, y = fitted_x1, color = "fitted")) +
               geom_point(data = tmp[tmp_x_chg_pts,],  aes(x = norm_t, y = x1, color = "chg_pts")), ncol = 2)
dev.off()

png("cross_validated_results.png")
get_opt_number_chg_pts(tmp$norm_t, tmp$x1)
dev.off()

tmp_y1_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$y1)
png("y1_curve_with_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = norm_t, y = y1)) + geom_point(data = tmp[tmp_y1_chg_pts,], 
                                                              aes(x = norm_t, y = y1, color = "chg_pts")) 
dev.off()

tmp_x2_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$x2)
png("x2_curve_with_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = norm_t, y = x2)) + geom_point(data = tmp[tmp_x2_chg_pts,], 
                                                              aes(x = norm_t, y = x2, color = "chg_pts")) 
dev.off()

tmp_y2_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$y2)
png("y2_curve_with_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = norm_t, y = y2)) + geom_point(data = tmp[tmp_y2_chg_pts,], 
                                                              aes(x = norm_t, y = y2, color = "chg_pts")) 
dev.off()

tmp_x1_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$x1)
tmp_y1_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$y1)
png("curve_with_x_y_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = norm_t, y = x1)) + geom_point(data = tmp[merge_chg_pts(tmp_x1_chg_pts, tmp_y1_chg_pts),], 
                                                              aes(x = norm_t, y = x1, color = "chg_pts")) 
dev.off()



tmp_x2_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$x2)
tmp_y2_chg_pts <- get_opt_number_chg_pts(tmp$norm_t, tmp$y2)

ggplot(tmp) + geom_path(aes(x = x1, y = y1, color = "Curve 1")) +
  geom_path(aes(x = x2, y = y2, color = "Curve 2")) +
  geom_point(data = tmp[c(tmp_x1_chg_pts, tmp_y1_chg_pts),],  
             aes(x = x1, y = y1, color = "Curve 1")) + 
  geom_point(data = tmp[c(tmp_x2_chg_pts, tmp_y2_chg_pts),],  
             aes(x = x2, y = y2, color = "Curve 2"))

png("encounter_with_merged_chg_pts.png")
ggplot(tmp) + geom_path(aes(x = x1, y = y1, color = "Curve 1")) +
  geom_path(aes(x = x2, y = y2, color = "Curve 2")) +
  geom_point(data = tmp[merge_chg_pts(tmp_x1_chg_pts, tmp_y1_chg_pts),],  
             aes(x = x1, y = y1, color = "Curve 1")) + 
  geom_point(data = tmp[merge_chg_pts(tmp_x2_chg_pts, tmp_y2_chg_pts),],  
             aes(x = x2, y = y2, color = "Curve 2"))
dev.off()

# Here is where the actual splitting occurs -> Ben

#Apply to all curves
start.time <- Sys.time()
chg_pt_list <- lapply(clean_encounters, function(encounter) {
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  if (nrow(tmp) == 0) {
    return(NA)
  }
  list(merge_chg_pts(get_opt_number_chg_pts(tmp$norm_t, tmp$x1), 
                     get_opt_number_chg_pts(tmp$norm_t, tmp$y1)),
       merge_chg_pts(get_opt_number_chg_pts(tmp$norm_t, tmp$x2), 
                     get_opt_number_chg_pts(tmp$norm_t, tmp$y2)))
})
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken
save(chg_pt_list, file = "full_segmented_chg_pts.Rdata")

load("segmented_chg_pts.Rdata")
chg_pt_interpolated_list <- lapply(1:length(clean_encounters), function(i) {
  cat(i, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(chg_pt_interpolate_data(tmp$norm_t, tmp$x1, chg_pt_list[[i]][[1]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y1, chg_pt_list[[i]][[1]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$x2, chg_pt_list[[i]][[2]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y2, chg_pt_list[[i]][[2]]))
})

norm_chg_pt_interpolated_list <- lapply(1:length(clean_encounters), function(i) {
  cat(i, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(chg_pt_interpolate_data(tmp$norm_t, tmp$x1, chg_pt_list[[i]][[1]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y1, chg_pt_list[[i]][[1]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$x2, chg_pt_list[[i]][[2]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y2, chg_pt_list[[i]][[2]]) / max(tmp$reldist))
})

time_pt_list <- lapply(1:length(clean_encounters), function(i) {
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(tmp$norm_t[chg_pt_list[[i]][[1]]],
       tmp$norm_t[chg_pt_list[[i]][[2]]])
})

p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(chg_pt_interpolated_list, time_pt_list)
save(p_chg_pt_dist_m, file = "p_chg_pt_dist_m.Rdata")

load("p_chg_pt_dist_m.Rdata")
chg_pt_mds <- cmdscale(p_chg_pt_dist_m, k = 10)
cv_folds <- split(sample(nrow(chg_pt_mds)), 1:10)
sapply(2:30, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(chg_pt_mds), fold, k)
}))})
png("avg_segment_p_dist.png")
make_group_plots(clean_rel_dist_m, kmeans_result$cluster, clean_encounters)
dev.off()


norm_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(chg_pt_interpolated_list, time_pt_list, norm = T)
save(norm_p_chg_pt_dist_m, file = "norm_p_chg_pt_dist_m.Rdata")

load("norm_p_chg_pt_dist_m.Rdata")
norm_chg_pt_mds <- cmdscale(norm_p_chg_pt_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_chg_pt_mds)), 1:10)
sapply(2:30, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(norm_chg_pt_mds), fold, k)
}))})
kmeans_result <- kmeans(norm_chg_pt_mds, 9)
png("avg_segment_p_dist.png")
make_group_plots(clean_rel_dist_m, kmeans_result$cluster, clean_encounters)
dev.off()

norm_avg_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(chg_pt_interpolated_list, time_pt_list, norm = T, avg = T)
save(norm_avg_p_chg_pt_dist_m, file = "norm_avg_p_chg_pt_dist_m.Rdata")

load("norm_avg_p_chg_pt_dist_m.Rdata")
norm_avg_chg_pt_mds <- cmdscale(norm_avg_p_chg_pt_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_avg_chg_pt_mds)), 1:10)
sapply(2:30, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(norm_avg_chg_pt_mds), fold, k)
}))})
kmeans_result <- kmeans(norm_avg_chg_pt_mds, 5)
png("norm_avg_segment_p_dist.png")
make_group_plots(clean_rel_dist_m, kmeans_result$cluster, clean_encounters)
dev.off()

total_norm_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(norm_chg_pt_interpolated_list, time_pt_list)
save(total_norm_p_chg_pt_dist_m, file = "total_norm_p_chg_pt_dist_m.Rdata")

total_norm_mds_m <- cmdscale(total_norm_p_chg_pt_dist_m)
kmeans_result <- kmeans(total_norm_mds_m, 6)
png("total_norm_segment_p_dist.png")
make_group_plots(clean_rel_dist_m, kmeans_result$cluster, clean_encounters)
dev.off()


total_norm_avg_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(norm_chg_pt_interpolated_list, time_pt_list, avg = T)
save(total_norm_avg_p_chg_pt_dist_m, file = "total_norm_avg_p_chg_pt_dist_m.Rdata")

total_norm_avg_mds_m <- cmdscale(total_norm_avg_p_chg_pt_dist_m)
sapply(2:30, function(k) {mean(sapply(cv_folds, function(fold) {
  cv_for_fold(as.matrix(total_norm_avg_mds_m), fold, k)
}))})
kmeans_result <- kmeans(total_norm_avg_mds_m, 5)
png("total_norm_avg_segment_p_dist.png")
make_group_plots(clean_rel_dist_m, kmeans_result$cluster, clean_encounters)
dev.off()

save(p_chg_pt_dist_m, norm_p_chg_pt_dist_m, norm_avg_p_chg_pt_dist_m,
     total_norm_p_chg_pt_dist_m, total_norm_avg_p_chg_pt_dist_m, file = "seg_p_chg_pts.Rdata")

#Try jointly
joint_chg_pt_list <- lapply(clean_encounters, function(encounter) {
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  if (nrow(tmp) == 0) {
    return(NA)
  }
  list(get_opt_number_chg_pts_jointly(tmp$norm_t, tmp$x1, tmp$y1),
       get_opt_number_chg_pts_jointly(tmp$norm_t, tmp$x2, tmp$y2))
})
save(joint_chg_pt_list, file = "jointly_segmented_chg_pts.Rdata")

all_joint_chg_pt_list <- lapply(clean_encounters, function(encounter) {
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  if (nrow(tmp) == 0) {
    return(NA)
  }
  list(get_opt_number_chg_pts_jointly(tmp$norm_t, list(tmp$x1, tmp$y1, tmp$x2, tmp$y2)))
})
save(joint_chg_pt_list, file = "jointly_segmented_chg_pts.Rdata")

joint_chg_pt_interpolated_list <- lapply(1:length(clean_encounters), function(i) {
  cat(i, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(chg_pt_interpolate_data(tmp$norm_t, tmp$x1, all_joint_chg_pt_list[[i]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y1, all_joint_chg_pt_list[[i]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$x2, all_joint_chg_pt_list[[i]]),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y2, all_joint_chg_pt_list[[i]]))
})

joint_chg_pt_norm_interpolated_list <- lapply(1:length(clean_encounters), function(i) {
  cat(i, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(chg_pt_interpolate_data(tmp$norm_t, tmp$x1, all_joint_chg_pt_list[[i]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y1, all_joint_chg_pt_list[[i]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$x2, all_joint_chg_pt_list[[i]]) / max(tmp$reldist),
       chg_pt_interpolate_data(tmp$norm_t, tmp$y2, all_joint_chg_pt_list[[i]]) / max(tmp$reldist))
})

joint_time_pt_list <- lapply(1:length(clean_encounters), function(i) {
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == clean_encounters[i],];
  list(tmp$norm_t[all_joint_chg_pt_list[[i]]],
       tmp$norm_t[all_joint_chg_pt_list[[i]]])
})

total_avg_norm_joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_norm_interpolated_list, joint_time_pt_list, norm = F, avg = T)
avg_norm_joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, norm = T, avg = T)

load("all_joint_p_chg_pt_dist_m.Rdata")
#6
setwd("~/Documents/Michigan/Research/Driving/plots/p_dist_clusters/seg_clusters/long_lat/")
run_cv(joint_p_chg_pt_dist_m, norm = T)
png("joint_p_dist.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

#6 minimizes
run_cv(total_norm_joint_p_chg_pt_dist_m, norm = T)
png("joint_total_norm_p_dist.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

png("joint_total_norm_p_dist_mds.png", width = 4, height = 4, res = 300)
plot(cmdscale(total_norm_joint_p_chg_pt_dist_m, k = 10))
dev.off()

# 6 minimizes
run_cv(total_avg_norm_joint_p_chg_pt_dist_m)
png("joint_total_avg_norm_p_dist.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_avg_norm_joint_p_chg_pt_dist_m, 10), 8)$cluster,
                 clean_encounters)
dev.off()
png("joint_total_avg_norm_p_dist_mds.png")
plot(cmdscale(total_avg_norm_joint_p_chg_pt_dist_m, k = 10))
dev.off()

#6
run_cv(norm_joint_p_chg_pt_dist_m, norm = T)
png("joint_norm_p_dist.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

#5
run_cv(avg_norm_joint_p_chg_pt_dist_m, norm = T)
png("joint_avg_norm_p_dist.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(avg_norm_joint_p_chg_pt_dist_m, 10), 5)$cluster,
                 clean_encounters)
dev.off()

#Joint trajectory
joint_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, traj = T)
total_norm_joint_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(joint_chg_pt_norm_interpolated_list, joint_time_pt_list, norm = F, traj = T)
norm_joint_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, norm = T, traj = T)
joint_avg_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, avg = T, traj = T)
total_avg_norm_joint_p_chg_pt_dist_m <- get_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, norm = T, avg = T, traj = T)
save(joint_p_chg_pt_dist_m, norm_joint_p_chg_pt_dist_m, joint_avg_p_chg_pt_dist_m, total_avg_norm_joint_p_chg_pt_dist_m, 
     total_norm_joint_p_chg_pt_dist_m, file = "all_traj_joint_p_chg_pt_dist_m.Rdata")

#Joint entire
joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list)
total_norm_joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_norm_interpolated_list, joint_time_pt_list, norm = F)
norm_joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, norm = T)
joint_avg_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, avg = T)
total_avg_norm_joint_p_chg_pt_dist_m <- get_entire_p_dist_with_chg_pts_m(joint_chg_pt_interpolated_list, joint_time_pt_list, norm = T, avg = T)
save(joint_p_chg_pt_dist_m, norm_joint_p_chg_pt_dist_m, joint_avg_p_chg_pt_dist_m, total_avg_norm_joint_p_chg_pt_dist_m, 
     total_norm_joint_p_chg_pt_dist_m, file = "all_entire_joint_p_chg_pt_dist_m.Rdata")

load("~/Documents/Michigan/Research/Driving/matlab_data/all_entire_joint_p_chg_pt_dist_m.Rdata")
setwd("~/Documents/Michigan/Research/Driving/plots/p_dist_clusters/seg_clusters/entire/")
run_cv(joint_p_chg_pt_dist_m, norm = T)
png("entire_joint_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(joint_p_chg_pt_dist_m, 10), 10)$cluster,
                 clean_encounters)
dev.off()

run_cv(total_norm_joint_p_chg_pt_dist_m, norm = T)
png("entire_joint_total_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_norm_joint_p_chg_pt_dist_m, 10), 7)$cluster,
                 clean_encounters)
dev.off()

run_cv(norm_joint_p_chg_pt_dist_m, norm = T)
png("entire_joint_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

run_cv(joint_avg_p_chg_pt_dist_m, norm = T)
png("entire_joint_avg_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(joint_avg_p_chg_pt_dist_m, 10), 7)$cluster,
                 clean_encounters)
dev.off()

run_cv(total_avg_norm_joint_p_chg_pt_dist_m, norm = T)
png("entire_joint_avg_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_avg_norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

#Loads all trajectory
load("~/Documents/Michigan/Research/Driving/matlab_data/all_traj_joint_p_chg_pt_dist_m.Rdata")
setwd("~/Documents/Michigan/Research/Driving/plots/p_dist_clusters/seg_clusters/traj/")

run_cv(joint_p_chg_pt_dist_m, norm = T)
png("traj_joint_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(joint_p_chg_pt_dist_m, 10), 8)$cluster,
                 clean_encounters)
dev.off()

run_cv(total_norm_joint_p_chg_pt_dist_m, norm = T)
png("traj_joint_total_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_norm_joint_p_chg_pt_dist_m, 10), 10)$cluster,
                 clean_encounters)
dev.off()

run_cv(norm_joint_p_chg_pt_dist_m, norm = T)
png("traj_joint_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

run_cv(joint_avg_p_chg_pt_dist_m, norm = T)
png("traj_joint_avg_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(joint_avg_p_chg_pt_dist_m, 10), 7)$cluster,
                 clean_encounters)
dev.off()

run_cv(total_avg_norm_joint_p_chg_pt_dist_m, norm = T)
png("traj_joint_avg_norm_p_chg_pt_groups.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(clean_rel_dist_m, 
                 kmeans(cmdscale(total_avg_norm_joint_p_chg_pt_dist_m, 10), 6)$cluster,
                 clean_encounters)
dev.off()

#Joint

# JK, Ben, here is the actual joint segmented things

all_joint_chg_pt_list <- mclapply(clean_encounters, function(encounter) {
  cat(encounter, "\n")
  tmp <- rel_dist_m[rel_dist_m$encounter == encounter,];
  if (nrow(tmp) == 0) {
    return(NA)
  }
  if (nrow(tmp) < 4) {
    return(NA)
  }
  get_opt_number_chg_pts_jointly(tmp$norm_t, list(tmp$x1, tmp$x2, tmp$y1, tmp$y2))
}, mc.cores = 3)
save(all_joint_chg_pt_list, file = "new_all_jointly_segmented_chg_pts.Rdata")

interpolated_prim_list <- list()
for (i in 1:length(clean_encounters)) {
  encounter = clean_encounters[i]
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  if (nrow(tmp) < 4) {
    next
  }
  interpolated_prim_list  <- c(interpolated_prim_list,
                 chg_pt_interpolate_prims(tmp$norm_t, 
                                          list("x1" = tmp$x1, "y1" = tmp$y1, 
                                               "x2" = tmp$x2, "y2" = tmp$y2), 
                                          all_joint_chg_pt_list[[i]],
                                          encounter))
}
save(interpolated_prim_list, file = "new_cubic_poly_interpolated_prim.Rdata")
prim_p_dist_m <- get_p_dist_m(interpolated_prim_list)
save(interpolated_prim_list, prim_p_dist_m, file = "prim_p_dist_m.Rdata")

prim_p_dist_m <- get_p_dist_m(interpolated_prim_list, traj = T)
save(interpolated_prim_list, prim_p_dist_m, file = "traj_prim_p_dist_m.Rdata")

prim_p_dist_m <- get_entire_p_dist_m(interpolated_prim_list)
save(interpolated_prim_list, prim_p_dist_m, file = "entire_prim_p_dist_m.Rdata")

load("prim_p_dist_m.Rdata")
prim_p_dist_mds <- cmdscale(prim_p_dist_m, k = 10)

cv_folds <- split(sample(nrow(prim_p_dist_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(prim_p_dist_mds, fold, k, iter.max = 30)
  }))})

prim_mds_cv <- kmeans(prim_p_dist_mds, 14)
prim_m <- do.call(rbind, prim_list)
prim_m$encounter <- as.factor(prim_m$prim_name)
prim_assignment <- data.frame("id" = unique(prim_m$prim_name), 
                              "cluster" = prim_mds_cv$cluster, 
                              stringsAsFactors = F)
random_rep = sapply(1:length(unique(prim_assignment$cluster)), function(i) {
  sample(prim_assignment$id[prim_assignment$cluster == i], 1)})

png("prim_p_dist_plots.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_m, prim_mds_cv$cluster, unique(prim_m$prim_name))
dev.off()
png("prim_p_dist_mds.png")
plot(prim_p_dist_mds)
dev.off()

load("entire_prim_p_dist_m.Rdata")
entire_prim_p_dist_mds <- cmdscale(prim_p_dist_m, k = 10)

prim_mds_cv <- kmeans(entire_prim_p_dist_mds, 7)
prim_m <- do.call(rbind, interpolated_prim_list)
prim_m$encounter <- as.factor(prim_m$prim_name)
prim_assignment <- data.frame("id" = unique(prim_m$prim_name), 
                              "cluster" = prim_mds_cv$cluster, 
                              stringsAsFactors = F)
random_rep = sapply(1:length(unique(prim_assignment$cluster)), function(i) {
  sample(prim_assignment$id[prim_assignment$cluster == i], 1)})

png("entire_prim_p_dist_plots.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_m, prim_mds_cv$cluster, unique(prim_m$prim_name))
dev.off()
png("entire_prim_p_dist_mds.png")
plot(entire_prim_p_dist_mds)
dev.off()

load("traj_prim_p_dist_m.Rdata")
traj_prim_p_dist_mds <- cmdscale(prim_p_dist_m, k = 10)

cv_folds <- split(sample(nrow(prim_p_dist_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(prim_p_dist_mds / max(prim_p_dist_mds), fold, k, iter.max = 30)
  }))})

prim_mds_cv <- kmeans(traj_prim_p_dist_mds, 12)
prim_m <- do.call(rbind, interpolated_prim_list)
prim_m$encounter <- as.factor(prim_m$prim_name)
prim_assignment <- data.frame("id" = unique(prim_m$prim_name), 
                              "cluster" = prim_mds_cv$cluster, 
                              stringsAsFactors = F)
random_rep = sapply(1:12, function(i) {
  sample(prim_assignment$id[prim_assignment$cluster == i], 1)})

png("traj_prim_p_dist_plots.png", units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_m, prim_mds_cv$cluster, unique(prim_m$prim_name))
dev.off()



#run_cv(prim_p_dist_m, 50)

prim_info_list <- list()
aug_df <- clean_rel_dist_m
aug_df$prim_name <- NA
for (i in 1:length(clean_encounters)) {
  encounter = clean_encounters[i]
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  if (nrow(tmp) < 4) {
    next
  }
  chg_pts <- all_joint_chg_pt_list[[i]]
  for (j in 1:(length(chg_pts) - 1)) {
    prim_list <- list(
      "x1" = fit_cubic_poly(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]], 
                              tmp$x1[chg_pts[j]:chg_pts[(j + 1)]]),
      "x2" = fit_cubic_poly(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]], 
                              tmp$x2[chg_pts[j]:chg_pts[(j + 1)]]),
      "y1" = fit_cubic_poly(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]], 
                              tmp$y1[chg_pts[j]:chg_pts[(j + 1)]]),
      "y2" = fit_cubic_poly(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]], 
                              tmp$y2[chg_pts[j]:chg_pts[(j + 1)]]),
      "length" = max(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]]) - 
                 min(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]]),
      "start" = min(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]]),
      "end" = max(tmp$Time[chg_pts[j]:chg_pts[(j + 1)]]),
      "encounter" = paste(encounter, j, sep = "_")
    )
    prim_info_list <- c(prim_info_list, list(prim_list))
    aug_df$prim_name[
      which(clean_rel_dist_m$encounter == encounter)[chg_pts[j]:chg_pts[(j + 1)]]] =
      paste(encounter, j, sep = "_")
  }
}
aug_df$ID <- aug_df$prim_name
save(aug_df, prim_info_list, file = "new_cubic_poly_prim_poly_info.Rdata")

max_prim_dist_m <- get_p_dist_for_p_info_list(prim_info_list, interpolated_prim_list)
min_prim_dist_m <- get_p_dist_for_p_info_list(prim_info_list, interpolated_prim_list, min)
mean_prim_dist_m <- get_p_dist_for_p_info_list(prim_info_list, interpolated_prim_list, mean)
save(max_prim_dist_m, min_prim_dist_m, mean_prim_dist_m, "prim_info_dist_m.Rdata")

#Ding's analysis
dtw <- read.csv("dtw_v_m.csv", header = F)
ding_prim_info <- read.csv("ding_prim_m.csv", header = F)

kmeans_results <- kmeans(dtw, 20)
ding_prim_info <- as.data.frame(ding_prim_info)
colnames(ding_prim_info) <- c("x1", "y1", "speed_1", "x2", "y2", "speed_2", "encounter")
ding_prim_info$cluster <- rep(kmeans_results$cluster, each = 50)
ding_prim_info_list <- lapply(sort(unique(ding_prim_info$encounter)), 
                              function(id) {ding_prim_info[ding_prim_info$encounter == id,]})
ding_p_dist <- get_entire_p_dist_m(trans_ding_prim_info_list)
speed_ding_p_dist <- get_entire_p_dist_m(trans_ding_prim_info_list, speed = T)
save(ding_prim_info, ding_p_dist, file = "entire_ding_p_dist.Rdata")

#Calculate Procrustes from tmp <- tmp[tmp$trtid10 %in% common_ids,]

get_silhouette_m <- function(prim_info_m, p_dist_m, num_obs = 50) {
  group_list <- 1:max(unique(prim_info_m$cluster))
  assignments <- prim_info_m$cluster
  assignments <- assignments[0:(length(unique(prim_info_m$encounter)) - 1) * num_obs + 1]
  silhouette_m <- t(sapply(1:length(assignments), function(i) {
    b_i = min(sapply(group_list[group_list != assignments[i]], 
                     function(a) mean(p_dist_m[i, assignments == a])))
    a_i = mean(p_dist_m[i, (assignments == assignments[i]) & (1:ncol(p_dist_m) != i)])
    c(assignments[i], (b_i - a_i) / max(a_i, b_i), unique(prim_info_m$encounter)[i]) 
  }))
  silhouette_m <- as.data.frame(silhouette_m)
  silhouette_m[, "V2"] <- as.numeric(as.character(silhouette_m[, "V2"] ))
  silhouette_m <- silhouette_m %>% arrange(V1, desc(V2))
}

get_silhouette_m_for_assignment <- function(p_dist_m, assignments,
                                            k = max(assignments)) {
  group_list <- 1:max(assignments)
  silhouette_m <- t(sapply(1:length(assignments), function(i) {
    b_i = min(sapply(group_list[group_list != assignments[i]],
                     function(a) mean(p_dist_m[i, assignments == a])), 
              na.rm = T)
    a_i = mean(p_dist_m[i, (assignments == assignments[i]) & (1:ncol(p_dist_m) != i)])
    c(assignments[i], (b_i - a_i) / max(a_i, b_i))
  }))
  silhouette_m <- as.data.frame(silhouette_m)
  silhouette_m[, "V2"] <- as.numeric(as.character(silhouette_m[, "V2"] ))
  silhouette_m <- silhouette_m %>% arrange(V1, desc(V2))
}

ggplot(silhouette_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")

load("trans_ding_p_dist.Rdata")
trans_silhouette_m <- get_silhouette_m(trans_ding_prim_info, trans_ding_p_dist)
png("transformed_ding_prim_p_dist_silhouette.png")
ggplot(trans_silhouette_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")
dev.off()

load("speed_trans_ding_p_dist.Rdata")
speed_trans_silhouette_m <- get_silhouette_m(trans_ding_prim_info, speed_trans_ding_p_dist)
png("transformed_ding_prim_p_dist_speed_silhouette.png")
ggplot(speed_trans_silhouette_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")
dev.off()

png("ding_prim_group_plots.png")
make_group_plots(trans_ding_prim_info, trans_ding_prim_info$encounter[0:4125 * 50 + 1],
                 trans_ding_prim_info$cluster[0:4125 * 50 + 1])
dev.off()
get_silhouette_stats_for_k <- function(prim_info_m, p_dist_m, dtw_m, k) {
  kmeans_results <- kmeans(dtw_m, k, iter.max = 30)
  prim_info_m$cluster <- rep(kmeans_results$cluster, each = 50)
  silhouette_m <- get_silhouette_m(prim_info_m, p_dist_m)
  #c(k, min(silhouette_m$V2), mean(silhouette_m$V2))
}

get_silhouette_stats_for_diff_cluster_num <- function(prim_info_m, p_dist_m, 
                                                      dtw_m, cluster_nums = 2:30) {
  sapply(cluster_nums, function(k) {cat(k, "\n");
    get_silhouette_stats_for_k(prim_info_m, p_dist_m, dtw_m, k)})
}

calc_min_max_ratio <- function(p_dist, assignment_m) {
  min_without_dist = Inf
  max_within_dist = -Inf
  for (i in 1:nrow(assignment_m)) {
    same_group <- which(assignment_m$cluster == assignment_m$cluster[i])
    tmp_min <- min(p_dist[i, -same_group])
    tmp_max <- max(p_dist[i, same_group])
    if (tmp_min < min_without_dist) {
      min_without_dist = tmp_min
    }
    if (tmp_max > max_within_dist) {
      max_within_dist = tmp_max
    }
  }
  return(min_without_dist / max_within_dist)
}

run_permuation_test <- function(p_dist, assignment_m, num_permutations = 100) {
  assignment_val <- calc_min_max_ratio(p_dist, assignment_m)
  print(assignment_val)
  tmp <- assignment_m
  p_val_count = 0
  for (i in 1:num_permutations) {
    print(i)
    tmp[,2] <- assignment_m$cluster[sample(nrow(tmp))]
    if (calc_min_max_ratio(p_dist, tmp) > assignment_val) {
      p_val_count = p_val_count + 1
    }
    print(calc_min_max_ratio(p_dist, tmp))
  }
  return(p_val_count / num_permutations)
}

#Load template_prim_dist_m
load("prim_info_dist_m.Rdata")
prim_info <- do.call(rbind, lapply(prim_info_list, function(p) {
  new_x <- seq(0, 1, by = 0.01);
  new_x <- new_x * p$end + (1 - new_x) * p$start;
  tmp <- as.data.frame(sapply(p[c("x1", "x2", "y1", "y2")], function(p_lm) {
    predict(p_lm, data.frame("x" = new_x), type = "response");
  }))
  tmp$encounter = p$encounter;
  rownames(tmp) <- NULL;
  tmp;
}))

min_prim_dist_m <- do.call(pmin, template_prim_dist_m)
max_prim_dist_m <- do.call(pmax, template_prim_dist_m)
mean_prim_dist_m <- Reduce('+', template_prim_dist_m) / length(template_prim_dist_m)

norm_min_prim_dist_m <- min_prim_dist_m / max(min_prim_dist_m)
norm_max_prim_dist_m <- max_prim_dist_m / max(max_prim_dist_m)
norm_mean_prim_dist_m <- mean_prim_dist_m / max(mean_prim_dist_m)

norm_min_prim_dist_m_mds <- cmdscale(norm_min_prim_dist_m, k = 10)
png("norm_min_prim_dist_m_mds_plot.png")
plot(norm_min_prim_dist_m_mds)
dev.off()
cv_folds <- split(sample(nrow(norm_min_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_min_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_min_prim_dist_m_mds, centers = 4)
png('norm_min_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

assignment_m <- data.frame("id" = prim_info$encounter[((0:5621) * 101 + 1)], "cluster" = kmeans_result$cluster)

do.call("grid.arrange", c(lapply(3:7, function(k) {
  kmeans_result <- kmeans(norm_min_prim_dist_m_mds, centers = k)
  prim_info$cluster <- rep(kmeans_result$cluster, each = 101)
  s_m <- get_silhouette_m(prim_info, norm_min_prim_dist_m, 101)
  ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")
}), "ncol" = 2))

norm_max_prim_dist_m_mds <- cmdscale(norm_max_prim_dist_m, k = 10)
png("norm_max_prim_dist_m_mds_plot.png")
plot(norm_max_prim_dist_m_mds)
dev.off()

cv_folds <- split(sample(nrow(norm_max_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_max_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = 4)
png('norm_max_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

do.call("grid.arrange", c(lapply(3:7, function(k) {
kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = k)
prim_info$cluster <- rep(kmeans_result$cluster, each = 101)
s_m <- get_silhouette_m(prim_info, norm_min_prim_dist_m, 101)
ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")
}), "ncol" = 2))

norm_mean_prim_dist_m_mds <- cmdscale(norm_mean_prim_dist_m, k = 10)
png("norm_mean_prim_dist_m_mds_plot.png")
plot(norm_mean_prim_dist_m_mds)
dev.off()
cv_folds <- split(sample(nrow(norm_mean_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_mean_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_mean_prim_dist_m_mds, centers = 8)
png('norm_mean_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

do.call("grid.arrange", c(lapply(3:7, function(k) {
  kmeans_result <- kmeans(norm_mean_prim_dist_m_mds, centers = k)
  prim_info$cluster <- rep(kmeans_result$cluster, each = 101)
  s_m <- get_silhouette_m(prim_info, norm_min_prim_dist_m, 101)
  ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity")
}), "ncol" = 2))

load("entire_prim_info_dist_m.Rdata")
prim_info <- do.call(rbind, interpolated_prim_list)
prim_info$encounter <- as.factor(prim_info$prim_name)
min_prim_dist_m <- do.call(pmin, entire_template_prim_dist_m)
max_prim_dist_m <- do.call(pmax, entire_template_prim_dist_m)
mean_prim_dist_m <- Reduce('+', entire_template_prim_dist_m) / length(entire_template_prim_dist_m)

norm_min_prim_dist_m <- min_prim_dist_m / max(min_prim_dist_m)
norm_max_prim_dist_m <- max_prim_dist_m / max(max_prim_dist_m)
norm_mean_prim_dist_m <- mean_prim_dist_m / max(mean_prim_dist_m)

norm_min_prim_dist_m_mds <- cmdscale(norm_min_prim_dist_m, k = 10)
png("entire_norm_min_prim_dist_m_mds_plot.png")
plot(norm_min_prim_dist_m_mds)
dev.off()
cv_folds <- split(sample(nrow(norm_min_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_min_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_min_prim_dist_m_mds, centers = 4)
png('entire_norm_min_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

norm_max_prim_dist_m_mds <- cmdscale(norm_max_prim_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_max_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_max_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = 10)
png('entire_norm_max_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

png('entire_norm_max_prim_interpolate_p_dist_silhouette_plots.png', units = 'in', width = 4, height = 1, res = 300)
do.call("grid.arrange", c(lapply(7:10, function(k) {
  kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = k)
  prim_info$cluster <- rep(kmeans_result$cluster, each = 101)
  s_m <- get_silhouette_m(prim_info, norm_max_prim_dist_m, 101)
  ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity") + 
    theme(legend.position="none") + labs(x = "", y = "")
}), "ncol" = 4))
dev.off()

for (k in 7:10) {
  kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = k)
  prim_info$cluster <- rep(kmeans_result$cluster, each = 101)
  s_m <- get_silhouette_m(prim_info, norm_max_prim_dist_m, 101)
  png(paste("entire_norm_max_prim_interpolate_p_dist_silhouette_plots_", k, ".png", sep = ""),
      units = 'in', width = 1, height = 1, res = 300)
  print(ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity") + 
    theme(legend.position="none") + labs(x = "", y = ""))
  dev.off()
}

norm_mean_prim_dist_m_mds <- cmdscale(norm_mean_prim_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_max_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_max_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_mean_prim_dist_m_mds, centers = 8)
png('entire_norm_mean_prim_interpolate_p_dist_plot.png')
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

load("traj_prim_info_dist_m.Rdata")
prim_info <- do.call(rbind, interpolated_prim_list)
prim_info$encounter <- as.factor(prim_info$prim_name)
min_prim_dist_m <- do.call(pmin, entire_template_prim_dist_m)
max_prim_dist_m <- do.call(pmax, entire_template_prim_dist_m)
mean_prim_dist_m <- Reduce('+', entire_template_prim_dist_m) / length(entire_template_prim_dist_m)

norm_min_prim_dist_m <- min_prim_dist_m / max(min_prim_dist_m)
norm_max_prim_dist_m <- max_prim_dist_m / max(max_prim_dist_m)
norm_mean_prim_dist_m <- mean_prim_dist_m / max(mean_prim_dist_m)

norm_min_prim_dist_m_mds <- cmdscale(norm_min_prim_dist_m, k = 10)
png("traj_norm_min_prim_dist_m_mds_plot.png")
plot(norm_min_prim_dist_m_mds)
dev.off()
cv_folds <- split(sample(nrow(norm_min_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_min_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_min_prim_dist_m_mds, centers = 8)
png('traj_norm_min_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

norm_max_prim_dist_m_mds <- cmdscale(norm_max_prim_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_max_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_max_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_max_prim_dist_m_mds, centers = 10)
png('traj_norm_max_prim_interpolate_p_dist_plot.png', units = 'in', width = 4, height = 4, res = 300)
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

norm_mean_prim_dist_m_mds <- cmdscale(norm_mean_prim_dist_m, k = 10)
cv_folds <- split(sample(nrow(norm_mean_prim_dist_m_mds)), 1:10)
cv_results <- sapply(2:20, function(k) {
  cat(k, "\n");
  mean(sapply(cv_folds, function(fold) {
    cv_for_fold(norm_mean_prim_dist_m_mds, fold, k, iter.max = 30)
  }))})
kmeans_result <- kmeans(norm_mean_prim_dist_m_mds, centers = 8)
png('traj_norm_mean_prim_interpolate_p_dist_plot.png')
make_group_plots(prim_info, kmeans_result$cluster, unique(prim_info$encounter))
dev.off()

# Stuff from paper here too, but still unsure where distance calculated

orient_and_centered_to_first <- function(traj_1_a, traj_1_b, traj_2_a, traj_2_b) {
  traj_1 <- rbind(data.matrix(traj_1_a), data.matrix(traj_1_b))
  traj_2 <- rbind(data.matrix(traj_2_a), data.matrix(traj_2_b))
  traj_1 <- scale(traj_1, scale = F)
  traj_2 <- scale(traj_2, scale = F)
  svd_decomp <- svd(crossprod(traj_2, traj_1))
  rotation_matrix <- crossprod(svd_decomp$v, svd_decomp$u)
  reoriented_traj_2 <- traj_2 %*% t(rotation_matrix)
  return(cbind(reoriented_traj_2[1:nrow(traj_2_a), 1:2],
               reoriented_traj_2[(nrow(traj_2_a) + 1):(2 * nrow(traj_2_a)),1:2]))
}

orient_and_centered_to_first_no_reflection <- function(traj_1_a, traj_1_b, traj_2_a, traj_2_b) {
  traj_1 <- rbind(data.matrix(traj_1_a), data.matrix(traj_1_b))
  traj_2 <- rbind(data.matrix(traj_2_a), data.matrix(traj_2_b))
  traj_1 <- scale(traj_1, scale = F)
  traj_2 <- scale(traj_2, scale = F)
  svd_decomp <- svd(crossprod(traj_2, traj_1))
  diag_m <- diag(1, nrow = nrow(svd_decomp$u))
  diag_m[nrow(svd_decomp$u), nrow(svd_decomp$u)] = det(tcrossprod(svd_decomp$v, svd_decomp$u))
  rotation_matrix <- svd_decomp$v %*% tcrossprod(diag_m, svd_decomp$u)
  reoriented_traj_2 <- traj_2 %*% t(rotation_matrix)
  return(cbind(reoriented_traj_2[1:nrow(traj_2_a), 1:2],
               reoriented_traj_2[(nrow(traj_2_a) + 1):(2 * nrow(traj_2_a)),1:2]))
}


update_assignment_oriented_m <- function(prim_oriented_m, prim_list, 
                                         interested_inds, ref_ind, reflection = F) {
  for (i in interested_inds) {
    if (reflection) {
      prim_oriented_m[i, ] <- as.vector(orient_and_centered_to_first(
        prim_list[[ref_ind]][, c("x1", "y1")], prim_list[[ref_ind]][, c("x2", "y2")],
        prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))
    } else {
      prim_oriented_m[i, ] <- as.vector(orient_and_centered_to_first_no_reflection(
        prim_list[[ref_ind]][, c("x1", "y1")], prim_list[[ref_ind]][, c("x2", "y2")],
        prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))
    }
  }
  return(prim_oriented_m)
}

procrustes_kmeans <- function(prim_list, num_clusters, reflection = F) {
  new_assignment <- sample(num_clusters, length(prim_list), replace = T)
  old_assignment <- rep(0, length(new_assignment))
  oriented_prim_m <- matrix(0, nrow = length(prim_list),
                            ncol = 4 * nrow(prim_list[[1]]))
  new_total_error = Inf
  prev_total_error = new_total_error
  mean_matrix_list <- lapply(1:num_clusters, function(i) {matrix(0, nrow = nrow(prim_list[[1]]), 
                                                                 ncol = ncol(prim_list[[1]]))})
  repeat {
    for (k in 1:num_clusters) {
      new_k_assignment <- which(new_assignment == k)
      #old_k_assignment <- which(old_assignment == k)
      update_assignment <- new_k_assignment
      #if (length(old_k_assignment) != 0) {
        #update_assignment <- new_k_assignment[-which(new_k_assignment %in% old_k_assignment)]
      #}
      #if (length(update_assignment) != 0) {
        oriented_prim_m <- update_assignment_oriented_m(oriented_prim_m, prim_list,
                                                        update_assignment, new_k_assignment[1], 
                                                        reflection)
        mean_matrix_list[[k]] <- matrix(colMeans(oriented_prim_m[new_k_assignment,, drop = F]), ncol = 4)
      #}
    }
    new_total_error = 0
    for (i in 1:nrow(oriented_prim_m)) {
      min_error = Inf
      min_ind = 0
      for (k in 1:num_clusters) {
        if (reflection) {
          error_for_k <- mean((mean_matrix_list[[k]] - 
                                 orient_and_centered_to_first(mean_matrix_list[[k]][, 1:2],
                                                              mean_matrix_list[[k]][, 3:4],
                                                              prim_list[[i]][, c("x1", "y1")], 
                                                              prim_list[[i]][, c("x2", "y2")]))^2)
        } else {
          error_for_k <- mean((mean_matrix_list[[k]] - 
                                 orient_and_centered_to_first_no_reflection(
                                   mean_matrix_list[[k]][, 1:2], mean_matrix_list[[k]][, 3:4],
                                   prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))^2)
        }
        if (error_for_k < min_error) {
          min_error = error_for_k
          min_ind = k
        }
      }
      if (sum(new_assignment == new_assignment[i]) != 1) {
        new_total_error = new_total_error + min_error
        new_assignment[i] = min_ind
      }
    }
    print(new_total_error)
    if (prev_total_error - new_total_error < 1e-6) {
      break
    }
    old_assignment = new_assignment
    prev_total_error = new_total_error
  }
  return(list("cluster" = new_assignment, 
              "centers" = mean_matrix_list,
              "oriented_m" = oriented_prim_m))
}

load("prim_p_dist_m.Rdata")
oriented_prim_m <- matrix(0, nrow = length(prim_list),
                          ncol = 4 * nrow(prim_list[[1]]))
for (i in 1:length(prim_list)) {
  print(i)
  oriented_prim_m[i,] <- as.vector(orient_and_centered_to_first(
    prim_list[[1]][, c("x1", "y1")], prim_list[[1]][, c("x2", "y2")],
    prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))
}
cosac_results <- cosac(oriented_prim_m, 
                       median(apply(scale(oriented_prim_m, scale = F), 1,
                                    function(row) {norm(row, type = "2")})),
                       .6, .005, T, T)
cosac_plot_df <- as.data.frame(do.call(rbind, lapply(1:length(cosac_results), 
                                                     function(i) {cbind(matrix(cosac_results[[i]], ncol = 4), i)})))
ggplot(cosac_plot_df) + geom_path(aes(x = V1, y = V2, color = "Traj 1")) + 
  geom_path(aes(x = V3, y = V4, color = "Traj 2")) + facet_wrap(~i)
kmeans_results <- kmeans(oriented_prim_m, 7)
kmeans_plot_df <- as.data.frame(do.call(rbind, lapply(1:7, function(i) {
  cbind(matrix(kmeans_results$centers[i,], ncol = 4), i)
})))
ggplot() + geom_path(data = kmeans_plot_df, aes(x = V1, y = V2, color = "Traj 1")) + 
  geom_path(data = kmeans_plot_df, aes(x = V3, y = V4, color = "Traj 2")) + 
  geom_point(data = kmeans_plot_df[101 * (0:6) + 1,], aes(x = V1, y = V2, color = "Traj 1")) +
  geom_point(data = kmeans_plot_df[101 * (0:6) + 1,], aes(x = V3, y = V4, color = "Traj 2")) +
  facet_wrap(~i, scales = "free")


prim_plot_df <- do.call(rbind, prim_list)
prim_plot_df$encounter <- as.factor(prim_plot_df$prim_name)
make_group_plots(prim_plot_df, kmeans_results$cluster, unique(prim_plot_df$prim_name))

prim_plot_df$cluster <- rep(kmeans_results$cluster, each = 101)
kmeans_plot_df$cluster = kmeans_plot_df$i
example_encounters <- as.vector(sapply(1:7, function(i) {sample(which(kmeans_results$cluster == i), 5)}))
example_encounters <- unique(prim_plot_df$prim_name)[example_encounters]
example_prim_plot_df1 <- prim_plot_df[prim_plot_df$prim_name %in% example_encounters,]
example_prim_plot_df1$prim_name <- paste(example_prim_plot_df1$prim_name, 1, sep = "_") 
example_prim_plot_df2 <- prim_plot_df[prim_plot_df$prim_name %in% example_encounters,]
example_prim_plot_df2$prim_name <- paste(example_prim_plot_df1$prim_name, 2, sep = "_") 
ggplot() + geom_path(data = kmeans_plot_df, aes(x = V1, y = V2), color = "red") + 
  geom_path(data = kmeans_plot_df, aes(x = V3, y = V4), color = "blue") + 
  geom_path(data = prim_plot_df[prim_plot_df$prim_name %in% example_encounters,], 
            aes(x = x1, y = y1, color = prim_name)) + 
  geom_path(data = prim_plot_df[prim_plot_df$prim_name %in% example_encounters,], 
            aes(x = x2, y = y2, color = prim_name)) + 
  guides(color = F) + facet_wrap(~cluster)

gdm_results <- run_tuned_gdm(oriented_prim_m, 1:ncol(oriented_prim_m), 7, tune = T)

p_kmeans_result <- procrustes_kmeans(prim_list, 7)
p_kmeans_df <- as.data.frame(
  do.call(rbind, lapply(1:length(p_kmeans_result$centers), 
                        function(i) {cbind(p_kmeans_result$centers[[i]], i)})))
ggplot() + geom_path(data = p_kmeans_df, aes(x = V1, y = V2, color = "Traj 1")) + 
  geom_path(data = p_kmeans_df, aes(x = V3, y = V4, color = "Traj 2")) + 
  geom_point(data = p_kmeans_df[101 * (0:6) + 1,], aes(x = V1, y = V2, color = "Traj 1")) +
  geom_point(data = p_kmeans_df[101 * (0:6) + 1,], aes(x = V3, y = V4, color = "Traj 2")) +
  facet_wrap(~i, scales = "free")

#Try with time sliding
load("all_jointly_segmented_chg_pts.Rdata")

renorm_prim_poly_list <- list()
for (i in 1:length(clean_encounters)) {
  tmp <- rel_dist_m[rel_dist_m$encounter == clean_encounters[i],]
  encounter_chg_pt_list <- all_joint_chg_pt_list[[i]]
  for (j in 1:(length(encounter_chg_pt_list) - 1)) {
    #new_x <- seq(0, 1, length.out = (encounter_chg_pt_list[j + 1] - 
                                       #encounter_chg_pt_list[j] + 1))
    new_x = tmp$t[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]
    renorm_prim_poly_list <- c(renorm_prim_poly_list, list(list(
      "x1" = fit_cubic_poly(new_x, tmp$x1[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
      "y1" = fit_cubic_poly(new_x, tmp$y1[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
      "x2" = fit_cubic_poly(new_x, tmp$x2[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
      "y2" = fit_cubic_poly(new_x, tmp$y2[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
      "start_time" = tmp$t[encounter_chg_pt_list[j]],
      "time" = tmp$t[encounter_chg_pt_list[j + 1]] - tmp$t[encounter_chg_pt_list[j]])))
  }
}

min_time = min(sapply(renorm_prim_poly_list, function(info) {info$time}))

interpolate_and_orient_prim_poly <- function(prim_poly, time_scale, 
                                             start_time, first_poly) {
  new_x <- (1 - seq(0, 1, by = 0.01)) * start_time + 
    (start_time + time_scale) * seq(0, 1, by = 0.01)
  interpolate_poly <- matrix(0, nrow = length(new_x), ncol = 4)
  for (i in 1:4) {
    interpolate_poly[, i] <- 
      predict(prim_poly[[i]], data.frame("x" = new_x), type = "response")
  }
  orient_and_centered_to_first(first_poly[,1:2], first_poly[,3:4],
                               interpolate_poly[, 1:2], interpolate_poly[, 1:2])
}

interpolate_and_orient_prim_poly_info <- function(prim_poly_info, time_scale, 
                                                  start_time_list) {
  new_x <- (1 - seq(0, 1, by = 0.01)) * start_time_list[1] + 
    (start_time_list[1] + time_scale) * seq(0, 1, by = 0.01)
  first_poly <- matrix(0, nrow = length(new_x), ncol = 4)
  for (i in 1:4) {
    first_poly[, i] <- predict(prim_poly_info[[1]][[i]], data.frame("x" = new_x), type = "response")
  }
  prim_matrix <- matrix(0, nrow = length(prim_poly_info), ncol = 4 * length(new_x))
  for (i in 1:length(prim_poly_info)) {
    prim_matrix[i,] <- as.vector(
      interpolate_and_orient_prim_poly(prim_poly_info[[i]], time_scale, 
                                       start_time_list[i], first_poly))
  }
  return(prim_matrix)
} 

hier_sample_sliding_procrustes_dist_approx <- function(prim_poly_info, time_scale, 
                                                       num_clusters, num_iter = 100) {
  dist_matrix <- matrix(0, nrow = length(prim_poly_info), ncol = length(prim_poly_info))
  time_lengths <- sapply(prim_poly_info, function(info) {info$time})
  start_times_list <- sapply(prim_poly_info, function(info) {info$start_time})
  for (i in 1:num_iter) {
    print(i)
    start_point = runif(length(prim_poly_info), start_times_list, 
                        start_times_list + time_lengths - time_scale)
    if (any(time_lengths == time_scale)) {
      start_point[which(time_lengths == time_scale)] = 
        start_times_list[which(time_lengths == time_scale)]
    }
    interpolated_prim_matrix <- 
      interpolate_and_orient_prim_poly_info(prim_poly_info, time_scale, start_point)
    kmeans_results <- kmeans(interpolated_prim_matrix, centers = num_clusters)
    for (k in 1:num_clusters) {
      dist_matrix[which(kmeans_results$cluster == k),
                  which(kmeans_results$cluster == k)] = 
        dist_matrix[which(kmeans_results$cluster == k),
                    which(kmeans_results$cluster == k)] + 1
    }
  }
  dist_matrix <- (1 - dist_matrix / num_iter)
  return(list("cluster" = cutree(hclust(as.dist(dist_matrix)), num_clusters), 
              "dist_m" = dist_matrix))
}



# Ben: here we go too!

#Paper code
k_list = c(2, 3, 5, 10, 20)
load("entire_prim_p_dist_m.Rdata")
load("all_jointly_segmented_chg_pts.Rdata")

max_prim_dist <- max(prim_p_dist_m)
prim_p_dist_m <- prim_p_dist_m / max_prim_dist

renorm_prim_poly_list <- list()
for (i in 1:length(clean_encounters)) {
  tmp <- rel_dist_m[rel_dist_m$encounter == clean_encounters[i],]
  encounter_chg_pt_list <- all_joint_chg_pt_list[[i]]
  for (j in 1:(length(encounter_chg_pt_list) - 1)) {
    new_x <- seq(0, 1, length.out = (encounter_chg_pt_list[j + 1] - 
                                       encounter_chg_pt_list[j] + 1))
    renorm_prim_poly_list <- c(renorm_prim_poly_list, list(list(
           "x1" = fit_cubic_poly(new_x, tmp$x1[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
           "y1" = fit_cubic_poly(new_x, tmp$y1[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
           "x2" = fit_cubic_poly(new_x, tmp$x2[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]),
           "y2" = fit_cubic_poly(new_x, tmp$y2[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]]))))
  }
}

unprocessed_poly <- do.call(rbind, lapply(renorm_prim_poly_list, function(p_info) {
  c(p_info$x1$coefficients,
    p_info$y1$coefficients,
    p_info$x2$coefficients,
    p_info$y2$coefficients)
}))
unprocessed_poly[is.na(unprocessed_poly)] <- 0
unprocessed_poly_cluster_results <- lapply(k_list, function(k) {
  kmeans(unprocessed_poly, k, iter.max = 30)
})
unprocessed_poly_cluster_silhouettes <- 
  lapply(unprocessed_poly_cluster_results[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })

poly_procrustes_center_info <- lapply(1:length(k_list), function(i) {
  mean_m <- unprocessed_poly_cluster_results[[i]]$centers
  cluster_results <- unprocessed_poly_cluster_results[[i]]$cluster
  new_x <- seq(0, 1, by = 0.01)
  dist_var_m <- matrix(0, ncol = 2, nrow = k_list[i]) 
  for (j in 1:k_list[i]) {
    interpolated_center <- sapply(1:4, function(l) {
      coeff_list <- mean_m[j, (l - 1)*4 + 1:4]
      interpolated_vals <- coeff_list[1]
      for (m in 2:4) {
        interpolated_vals <- interpolated_vals + coeff_list[m] * new_x^(m - 1)
      }
      return(interpolated_vals)
    })
    p_dist <- sapply(interpolated_prim_list[cluster_results == j], function(prim_info) {
      calc_p_dist(c(interpolated_center[, 1], interpolated_center[, 2]),
                  c(interpolated_center[, 3], interpolated_center[, 4]),
                  c(prim_info[, c("x1")], prim_info[, "y1"]),
                  c(prim_info[, c("x2")], prim_info[, "y2"])) / max_prim_dist
      
    })
    dist_var_m[j,] <- c(mean(p_dist), var(p_dist))
  }
  return(dist_var_m)
})

#Processed polynomial
load("prim_p_dist_m.Rdata")
oriented_prim_m <- matrix(0, nrow = length(prim_list),
                          ncol = 4 * nrow(prim_list[[1]]))
poly_coef_m <- matrix(0, nrow = length(prim_list), ncol = 16)
for (i in 1:length(prim_list)) {
  print(i)
  oriented_prim_m <- orient_and_centered_to_first(
    prim_list[[1]][, c("x1", "y1")], prim_list[[1]][, c("x2", "y2")],
    prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")])
  for (j in 1:ncol(oriented_prim_m)) {
    poly_coef_m[i, ((j - 1) * 4 + 1):(4 * j)] <- fit_cubic_poly(seq(0, 1, by = 0.01),
                                                                oriented_prim_m[, j])$coefficients 
  }
}
processed_poly_cluster_results <- lapply(k_list, function(k) {
  kmeans(poly_coef_m, k, iter.max = 30)
})
processed_poly_cluster_silhouettes <- 
  lapply(processed_poly_cluster_results[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })

for (s_m in processed_poly_cluster_silhouettes) {
  #png( , width = 960, height = 960)
  print(ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity") + 
          theme(legend.position="none") + labs(x = "", y = ""))
}

#Processed DTW (with speed)
load("entire_prim_p_dist_m.Rdata")
load("all_jointly_segmented_chg_pts.Rdata")

num_cols <- ncol(interpolated_prim_list[[1]])
k = 0
for (i in 1:length(clean_encounters)) {
  tmp <- rel_dist_m[rel_dist_m$encounter == clean_encounters[i],]
  encounter_chg_pt_list <- all_joint_chg_pt_list[[i]]
  for (j in 1:(length(encounter_chg_pt_list) - 1)) {
    k = k + 1
    print(k)
    new_x <- seq(0, 1, length.out = (encounter_chg_pt_list[j + 1] - 
                                       encounter_chg_pt_list[j] + 1))
    proj_x <- seq(0, 1, by = 0.01)
    speed1_poly <- fit_cubic_poly(new_x, tmp$Speed_1[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]])
    speed2_poly <- fit_cubic_poly(new_x, tmp$Speed_2[encounter_chg_pt_list[j]:encounter_chg_pt_list[j + 1]])
    interpolated_prim_list[[k]] <- cbind(interpolated_prim_list[[k]],
                                         predict(speed1_poly, data.frame("x" = proj_x), 
                                                 type = "response"),
                                         predict(speed2_poly, data.frame("x" = proj_x), 
                                                 type = "response"))
    colnames(interpolated_prim_list[[k]])[num_cols + 1:2] <- 
      c("speed_1", "speed_2")
  }
}

dtw_cost_matrix_with_speed <- matrix(0, nrow = length(interpolated_prim_list),
                                     ncol = 51005)
for (i in 1:length(interpolated_prim_list)) {
  print(i)
  prim_m <- interpolated_prim_list[[i]]
  traj_cost_m <- dtw(c(prim_m$x1, prim_m$y1), c(prim_m$x2, prim_m$y2), 
                     keep.internals = T)$costMatrix
  if (max(traj_cost_m) != 0) {
    traj_cost_m <- traj_cost_m / max(traj_cost_m)
  }
  speed_cost_m <- dtw(prim_m$speed_1, prim_m$speed_2, 
                      keep.internals = T)$costMatrix
  if (max(speed_cost_m) != 0) {
    speed_cost_m <- speed_cost_m / max(speed_cost_m)
  }
  dtw_cost_matrix_with_speed[i,] <- c(as.vector(traj_cost_m), as.vector(speed_cost_m))
}

dtw_cluster_results <- lapply(k_list, function(k) {
  kmeans(dtw_cost_matrix_with_speed, k, iter.max = 30)
})
dtw_poly_cluster_silhouettes <- 
  lapply(dtw_cluster_results[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })
dtw_centers_info <-
  lapply(dtw_cluster_results, function(result) {
    center_inds <- find_closest_mds(dtw_cost_matrix_with_speed, result)
    t(sapply(1:length(center_inds), function(i) {
      c(mean(prim_p_dist_m[center_inds[i], result$cluster == i & 
                             (1:length(result$cluster) != center_inds[i])]), 
        var(prim_p_dist_m[center_inds[i], result$cluster == i & 
                            (1:length(result$cluster) != center_inds[i])]))
    }))})

#MDS
entire_prim_p_dist_mds <- cmdscale(prim_p_dist_m, k = 10)
mds_cluster_results <- lapply(k_list, function(k) {
  kmeans(entire_prim_p_dist_mds, k, iter.max = 30)
})
mds_cluster_silhouettes <- 
  lapply(mds_cluster_results[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })

mds_centers_info <- 
  lapply(mds_cluster_results, function(result) {
    center_inds <- find_closest_mds(entire_prim_p_dist_mds, result)
    center_inds <- t(sapply(1:length(center_inds), function(i) {
      c(mean(prim_p_dist_m[center_inds[i], result$cluster == i & 
                             (1:length(result$cluster) != center_inds[i])]), 
        var(prim_p_dist_m[center_inds[i], result$cluster == i & 
                            (1:length(result$cluster) != center_inds[i])]))
    }))})

#First approximation
load("prim_p_dist_m.Rdata")
oriented_prim_m <- matrix(0, nrow = length(prim_list),
                          ncol = 4 * nrow(prim_list[[1]]))
for (i in 1:length(prim_list)) {
  print(i)
  oriented_prim_m[i,] <- as.vector(orient_and_centered_to_first(
    prim_list[[1]][, c("x1", "y1")], prim_list[[1]][, c("x2", "y2")],
    prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))
}
first_approx_cluster_results <- lapply(k_list, function(k) {
  kmeans(oriented_prim_m, k, iter.max = 30)
})
first_approx_silhouettes <- 
  lapply(first_approx_cluster_results[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })
first_approx_centers_info <-
  lapply(first_approx_cluster_results, function(results) {
    mean_var_m <- matrix(0, nrow = nrow(results$center), ncol = 2)
    for (i in 1:nrow(results$center)) {
      mean_traj <- matrix(results$center[i,], ncol = 4)
      p_dist <- sapply(interpolated_prim_list[results$cluster == i], function(traj_info) {
        calc_p_dist(c(mean_traj[, 1], mean_traj[, 2]),
                    c(mean_traj[, 3], mean_traj[, 4]),
                    c(traj_info[, "x1"], traj_info[, "y1"]),
                    c(traj_info[, "x2"], traj_info[, "y2"])) / max_prim_dist
      })
      mean_var_m[i,] <- c(mean(p_dist), var(p_dist))
    }
    return(mean_var_m)
  })

#Second approximation
second_approx_cluster_result <- lapply(k_list, function(k) {procrustes_kmeans(prim_list, k)})
second_approx_silhouettes <- 
  lapply(second_approx_cluster_result[1:3], function(cluster_result) {
    get_silhouette_m_for_assignment(prim_p_dist_m, cluster_result$cluster) 
  })

second_approx_centers_info <-
  lapply(second_approx_cluster_result, function(results) {
    t(mapply(function(center, i) {
      p_dist <- sapply(interpolated_prim_list[results$cluster == i], function(traj_info) {
        calc_p_dist(c(center[, 1], center[, 2]), 
                    c(center[, 3], center[, 4]),
                    c(traj_info[, "x1"], traj_info[, "y1"]),
                    c(traj_info[, "x2"], traj_info[, "y2"]))
      }) / max_prim_dist
      c(mean(p_dist), var(p_dist))
    }, center = results$centers, i = 1:length(results$centers)))
})

make_silhouette_plots <- function(s_m) {
  print(ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + 
          geom_bar(stat="identity") + labs(x = "", y = "") + 
          coord_cartesian(ylim = c(-1, 1)) +
          theme(legend.position="none",
                axis.title.x=element_blank(), 
                axis.text.x=element_blank(),
                axis.ticks.x=element_blank(),
                axis.text.y = element_text(size = 20), 
                axis.title.y = element_text(size = 25)))
}

make_silhouette_plots_for_method <- function(silhouette_m_list, file_prefix, order = F) {
  k_list = c(2, 3, 5, 10, 20)
  k = 1
  for (s_m in silhouette_m_list) {
    if (order) {
      s_m$V1 <- reorder_cluster_results(s_m$V1, k_list[k])
      s_m <- s_m %>% arrange(V1, desc(V2))
    }
    png(paste(file_prefix, k_list[k], ".png", sep = ""), 
        width = 3000, height = 3000, res = 600)
    make_silhouette_plots(s_m)
    dev.off()
    k = k + 1
  }
  
}

make_plots <- function(cluster_results_file, data_dir, plot_dir, order = F) {
  setwd(data_dir)
  load(cluster_results_file)
  
  setwd(plot_dir)
  make_silhouette_plots_for_method(unprocessed_poly_cluster_silhouettes,
                                   "poly_cluster_", order)
  make_silhouette_plots_for_method(dtw_poly_cluster_silhouettes,
                                   "dtw_cluster_", order)
  make_silhouette_plots_for_method(mds_cluster_silhouettes,
                                   "mds_cluster_", order)
  make_silhouette_plots_for_method(mds_cluster_no_reflection_silhouettes,
                                   "mds_cluster_no_reflection_", order)
  make_silhouette_plots_for_method(mds_cluster_no_reflection_tf_silhouettes,
                                   "mds_no_reflection_tf_cluster_", order)
  make_silhouette_plots_for_method(mds_cluster_no_reflection_tf_10_silhouettes,
                                   "mds_no_reflection_tf_10_cluster_", order)
  make_silhouette_plots_for_method(first_approx_silhouettes,
                                   "first_approx_cluster_", order)
  make_silhouette_plots_for_method(first_approx_no_reflection_silhouettes,
                                   "first_approx_no_reflection_cluster_", order)
  make_silhouette_plots_for_method(second_approx_silhouettes,
                                   "second_approx_cluster_", order)
  make_silhouette_plots_for_method(second_approx_no_reflection_silhouettes,
                                   "second_approx_no_reflection_cluster_", order)
}

#40543 comparison
make_plots("40543_comparison.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/40543/p_dist/")

make_plots("40543_comparison_no_reflection.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/40543/p_dist_no_reflection/")

make_plots("40543_comparison_no_reflection_tf.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/40543/p_dist_no_reflection_tf/")

#Cubic poly comparison
make_plots("cubic_poly_comparison.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist/")

make_plots("cubic_poly_results/cubic_poly_comparison_no_reflection.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection/")
make_plots("cubic_poly_results/cubic_poly_comparison_no_reflection.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection_ordered/",
           order = T)

make_plots("cubic_poly_results/cubic_poly_sq_dist_comparison_no_reflection.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection_ordered_2/",
           order = T)

make_plots("cubic_poly_results/cubic_poly_sq_dist_all_approx_centers_corrected_comparison_no_reflection.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection_corrected_ordered/",
           order = T)

make_plots("cubic_poly_comparison_no_reflection_tf.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection_tf/")

make_plots("cubic_poly_comparison_no_reflection_tf_10.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/poly_fitting/p_dist_no_reflection_tf_10/")

#BNP comparison
make_plots("K_10_I_100_K_0.25_A_2.0_G_10.0_comparison.Rdata", 
           "~/Documents/Michigan/Research/Driving/matlab_data/", 
           "~/Documents/Michigan/Research/Driving/plots/paper_plots/silhouettes/bnp/")

reorder_based_on_silhouettes_m <- function(silhouette_m, k) {
  order_count <- rep(0, k)
  cluster_result_count <- table(silhouette_m$V1)
  order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  return(order(order_count, decreasing = T))
}

create_unprocessed_poly_cluster_mean_m_list <- function(cluster_result, k) {
  mean_m <- cluster_result$centers
  cluster_results <- cluster_result$cluster
  new_x <- seq(0, 1, by = 0.01)
  interpolated_center <- lapply(1:k, function(j) {
    tmp <- sapply(1:4, function(l) {
      coeff_list <- mean_m[j, (l - 1)*4 + 1:4]
      interpolated_vals <- coeff_list[1]
      for (m in 2:4) {
        interpolated_vals <- interpolated_vals + coeff_list[m] * new_x^(m - 1)
      }
      return(interpolated_vals)
    })
    colnames(tmp) <- c("x1", "y1", "x2", "y2")
    return(as.data.frame(tmp))
  })
}

create_first_approx_centers_m_list <- function(cluster_result) {
  mean_m <- cluster_result$centers
  mean_m_list <- lapply(1:nrow(mean_m), function(i) {
    tmp <- matrix(mean_m[i,], ncol = 4)
    colnames(tmp) <- c("x1", "y1", "x2", "y2")
    return(as.data.frame(tmp))
  })
}

create_approx_center_mean_m_list <- function(interpolated_prim_list, center_inds) {
  mean_m_list <- interpolated_prim_list[center_inds]
  for (i in 1:length(mean_m_list)) {
    mean_m_list[[i]] = mean_m_list[[i]][, c("x1", "y1", "x2", "y2")]
  }
  return(mean_m_list)
}

find_appropriate_scale_for_approx <- function(
  interpolated_prim_list, dtw_center_inds, mds_no_reflection_center_inds, 
  first_approx_center_inds, second_approx_center_inds) {
  
  all_mean_trj <- do.call(rbind, interpolated_prim_list[c(
    dtw_center_inds, mds_no_reflection_center_inds, 
    first_approx_center_inds, second_approx_center_inds
  )])
  return(c(min(all_mean_trj[, c("x1", "x2")]),
           max(all_mean_trj[, c("x1", "x2")]),
           min(all_mean_trj[, c("y1", "y2")]),
           max(all_mean_trj[, c("y1", "y2")])))
}


make_mean_plots <- function(centers_m_list, xlimits = NULL, ylimits = NULL) {
  start_plot_df <- matrix(data = 0, nrow = length(centers_m_list),
                          ncol = 5)
  for (i in 1:length(centers_m_list)) {
    centers_m_list[[i]] = cbind(centers_m_list[[i]], i)
    start_plot_df[i,] <- unlist(c(centers_m_list[[i]][1,]))
  }
  plot_df <- do.call(rbind, centers_m_list)
  start_plot_df <- as.data.frame(start_plot_df)
  colnames(start_plot_df) <- colnames(plot_df)
  g <- ggplot(data = plot_df) + 
    geom_path(aes(x = x1, y = y1, color = factor(i)), size = 2) +
    geom_path(aes(x = x2, y = y2, color = factor(i)), size = 2) +
    geom_point(data = start_plot_df,
               aes(x = x1, y = y1, color = factor(i)), size = 4) +
    geom_point(data = start_plot_df,
               aes(x = x2, y = y2, color = factor(i)), size = 4) +
    xlab("") + ylab("") + labs(color = "Cluster") +
    theme(plot.title = element_text(size = 30, face = "bold"),
          legend.text = element_text(size = 20), 
          legend.title = element_text(size = 20, face = "bold"), 
          axis.text.x = element_text(size = 20), 
          axis.text.y = element_text(size = 20),
          axis.title.x = element_text(size = 25), 
          axis.title.y = element_text(size = 25))
  if (!is.null(xlimits)) {
    g <- g + coord_cartesian(xlim = xlimits, ylim = ylimits)
  }
  print(g)
}

make_mean_plots_for_all_results <- function(
  interpolated_prim_list,cluster_results_file, data_dir, plot_dir, order = F, approx = F) {
  k_list = c(2, 3, 5, 10, 20)
  
  setwd(data_dir)
  load(cluster_results_file)
  
  if (approx) {
    appropriate_scale_list <- lapply(1:length(k_list), function(k_ind) {
      find_appropriate_scale_for_approx(
        interpolated_prim_list, dtw_cluster_results[[k_ind]]$center_index,
        mds_cluster_results_no_reflection[[k_ind]]$center_index,
        first_approx_cluster_results[[k_ind]]$center_index,
        second_approx_cluster_result[[k_ind]]$center_index)
    })
  }
  
  setwd(plot_dir)
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    if (approx) {
      interpolated_center <- create_approx_center_mean_m_list(
        interpolated_prim_list, unprocessed_poly_cluster_results[[k_ind]]$center_index
      )
    } else {
      interpolated_center <- create_unprocessed_poly_cluster_mean_m_list(
        unprocessed_poly_cluster_results[[k_ind]], k)
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          unprocessed_poly_cluster_results[[k_ind]]$cluster, k)]
    }
    png(paste("poly_cluster_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    interpolated_center <- create_approx_center_mean_m_list(
      interpolated_prim_list, dtw_cluster_results[[k_ind]]$center_index
    )
    if (approx) {
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          dtw_cluster_results[[k_ind]]$cluster, k)]
    }
    png(paste("dtw_cluster_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    interpolated_center <- create_approx_center_mean_m_list(
      interpolated_prim_list, mds_cluster_results[[k_ind]]$center_index
    )
    if (approx) {
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          mds_cluster_results[[k_ind]]$cluster, k)]
    }
    png(paste("mds_cluster_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    interpolated_center <- create_approx_center_mean_m_list(
      interpolated_prim_list, mds_cluster_results_no_reflection[[k_ind]]$center_index
    )
    if (approx) {
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          mds_cluster_results_no_reflection[[k_ind]]$cluster, k)]
    }
    png(paste("mds_cluster_no_reflection_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    interpolated_center <- create_approx_center_mean_m_list(
      interpolated_prim_list, mds_cluster_results_no_reflection_tf[[k_ind]]$center_index
    )
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          mds_cluster_results_no_reflection_tf[[k_ind]]$cluster, k)]
    }
    png(paste("mds_cluster_no_reflection_tf_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    interpolated_center <- create_approx_center_mean_m_list(
      interpolated_prim_list, mds_cluster_results_no_reflection_tf_10[[k_ind]]$center_index
    )
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          mds_cluster_results_no_reflection_tf_10[[k_ind]]$cluster, k)]
    }
    png(paste("mds_cluster_no_reflection_tf_10_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    if (approx) {
      interpolated_center <- create_approx_center_mean_m_list(
        interpolated_prim_list, first_approx_cluster_results[[k_ind]]$center_index
      )
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    } else {
      interpolated_center <- create_first_approx_centers_m_list(
        first_approx_cluster_results[[k_ind]])
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          first_approx_cluster_results[[k_ind]]$cluster, k)]
    }
    png(paste("first_approx_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    if (approx) {
      interpolated_center <- create_approx_center_mean_m_list(
        interpolated_prim_list, first_approx_no_reflection_cluster_results[[k_ind]]$center_index
      )
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    } else {
      interpolated_center <- create_first_approx_centers_m_list(
        first_approx_no_reflection_cluster_results[[k_ind]])
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          first_approx_no_reflection_cluster_results[[k_ind]]$cluster, k)]
    }
    png(paste("first_approx_no_reflection_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    if (approx) {
      interpolated_center <- create_approx_center_mean_m_list(
        interpolated_prim_list, second_approx_cluster_result[[k_ind]]$center_index
      )
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    } else {
      interpolated_center <- lapply(second_approx_cluster_result[[k_ind]]$centers, function(center_m) {
        colnames(center_m) <- c("x1", "y1", "x2", "y2")
        return(as.data.frame(center_m))
      })
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          second_approx_cluster_result[[k_ind]]$cluster, k)]
    }
    png(paste("second_approx_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    xlimits = NULL
    ylimits = NULL
    if (approx) {
      interpolated_center <- create_approx_center_mean_m_list(
        interpolated_prim_list, second_approx_no_reflection_cluster_result[[k_ind]]$center_index
      )
      xlimits <- appropriate_scale_list[[k_ind]][1:2]
      ylimits <- appropriate_scale_list[[k_ind]][3:4]
    } else {
      interpolated_center <- lapply(
        second_approx_no_reflection_cluster_result[[k_ind]]$centers, function(center_m) {
        colnames(center_m) <- c("x1", "y1", "x2", "y2")
        return(as.data.frame(center_m))
      })
    }
    if (order) {
      interpolated_center <- interpolated_center[
        reorder_based_on_cluster_results(
          second_approx_no_reflection_cluster_result[[k_ind]]$cluster, k)]
    }
    png(paste("second_approx_no_reflection_mean_plot_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_mean_plots(interpolated_center, xlimits, ylimits)
    dev.off()
  }
}

make_mean_plots_for_all_results(
  interpolated_prim_list,
  "cubic_poly_results/cubic_poly_cluster_results.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_ordered/",
  order = T)

make_mean_plots_for_all_results(
  interpolated_prim_list,
  "cubic_poly_results/cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_corrected_ordered/",
  order = T)


make_mean_plots_for_all_results(interpolated_prim_list,
                                "cubic_poly_results/cubic_poly_cluster_results_with_all_approx_centers.Rdata", 
                                "~/Documents/Michigan/Research/Driving/matlab_data/", 
                                "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_all_approx_ordered/",
                                order = T, approx = T)

create_plot_dfs_for_typical_encounters <- function(
  typical_encounters_m_list, num_encounters, orient_m) {
  start_plot_df <- matrix(
    data = 0, nrow = 
      length(typical_encounters_m_list) * num_encounters,
    ncol = 6)
  mid_plot_df <- matrix(
    data = 0, nrow = 
      length(typical_encounters_m_list) * num_encounters,
    ncol = 6)
  for (i in 1:length(typical_encounters_m_list)) {
      tmp <- typical_encounters_m_list[[i]]
      for (id in 1:num_encounters) {
        tmp[tmp$prim_name == id, c("x1", "y1", "x2", "y2")] <-
          orient_and_centered_to_first_no_reflection(
            orient_m[, 1:2], orient_m[, 3:4],
            tmp[tmp$prim_name == id, c("x1", "y1")], 
            tmp[tmp$prim_name == id, c("x2", "y2")])
      }
      typical_encounters_m_list[[i]]<- tmp
    } 
    typical_encounters_m_list[[i]] <- cbind(typical_encounters_m_list[[i]], i)
    start_plot_df[num_encounters * (i - 1) + 1:num_encounters,] <- 
      data.matrix(typical_encounters_m_list[[i]])[
        nrow(typical_encounters_m_list[[i]]) / num_encounters * (1:num_encounters - 1) + 1,]
    mid_plot_df[num_encounters * (i - 1) + 1:num_encounters,] <-
      data.matrix(typical_encounters_m_list[[i]])[
        nrow(typical_encounters_m_list[[i]]) / num_encounters * (1:num_encounters - 1) + 51,]
  }
  plot_df <- do.call(rbind, typical_encounters_m_list)
  start_plot_df <- as.data.frame(start_plot_df)
  colnames(start_plot_df) <- colnames(plot_df)
  mid_plot_df <- as.data.frame(mid_plot_df)
  colnames(mid_plot_df) <- colnames(plot_df)
  plot_df <- plot_df %>% unite(traj_id, c(i, prim_name), remove = F)
  
  return(list(plot_df, start_plot_df, mid_plot_df))
}

create_typical_encounters_plot <- function(plot_df, start_plot_df, mid_plot_df,
                                           xlimits = NULL, ylimits = NULL, 
                                           hide_legend = F) {
  g <- ggplot(data = plot_df) + 
    geom_path(aes(x = x1, y = y1, color = i, group = factor(traj_id)), 
              size = 2, alpha = 0.6) +
    geom_path(aes(x = x2, y = y2, color = i, group = factor(traj_id)), 
              size = 2, alpha = 0.6) +
    geom_point(data = start_plot_df,
               aes(x = x1, y = y1, color = i, 
                   shape = factor(prim_name)), size = 4) +
    geom_point(data = start_plot_df,
               aes(x = x2, y = y2, color = i, 
                   shape = factor(prim_name)), size = 4) +
    geom_point(data = start_plot_df,
               aes(x = x2, y = y2, 
                   shape = factor(prim_name)), size = 1.5, color = "black") +
    geom_point(data = mid_plot_df,
               aes(x = x1, y = y1, color = i, shape = factor(prim_name)), 
               size = 4) +
    geom_point(data = mid_plot_df,
               aes(x = x2, y = y2, color = i, shape = factor(prim_name)), 
               size = 4) +
    geom_point(data = mid_plot_df,
               aes(x = x1, y = y1, shape = factor(prim_name)), 
               color = "grey90", size = 1.5) +
    geom_point(data = mid_plot_df,
               aes(x = x2, y = y2, shape = factor(prim_name)), 
               color = "grey90", size = 1.5) +
    xlab("") + ylab("") + labs(color = "Cluster") + guides(shape = F) + 
    scale_colour_discrete(drop=TRUE,
                          limits = levels(plot_df$i)) +
    theme(plot.title = element_text(size = 30, face = "bold"),
          legend.text = element_text(size = 20), 
          legend.title = element_text(size = 20, face = "bold"), 
          axis.text.x = element_text(size = 20), 
          axis.text.y = element_text(size = 20),
          axis.title.x = element_blank(), 
          axis.title.y = element_blank())
  if (!is.null(xlimits)) {
    g <- g + coord_cartesian(xlim = xlimits, ylim = ylimits)
  }
  if (hide_legend) {
    g <- g + theme(
      legend.text = element_text(color = "white"),
      legend.title = element_text(color = "white"),
      legend.key = element_rect(fill = "white")
    ) + scale_color_discrete(
      drop=TRUE,
      limits = levels(plot_df$i),
      guide = guide_legend(override.aes = list(color = "white"))
    ) + scale_shape_discrete(
      guide = guide_legend(override.aes = list(shape = "white"))
    )
  }
  print(g)
}

make_typical_encounters_plots <- function(
  typical_encounters_m_list, num_encounters,
  orient_m = NULL, xlimits = NULL, ylimits = NULL, 
  k = NULL, k_list = NULL, hide_legend = F) {
  
  plot_info <- create_plot_dfs_for_typical_encounters(
    typical_encounters_m_list, num_encounters, orient_m)
  plot_df <- plot_info[[1]]
  start_plot_df <- plot_info[[2]]
  mid_plot_df <- plot_info[[3]]
  
  if (is.null(k_list)) {
    k_list <- 1:max(plot_df$i)
  }
  if (!is.null(k)) {
    plot_df$i <- k
    start_plot_df$i <- k
    mid_plot_df$i <- k
  }
  plot_df$i <- factor(plot_df$i, levels = k_list)
  start_plot_df$i <- factor(start_plot_df$i, levels = k_list)
  mid_plot_df$i <- factor(mid_plot_df$i, levels = k_list)
  
  create_typical_encounters_plot(plot_df, start_plot_df, mid_plot_df,
                                 xlimits, ylimits, hide_legend)
}

find_oriented_bounds <- function(
  interpolated_prim_list, results_list, orient_m, num_encounters) {
  
  encounter_list <- lapply(1:length(results_list[[1]]), function(i) c())
  for (i in 1:length(results_list)) {
    for (j in 1:length(results_list[[i]])) {
      encounter_list[[j]] <- 
        c(encounter_list[[j]],
          as.vector(
            results_list[[i]][[j]]$order_encounter[1:num_encounters,]))
    }
  }
  lapply(encounter_list, function(enc_list) {
    min_max_list <- sapply(enc_list, function(e_ind) {
      encounter_m <- interpolated_prim_list[[e_ind]]
      tmp <- orient_and_centered_to_first_no_reflection(
        orient_m[, 1:2], orient_m[, 3:4],
        encounter_m[, c("x1", "y1")],
        encounter_m[, c("x2", "y2")]
      )
      c(min(tmp[, c(1, 3)]),
        max(tmp[, c(1, 3)]),
        min(tmp[, c(2, 4)]),
        max(tmp[, c(2, 4)]))
    })
    return(c(min(min_max_list[1,]),
             max(min_max_list[2,]),
             min(min_max_list[3,]),
             max(min_max_list[4,])))
  })
}

create_approx_typical_encounter_m_list <- function(
  interpolated_prim_list, typical_inds_m) {
  mean_m_list <- list()
  n_obs <- nrow(typical_inds_m)
  for (i in 1:ncol(typical_inds_m)) {
    tmp <- do.call(rbind, interpolated_prim_list[typical_inds_m[, i]])
    tmp <- as.data.frame(tmp[, c("x1", "y1", "x2", "y2", "prim_name")])
    tmp$prim_name <- rep(1:n_obs, each = nrow(tmp) / n_obs)
    mean_m_list <- c(mean_m_list, list(tmp))
  }
  return(mean_m_list)
}

make_typical_encounter_plots_for_method <- function(
  cluster_result_list, orient_m, orient_bounds,
  file_prefix, interpolated_prim_list, num_encounter = 3, order = F,
  hide_legend = F) {
  k_list = c(2, 3, 5, 10, 20)
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    interpolated_encounter <- create_approx_typical_encounter_m_list(
      interpolated_prim_list, 
      cluster_result_list[[k_ind]]$order_encounter[1:num_encounter,, drop = F])
    if (order) {
      interpolated_encounter <- interpolated_encounter[
        reorder_based_on_cluster_results(cluster_result_list[[k_ind]]$cluster, k)]
    }
    png(paste(file_prefix, k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    if (!is.null(orient_bounds)) {
      make_typical_encounters_plots(interpolated_encounter, 
                                    num_encounter, orient_m,
                                    xlimits = orient_bounds[[k_ind]][1:2], 
                                    ylimits = orient_bounds[[k_ind]][3:4], 
                                    hide_legend = hide_legend)
      
    } else {
      make_typical_encounters_plots(interpolated_encounter, 
                                    num_encounter, orient_m, 
                                    hide_legend = hide_legend)
    }
    dev.off()
  }
}

make_typical_encounter_plots_for_method_and_cluster <- function(
  cluster_result, k, orient_m, file_prefix, interpolated_prim_list,
  num_encounter = 3, order = F, common_scale = F, hide_legend = F) {
  
  interpolated_encounter <- create_approx_typical_encounter_m_list(
    interpolated_prim_list, 
    cluster_result$order_encounter[1:num_encounter,, drop = F])
  if (order) {
    interpolated_encounter <- interpolated_encounter[
      reorder_based_on_cluster_results(cluster_result$cluster, k)]
  }
  if (common_scale) {
    orient_bounds <- find_oriented_bounds(interpolated_prim_list,
                                          list(list(cluster_result)),
                                          orient_m, num_encounter)
    orient_bounds <- orient_bounds[[1]]
  }
  for (i in 1:k) {
    png(paste(file_prefix, k, "_cluster_", i, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    if (common_scale) {
      make_typical_encounters_plots(interpolated_encounter[i], 
                                    num_encounter, orient_m,
                                    xlimits = orient_bounds[1:2], 
                                    ylimits = orient_bounds[3:4],
                                    k = i, k_list = 1:k, hide_legend)
      
    } else {
      make_typical_encounters_plots(interpolated_encounter[i], 
                                    num_encounter, orient_m, 
                                    k = i, k_list = 1:k, hide_legend = hide_legend)
    }
    dev.off()
  }
}

create_orient_m <- function() {
  t_vector <- seq(0, 1, by = 0.01)
  orient_interaction <- matrix(0, nrow = 101, ncol = 4)
  orient_interaction[, 1] <- t_vector
  orient_interaction[, 2] <- 2 * t_vector - 1
  orient_interaction[, 3] <- -t_vector
  orient_interaction[, 4] <- 1 - 2 * t_vector
  return(orient_interaction)
}

make_typical_encounter_plots_for_all_results <- function(
  interpolated_prim_list, cluster_results_file, data_dir, plot_dir, 
  num_encounters = 3, order = F) {
  
  setwd(data_dir)
  load(cluster_results_file)
  orient_interaction <- create_orient_m()

  orient_bounds <- find_oriented_bounds(
    interpolated_prim_list, 
    list(
         #unprocessed_poly_cluster_results,
         dtw_cluster_results, 
         mds_cluster_results_no_reflection,
         first_approx_no_reflection_cluster_results,
         second_approx_no_reflection_cluster_result), 
    orient_interaction, num_encounters)
  
  
  setwd(plot_dir)
  make_typical_encounter_plots_for_method(
    unprocessed_poly_cluster_results, orient_interaction, NULL,
    paste("poly_cluster_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  make_typical_encounter_plots_for_method(
    dtw_cluster_results, orient_interaction, orient_bounds,
    paste("dtw_cluster_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  # make_typical_encounter_plots_for_method(
  #   mds_cluster_results, orient_interaction,
  #   "mds_cluster_typical_encounters_3_", 3, order)
  make_typical_encounter_plots_for_method(
    mds_cluster_results_no_reflection, orient_interaction, orient_bounds,
    paste("mds_cluster_no_reflection_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  # make_typical_encounter_plots_for_method(
  #   mds_cluster_results_no_reflection_tf, orient_interaction,
  #   "mds_cluster_no_reflection_tf_typical_encounters_3_", 3, order)
  # make_typical_encounter_plots_for_method(
  #   mds_cluster_results_no_reflection_tf_10, orient_interaction,
  #   "mds_cluster_no_reflection_tf_10_typical_encounters_3_", 3, order)
  make_typical_encounter_plots_for_method(
    first_approx_cluster_results, orient_interaction, NULL,
    paste("first_approx_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  make_typical_encounter_plots_for_method(
    first_approx_no_reflection_cluster_results, orient_interaction, orient_bounds,
    "first_approx_no_reflection_no_legend_typical_encounters_3_", 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  make_typical_encounter_plots_for_method(
    second_approx_cluster_results, orient_interaction, NULL,
    paste("second_approx_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
  make_typical_encounter_plots_for_method(
    second_approx_no_reflection_cluster_result, orient_interaction, orient_bounds,
    paste("second_approx_no_reflection_no_legend_typical_encounters_3_", num_encounters, "_", sep = ""), 
    interpolated_prim_list, num_encounters, order, hide_legend = T)
}

make_typical_encounter_plots_for_all_results(
  interpolated_prim_list, 
  "cubic_poly_results/cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_corrected_typical_examples/",
  order = T)

make_typical_encounter_plots_for_all_results(
  interpolated_prim_list, 
  "cubic_poly_results/cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_all_approx_corrected_ordered/",
  order = T, num_encounters = 1)

make_typical_encounter_plots_for_method_and_cluster(
  mds_cluster_results_no_reflection[[3]], 5, create_orient_m(), 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_corrected_typical_examples/mds_cluster_no_reflection_typical_encounters_3_ind_scale_", 
  interpolated_prim_list, order = T, common_scale = F)

make_typical_encounter_plots_for_method_and_cluster(
  mds_cluster_results_no_reflection[[3]], 5, create_orient_m(), 
  "~/Documents/Michigan/Research/Driving/plots/paper_plots/mean_plots/poly_fitting/p_dist_no_reflection_corrected_typical_examples/mds_cluster_no_reflection_typical_encounters_3_ind_scale_no_legend_", 
  interpolated_prim_list, order = T, common_scale = F, 
  hide_legend = T)


make_freq_plot <- function(plot_df, ylimits = NULL, hide_legend = F) {
  g <- ggplot(plot_df, aes(x = V1, y = ..count.., 
                           weight = weight_col, color = factor(i),
                           linetype = factor(i))) + 
    geom_freqpoly(size = 2) + 
    coord_cartesian(xlim = c(0, 1), ylim = ylimits) +
    xlab("") + ylab("") + labs(color = "Cluster", linetype = "Cluster") +
    theme(plot.title = element_text(size = 30, face = "bold"),
          legend.text = element_text(size = 20), 
          legend.title = element_text(size = 20, face = "bold"), 
          axis.text.x = element_text(size = 20), 
          axis.text.y = element_text(size = 20),
          axis.title.x = element_blank(), 
          axis.title.y = element_blank())
  if (hide_legend) {
    g <- g + theme(
      legend.text = element_text(color = "white"),
      legend.title = element_text(color = "white"),
      legend.key = element_rect(fill = "white")
    ) + scale_color_discrete(
      guide = guide_legend(override.aes = list(color = "white"))
    ) + scale_linetype_discrete(
      guide = guide_legend(override.aes = list(color = "white"))
    )
  }
  print(g)
}

reorder_based_on_cluster_results <- function(cluster_results, K) {
  cluster_counts <- rep(0, K)
  results_cluster_counts <- table(cluster_results)
  cluster_counts[as.integer(names(results_cluster_counts))] <- results_cluster_counts
  count_order <- order(cluster_counts, decreasing = T)
}

make_approx_dist_freq_plots <- function(cluster_result, prim_p_dist_m, 
                                        k, plot_dir, file_prefix, 
                                        order = T, all_ylimits = NULL,
                                        cluster_ylimits = NULL,
                                        hide_legend = F) {
    center_inds <- cluster_result$center_index
    cluster_assignment <- cluster_result$cluster
    if (order) {
      center_inds <- center_inds[reorder_based_on_cluster_results(cluster_result$cluster, k)]
      cluster_assignment <- reorder_cluster_results(cluster_assignment, k)
    }
    
    all_approx_plot_df <- do.call(rbind, lapply(1:k, function(i) {
      cbind(prim_p_dist_m[center_inds[i], 1:length(cluster_assignment) != center_inds[i]], 
            i, 1 / (length(cluster_assignment - 1)))
    }))
    all_approx_plot_df <- as.data.frame(all_approx_plot_df)
    colnames(all_approx_plot_df) <- c("V1", "i", "weight_col")
    png(paste(plot_dir, file_prefix, "_all_approx_dist_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_freq_plot(all_approx_plot_df, all_ylimits, hide_legend)
    dev.off()
    
    cluster_approx_plot_df <- do.call(rbind, 
      lapply(1:k, function(i) {
               cbind(prim_p_dist_m[center_inds[i],
                                   cluster_assignment == i & 
                                   (1:length(cluster_assignment) != center_inds[i])], 
                     i, 1 / (sum(cluster_assignment == i) - 1))
             }))
    cluster_approx_plot_df <- as.data.frame(cluster_approx_plot_df)
    colnames(cluster_approx_plot_df) <- c("V1", "i", "weight_col")
    png(paste(plot_dir, file_prefix, "_cluster_approx_dist_", k, ".png", sep = ""), width = 3000, height = 3000, res = 600)
    make_freq_plot(cluster_approx_plot_df, cluster_ylimits, hide_legend)
    dev.off()
}

calc_p_dist_no_reflection_between_center_and_encounters <- function(
  interpolated_center, prim_info, max_prim_dist) {
  sapply(prim_info, function(info) {
    calc_p_dist_no_reflection(
      interpolated_center[, 1:2], interpolated_center[, 3:4],
      info[, c("x1", "y1")], info[, c("x2", "y2")]) /
      max_prim_dist
  })
}

#Assumes center_m_list is properly ordered
make_mean_dist_freq_plots <- function(cluster_result, center_m_list,
                                      interpolated_prim_list,
                                      max_prim_dist, k, 
                                      plot_dir, file_prefix,
                                      order = T, all_ylimits = NULL,
                                      cluster_ylimits = NULL,
                                      hide_legend = F) {
  if (order) {
    center_m_list <- center_m_list[reorder_based_on_cluster_results(cluster_result$cluster, k)]
    cluster_result$cluster <- reorder_cluster_results(cluster_result$cluster, k)
  }
  all_mean_plot_info <- lapply(1:k, function(i) {
    center_m <- data.matrix(center_m_list[[i]])
    cbind(calc_p_dist_no_reflection_between_center_and_encounters(
      center_m, interpolated_prim_list, max_prim_dist
    ), i)})
  all_mean_plot_df <- do.call(rbind, all_mean_plot_info)
  all_mean_plot_df <- as.data.frame(all_mean_plot_df)
  all_mean_plot_df$weight_col <- 1 / length(cluster_result$cluster) 
  colnames(all_mean_plot_df) <- c("V1", "i", "weight_col")
  png(paste(plot_dir, file_prefix, "_all_mean_dist_", k, ".png", sep = ""),
      width = 3000, height = 3000, res = 600)
  make_freq_plot(all_mean_plot_df, all_ylimits, hide_legend)
  dev.off()
  
  cluster_mean_plot_df <- do.call(rbind, 
    lapply(1:k, function(i) {
      cbind(all_mean_plot_info[[i]][cluster_result$cluster == i,], 
        1 / sum(cluster_result$cluster == i))
    }))
  cluster_mean_plot_df <- as.data.frame(cluster_mean_plot_df)
  colnames(cluster_mean_plot_df) <- c("V1", "i", "weight_col")
  png(paste(plot_dir, file_prefix, "_cluster_mean_dist_", k, ".png", sep = ""),
      width = 3000, height = 3000, res = 600)
  make_freq_plot(cluster_mean_plot_df, cluster_ylimits, hide_legend)
  dev.off()
}

make_dist_freq_plots_for_k <- function(cluster_results_file, data_dir, 
                                       plot_dir, interpolated_prim_list, 
                                       prim_p_dist_m, max_prim_dist, 
                                       k, k_ind, order = T,
                                       all_ylimits = NULL,
                                       cluster_ylimits = NULL) {
  load(paste(data_dir, cluster_results_file, sep = ""))
  
  cluster_result <- unprocessed_poly_cluster_results[[k_ind]]
  centers_m_list <- create_unprocessed_poly_cluster_mean_m_list(cluster_result, k)
  make_mean_dist_freq_plots(cluster_result, centers_m_list, 
                            interpolated_prim_list, max_prim_dist, k,
                            plot_dir, "poly_cluster", 
                            order, all_ylimits = all_ylimits,
                            cluster_ylimits = cluster_ylimits)
  make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                              k, plot_dir, "poly_cluster", 
                              order, all_ylimits = all_ylimits,
                              cluster_ylimits = cluster_ylimits)
  
  cluster_result <- first_approx_no_reflection_cluster_results[[k_ind]]
  centers_m_list <- create_first_approx_centers_m_list(cluster_result)
  make_mean_dist_freq_plots(cluster_result, centers_m_list, 
                            interpolated_prim_list, max_prim_dist, k,
                            plot_dir, "first_approx_no_reflection", 
                            order, all_ylimits = all_ylimits,
                            cluster_ylimits = cluster_ylimits)
  make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                              k, plot_dir, "first_approx_no_reflection", 
                              order, all_ylimits = all_ylimits,
                              cluster_ylimits = cluster_ylimits)
  
  cluster_result <- second_approx_no_reflection_cluster_result[[k_ind]]
  make_mean_dist_freq_plots(cluster_result, cluster_result$centers, 
                            interpolated_prim_list, max_prim_dist, k,
                            plot_dir, "second_approx_no_reflection", 
                            order, all_ylimits = all_ylimits,
                            cluster_ylimits = cluster_ylimits)
  make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                              k, plot_dir, "second_approx_no_reflection", 
                              order, all_ylimits = all_ylimits,
                              cluster_ylimits = cluster_ylimits)
}

make_dist_freq_plots <- function(cluster_results_file, data_dir, 
                                 plot_dir, interpolated_prim_list, 
                                 prim_p_dist_m, max_prim_dist, order = T) {
  load(paste(data_dir, cluster_results_file, sep = ""))
  k_list <- c(2, 3, 5, 10, 20)
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    cluster_result <- unprocessed_poly_cluster_results[[k_ind]]
    centers_m_list <- create_unprocessed_poly_cluster_mean_m_list(cluster_result, k)
    make_mean_dist_freq_plots(cluster_result, centers_m_list, 
                              interpolated_prim_list, max_prim_dist, k,
                              plot_dir, "poly_cluster", order)
    make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                                k, plot_dir, "poly_cluster", order)
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    cluster_result <- first_approx_cluster_results[[k_ind]]
    centers_m_list <- create_first_approx_centers_m_list(cluster_result)
    make_mean_dist_freq_plots(cluster_result, centers_m_list, 
                              interpolated_prim_list, max_prim_dist, k,
                              plot_dir, "first_approx", order)
    make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                                k, plot_dir, "first_approx", order)
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    cluster_result <- first_approx_no_reflection_cluster_results[[k_ind]]
    centers_m_list <- create_first_approx_centers_m_list(cluster_result)
    make_mean_dist_freq_plots(cluster_result, centers_m_list, 
                              interpolated_prim_list, max_prim_dist, k,
                              plot_dir, "first_approx", order)
    make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                                k, plot_dir, "first_approx", order)
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    cluster_result <- second_approx_cluster_result[[k_ind]]
    make_mean_dist_freq_plots(cluster_result, cluster_result$centers, 
                              interpolated_prim_list, max_prim_dist, k,
                              plot_dir, "second_approx", order, hide_legend = T)
    make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                                k, plot_dir, "second_approx", order, hide_legend = T)
  }
  
  for (k_ind in 1:length(k_list)) {
    k = k_list[k_ind]
    cluster_result <- second_approx_no_reflection_cluster_result[[k_ind]]
    make_mean_dist_freq_plots(cluster_result, cluster_result$centers, 
                              interpolated_prim_list, max_prim_dist, k,
                              plot_dir, "second_approx_no_reflection", order, hide_legend = T)
    make_approx_dist_freq_plots(cluster_result, prim_p_dist_m, 
                                k, plot_dir, "second_approx_no_reflection", order, hide_legend = T)
  }
}

setwd("~/Documents/Michigan/Research/Driving/matlab_data/")
load("cubic_poly_results/cubic_poly_cluster_results_with_all_approx_centers.Rdata")
load("cubic_poly_results/cubic_poly_entire_p_dist_m_no_reflection.Rdata")
load("cubic_poly_results/cubic_poly_interpolated_prim.Rdata")
max_prim_p_dist <- max(prim_p_dist_m)
prim_p_dist_m <- prim_p_dist_m / max_prim_p_dist

make_dist_freq_plots("cubic_poly_cluster_results_with_all_approx_centers.Rdata", 
                     "~/Documents/Michigan/Research/Driving/matlab_data/cubic_poly_results/", 
                     "~/Documents/Michigan/Research/Driving/plots/paper_plots/freq_plots/poly_fitting/p_dist_no_reflection_ordered/", 
                     interpolated_prim_list, 
                     prim_p_dist_m, max_prim_p_dist, T)

make_dist_freq_plots("cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata", 
                     "~/Documents/Michigan/Research/Driving/matlab_data/cubic_poly_results/", 
                     "~/Documents/Michigan/Research/Driving/plots/paper_plots/freq_plots/poly_fitting/p_dist_no_reflection_corrected_ordered/", 
                     interpolated_prim_list, 
                     prim_p_dist_m, max_prim_p_dist, T)

make_dist_freq_plots_for_k("cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata", 
                           "~/Documents/Michigan/Research/Driving/matlab_data/cubic_poly_results/", 
                           "~/Documents/Michigan/Research/Driving/plots/paper_plots/freq_plots/poly_fitting/p_dist_no_reflection_corrected_ordered/", 
                           interpolated_prim_list, prim_p_dist_m, max_prim_p_dist, 5, 3, T, 
                           c(0, 0.3), c(0, 0.8))

#Preprocess results
create_cell_text <- function(val_m) {
  lapply(1:nrow(val_m), function(i) {
    row <- sprintf("%0.2e", val_m[i,]) 
    paste("\\multirow{2}{*}{\\shortstack[c]{", 
          row[1], " (", row[2], ")\\\\", 
          row[3], " (", row[4], ")}}", sep = "")
  })
}

print_tables <- function(cluster_results_file, data_dir, plot_dir) {
  k_list = c(2, 3, 5, 10, 20)
  #k_list <- c(5)
  
  setwd(data_dir)
  load(cluster_results_file)
  
  poly_procrustes_center_info <- lapply(poly_procrustes_center_info, function(result) {
    result[which(is.na(result[, 2])),] <- 0 
    result
  })
  dtw_centers_info <- lapply(dtw_centers_info, function(result) {
    result[which(is.na(result[, 2])),] <- 0
    result
  })
  # mds_centers_info <- lapply(mds_centers_info, function(result) {
  #   result[which(is.na(result[, 2])),] <- 0 
  #   result
  # })
  mds_centers_no_reflection_info <- lapply(mds_centers_no_reflection_info, function(result) {
    result[which(is.na(result[, 2])),] <- 0
    result
  })
  # mds_centers_no_reflection_tf_info <- lapply(mds_centers_no_reflection_tf_info, function(result) {
  #   result[which(is.na(result[, 2])),] <- 0 
  #   result
  # })
  
  first_approx_centers_info <- lapply(first_approx_centers_info, function(result) {
    result[which(is.na(result[, 2])),] <- 0 
    result
  })
  second_approx_centers_info <- lapply(second_approx_centers_info, function(result) {
    result[which(is.na(result[, 2])),] <- 0 
    result
  })
  
  # summary_table <-
  #   mapply(function(a, b, c, d, e, f, g) {
  #     do.call(rbind, list(create_cell_text(a),
  #                         create_cell_text(b),
  #                         create_cell_text(c),
  #                         create_cell_text(d),
  #                         create_cell_text(e),
  #                         create_cell_text(f),
  #                         create_cell_text(g)))},
  #        a = poly_procrustes_center_info, b = dtw_centers_info,
  #        c = mds_centers_info, d = mds_centers_no_reflection_info,
  #        e = mds_centers_no_reflection_tf_info,
  #        f = first_approx_centers_info, g = second_approx_centers_info)
  summary_table <-
    mapply(function(a, b, d, f, g) {
      do.call(rbind, list(create_cell_text(a[, 1:4]),
                          create_cell_text(b[, 1:4]),
                          #create_cell_text(c[, 1:4]),
                          create_cell_text(d[, 1:4]),
                          #create_cell_text(e),
                          create_cell_text(f[, 1:4]),
                          create_cell_text(g[, 1:4])))},
      a = poly_procrustes_center_info, 
      b = dtw_centers_info,
      # c = mds_centers_info, 
      d = mds_centers_no_reflection_info,
      #e = mds_centers_no_reflection_tf_info,
      f = first_approx_centers_info, g = second_approx_centers_info)
  
  print(summary_table)
  # for (st in summary_table) {
  #   # rownames(st) <- c("Spline", "DTW", "MDS", "MDS No Refl", "MDS No Refl Tf", 
  #   #                   "First Approx", "Second Approx")
  #   rownames(st) <- c("Spline", "DTW", "MDS", "First Approx", "Second Approx")
  #   colnames(st) <- sapply(1:ncol(st), function(i) {paste("Cluster", i, sep = " ")})
  #   stargazer(st)
  # }
}

print_sq_dist_ordered_tables_for_k <- function(cluster_results_file, data_dir, 
                                               k, k_ind) {
  #k_list <- c(5)
  
  setwd(data_dir)
  load(cluster_results_file)
  
  # print("Unprocessed cluster")
  # order_count <- rep(0, k)
  # cluster_result_count <- table(unprocessed_poly_cluster_silhouettes[[k_ind]]$V1)
  # order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  # print(paste(paste(
  #   create_cell_text(poly_procrustes_center_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
  #             collapse = "&"), "\\\\", sep = ""))
  # print(sum(poly_procrustes_center_info[[k_ind]][, 5]))
  # 
  # print("DTW")
  # order_count <- rep(0, k)
  # cluster_result_count <- table(dtw_poly_cluster_silhouettes[[k_ind]]$V1)
  # order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  # print(paste(paste(
  #   create_cell_text(
  #     dtw_centers_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
  #                   collapse = "&"), "\\\\", sep = ""))
  # print(sum(dtw_centers_info[[k_ind]][, 5]))
  # 
  # 
  # print("MDS")
  # order_count <- rep(0, k)
  # cluster_result_count <- table(mds_cluster_no_reflection_silhouettes[[k_ind]]$V1)
  # order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  # print(paste(paste(
  #   create_cell_text(
  #     mds_centers_no_reflection_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
  #                   collapse = "&"), "\\\\", sep = ""))
  # print(sum(mds_centers_no_reflection_info[[k_ind]][, 5]))
  # # mds_centers_no_reflection_tf_info <- lapply(mds_centers_no_reflection_tf_info, function(result) {
  # #   result[which(is.na(result[, 2])),] <- 0 
  # #   result
  # # })
  
  print("First approx")
  order_count <- rep(0, k)
  cluster_result_count <- table(first_approx_silhouettes[[k_ind]]$V1)
  order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  print(paste(paste(
    create_cell_text(
      first_approx_centers_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
                    collapse = "&"), "\\\\", sep = ""))
  print(sum(first_approx_centers_info[[k_ind]][, 5]))
  
  print("First approx no reflection")
  order_count <- rep(0, k)
  cluster_result_count <- table(first_approx_no_reflection_silhouettes[[k_ind]]$V1)
  order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  print(paste(paste(
    create_cell_text(
      first_approx_no_reflection_centers_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
    collapse = "&"), "\\\\", sep = ""))
  print(sum(first_approx_no_reflection_centers_info[[k_ind]][, 5]))
  
  print("Second approx")
  order_count <- rep(0, k)
  cluster_result_count <- table(second_approx_silhouettes[[k_ind]]$V1)
  order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  print(paste(paste(
    create_cell_text(
      second_approx_centers_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
                    collapse = "&"), "\\\\", sep = ""))
  print(sum(second_approx_centers_info[[k_ind]][, 5]))
  
  print("Second approx no reflection")
  order_count <- rep(0, k)
  cluster_result_count <- table(second_approx_no_reflection_silhouettes[[k_ind]]$V1)
  order_count[as.integer(names(cluster_result_count))] <- cluster_result_count
  print(paste(paste(
    create_cell_text(
      second_approx_no_reflection_centers_info[[k_ind]][order(order_count, decreasing = T), 1:4]), 
    collapse = "&"), "\\\\", sep = ""))
  print(sum(second_approx_no_reflection_centers_info[[k_ind]][, 5]))
  
}

print_tables("cubic_poly_comparison.Rdata", 
             "~/Documents/Michigan/Research/Driving/matlab_data/")

print_tables("cubic_poly_comparison_no_reflection.Rdata", 
             "~/Documents/Michigan/Research/Driving/matlab_data/")

print_tables("cubic_poly_comparison_no_reflection_tf.Rdata", 
             "~/Documents/Michigan/Research/Driving/matlab_data/")

print_sq_dist_ordered_tables_for_k(
  "./cubic_poly_results/cubic_poly_sq_dist_comparison_no_reflection.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 5, 3)

print_sq_dist_ordered_tables_for_k(
  "./cubic_poly_results/cubic_poly_sq_dist_all_approx_centers_comparison_no_reflection.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 5, 3)

print_sq_dist_ordered_tables_for_k(
  "cubic_poly_results/cubic_poly_update_geom_approx_centers_info.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 5, 3)

print_sq_dist_ordered_tables_for_k(
  "cubic_poly_results/cubic_poly_update_geom_approx_all_approx_centers_info.Rdata", 
  "~/Documents/Michigan/Research/Driving/matlab_data/", 5, 3)


#Comparison of centers
test_kmeans <- kmeans(entire_prim_p_dist_mds, centers = 2, iter.max = 30)

find_closest_mds <- function(prim_mds, kmeans_results) {
  k = max(kmeans_results$cluster)
  closest_to_center <- rep(0, k)
  min_l2_dist <- rep(Inf, k)
  for (i in 1:nrow(prim_mds)) {
    cluster_i = kmeans_results$cluster[i]
    l2_dist = norm(prim_mds[i,] - (kmeans_results$centers)[cluster_i,], type = "2")
    if (l2_dist < min_l2_dist[cluster_i]) {
      closest_to_center[cluster_i] = i
      min_l2_dist[cluster_i] = l2_dist
    }
  }
  return(closest_to_center)
}



closest_mds <- find_closest_mds(entire_prim_p_dist_mds, prim_mds_cv)
closest_centers_df <- do.call(rbind, interpolated_prim_list[closest_mds])
ggplot() + geom_path(data = closest_centers_df, aes(x = x1, y = y1, color = "Traj 1")) + 
  geom_path(data = closest_centers_df, aes(x = x2, y = y2, color = "Traj 2")) + 
  geom_point(data = closest_centers_df[101 * (0:6) + 1,], aes(x = x1, y = y1, color = "Traj 1")) +
  geom_point(data = closest_centers_df[101 * (0:6) + 1,], aes(x = x2, y = y2, color = "Traj 2")) +
  facet_wrap(~prim_name, scales = "free")


#Prep other data sets
df <- read.csv('38028_updated.csv')
df <- df[-which(is.na(df$ID)),]
interpolated_prim_list <- list()
for (encounter in unique(df$ID)) {
  cat(encounter, "\n")
  tmp <- df[df$ID == encounter,];
  if (nrow(tmp) == 0) {
    next
  }
  tmp$norm_t <- (tmp$Time - min(tmp$Time)) / max(tmp$Time - min(tmp$Time))
  tmp$x1 <- (tmp$LO_1 - tmp$LO_1[1]) * lond
  tmp$y1 <- (tmp$LA_1 - tmp$LA_1[1]) * latd
  tmp$x2 <- (tmp$LO_2 - tmp$LO_1[1]) * lond
  tmp$y2 <- (tmp$LA_2 - tmp$LA_1[1]) * latd
  interpolated_prim_list  <- c(interpolated_prim_list,
                               chg_pt_interpolate_prims(tmp$norm_t, 
                                                        list("x1" = tmp$x1, "y1" = tmp$y1, 
                                                             "x2" = tmp$x2, "y2" = tmp$y2), 
                                                        c(1, sum(df$ID == encounter)),
                                                        encounter))
}
save(interpolated_prim_list, file = "38028_interpolated_prim.Rdata")

df <- read.csv("30295_updated.csv")
df <- df[-which(is.na(df$ID)),]
interpolated_prim_list <- list()
for (encounter in unique(df$ID)) {
  cat(encounter, "\n")
  tmp <- df[df$ID == encounter, , drop = F];
  if (nrow(tmp) == 0) {
    next
  }
  tmp$norm_t <- (tmp$Time - min(tmp$Time)) / max(tmp$Time - min(tmp$Time))
  tmp$x1 <- (tmp$LO_1 - tmp$LO_1[1]) * lond
  tmp$y1 <- (tmp$LA_1 - tmp$LA_1[1]) * latd
  tmp$x2 <- (tmp$LO_2 - tmp$LO_1[1]) * lond
  tmp$y2 <- (tmp$LA_2 - tmp$LA_1[1]) * latd
  if (nrow(tmp) == 1) {
    encounter_m <- as.data.frame(
      matrix(c(tmp$x1, tmp$y1, tmp$x2, tmp$y2), 
             nrow = 101, ncol = 4, byrow = T))
    colnames(encounter_m) <- c("x1", "y1", "x2", "y2")
    encounter_m$prim_name = paste(encounter, 1, sep = "_")
    encounter_m$norm_t = seq(0, 1, by = 0.01)
    interpolated_prim_list  <- c(interpolated_prim_list, 
                                 list(encounter_m))
    next
  }
  interpolated_prim_list  <- c(interpolated_prim_list,
                               chg_pt_interpolate_prims(tmp$norm_t, 
                                                        list("x1" = tmp$x1, "y1" = tmp$y1, 
                                                             "x2" = tmp$x2, "y2" = tmp$y2), 
                                                        c(1, sum(df$ID == encounter)),
                                                        encounter))
}
save(interpolated_prim_list, file = "30295_interpolated_prim.Rdata")

df <- read.csv("40543_updated.csv")
df <- df[-which(is.na(df$ID)),]
interpolated_prim_list <- list()
for (encounter in unique(df$ID)) {
  cat(encounter, "\n")
  tmp <- df[df$ID == encounter,];
  if (nrow(tmp) == 0) {
    next
  }
  tmp$norm_t <- (tmp$Time - min(tmp$Time)) / max(tmp$Time - min(tmp$Time))
  tmp$x1 <- (tmp$LO_1 - tmp$LO_1[1]) * lond
  tmp$y1 <- (tmp$LA_1 - tmp$LA_1[1]) * latd
  tmp$x2 <- (tmp$LO_2 - tmp$LO_1[1]) * lond
  tmp$y2 <- (tmp$LA_2 - tmp$LA_1[1]) * latd
  interpolated_prim_list  <- c(interpolated_prim_list,
                               chg_pt_interpolate_prims(tmp$norm_t, 
                                                        list("x1" = tmp$x1, "y1" = tmp$y1, 
                                                             "x2" = tmp$x2, "y2" = tmp$y2), 
                                                        c(1, sum(df$ID == encounter)),
                                                        encounter))
}
save(interpolated_prim_list, file = "40543_interpolated_prim.Rdata")


aug_df <- do.call(rbind, lapply(unique(df$ID), function(encounter) {
  cat(encounter, "\n")
  tmp <- df[df$ID == encounter,];
  if (nrow(tmp) == 0) {
    next
  }
  tmp$norm_t <- (tmp$Time - min(tmp$Time)) / max(tmp$Time - min(tmp$Time))
  tmp$x1 <- (tmp$LO_1 - tmp$LO_1[1]) * lond
  tmp$y1 <- (tmp$LA_1 - tmp$LA_1[1]) * latd
  tmp$x2 <- (tmp$LO_2 - tmp$LO_1[1]) * lond
  tmp$y2 <- (tmp$LA_2 - tmp$LA_1[1]) * latd
  tmp
}))

#Poly list
renorm_prim_poly_list <- list()
for (encounter in unique(aug_df$ID)) {
  tmp <- aug_df[aug_df$ID == encounter,]
  new_x <- seq(0, 1, length.out = nrow(tmp))
  renorm_prim_poly_list <- c(renorm_prim_poly_list, list(list(
    "x1" = fit_cubic_poly(new_x, tmp$x1),
    "y1" = fit_cubic_poly(new_x, tmp$y1),
    "x2" = fit_cubic_poly(new_x, tmp$x2),
    "y2" = fit_cubic_poly(new_x, tmp$y2))))
}

#DTW
num_cols <- ncol(interpolated_prim_list[[1]])
for (i in 1:length(interpolated_prim_list)) {
  tmp <- aug_df[aug_df$ID == unique(aug_df$ID)[i],]
  new_x <- seq(0, 1, length.out = nrow(tmp))
  proj_x <- seq(0, 1, by = 0.01)
  speed1_poly <- fit_cubic_poly(new_x, tmp$Speed_1)
  speed2_poly <- fit_cubic_poly(new_x, tmp$Speed_2)
  interpolated_prim_list[[i]] <- cbind(interpolated_prim_list[[i]],
                                         predict(speed1_poly, data.frame("x" = proj_x), 
                                                 type = "response"),
                                         predict(speed2_poly, data.frame("x" = proj_x), 
                                                 type = "response"))
  colnames(interpolated_prim_list[[i]])[num_cols + 1:2] <- 
      c("speed_1", "speed_2")
}

make_stability_plots <- function(stability_summary, file_prefix, k_list, 
                                 subset = F) {
  titles <- c("Mean Within Distance", "Variance Within Distance",
              "Mean Without Distance", "Variance Without Distance")
  for (k in k_list) {
    png(paste(file_prefix, k, ".png", sep = ""), height = 1096, width = 1096)
    do.call(grid.arrange, lapply(1:length(titles), function(j) {
      plot_info <- t(stability_summary[[k - 1]][,j,])
      plot_df <- melt(plot_info)
      plot_df$Var1 <- 2:20
      if (subset) {
        plot_df <- plot_df[plot_df$Var2 %in% c(1, median(1:ncol(plot_info)),
                                               ncol(plot_info)),]
      }
      ggplot(plot_df, aes(x = Var1, y = value, color = as.factor(Var2))) + 
        geom_path(size = 5) + xlab("MDS dim") + ylab("") + 
        guides(color = guide_legend(title="Cluster")) + ggtitle(titles[j]) +
        theme(plot.title = element_text(size = 30, face = "bold"),
              legend.text = element_text(size = 20), 
              legend.title = element_text(size = 20, face = "bold"), 
              axis.text.x = element_text(size = 20), 
              axis.text.y = element_text(size = 20),
              axis.title.x = element_text(size = 25), 
              axis.title.y = element_text(size = 25))
    }))
    dev.off()
  }
}

make_stability_heatmap <- function(stability_summary, order = 1, sum = F) {
  titles <- c("Mean Within Distance", "Variance Within Distance",
              "Mean Without Distance", "Variance Without Distance")
  do.call(grid.arrange, lapply(1:length(titles), function(j) {
    if (sum) {
      plot_info <- sapply(stability_summary, function(info_m) {colSums(info_m[, j,])})
    } else {
      plot_info <- sapply(stability_summary, function(info_m) {info_m[order, j,]})
    }
    plot_info_df <- melt(plot_info)
    colnames(plot_info_df) <- c("d", "k", "value")
    plot_info_df[, c(1,2)] = plot_info_df[, c(1,2)] + 1
    ggplot(plot_info_df, aes(x = d, y = k, fill = value)) + 
      geom_raster(interpolate = T) + 
      scale_fill_gradient(low = "blue", high = "red") + 
      guides(fill = guide_colorbar()) + 
      ggtitle(titles[j]) +
      theme(plot.title = element_text(size = 30, face = "bold"),
            legend.text = element_text(size = 20), 
            legend.title = element_text(size = 20), 
            axis.text.x = element_text(size = 20), 
            axis.text.y = element_text(size = 20),
            axis.title.x = element_text(size = 25), 
            axis.title.y = element_text(size = 25))
  }))
}

k_list <- c(2, 3, 5, 10, 20)
setwd("~/Documents/Michigan/Research/Driving/matlab_data/")
load("cubic_poly_stability_comparison_no_reflection_tf.Rdata")
setwd("~/Documents/Michigan/Research/Driving/plots/paper_plots/stability_plots/")
make_stability_plots(stability_summary, "mds_no_reflection_tf_stability_plots_cluster", k_list)

setwd("~/Documents/Michigan/Research/Driving/matlab_data/")
load("cubic_poly_stability_comparison_no_reflection_tf_10.Rdata")
setwd("~/Documents/Michigan/Research/Driving/plots/paper_plots/stability_plots/")
make_stability_plots(stability_summary, "mds_no_reflection_tf_10_stability_plots_cluster", k_list)

load("cubic_poly_stability_all_comparison_no_reflection_tf.Rdata")
make_stability_heatmap(stability_summary, sum = T)

for (i in 3:5) {
  load(paste("cubic_poly_stability_all_comparison_no_reflection_tf_", i,".Rdata", sep = ""))
  png(paste("mds_no_reflection_tf_all_stability_plots_sum", i, ".png", sep = ""))
  make_stability_heatmap(stability_summary, sum = T)
  dev.off()
}

load("K_10_A_2.0_G_stability_results_2.Rdata")
prim_p_dist_files <- dir("BNP_entire_prim_m/", full.names = T)
prim_p_dist_files <- prim_p_dist_files[
  order(as.numeric(sub(".*G_(.*?)_.*", "\\1", basename(prim_p_dist_files))))]
interested_files <- which(as.numeric(sub(".*A_(.*?)_.*", "\\1",
                                         basename(prim_p_dist_files))) == 2)
prim_p_dist_files <- prim_p_dist_files[interested_files]
prim_p_dist_m_list <- lapply(prim_p_dist_files, function(filename) {
  load(filename) 
  max_prim_dist <- max(prim_p_dist_m)
  prim_p_dist_m / max_prim_dist
})

create_stability_stat_m <- function(stability_result, prim_p_dist_m, list = F) {
  num_k = length(stability_result)
  num_d = length(stability_result[[1]])
  stability_stat_plot_m <- matrix(0, nrow = num_k * num_d, ncol = 3)
  for (i in 1:num_k) {
    result_cluster <- stability_result[[i]]
    for (j in 1:num_d) {
      if (list) {
        prim_p_dist_m = prim_p_dist_m_list[[j]]
      }
      n = nrow(prim_p_dist_m)
      result <- result_cluster[[j]]
      stability_stat = 0
      for (k in 1:max(result$cluster)) {
        stability_stat = stability_stat + 
          sum(prim_p_dist_m[result$cluster == k, result$cluster == k]^2)
      }
      # print(c(i, j))
      stability_stat_plot_m[(i - 1) * num_d + j,] <-
        c(i + 1, j + 1, stability_stat / (2 * n))
    }
  }
  stability_stat_plot_m <- as.data.frame(stability_stat_plot_m)
  colnames(stability_stat_plot_m) <- c("k", "d", "value")
  return(stability_stat_plot_m)
}

create_stability_stat_change_m <- function(statbility_stat_plot_m, num_k, num_d) {
  stability_change_stat_plot_m <- stability_stat_plot_m
  for (i in 1:nrow(stability_change_stat_plot_m)) {
    # print(i)
    row <- stability_stat_plot_m[i,]
    neighbors_m <- matrix(unlist(row[-3]), nrow = 4, ncol = 2, byrow = T)
    neighbors_m[1,1] <- neighbors_m[1,1] - 1
    neighbors_m[2,1] <- neighbors_m[2,1] + 1
    neighbors_m[3,2] <- neighbors_m[3,2] - 1
    neighbors_m[4,2] <- neighbors_m[4,2] + 1
    neighbors_m <- neighbors_m[neighbors_m[,1] >= 2 & 
                                 neighbors_m[,1] <= num_k + 1 &
                                 neighbors_m[,2] >= 2 & 
                                 neighbors_m[,2] <= num_d + 1,]
    abs_change = 0
    for (j in 1:nrow(neighbors_m)) {
      abs_change = abs_change +
        abs(stability_stat_plot_m$value[
          stability_stat_plot_m$k == neighbors_m[j, 1] &
            stability_stat_plot_m$d == neighbors_m[j, 2]] -
            row[[3]])
    }
    stability_change_stat_plot_m$value[i] = abs_change / nrow(neighbors_m) 
  }
  return(stability_change_stat_plot_m)
}

make_stability_stat_plot <- function(stability_stat_plot_m, xlabel, low_color, high_color) {
  print(ggplot(stability_stat_plot_m, aes(x = d, y = k, fill = value)) +
          geom_raster(interpolate = T) + xlab(xlabel) +
          scale_fill_gradient(low = low_color, high = high_color) +
          guides(fill = guide_colorbar()) + 
          theme(legend.text = element_text(size = 20), 
                legend.title=element_blank(), 
                axis.text.x = element_text(size = 20), 
                axis.text.y = element_text(size = 20),
                axis.title.x = element_text(size = 25), 
                axis.title.y = element_text(size = 25)))
}

make_common_stability_stat_plot <- function(stability_stat_plot_m, xlabel, min_k_val, max_k_val) {
  print(ggplot(stability_stat_plot_m, 
               aes(x = d, y = k, fill = value)) +
          geom_raster(interpolate = T) + xlab(xlabel) + 
          scale_fill_gradient2(low = "yellow", mid = "blue", high = "dark blue", 
                               midpoint = max_k_val, limits = c(min_k_val, NA)) +
          guides(fill = guide_colorbar()) + 
          theme(legend.text = element_text(size = 20), 
                legend.title=element_blank(), 
                axis.text.x = element_text(size = 20), 
                axis.text.y = element_text(size = 20),
                axis.title.x = element_text(size = 25), 
                axis.title.y = element_text(size = 25)))
}

make_all_stability_stat_plots <- function(stability_stat_plot_m, xlabel, file_prefix) {
  png(paste(file_prefix, "all_stability_plots_stability_stat.png", sep = "_"),
      width = 3000, height = 3000, res = 600)
  make_stability_stat_plot(stability_stat_plot_m, xlabel, "yellow", "red")
  dev.off()
  
  png(paste(file_prefix, "all_stability_plots_stability_stat_k_gt_10.png", sep = "_"),
      width = 3000, height = 3000, res = 600)
  make_stability_stat_plot(stability_stat_plot_m[stability_stat_plot_m$k >= 10,], 
                           xlabel, "yellow", "blue")
  dev.off()
  
  png(paste(file_prefix, "all_stability_plots_stability_stat_k_lt_10.png", sep = "_"),
      width = 3000, height = 3000, res = 600)
  make_stability_stat_plot(stability_stat_plot_m[stability_stat_plot_m$k <= 10,], 
                           xlabel, "blue", "red")
  dev.off()
}

make_stability_stat_change_plots <- function(
  stability_change_stat_plot_m, xlabel, low_color, high_color) {
  
  print(ggplot(stability_change_stat_plot_m, 
         aes(x = d, y = k, fill = value)) +
    xlab(xlabel) +
    geom_raster(interpolate = T) +
    scale_fill_gradient(low = low_color, high = high_color) +
    guides(fill = guide_colorbar()) + 
    theme(legend.text = element_text(size = 20), 
          legend.title=element_blank(), 
          axis.text.x = element_text(size = 20), 
          axis.text.y = element_text(size = 20),
          axis.title.x = element_text(size = 25), 
          axis.title.y = element_text(size = 25)))
}

make_all_stability_stat_change_plots <- function(
  stability_change_stat_plot_m, xlabel, file_prefix) {
  
  png(paste(file_prefix, "stability_change_no_reflection_plot.png", sep = "_"), 
      width = 3000, height = 3000, res = 600)
  make_stability_stat_change_plots(stability_change_stat_plot_m, xlabel, "yellow", "red")
  dev.off()
  
  png(paste(file_prefix, "stability_change_no_reflection_plot_k_lt_10.png", sep = "_"), 
      width = 3000, height = 3000, res = 600)
  make_stability_stat_change_plots(
    stability_change_stat_plot_m[stability_change_stat_plot_m$k <= 10,], 
    xlabel, "blue", "red")
  dev.off()
  
  png(paste(file_prefix, "stability_change_no_reflection_plot_k_gt_10.png", sep = "_"), 
      width = 3000, height = 3000, res = 600)
  make_stability_stat_change_plots(
    stability_change_stat_plot_m[stability_change_stat_plot_m$k >= 10,], 
    xlabel, "yellow", "blue")
  dev.off()
}

#MDS
load("cubic_poly_results/cubic_poly_entire_p_dist_m_no_reflection.Rdata")
load("cubic_poly_results/cubic_poly_stability_all_cluster_results_25_restarts.Rdata")
#prim_p_dist_m <- do.call(pmax, prim_p_dist_m_list)
#rm(prim_p_dist_m_list)
max_prim_dist <- max(prim_p_dist_m)
prim_p_dist_m <- prim_p_dist_m / max_prim_dist
stability_stat_plot_m <- 
  create_stability_stat_m(stability_result, prim_p_dist_m)
make_all_stability_stat_plots(stability_stat_plot_m, expression(beta), "mds_no_reflection")
stability_change_stat_plot_m <-
  create_stability_stat_change_m(stability_stat_plot_m, 
                                 length(stability_result), length(stability_result[[1]]))
make_all_stability_stat_change_plots(
  stability_change_stat_plot_m, expression(beta), "mds_no_reflection")
save(stability_stat_plot_m, stability_change_stat_plot_m,
     file = "cubic_poly_results/cubic_poly_mds_stability_stat_m.Rdata")

#BNP

bnp_prim_file_list <- c()
prim_p_dist_m_list <- lapply(bnp_prim_file_list, function(file) {
  load(file)
  prim_p_dist_m / max(prim_p_dist_m)
})
stability_stat_plot_m <- 
  create_stability_stat_plot_m(comparison_info, prim_p_dist_m_list, list = T)
make_all_stability_stat_plots(stability_stat_plot_m, expression(gamma), "bnp_no_reflection")
stability_change_stat_plot_m <-
  create_stability_stat_change_m(stability_stat_plot_m)
make_all_stability_stat_change_plots(
  stability_change_stat_plot_m, expression(gamma), "bnp_no_reflection")
save(stability_stat_plot_m, stability_change_stat_plot_m,
     file = "bnp_stability_stat_m.Rdata")

#Comparison
load("K_10_A_2.0_G_comparison_results_2.Rdata")
load("cubic_poly_results/cubic_poly_entire_p_dist_m_no_reflection.Rdata")
max_prim_dist <- max(prim_p_dist_m)
prim_p_dist_m <- prim_p_dist_m / max_prim_dist
stability_stat_plot_m <- 
  create_stability_stat_plot_m(comparison_info, prim_p_dist_m)
make_all_stability_stat_plots(stability_stat_plot_m, expression(gamma), "comparison_bnp_no_reflection")
stability_change_stat_plot_m <-
  create_stability_stat_change_m(stability_stat_plot_m)
make_all_stability_stat_change_plots(
  stability_change_stat_plot_m, expression(gamma), "comparison_bnp_no_reflection")
save(stability_stat_plot_m, stability_change_stat_plot_m, 
     file = "comparison_bnp_stability_stat_m.Rdata")

#Make simultaneous plot
load("cubic_poly_results/cubic_poly_mds_stability_stat_m.Rdata")
min_k_val = min(stability_stat_plot_m[stability_stat_plot_m$k >= 10,]$value)
max_k_val = max(stability_stat_plot_m[stability_stat_plot_m$k >= 10,]$value)

load("bnp_stability_stat_m.Rdata")
min_k_val = min(min_k_val, min(
  stability_stat_plot_m[stability_stat_plot_m$k >= 10,]$value))
max_k_val = max(max_k_val, max(
  stability_stat_plot_m[stability_stat_plot_m$k >= 10,]$value))

load("cubic_poly_results/cubic_poly_mds_stability_stat_m.Rdata")
png("mds_no_reflection_all_stability_plots_stability_stat_k_gt_10_comparison.png",
    width = 3000, height = 3000, res = 600)
make_common_stability_stat_plot(
  stability_stat_plot_m[stability_stat_plot_m$k >= 10,], expression(beta),
  min_k_val, max_k_val)
dev.off()

load("bnp_stability_stat_m.Rdata")
png("bnp_no_reflection_all_stability_plots_stability_stat_k_gt_10_comparison.png",
    width = 3000, height = 3000, res = 600)
make_common_stability_stat_plot(
  stability_stat_plot_m[stability_stat_plot_m$k >= 10,], expression(gamma),
  min_k_val, max_k_val)
dev.off()

load("comparison_bnp_stability_stat_m.Rdata")
png("comparison_bnp_no_reflection_all_stability_plots_stability_stat_k_gt_10_comparison.png",
    width = 3000, height = 3000, res = 600)
make_common_stability_stat_plot(
  stability_stat_plot_m[stability_stat_plot_m$k >= 10,], expression(gamma),
  min_k_val, max_k_val)
dev.off()


save(stability_stat_plot_m, stability_change_stat_plot_m, 
     file = "comparison_bnp_stability_stat_m.Rdata")

load("comparison_bnp_stability_stat_m.Rdata")

k = 5
cluster_prop <- t(sapply(stability_result[[4]], function(result) {
    cluster_prop <- table(result$cluster) / length(result$cluster)
    sort(cluster_prop, decreasing = T)[c(1, 3, 5)]
}))
colnames(cluster_prop) <- c(1, 3, 5)
plot_df <- melt(cluster_prop)
colnames(plot_df) <- c("d", "Order", "value")
png("subset_cluster_proportion_5_2.png")
ggplot(plot_df, aes(x = d, y = value, color = factor(Order))) + geom_path(size = 5) + 
  xlab("MDS dim") + ylab("") + 
  guides(color = guide_legend(title="Cluster")) +
  theme(legend.text = element_text(size = 20), 
        legend.title = element_text(size = 20), 
        axis.text.x = element_text(size = 20), 
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 25), 
        axis.title.y = element_text(size = 25))
dev.off()

load("cubic_poly_stability_all_comparison_no_reflection_tf_25_restarts.Rdata")
png("mds_no_reflection_tf_all_stability_plots_sum_25_restarts.png")
make_stability_heatmap(stability_summary, sum = T)
dev.off()

load("cluster_size_ordered_cubic_poly_stability_all_comparison_no_reflection_tf_25_restarts.Rdata")

png("cluster_size_ordered_subset_stability_plot_5.png")
make_stability_plots(stability_summary, "cluster_size_ordered_subset_stability_plot", 5, subset = T)
dev.off()

#Other work
df <- read.csv("new_input.csv")
df$LO_1 <- df$host_long
df$LA_1 <- df$host_lat
df$LA_2 <- df$lat
df$LO_2 <- df$long
df$ID <- df$id
df <- do.call(rbind, lapply(unique(df$id), function(id) {
  tmp <- df[df$id == id,]
  tmp$Time <- seq(0, (nrow(tmp) - 1) * 0.1, by = 0.1)
  tmp
}))
latd <- 111321   # meters / degree latitude
latitude <- mean(df$LA_1)  # average latitude of dataset
beta <- atan(0.99664719 * tan(latitude * pi/180))
lond <- pi/180 * 6378137 * cos(beta)  # meters / degree longitude

for (k in 1:19) {
  print(k)
  s_m <- get_silhouette_m_for_assignment(prim_p_dist_m, stability_result[[k]][[9]]$cluster) 
  png(paste("new_cubic_poly_silhouette_plots_", k + 1, ".png", sep = ""), width = 3000, height = 3000, res = 600)
  print(ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity") + 
          theme(legend.position="none") + labs(x = "", y = ""))
  dev.off()
}

prim_info_m <- do.call(rbind, interpolated_prim_list[-c(5111, 5130, 4249)])
clean_encounters <- unique(prim_info_m$prim_name)
prim_info_m$encounter <- as.factor(prim_info_m$prim_name)
png("new_cubic_poly_clusters_8.png")
make_group_plots(prim_info_m, stability_result[[7]][[9]]$cluster, clean_encounters)
dev.off()

png("new_cubic_poly_clusters_10.png")
make_group_plots(prim_info_m, stability_result[[9]][[9]]$cluster, clean_encounters)
dev.off()

load("cubic_poly_stability_all_cluster_results_25_restarts_no_reflection.Rdata")
load("cubic_poly_results/cubic_poly_entire_p_dist_m_no_reflection.Rdata")

#Evaluate BNP results

create_needed_files <- function(df, file_prefix) {
  latd <- 111321   # meters / degree latitude
  latitude <- 42.28  # average latitude of dataset
  beta <- atan(0.99664719 * tan(latitude * pi/180))
  lond <- pi/180 * 6378137 * cos(beta)  # meters / degree longitude

  aug_df <- do.call(rbind, lapply(unique(df$ID), function(encounter) {
   cat(encounter, "\n")
   tmp <- df[df$ID == encounter,];
   new_tmp <- tmp;
   if (nrow(tmp) == 0) {
     next
   }
   # tmp$norm_t <- (tmp$Time - min(tmp$Time)) / max(tmp$Time - min(tmp$Time))
   # tmp$centered_t <- (tmp$Time - min(tmp$Time))
   #tmp <- tmp %>% arrange(centered_t)
   new_tmp$x1 <- (tmp$x1 - tmp$x1[1]) * lond
   new_tmp$y1 <- (tmp$y1 - tmp$y1[1]) * latd
   new_tmp$x2 <- (tmp$x2 - tmp$x2[1]) * lond
   new_tmp$y2 <- (tmp$y2 - tmp$y2[1]) * latd
   new_tmp$centered_t <- 0.1 * 1:nrow(tmp) - 0.1
   new_tmp
  }))

  prim_info_list <- list()
  for (id in unique(aug_df$ID)) {
   print(id)
   tmp <- aug_df[aug_df$ID == id,];
   prim_list <- list(
     "x1" = fit_cubic_poly(tmp$centered_t, tmp$x1),
     "x2" = fit_cubic_poly(tmp$centered_t, tmp$x2),
     "y1" = fit_cubic_poly(tmp$centered_t, tmp$y1),
     "y2" = fit_cubic_poly(tmp$centered_t, tmp$y2),
     "length" = max(tmp$centered_t),
     "start" = min(tmp$centered_t),
     "end" = max(tmp$centered_t) - min(tmp$centered_t),
     "encounter" = id
   )
   prim_info_list <- c(prim_info_list, list(prim_list))
  }
  data_save_file = paste(file_prefix,
                        "prim_poly_info.Rdata", sep = "_")
  save(aug_df, prim_info_list, file = data_save_file)
  
  interpolated_prim_list <- list()
  for (id in unique(aug_df$ID)) {
    tmp <- aug_df[aug_df$ID == id,];
    if (nrow(tmp) < 4) {
      next
    }
    interpolated_prim_list  <- c(interpolated_prim_list,
                                 chg_pt_interpolate_prims(tmp$centered_t, 
                                                          list("x1" = tmp$x1, "y1" = tmp$y1, 
                                                               "x2" = tmp$x2, "y2" = tmp$y2), 
                                                          c(1, nrow(tmp)),
                                                          id))
  }
  save(aug_df, interpolated_prim_list, 
       file = paste(file_prefix, "interpolated_prim_list.Rdata", sep = "_"))
  return(interpolated_prim_list)
}

source_python("~/Documents/Michigan/Research/Driving/convert_to_prim_m.py")
prim_list <- dir("~/Downloads/primitives_from_BNP/", full.names = T)

for (prim_filename in prim_list) {
  prim_file_prefix <- sub("(.*)\\..*", replacement = "\\1", basename(prim_filename))
  prim_m <- convert_prim_list_to_prim_m(prim_filename)
  prim_m <- as.data.frame(prim_m)
  colnames(prim_m) <- c("x1", "y1", "x2", "y2", "ID")
  interpolated_prim_list <- create_needed_files(prim_m, prim_file_prefix)
  prim_p_dist_m <- get_entire_p_dist_m(interpolated_prim_list, F)
  save(prim_p_dist_m, 
       file = paste(prim_file_prefix, 
                    "entire_p_dist_m_no_reflection.Rdata", sep = "_"))
    
}

load("K_10_A_2.0_G_comparison_cluster_silhouettes_2.Rdata")
#load("K_10_A_2.0_G_cluster_silhouettes_2.Rdata")
k_list <- c(2, 3, 5, 10, 20) - 1
for (k in k_list) {
  print(k)
  s_m <- bnp_cluster_silhouettes[[k]][[9]]
  png(paste("comparison_bnp_silhouette_plots_", k + 1, "_g_10.png", sep = ""), width = 3000, height = 3000, res = 600)
  print(ggplot(s_m, aes(x = seq_along(V2), y = V2, color = factor(V1))) + geom_bar(stat="identity") + 
          theme(legend.position="none") + labs(x = "", y = ""))
  dev.off()
}

#Update clusters
reorder_cluster_results <- function(cluster_results, K) {
  cluster_counts <- rep(0, K)
  results_cluster_counts <- table(cluster_results)
  cluster_counts[as.integer(names(results_cluster_counts))] <- results_cluster_counts
  count_order <- order(cluster_counts, decreasing = T)
  for (k in 1:K) {
    cluster_results[cluster_results == count_order[k]] = K + k    
  }
  return(cluster_results - K)
}

calc_approx_sq_dist_stats_for_results <- function(cluster_results, prim_p_dist_m) {
  lapply(cluster_results, function(result) {
    center_inds <- result$center_index
    t(sapply(1:length(center_inds), function(i) {
      c(mean(prim_p_dist_m[center_inds[i], result$cluster == i &
                             (1:length(result$cluster) != center_inds[i])]^2),
        var(prim_p_dist_m[center_inds[i], result$cluster == i &
                            (1:length(result$cluster) != center_inds[i])]^2),
        mean(prim_p_dist_m[center_inds[i], result$cluster != i]^2),
        var(prim_p_dist_m[center_inds[i], result$cluster != i]^2),
        sum(prim_p_dist_m[center_inds[i], result$cluster == i]^2))
    }))})
}


setwd("~/Documents/Michigan/Research/Driving/matlab_data/")
load("cubic_poly_results/cubic_poly_entire_p_dist_m_no_reflection.Rdata")
load("cubic_poly_results/cubic_poly_cluster_results.Rdata")
load("cubic_poly_results/cubic_poly_sq_dist_comparison_no_reflection.Rdata")
load("cubic_poly_results/cubic_poly_interpolated_prim.Rdata")
load("cubic_poly_results/cubic_poly_prim_poly_info.Rdata")

create_unprocessed_poly_data_m <- function(aug_df) {
  renorm_prim_poly_list <- list()
  for (encounter in unique(aug_df$ID)) {
    tmp <- aug_df[aug_df$ID == encounter,]
    new_x <- seq(0, 1, length.out = nrow(tmp))
    renorm_prim_poly_list <- c(renorm_prim_poly_list, list(list(
      "x1" = fit_cubic_poly(new_x, tmp$x1),
      "y1" = fit_cubic_poly(new_x, tmp$y1),
      "x2" = fit_cubic_poly(new_x, tmp$x2),
      "y2" = fit_cubic_poly(new_x, tmp$y2))))
  }
  unprocessed_poly <- do.call(rbind, lapply(renorm_prim_poly_list, function(p_info) {
    c(p_info$x1$coefficients,
      p_info$y1$coefficients,
      p_info$x2$coefficients,
      p_info$y2$coefficients)
  }))
} 

create_dtw_data_m <- function(interpolated_prim_list, aug_df) {
  num_cols <- ncol(interpolated_prim_list[[1]])
  for (i in 1:length(interpolated_prim_list)) {
   tmp <- aug_df[aug_df$ID == unique(aug_df$ID)[i],]
   new_x <- seq(0, 1, length.out = nrow(tmp))
   proj_x <- seq(0, 1, by = 0.01)
   speed1_poly <- fit_cubic_poly(new_x, tmp$Speed_1)
   speed2_poly <- fit_cubic_poly(new_x, tmp$Speed_2)
   interpolated_prim_list[[i]] <- cbind(interpolated_prim_list[[i]],
                                        predict(speed1_poly, data.frame("x" = proj_x),
                                                type = "response"),
                                        predict(speed2_poly, data.frame("x" = proj_x),
                                                type = "response"))
   colnames(interpolated_prim_list[[i]])[num_cols + 1:2] <-
     c("speed_1", "speed_2")
  }

  dtw_cost_matrix_with_speed <- matrix(0, nrow = length(interpolated_prim_list),
                                      ncol = 51005)
  for (i in 1:length(interpolated_prim_list)) {
   prim_m <- interpolated_prim_list[[i]]
   traj_cost_m <- dtw(c(prim_m$x1, prim_m$y1), c(prim_m$x2, prim_m$y2),
                      keep.internals = T)$costMatrix
   if (max(traj_cost_m) != 0) {
     traj_cost_m <- traj_cost_m / max(traj_cost_m)
   }
   speed_cost_m <- dtw(prim_m$speed_1, prim_m$speed_2,
                       keep.internals = T)$costMatrix
   if (max(speed_cost_m) != 0) {
     speed_cost_m <- speed_cost_m / max(speed_cost_m)
   }
   dtw_cost_matrix_with_speed[i,] <- c(as.vector(traj_cost_m), as.vector(speed_cost_m))
  }
  return(dtw_cost_matrix_with_speed)
}

update_cluster_results_with_ordered_encounter <- function(data_m, cluster_results) {
  for (i in 1:length(cluster_results)) {
    cluster_results[[i]]$order_encounter <- 
      apply(cluster_results[[i]]$centers, 1, function(center) {
        order(apply(data_m, 1, function(row) {
          norm(row - center, type = "2")
        }))
      })
  }
  return(cluster_results)
} 

max_prim_dist <- max(prim_p_dist_m)
prim_p_dist_m <- prim_p_dist_m / max(prim_p_dist_m)

#Polynomial
unprocessed_poly <- create_unprocessed_poly_data_m(aug_df)
unprocessed_poly_cluster_results <- 
  update_cluster_results_with_ordered_encounter(
    unprocessed_poly, unprocessed_poly_cluster_results)

poly_procrustes_center_info <- calc_approx_sq_dist_stats_for_results(
  unprocessed_poly_cluster_results, prim_p_dist_m
)

#DTW
dtw_cost_matrix_with_speed <- create_dtw_data_m(interpolated_prim_list, aug_df)
dtw_cluster_results <- 
  update_cluster_results_with_ordered_encounter(
    dtw_cost_matrix_with_speed, dtw_cluster_results)

#MDS
entire_prim_p_dist_mds <- cmdscale(prim_p_dist_m, k = 10)
mds_cluster_results_no_reflection <- 
  update_cluster_results_with_ordered_encounter(
    entire_prim_p_dist_mds, mds_cluster_results_no_reflection
  )


prim_list <- interpolated_prim_list
#Find closest index for first approximation
oriented_prim_m <- matrix(0, nrow = length(prim_list),
                          ncol = 4 * nrow(prim_list[[1]]))
for (i in 1:length(prim_list)) {
  oriented_prim_m[i,] <- as.vector(orient_and_centered_to_first(
    prim_list[[1]][, c("x1", "y1")], prim_list[[1]][, c("x2", "y2")],
    prim_list[[i]][, c("x1", "y1")], prim_list[[i]][, c("x2", "y2")]))
}
first_approx_cluster_results <-
  update_cluster_results_with_ordered_encounter(
    oriented_prim_m, first_approx_cluster_results
  )
for (i in 1:length(first_approx_cluster_results)) {
  # first_approx_cluster_results[[i]]$center_index <- 
  #   apply(first_approx_cluster_results[[i]]$centers, 1, function(center) {
  #     which.min(apply(oriented_prim_m, 1, function(row) {
  #       norm(row - center, type = "2")
  #     }))
  #   })
  first_approx_cluster_results[[i]]$ordered_encounter <- 
    apply(first_approx_cluster_results[[i]]$centers, 1, function(center) {
      order(apply(oriented_prim_m, 1, function(row) {
        norm(row - center, type = "2")
      }))
    })
}
first_approx_centers_info <- calc_approx_sq_dist_stats_for_results(
  first_approx_cluster_results, prim_p_dist_m
)

#Find closest index for second approximation
for (i in 1:length(second_approx_cluster_result)) {
  # second_approx_cluster_result[[i]]$center_index <- 
  #   sapply(second_approx_cluster_result[[i]]$centers, function(center) {
  #     which.min(sapply(prim_list, function(prim_m) {
  #       oriented_prim_interaction_m <- orient_and_centered_to_first(
  #         center[, 1:2], center[, 3:4], 
  #         prim_m[, c("x1", "y1")], prim_m[, c("x1", "y1")])
  #       mean((center - oriented_prim_interaction_m)^2)
  #     }))
  #   })
  
  second_approx_cluster_result[[i]]$ordered_encounter <- 
    sapply(second_approx_cluster_result[[i]]$centers, function(center) {
      order(sapply(prim_list, function(prim_m) {
        oriented_prim_interaction_m <- orient_and_centered_to_first(
          center[, 1:2], center[, 3:4], 
          prim_m[, c("x1", "y1")], prim_m[, c("x1", "y1")])
        mean((center - oriented_prim_interaction_m)^2)
      }))
    })
}
second_approx_centers_info <- calc_approx_sq_dist_stats_for_results(
  second_approx_cluster_result, prim_p_dist_m
)

save(unprocessed_poly_cluster_results,
     dtw_cluster_results,
     mds_cluster_results,
     mds_cluster_results_no_reflection,
     mds_cluster_results_no_reflection_tf, 
     first_approx_cluster_results,
     second_approx_cluster_result, 
     file = "cubic_poly_results/cubic_poly_cluster_results_with_all_approx_centers.Rdata")

save(unprocessed_poly_cluster_silhouettes, poly_procrustes_center_info,
     dtw_poly_cluster_silhouettes, dtw_centers_info,
     mds_cluster_silhouettes, mds_centers_info,
     mds_cluster_no_reflection_silhouettes, mds_centers_no_reflection_info,
     mds_cluster_no_reflection_tf_silhouettes,
     mds_centers_no_reflection_tf_info,
     mds_cluster_no_reflection_tf_10_silhouettes,
     mds_centers_no_reflection_tf_10_info,
     first_approx_silhouettes, first_approx_centers_info,
     second_approx_silhouettes, second_approx_centers_info,
     file = "cubic_poly_results/cubic_poly_sq_dist_all_approx_centers_comparison_no_reflection.Rdata")

load("cubic_poly_results/cubic_poly_sq_dist_all_approx_centers_comparison_no_reflection.Rdata")
first_approx_mean_dist_5 <- lapply(1:5, function(k) {
    center_m <- matrix(first_approx_cluster_results[[3]]$centers[k,], ncol = 4)
    calc_p_dist_no_reflection_between_center_and_encounters(
      center_m, interpolated_prim_list[
        first_approx_cluster_results[[3]]$cluster == k
      ], max(prim_p_dist_m))
  })

first_approx_dist_list <- lapply(1:5, function(k) {
  calc_p_dist_no_reflection_between_center_and_encounters(
    matrix(first_approx_cluster_results[[k]]$centers[k,], ncol = 4), interpolated_prim_list,
    max_prim_dist)
})


test_t <- seq(0, 1, by = 0.01)
sq_exp_kernel <- 0.05^2 * exp(-1 / (2 * 0.2^2) * as.matrix(dist(test_t))^2) + 
  diag(rep(1e-9, length(test_t)))

make_encounter_plot <- function(encounter_m) {
  plot(encounter_m[,1], encounter_m[,2], 
       type = "l", 
       xlim = c(min(encounter_m[, c(1,3)]) - 0.5, max(encounter_m[, c(1,3)]) + 0.5), 
       ylim = c(min(encounter_m[, c(2,4)]) - 0.5, max(encounter_m[, c(2,4)]) + 0.5))
  lines(encounter_m[,3], encounter_m[,4])
  points(rbind(encounter_m[1, 1:2], encounter_m[1, 3:4]))
}

make_encounter_ggplot <- function(encounter_m_list, color_list = c("red", "blue"),
                                  xlim = NULL, ylim = NULL, 
                                  centered = F, annotate_text = NULL) {
  g <- ggplot()
  for (i in 1:length(encounter_m_list)) {
    encounter_m = encounter_m_list[[i]]
    if (centered) {
      encounter_colnames <- colnames(encounter_m)
      tmp <- rbind(data.matrix(encounter_m[, c(1:2)]),
                   data.matrix(encounter_m[, c(3:4)]))
      tmp <- apply(tmp, 2, function(col) {
        col - mean(col)
      })
      encounter_m <- cbind(tmp[1:nrow(encounter_m),],
                           tmp[1:nrow(encounter_m) + nrow(encounter_m),])
      colnames(encounter_m) <- encounter_colnames
    }
    plot_df <- as.data.frame(encounter_m)
    g <- g + geom_path(data = plot_df,
      aes_string(x = colnames(plot_df)[1], 
                 y = colnames(plot_df)[2]), size = 2, 
      color = color_list[i], linetype = i) +
      geom_path(data = plot_df,
        aes_string(x = colnames(plot_df)[3], 
                   y = colnames(plot_df)[4]), size = 2, 
        color = color_list[i], linetype = i) +
      geom_point(data = plot_df, x = plot_df[1,1], y = plot_df[1,2], 
                 size = 5, color = color_list[i], shape = 5) +
      geom_point(data = plot_df, x = plot_df[1,3], y = plot_df[1,4], 
                 size = 5, color = color_list[i], shape = 5) +
      geom_point(data = plot_df, x = plot_df[nrow(encounter_m),1], 
                 y = plot_df[nrow(encounter_m),2], 
                 size = 5, color = color_list[i], shape = 8) +
      geom_point(data = plot_df, x = plot_df[nrow(encounter_m),3], 
                 y = plot_df[nrow(encounter_m),4], 
                 size = 5, color = color_list[i], shape = 8)
  }
  if (!is.null(xlim) && !is.null(ylim)) {
    g <- g + coord_cartesian(xlim = xlim, y = ylim)
  }
  if (!is.null(annotate_text)) {
    g <- g + geom_label(data = plot_df, x = Inf, y = Inf, hjust = 1, vjust = 1,
                        label = annotate_text, size = 7.5, fontface = "bold")
    # g <- g + geom_label(data = plot_df, x = annotate_x, y = annotate_y, 
    #                      label = annotate_text, size = 5, fontface = "bold")
  }
  g <- g + xlab("Miles") + ylab("Miles") +
    annotation_north_arrow(which_north = "grid", location = "br", 
                           style = north_arrow_fancy_orienteering) + 
  theme(legend.text = element_text(size = 20), 
        legend.title = element_text(size = 20), 
        axis.text = element_text(size = 20),
        axis.title = element_text(size = 20))
  print(g)
}


test_interaction <- matrix(0, nrow = 101, ncol = 4)
test_interaction[, 1] <- test_t
test_interaction[, 2] <- 2 * test_t - 1
test_interaction[, 3] <- -test_t
test_interaction[, 4] <- 1 - 2 * test_t
plot(test_interaction[,1], test_interaction[,2], 
     type = "l", xlim = c(-2, 2), ylim = c(-1, 1))
lines(test_interaction[,3], test_interaction[,4])
points(rbind(test_interaction[1, 1:2], test_interaction[1, 3:4]))

test_interaction_2 <- matrix(0, nrow = 101, ncol = 4)
test_interaction_2[, 1] <- 1 - test_t
test_interaction_2[, 2] <- 1 - 2 * test_t
test_interaction_2[, 3] <- test_t - 1
test_interaction_2[, 4] <- 2 * test_t - 1
plot(test_interaction_2[,1], test_interaction_2[,2], 
     type = "l", xlim = c(-2, 2), ylim = c(-1, 1))
lines(test_interaction_2[,3], test_interaction_2[,4])
points(rbind(test_interaction_2[1, 1:2], test_interaction_2[1, 3:4]))

test_interaction_3 <- matrix(0, nrow = 101, ncol = 4)
test_interaction_3[, 1] <- cos(test_t * -(pi / 2) + pi) + 1
test_interaction_3[, 2] <- sin(test_t * -(pi / 2) + pi)
test_interaction_3[, 3] <- cos(test_t * -(pi / 2) + pi) + 2
test_interaction_3[, 4] <- sin(test_t * -(pi / 2) + pi)
plot(test_interaction_3[,1], test_interaction_3[,2], 
     type = "l", xlim = c(-2, 2), ylim = c(-1, 1))
lines(test_interaction_3[,3], test_interaction_3[,4])
points(rbind(test_interaction_3[1, 1:2], test_interaction_3[1, 3:4]))

test_interaction_4 <- matrix(0, nrow = 101, ncol = 4)
test_interaction_4[, 1] <- cos(test_t * (pi / 2)) - 1
test_interaction_4[, 2] <- sin(test_t * (pi / 2))
test_interaction_4[, 3] <- cos(test_t * (pi / 2))
test_interaction_4[, 4] <- sin(test_t * (pi / 2))
plot(test_interaction_4[,1], test_interaction_4[,2], 
     type = "l", xlim = c(-2, 2), ylim = c(-1, 1))
lines(test_interaction_4[,3], test_interaction_4[,4])
points(rbind(test_interaction_4[1, 1:2], test_interaction_4[1, 3:4]))

test_interaction_5 <- matrix(0, nrow = 101, ncol = 4)
test_interaction_5[, 1] <- cos(test_t * (pi / 2)) - 1
test_interaction_5[, 2] <- sin(test_t * (pi / 2))
test_interaction_5[, 3] <- cos(test_t * -(pi / 2) + pi) + 2
test_interaction_5[, 4] <- sin(test_t * -(pi / 2) + pi)
plot(test_interaction_5[,1], test_interaction_5[,2], 
     type = "l", xlim = c(-2, 2), ylim = c(-1, 1))
lines(test_interaction_5[,3], test_interaction_5[,4])
points(rbind(test_interaction_5[1, 1:2], test_interaction_5[1, 3:4]))

generate_test_encounter_set <- function(test_interaction, sq_exp_kernel, num_example = 100) {
  lapply(1:num_example, function(i) {
    test_encounter <- apply(test_interaction, 2, function(col) {
      rmvn(1, col, sq_exp_kernel)
    })
    r_angle <- runif(1, 0, 2 * pi)
    rotation_m <- matrix(
      c(cos(r_angle), sin(r_angle), -sin(r_angle), cos(r_angle)), nrow = 2)
    offset <- rnorm(2, sd = 0.5)
    t(apply(test_encounter, 1, function(row) {
      as.vector(rotation_m %*% matrix(row, ncol = 2) + offset)
    }))
  })
}

test_set1 <- generate_test_encounter_set(test_interaction, sq_exp_kernel)
test_set2 <- generate_test_encounter_set(test_interaction_2, sq_exp_kernel)
test_set3 <- generate_test_encounter_set(test_interaction_3, sq_exp_kernel)
test_set4 <- generate_test_encounter_set(test_interaction_4, sq_exp_kernel)
test_set5 <- generate_test_encounter_set(test_interaction_5, sq_exp_kernel)
save(test_set1, test_set2, test_set3, test_set4, test_set5, 
     file = "~/Documents/Michigan/Research/Driving/matlab_data/sim_data_test_sets.Rdata")

test_2_list <- c(test_set1, test_set2)
test_2_order <- sample(length(test_2_list))
test_2_list <- test_2_list[test_2_order]
test_2_list <- lapply(test_2_list, function(test) {
  colnames(test) <- c("x1", "y1", "x2", "y2")
  test
})
test_2_p_dist_prim_m <- get_entire_p_dist_m(test_2_list, reflection = F)

test_3_list <- c(test_set1, test_set2, test_set3)
test_3_order <- sample(length(test_3_list))
test_3_list <- test_3_list[test_3_order]
test_3_list <- lapply(test_3_list, function(test) {
  colnames(test) <- c("x1", "y1", "x2", "y2")
  test
})
test_3_p_dist_prim_m <- get_entire_p_dist_m(test_3_list, reflection = F)

test_5_list <- c(test_set1, test_set2, test_set3, test_set4, test_set5)
test_5_order <- sample(length(test_5_list))
test_5_label <- (test_5_order - 1) %/% 100 + 1
group_assignment <- lapply(1:5, function(i) {which(test_5_label == i)})
test_5_list <- test_5_list[test_5_order]
test_5_list <- lapply(test_5_list, function(test) {
  colnames(test) <- c("x1", "y1", "x2", "y2")
  test
})
test_5_p_dist_prim_m <- get_entire_p_dist_m(test_5_list, reflection = F)
test_5_p_dist_prim_m_mds <- cmdscale(test_5_p_dist_prim_m, k = 10)
test_5_kmeans_result <- kmeans(test_5_p_dist_prim_m_mds, 5, nstart = 25)
mds_assignment <- lapply(1:5, function(i) {which(test_5_kmeans_result$cluster == i)})
get_optimal_match_hungarian(mds_assignment, group_assignment)

test_5_kmeans_result$center_inds <- find_closest_mds(test_5_p_dist_prim_m_mds, test_5_kmeans_result)
calc_within_cluster_dist(test_5_kmeans_result, test_5_p_dist_prim_m / max(test_5_p_dist_prim_m), 5)

test_5_p_dist_prim_m_mds_2 <- cmdscale(test_5_p_dist_prim_m, k = 2)
test_5_kmeans_result_2 <- kmeans(test_5_p_dist_prim_m_mds_2, 5, nstart = 25)
mds_assignment_2 <- lapply(1:5, function(i) {which(test_5_kmeans_result_2$cluster == i)})
get_optimal_match_hungarian(mds_assignment_2, group_assignment)

test_5_p_dist_prim_m_mds_3 <- cmdscale(test_5_p_dist_prim_m, k = 3)
test_5_kmeans_result_3 <- kmeans(test_5_p_dist_prim_m_mds_3, 5, nstart = 25)
mds_assignment_3 <- lapply(1:5, function(i) {which(test_5_kmeans_result_3$cluster == i)})
get_optimal_match_hungarian(mds_assignment_3, group_assignment)

mds_plots <- cbind(test_5_p_dist_prim_m_mds_3, test_5_label)

png("mds_sim_plots_3d.png", width = 4, height = 4, units = "in", res = 300)
points3D(x = mds_plots[,1], y = mds_plots[,2], z = mds_plots[,3],
         colvar = mds_plots[, 4], cex = 2, 
         col = c("navy", "lightblue", "green", "orange", "darkred"),
         ticktype = "detailed", clim = c(1, 5), clab = "Primitive",
         xlab = "", ylab = "", zlab = "", phi = 20, theta = 60,
         colkey = list(at = c(1.4, 2.2, 3, 3.8, 4.6), side = 4,
                       addlines = TRUE, labels = as.character(1:5)))
dev.off()

test_5_p_dist_prim_m_mds_5 <- cmdscale(test_5_p_dist_prim_m, k = 5)
test_5_kmeans_result_5 <- kmeans(test_5_p_dist_prim_m_mds_5, 5, nstart = 25)
mds_assignment_5 <- lapply(1:5, function(i) {which(test_5_kmeans_result_5$cluster == i)})
get_optimal_match_hungarian(mds_assignment_5, group_assignment)

l2_kmeans <- kmeans(t(sapply(test_5_list, as.vector)), 5, nstart = 25)
l2_assignment <- lapply(1:5, function(i) {which(l2_kmeans$cluster == i)})
get_optimal_match_hungarian(l2_assignment, group_assignment)

dtw_kmeans <- dtw_kmeans(test_5_list, 5)
dtw_assignment <- lapply(1:5, function(i) {
  which(dtw_kmeans$cluster == i)
})
get_optimal_match_hungarian(dtw_assignment, group_assignment)
calc_within_cluster_dist(dtw_kmeans, test_5_p_dist_prim_m / max(test_5_p_dist_prim_m), 5)

poly_kmeans <- poly_coeff_clustering(test_5_list, 5)
poly_assignment <- lapply(1:5, function(i) {
  which(poly_kmeans$cluster == i)
})
get_optimal_match_hungarian(poly_assignment, group_assignment)
calc_within_cluster_dist(poly_kmeans, test_5_p_dist_prim_m / max(test_5_p_dist_prim_m), 5)

oriented_prim_m <- matrix(0, nrow = length(test_5_list),
                          ncol = 4 * nrow(test_5_list[[1]]))
for (i in 1:length(test_5_list)) {
  print(i)
  oriented_prim_m[i,] <- as.vector(orient_and_centered_to_first_no_reflection(
    test_5_list[[1]][, c("x1", "y1")], test_5_list[[1]][, c("x2", "y2")],
    test_5_list[[i]][, c("x1", "y1")], test_5_list[[i]][, c("x2", "y2")]))
}
first_approx_cluster_results <- 
  kmeans(oriented_prim_m, 5, iter.max = 30, nstart = 25)
first_approx_assignment <- 
  lapply(1:5, function(i) {which(first_approx_cluster_results$cluster == i)})
get_optimal_match_hungarian(first_approx_assignment, group_assignment)
cluster_centers <- lapply(first_approx_assignment, function(group_ind) {
  matrix(colMeans(oriented_prim_m[group_ind,]), ncol = 4)
})
first_approx_cluster_results$center_inds <- find_closest_mds(oriented_prim_m, first_approx_cluster_results)
calc_within_cluster_dist(first_approx_cluster_results, test_5_p_dist_prim_m / max(test_5_p_dist_prim_m), 5)


second_approx_cluster_result <- procrustes_kmeans(test_5_list, 5, F)
second_approx_assignment <-
  lapply(1:5, function(i) {
    which(second_approx_cluster_result$cluster == i)
  })
get_optimal_match_hungarian(second_approx_assignment, group_assignment)
second_approx_cluster_result$centers <- t(sapply(second_approx_cluster_result$centers, as.vector))
second_approx_cluster_result$center_inds <- find_closest_mds(
  second_approx_cluster_result$oriented_m, second_approx_cluster_result)
calc_within_cluster_dist(second_approx_cluster_result, test_5_p_dist_prim_m / max(test_5_p_dist_prim_m), 5)


get_optimal_match_hungarian <- function(assignment1, assignment2) {
  
  cost_matrix <- build_zero_one_cost_matrix(assignment1, assignment2)
  opt_match <- solve_LSAP(cost_matrix)
  return(c(sum(sapply(1:length(opt_match), function(i) {
    cost_matrix[i, opt_match[i]]
  })), opt_match))
  
}

calc_within_cluster_dist <- function(kmeans_result, prim_p_dist_m, K) {
  prim_dist_sum = 0
  for (k in 1:K) {
    center_ind <- kmeans_result$center_inds[k]
    cluster_assignment <- which(kmeans_result$cluster == k)
    cluster_assignment <- cluster_assignment[cluster_assignment != center_ind]
    prim_dist_sum = prim_dist_sum + 
      sum(prim_p_dist_m[center_ind, cluster_assignment]^2)
  }
  print(prim_dist_sum)
}

dtw_kmeans <- function(prim_interpolated_list, k) {
  dtw_matrix <- do.call(rbind, 
    lapply(prim_interpolated_list, function(encounter) {
      dtw_cost <- as.vector(dtw(as.vector(encounter[, 1:2]), 
                                as.vector(encounter[, 3:4]), keep.internals = T)$costMatrix)
      if (max(dtw_cost) != 0) {
        dtw_cost / max(dtw_cost)
      }
    }))
  kmeans_results <- kmeans(dtw_matrix, k, nstart = 10)
  kmeans_results$center_inds <- find_closest_mds(dtw_matrix, kmeans_results)
  kmeans_results
}



build_zero_one_cost_matrix <- function(list1, list2) {
  cost_matrix <- matrix(0, nrow = length(list1), ncol = length(list2))
  for (i in 1:length(list1)) {
    for (j in 1:length(list2)) {
      cost_matrix[i, j] = sum(!(list1[[i]] %in% list2[[j]]))
    }
  }
  cost_matrix
}


get_optimal_match <- function(list1, list2, start_ind) {
  if (start_ind == 1) {
    list_order <- start_ind:length(list1)
  } else {
    list_order <- c(start_ind:length(list1), 1:(start_ind - 1))
  }
  list_2_inds <- 1:length(list2)
  map <- rep(0, length(list1))
  total_one_loss = 0
  for (i in list_order) {
    zero_one_loss <- sapply(list_2_inds, function(j) {
      sum(!(list1[[i]] %in% list2[[j]]))
    })
    map[i] = list_2_inds[which.min(zero_one_loss)]
    total_one_loss = total_one_loss + zero_one_loss[which.min(zero_one_loss)]
    list_2_inds <- list_2_inds[-which.min(zero_one_loss)]
  }
  c(total_one_loss, map)
}

find_optimal_group_match_zero_one_loss <- function(list1, list2) {
  opt_map <- rep(0, length(list1))
  zero_dist <- Inf
  for (i in 1:length(list1)) {
    opt_info <- get_optimal_match(list1, list2, i) 
    if (zero_dist > opt_info[1]) {
      zero_dist <- opt_info[1]
      opt_map <- opt_info[-1]
    }
  }
  return(c(zero_dist, opt_map))
}

poly_coeff_clustering <- function(prim_interpolated_list, k) {
  unprocessed_poly <- matrix(0, nrow = length(prim_interpolated_list),
                             ncol = 16)
  for (i in 1:length(prim_interpolated_list)) {
    for (j in 1:4) {
      unprocessed_poly[i, (j - 1) * 4 + 1:4] <-
        fit_cubic_poly(seq(0, 1, by = 0.01), prim_interpolated_list[[i]][, j])$coefficients
    }
  }
  kmeans_results <- kmeans(unprocessed_poly, k, iter.max = 30, nstart = 25)
  kmeans_results$center_inds <- find_closest_mds(unprocessed_poly, kmeans_results)
  return(kmeans_results)
}

#Test_set 13, 46 prim_interpolated, 1472, 4012,
#For comparison, test_set2 5, test set5 5, prim_interpolated, 3489, 3729
sim_ex <- do.call(rbind, test_set1[c(13, 46)])
max_x <- max(sim_ex[, c(1, 3)])
max_y <- max(sim_ex[, c(2, 4)])
min_x <- min(sim_ex[, c(1, 3)])
min_y <- min(sim_ex[, c(2, 4)])

png("../plots/paper_plots/sim_example_plots/sim_examples.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(test_set1[c(77, 46)], annotate_text = "Sim Example",
                      xlim = c(min_x - 0.5, max_x + 0.5),
                      ylim = c(min_y - 0.25, max_y + 0.25))
dev.off()

prim_examples <- lapply(c(1472, 4012), function(i) {
  interpolated_prim_list[[i]][, c("x1", "y1", "x2", "y2")]
})
make_encounter_ggplot(prim_examples, annotate_text = "Data Ex")

centered_prim_examples <- lapply(c(1472, 4012), function(i) {
  tmp <- interpolated_prim_list[[i]][, c("x1", "y1", "x2", "y2")]
  tmp_center <- colMeans(tmp)
  tmp_center <- (tmp_center[1:2] + tmp_center[3:4]) / 2
  cbind(sweep(tmp[, 1:2], 2, tmp_center, "-"),
        sweep(tmp[, 3:4], 2, tmp_center, "-"))
})
data_ex <- do.call(rbind, centered_prim_examples)
max_x <- max(data_ex[, c(1, 3)])
max_y <- max(data_ex[, c(2, 4)])
min_x <- min(data_ex[, c(1, 3)])
min_y <- min(data_ex[, c(2, 4)])
png("../plots/paper_plots/sim_example_plots/segmented_interpolated_prim.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(centered_prim_examples, annotate_text = "Re-Centered \n Data Example",
                      xlim = c(min_x - 1.5, max_x + 1.5),
                      ylim = c(min_y - 4, max_y + 4))
dev.off()

center_prim_example_1 <- interpolated_prim_list[[4]]
center_prim_example_1_center <- 
  colMeans(interpolated_prim_list[[4]][, c("x1", "y1", "x2", "y2")])
center_prim_example_1_center <- 
  (center_prim_example_1_center[1:2] + center_prim_example_1_center[3:4]) / 2
center_prim_example_1[, c("x1", "y1")] <-
  sweep(center_prim_example_1[, c("x1", "y1")], 2, center_prim_example_1_center, "-")
center_prim_example_1[, c("x2", "y2")] <-
  sweep(center_prim_example_1[, c("x2", "y2")], 2, center_prim_example_1_center, "-")

center_prim_example_2 <- interpolated_prim_list[[1472]]
center_prim_example_2_center <- 
  colMeans(center_prim_example_2[, c("x1", "y1", "x2", "y2")])
center_prim_example_2_center <- 
  (center_prim_example_2_center[1:2] + center_prim_example_2_center[3:4]) / 2
center_prim_example_2[, c("x1", "y1")] <-
  sweep(center_prim_example_2[, c("x1", "y1")], 2, center_prim_example_2_center, "-")
center_prim_example_2[, c("x2", "y2")] <-
  sweep(center_prim_example_2[, c("x2", "y2")], 2, center_prim_example_2_center, "-")

make_encounter_ggplot(list(center_prim_example_1, center_prim_example_2), 
                      annotate_text = "Sim Ex")

#Example is 1
png("../plots/paper_plots/sim_example_plots/test_interaction_1.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_interaction), color = "black", xlim = c(-1.25, 1.25), ylim = c(-1.25, 1.25), annotate_text = "Primitive 1")
dev.off()

png("../plots/paper_plots/sim_example_plots/test_interaction_2.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_interaction_2), color = "black", xlim = c(-1.25, 1.25), ylim = c(-1.25, 1.25), annotate_text = "Primitive 2")
dev.off()

png("../plots/paper_plots/sim_example_plots/test_interaction_3.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_interaction_3), color = "black", xlim = c(-.25, 2.25), ylim = c(-.125, 1.125), annotate_text = "Primitive 3")
dev.off()

png("../plots/paper_plots/sim_example_plots/test_interaction_4.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_interaction_4), color = "black", xlim = c(-1.25, 1.25), ylim = c(-.125, 1.125), annotate_text = "Primitive 4")
dev.off()

png("../plots/paper_plots/sim_example_plots/test_interaction_5.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_interaction_5), color = "black", xlim = c(-1.25, 2.25), ylim = c(-.125, 1.125), annotate_text = "Primitive 5")
dev.off()

png("../plots/paper_plots/sim_example_plots/test_interaction_1_example.png", height = 900, width = 900, res = 150)
make_encounter_ggplot(list(test_set1[[1]]), color = "black", xlim = c(-1.5 - .125, 1.625), ylim = c(-1.25 -.25, -.25 + .5), annotate_text = "Primitive 1 \n Interaction \n Trajectories")
dev.off()

#Find top 3 to export
coord_prim_list <- list()
for (i in 1:length(clean_encounters)) {
  encounter = clean_encounters[i]
  cat(encounter, "\n")
  tmp <- clean_rel_dist_m[clean_rel_dist_m$encounter == encounter,];
  
  chg_pts <- all_joint_chg_pt_list[[i]]
  for (j in 1:(length(chg_pts) - 1)) {
    prim_m <- data.frame("x1" = tmp$LO_1[chg_pts[j]:chg_pts[(j + 1)]],
                         "y1" = tmp$LA_1[chg_pts[j]:chg_pts[(j + 1)]],
                         "x2" = tmp$LO_2[chg_pts[j]:chg_pts[(j + 1)]],
                         "y2" = tmp$LA_2[chg_pts[j]:chg_pts[(j + 1)]],
                         "encounter" = paste(encounter, j, sep = "_"))
    coord_prim_list <- c(coord_prim_list, list(prim_m))
  }
}

load("cubic_poly_results/cubic_poly_cluster_results_corrected_with_ordered_encounters.Rdata")
load("cubic_poly_results/cubic_poly_interpolated_prim.Rdata")
load("cubic_poly_results/cubic_poly_coord_prim_encounters.Rdata")
mds_result_nr <- mds_cluster_results_no_reflection[[3]]
cluster_order <- order(table(mds_result_nr$cluster), decreasing = T)
encounter_m <- as.data.frame(do.call(rbind, lapply(1:length(cluster_order), function(i) {
  encounter_ind <- mds_result_nr$order_encounter[1:10, cluster_order[i]]
  do.call(rbind, lapply(encounter_ind, function(j) {
    cbind(coord_prim_list[[j]], i)
  }))
})))
for (i in 1:4) {
  encounter_m[, i] <- as.numeric(encounter_m[, i])
}
write.csv(encounter_m, file = "cubic_poly_results/cubic_poly_typical_encounters_10_ordered_coord.csv", row.names = F)



