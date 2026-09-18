args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4, args[4]=="binary64")
folder <- args[1]
labels <- read.csv(args[2], check.names=FALSE)
reported <- read.csv(args[3], check.names=FALSE)
stopifnot(!anyDuplicated(labels$sample_id))
anchors <- names(labels)[names(labels)!="sample_id"]
canonical_columns <- readLines(file.path(folder,"column-ids.txt"))
canonical_ids <- read.csv(file.path(folder,"labels.index.csv"),check.names=FALSE)$sample_id
stopifnot(identical(canonical_ids,labels$sample_id),setequal(canonical_columns,anchors))
canonical <- matrix(readBin(file.path(folder,"labels.f64"),"double",n=nrow(labels)*length(anchors),size=8,endian="little"),nrow=nrow(labels),byrow=TRUE)
stopifnot(max(abs(canonical-as.matrix(labels[canonical_columns])))<1e-12)
labels[canonical_columns] <- canonical
# Non-interpolated AP: tied scores share the same cumulative precision threshold.
average_precision <- function(y, score) {
  stopifnot(all(y %in% c(0,1)), all(is.finite(score)))
  if (sum(y)==0) return(NA_real_)
  order <- order(score, decreasing=TRUE)
  y <- y[order]; score <- score[order]
  end <- c(which(diff(score)!=0), length(score))
  positive <- cumsum(y)[end]
  sum(diff(c(0, positive)) / sum(y) * positive / end)
}
stopifnot(abs(average_precision(c(1,0,1), c(.8,.8,.1)) - 7/12)<1e-12)
checks <- list()
for (i in seq_len(nrow(reported))) {
  row <- reported[i,]
  prefix <- file.path(folder, paste0(row$kind,"--",row$method))
  ids <- read.csv(paste0(prefix,".index.csv"),check.names=FALSE)$sample_id
  columns <- readLines(file.path(folder,"column-ids.txt"))
  values <- readBin(paste0(prefix,".f64"),"double",n=length(ids)*length(columns),size=8,endian="little")
  stopifnot(length(values)==length(ids)*length(columns))
  x <- data.frame(sample_id=ids,matrix(values,nrow=length(ids),byrow=TRUE),check.names=FALSE)
  names(x) <- c("sample_id",columns)
  stopifnot(!anyDuplicated(x$sample_id), setequal(x$sample_id,labels$sample_id),
            setequal(names(x),names(labels)))
  x <- x[match(labels$sample_id,x$sample_id),]
  if (row$metric=="macro_ap") {
    per <- vapply(anchors, function(a) average_precision(labels[[a]],x[[a]]), numeric(1))
    point <- mean(per[is.finite(per)])
  } else {
    dimension <- sub("^spearman_","",row$metric)
    stopifnot(startsWith(row$metric,"spearman_"),dimension %in% anchors)
    point <- cor(labels[[dimension]],x[[dimension]],method="spearman")
    per <- point
  }
  error <- abs(point-row$value)
  stopifnot(is.finite(point), error<1e-12)
  checks[[length(checks)+1]] <- data.frame(kind=row$kind,method=row$method,
      metric=row$metric,value=point,max_error=error,n=nrow(labels),defined_coordinates=sum(is.finite(per)))
}
write.csv(do.call(rbind,checks),file.path(folder,"verification.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(folder,"R-session.txt"))
