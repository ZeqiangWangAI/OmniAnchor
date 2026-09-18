# Export selected data objects without executing the upstream R Markdown analysis.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2, !dir.exists(args[2]))
dir.create(args[2], recursive=TRUE)
environment <- new.env(parent=emptyenv())
load(args[1], envir=environment)
for (name in c("d1a", "d1b", "d3a", "stats.occupation", "stats.name")) {
  object <- get(name, envir=environment, inherits=FALSE)
  stopifnot(is.data.frame(object))
  write.csv(object, file=file.path(args[2], paste0(name, ".csv")), row.names=FALSE)
}
for (name in c("wefat.occupation", "wefat.name")) {
  object <- get(name, envir=environment, inherits=FALSE)$data.diff
  write.csv(object, file=file.path(args[2], paste0(name, ".csv")), row.names=FALSE)
}
writeLines(capture.output(sessionInfo()), file.path(args[2], "R-session.txt"))
