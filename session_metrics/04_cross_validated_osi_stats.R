# Fig. 5C OSI vs cross-validated OSI (09_cross_validated_osi.py): ART mixed models with mouse as a
# random intercept, as in Figs 4-6 (level = SF).
#   per group      value ~ training * sf + (1 | mouse)                 training effect
#   interaction    value ~ group * training * sf + (1 | mouse)         group x training
#   WT vs rescued  value ~ group * sf + (1 | mouse), pre- and post-training sessions separately
suppressPackageStartupMessages(library(ARTool))
options(warn = -1, width = 140)
f <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

d <- read.csv("osi_cross_validated.csv", stringsAsFactors = FALSE)
d$sf <- factor(d$sf); d$mouse <- factor(d$mouse)
d$training <- factor(d$training, c("pre", "post")); d$group <- factor(d$group, c("WT", "Rescued"))
p_of <- function(formula, data, term) {
  a <- anova(art(formula, data = droplevels(data)))
  a[a$Term == term, "Pr(>F)"]
}
res <- list()
for (sel in unique(d$selection)) for (measure in c("osi_fig5c", "osi_cv", "mean_response")) {
  x <- d[d$selection == sel, ]; x$y <- x[[measure]]
  res[[length(res) + 1]] <- data.frame(
    selection = sel, measure = measure,
    WT_training_p = p_of(y ~ training * sf + (1 | mouse), x[x$group == "WT", ], "training"),
    Rescued_training_p = p_of(y ~ training * sf + (1 | mouse), x[x$group == "Rescued", ], "training"),
    group_x_training_p = p_of(y ~ group * training * sf + (1 | mouse), x, "group:training"),
    pre_group_p = p_of(y ~ group * sf + (1 | mouse), x[x$training == "pre", ], "group"),
    post_group_p = p_of(y ~ group * sf + (1 | mouse), x[x$training == "post", ], "group"))
}
res <- do.call(rbind, res)
write.csv(res, "stats_osi_cross_validated.csv", row.names = FALSE)
fmt <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
print(data.frame(res[1:2], lapply(res[-(1:2)], fmt)), row.names = FALSE)
