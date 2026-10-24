# Statistics for each geometry measure, as in Fig. 6D: ART mixed models with mouse as a
# random intercept and SF as the repeated level.
#   WT vs rescued pre-training     value ~ group * sf + (1 | mouse)      (WT = pre-training sessions, as in Fig. 6D)
#   WT vs rescued post-training    value ~ group * sf + (1 | mouse)
#   rescued pre vs post            value ~ training * sf + (1 | mouse)
#   WT pre vs post                 value ~ training * sf + (1 | mouse)
#   group x training               value ~ group * training * sf + (1 | mouse)
suppressPackageStartupMessages(library(ARTool))
options(warn = -1, width = 140)
f <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

d <- read.csv("geometry_metrics_long.csv", stringsAsFactors = FALSE)
d$sf <- factor(d$sf); d$mouse <- factor(d$mouse)
p_term <- function(formula, data, term) {
  a <- anova(art(formula, data = droplevels(data)))
  a[a$Term == term, "Pr(>F)"]
}
res <- do.call(rbind, lapply(unique(d$metric), function(m) {
  x <- d[d$metric == m, ]
  wt_pre <- x[x$group == "WT" & x$training == "pre", ]
  r_pre <- x[x$group == "Rescued" & x$training == "pre", ]
  r_post <- x[x$group == "Rescued" & x$training == "post", ]
  g1 <- rbind(wt_pre, r_pre); g1$grp <- factor(g1$group)
  g2 <- rbind(wt_pre, r_post); g2$grp <- factor(g2$group)
  rs <- x[x$group == "Rescued", ]; rs$training <- factor(rs$training, c("pre", "post"))
  wt <- x[x$group == "WT", ]; wt$training <- factor(wt$training, c("pre", "post"))
  all <- x; all$training <- factor(all$training, c("pre", "post")); all$grp <- factor(all$group)
  data.frame(metric = m,
             WT_vs_rescued_pre_p = p_term(value ~ grp * sf + (1 | mouse), g1, "grp"),
             WT_vs_rescued_post_p = p_term(value ~ grp * sf + (1 | mouse), g2, "grp"),
             rescued_pre_vs_post_p = p_term(value ~ training * sf + (1 | mouse), rs, "training"),
             WT_pre_vs_post_p = p_term(value ~ training * sf + (1 | mouse), wt, "training"),
             group_x_training_p = p_term(value ~ grp * training * sf + (1 | mouse), all, "grp:training"))
}))
write.csv(res, "stats_geometry_alternatives.csv", row.names = FALSE)
fmt <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
print(data.frame(metric = res$metric, lapply(res[-1], fmt), check.names = FALSE), row.names = FALSE)
