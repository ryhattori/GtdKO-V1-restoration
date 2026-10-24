# Supplementary Table 1: every group comparison in Figs 4-6 and Supplementary Figs 3-4, refitted with the same
# aligned-rank-transform (ART) mixed models as the figures, with F, degrees of freedom, exact p, partial eta^2
# (F*df1 / (F*df1 + df2)) and group means +/- SEM over mice (each mouse averaged over the x-axis levels).
#   between groups      value ~ group * level + (1 | mouse)              term: group
#   pre vs post         value ~ training * level + (1 | mouse)           term: training
#   group x training    value ~ group * training * level + (1 | mouse)   term: group:training
# Inputs: fig4_fig5/fig45_session_metrics.csv, session_metrics/session_metrics_long.csv and
# osi_cross_validated.csv, suppfig4_dspca_alternatives/geometry_metrics_long.csv.  Output: stats_table.csv
suppressPackageStartupMessages(library(ARTool))
f <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(f)) dirname(normalizePath(f)) else getwd()
root <- dirname(here)

f45 <- read.csv(file.path(root, "fig4_fig5/fig45_session_metrics.csv"))
r16 <- read.csv(file.path(root, "session_metrics/session_metrics_long.csv"))
cvo <- read.csv(file.path(root, "session_metrics/osi_cross_validated.csv"))
cvo <- subset(cvo, selection == "responsive")
r16 <- rbind(r16, data.frame(group = cvo$group, mouse = cvo$mouse, training = cvo$training, metric = "osi_cv",
                             level = cvo$sf, value = cvo$osi_cv))
geo <- read.csv(file.path(root, "suppfig4_dspca_alternatives/geometry_metrics_long.csv"))
geo <- data.frame(group = geo$group, mouse = geo$mouse, training = geo$training, metric = geo$metric,
                  level = geo$sf, value = geo$value)

NEURONS <- c(WT_pre = 741, WT_post = 630, Rescued_pre = 513, Rescued_post = 455, GtdKO = 279)
LABEL <- c(resp_by_sf = "Responsive fraction by SF", resp_by_ori = "Responsive fraction by orientation",
           osi_cv = "Cross-validated OSI", sf_dec_vs_n = "SF decoding vs number of neurons",
           ori_dec_vs_n = "Orientation decoding vs number of neurons", sf_dec_by_ori = "SF decoding by orientation",
           ori_dec_by_sf = "Orientation decoding by SF", adj_dist_by_sf = "Adjacent-orientation distance (dsPCA)",
           pca3 = "Adjacent-orientation distance (top 3 PCs)", crossnobis = "Cross-validated Mahalanobis distance",
           lda_acc = "Cross-validated linear decoding accuracy")

fit <- function(formula, x, term) {
  x$level <- factor(x$level); x$mouse <- factor(x$mouse)
  for (v in intersect(c("group", "training"), names(x))) x[[v]] <- factor(x[[v]])
  a <- as.data.frame(anova(art(formula, data = droplevels(x))))
  r <- a[a$Term == term, ]
  c(F = r$F, df1 = r$Df, df2 = r$Df.res, p = r$`Pr(>F)`, eta2p = r$F * r$Df / (r$F * r$Df + r$Df.res))
}
msem <- function(x) {
  m <- aggregate(value ~ mouse, x, mean)$value
  sprintf("%.3f ± %.3f", mean(m), sd(m) / sqrt(length(m)))
}
rows <- list()
add <- function(fig, panel, metric, comparison, term, g1, g2, x, formula, n1, n2, ne1, ne2) {
  s <- fit(formula, x, term)
  rows[[length(rows) + 1]] <<- data.frame(
    figure = fig, panel = panel, measure = LABEL[[metric]], comparison = comparison, term = term,
    n_mice = paste(n1, n2, sep = " / "), neurons = paste(ne1, ne2, sep = " / "),
    mean_sem_1 = msem(g1), mean_sem_2 = msem(g2), F = s["F"], df1 = s["df1"], df2 = s["df2"], p = s["p"],
    eta2p = s["eta2p"], metric = metric)
}
nm <- function(x) length(unique(x$mouse))

# ---- Figs 4-5 (WT = WT mice before training; Pre/Post = GtdKO+AAV-Gat2)
m45 <- c(resp_by_sf = "C|A", resp_by_ori = "D|B", osi_cv = "F|D", sf_dec_vs_n = "H|E", ori_dec_vs_n = "I|F",
         sf_dec_by_ori = "J|G", ori_dec_by_sf = "K|H")
for (met in names(m45)) {
  pan <- strsplit(m45[[met]], "\\|")[[1]]
  d <- subset(f45, metric == met)
  wt <- subset(d, group == "WT"); pre <- subset(d, group == "Pre"); post <- subset(d, group == "Post")
  x <- rbind(wt, pre); x$group <- ifelse(x$group == "WT", "WT", "Rescued")
  add("Fig. 4", pan[1], met, "WT vs GtdKO+AAV-Gαt2 (pre)", "group", wt, pre, x, value ~ group * level + (1 | mouse),
      nm(wt), nm(pre), NEURONS[["WT_pre"]], NEURONS[["Rescued_pre"]])
  x <- rbind(wt, post)
  add("Fig. 5", pan[2], met, "WT vs GtdKO+AAV-Gαt2 (post)", "group", wt, post, x,
      value ~ group * level + (1 | mouse), nm(wt), nm(post), NEURONS[["WT_pre"]], NEURONS[["Rescued_post"]])
  x <- rbind(pre, post); x$training <- ifelse(x$group == "Pre", "pre", "post"); x$group <- NULL
  add("Fig. 5", pan[2], met, "GtdKO+AAV-Gαt2 pre vs post", "training", pre, post, x,
      value ~ training * level + (1 | mouse), nm(pre), nm(post), NEURONS[["Rescued_pre"]], NEURONS[["Rescued_post"]])
}

# ---- Fig. 6D and Supplementary Fig. 4 (WT = WT before training)
three_way <- function(d, fig, panel, met) {
  wt <- subset(d, group == "WT" & training == "pre"); pre <- subset(d, group == "Rescued" & training == "pre")
  post <- subset(d, group == "Rescued" & training == "post")
  add(fig, panel, met, "WT vs GtdKO+AAV-Gαt2 (pre)", "group", wt, pre, rbind(wt, pre)[, -3],
      value ~ group * level + (1 | mouse), nm(wt), nm(pre), NEURONS[["WT_pre"]], NEURONS[["Rescued_pre"]])
  add(fig, panel, met, "WT vs GtdKO+AAV-Gαt2 (post)", "group", wt, post, rbind(wt, post)[, -3],
      value ~ group * level + (1 | mouse), nm(wt), nm(post), NEURONS[["WT_pre"]], NEURONS[["Rescued_post"]])
  add(fig, panel, met, "GtdKO+AAV-Gαt2 pre vs post", "training", pre, post, rbind(pre, post)[, -1],
      value ~ training * level + (1 | mouse), nm(pre), nm(post), NEURONS[["Rescued_pre"]], NEURONS[["Rescued_post"]])
}
three_way(subset(r16, metric == "adj_dist_by_sf"), "Fig. 6", "D", "adj_dist_by_sf")
for (met in c(pca3 = "B", crossnobis = "D", lda_acc = "F")) NULL
for (met in c("pca3", "crossnobis", "lda_acc"))
  three_way(subset(geo, metric == met), "Suppl. Fig. 4", c(pca3 = "B", crossnobis = "D", lda_acc = "F")[[met]], met)

# ---- Supplementary Fig. 3 (WT before vs after training; group x training)
s3 <- c(resp_by_sf = "A|J", resp_by_ori = "B|K", osi_cv = "D|M", sf_dec_vs_n = "E|N", ori_dec_vs_n = "F|O",
        sf_dec_by_ori = "G|P", ori_dec_by_sf = "H|Q", adj_dist_by_sf = "I|R")
for (met in names(s3)) {
  pan <- strsplit(s3[[met]], "\\|")[[1]]
  d <- subset(r16, metric == met & group %in% c("WT", "Rescued"))
  wpre <- subset(d, group == "WT" & training == "pre"); wpost <- subset(d, group == "WT" & training == "post")
  add("Suppl. Fig. 3", pan[1], met, "WT pre vs post", "training", wpre, wpost, rbind(wpre, wpost)[, -1],
      value ~ training * level + (1 | mouse), nm(wpre), nm(wpost), NEURONS[["WT_pre"]], NEURONS[["WT_post"]])
  wt_d <- subset(d, group == "WT"); rs_d <- subset(d, group == "Rescued")
  dm <- function(z) { a <- aggregate(value ~ mouse + training, z, mean); w <- reshape(a, idvar = "mouse",
                       timevar = "training", direction = "wide"); data.frame(mouse = w$mouse, value = w$value.post - w$value.pre) }
  add("Suppl. Fig. 3", pan[2], met, "Group × training (change, WT vs GtdKO+AAV-Gαt2)", "group:training",
      dm(wt_d), dm(rs_d), d, value ~ group * training * level + (1 | mouse), nm(wt_d), nm(rs_d),
      paste0(NEURONS[["WT_pre"]], "+", NEURONS[["WT_post"]]), paste0(NEURONS[["Rescued_pre"]], "+", NEURONS[["Rescued_post"]]))
}

out <- do.call(rbind, rows)
rownames(out) <- NULL
write.csv(out, file.path(here, "stats_table.csv"), row.names = FALSE)
cat("rows:", nrow(out), "\n")
