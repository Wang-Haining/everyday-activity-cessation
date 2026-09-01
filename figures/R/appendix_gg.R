# The appendix forests, in the same house style as the manuscript figures.
#
# Four appendix tables were columns of risk ratios and intervals, which is a
# forest written out in words. They are drawn here instead, from CSVs that
# analysis/export_appendix_figure_data.py reads out of those same tables, so a
# figure cannot disagree with the table it replaces.
#
# The cohort-specific table is drawn as two figures rather than one. Forty rows
# and eight headings is 9 inches of forest, which does not fit on a page with a
# caption, and a figure that runs over is worse than two that do not.

MS_ROOT <- Sys.getenv("MS_ROOT", unset = ".")
DATA <- file.path(MS_ROOT, "figures", "data")
OUT  <- file.path(MS_ROOT, "figures", "lancet_r")
source(file.path(MS_ROOT, "figures", "R", "lancet_theme.R"))

read_appendix <- function(name)
  read.csv(file.path(DATA, paste0("appendix_", name, ".csv")),
           stringsAsFactors = FALSE, colClasses = "character")

num <- function(x) suppressWarnings(as.numeric(ifelse(x == "", NA, x)))

# Rows for lancet_forest(), grouped by the `group` column in the order the
# table had them. A row with no estimate keeps its label and prints why in the
# estimate column, which is what the table did.
build <- function(d, text_of, extra_of = NULL) {
  d$estimate <- num(d$estimate); d$lo <- num(d$lo); d$hi <- num(d$hi)
  groups <- unique(d$group)
  out <- do.call(rbind, lapply(groups, function(g) {
    body <- d[d$group == g, , drop = FALSE]
    head <- body[1, , drop = FALSE]
    head[] <- NA
    head$label <- g
    rbind(head, body)
  }))
  out$heading <- is.na(out$estimate) & out$label %in% groups
  out$text <- ifelse(out$heading, NA_character_, text_of(out))
  out$extra <- if (is.null(extra_of)) NA_character_ else
    ifelse(out$heading, NA_character_, extra_of(out))
  out[, c("label", "heading", "estimate", "lo", "hi", "text", "extra")]
}

# lancet_forest refuses an interval outside the axis, which is the behaviour we
# want; pick the axis from the data so that never happens by accident.
axis_for <- function(d, breaks) {
  lo <- min(num(d$lo), na.rm = TRUE); hi <- max(num(d$hi), na.rm = TRUE)
  c(min(lo, min(breaks)) * 0.96, max(hi, max(breaks)) * 1.04)
}

save_forest <- function(p, stem, n, favours = NULL, xlab = NULL) {
  save_lancet(p, file.path(OUT, stem), width = DOUBLE_COL,
              height = forest_height(n, xlab, favours))
}

# ------------------------------------------------------------------ 1. LOCO
loco <- read_appendix("leave_one_out")
# The blocks that would leave two cohorts are not synthesised. Twelve rows
# saying so is a design fact repeated twelve times: it goes in the caption, and
# the figure comes down from 8.6 inches, which no page can hold with a caption,
# to something that fits.
loco <- loco[loco$estimate != "", , drop = FALSE]
rows <- build(loco,
  text_of = function(d) ci_text(d$estimate, d$lo, d$hi),
  extra_of = function(d) md(d$i2))
br <- c(1, 1.5, 2, 3, 4)
p <- lancet_forest(rows, breaks = br, limits = axis_for(loco, br),
                   est_header = "Risk ratio (95% CI)",
                   extra_x = 1.0, extra_header = paste0("I", SUP2),
                   plot_x = c(0.40, 0.70), est_x = 0.90,
                   favours = c("Lower risk", "Higher risk"))
save_forest(p, "appendix_fig_leave_one_out", nrow(rows),
            favours = c("Lower risk", "Higher risk"))

# --------------------------------------------------- 2 and 3. per cohort
# One figure, not two. It was split into a diseases half and a
# death-function-multimorbidity half; the newly reported diagnoses have left
# this paper, so the first half would now draw an empty forest.
coh <- read_appendix("cohort_results")
{
  d <- coh
  rows <- build(d,
    text_of = function(x) ifelse(is.na(x$estimate), x$status,
                                 ci_text(x$estimate, x$lo, x$hi)),
    extra_of = function(x) paste0(x$n, " / ", x$events))
  br <- c(0.75, 1, 1.5, 2)
  p <- lancet_forest(rows, breaks = br, limits = axis_for(d, br),
                     est_header = "Risk ratio (95% CI)",
                     extra_x = 1.0, extra_header = "n / events",
                     plot_x = c(0.32, 0.64), est_x = 0.86,
                     favours = c("Lower risk", "Higher risk"))
  save_forest(p, "appendix_fig_cohort_results", nrow(rows),
              favours = c("Lower risk", "Higher risk"))
}

# ------------------------------------------------------------ 4. components
comp <- read_appendix("components")
rows <- build(comp,
  text_of = function(d) ci_text(d$estimate, d$lo, d$hi),
  extra_of = function(d) paste0(fmt(num(d$pi_lo)), EN_DASH, fmt(num(d$pi_hi))))
br <- c(1, 1.25, 1.5, 2)
p <- lancet_forest(rows, breaks = br, limits = axis_for(comp, br),
                   est_header = "Risk ratio (95% CI)",
                   extra_x = 1.0, extra_header = "Prediction interval",
                   plot_x = c(0.34, 0.66), est_x = 0.84,
                   favours = c("Lower risk", "Higher risk"))
save_forest(p, "appendix_fig_components", nrow(rows),
            favours = c("Lower risk", "Higher risk"))

# ------------------------------------------------------- 5. are they equal
eq <- read_appendix("component_equality")
rows <- build(eq,
  text_of = function(d) ci_text(d$estimate, d$lo, d$hi),
  extra_of = function(d) paste0(fmt(num(d$pi_lo)), EN_DASH, fmt(num(d$pi_hi))))
br <- c(0.75, 1, 1.5, 2)
p <- lancet_forest(rows, breaks = br, limits = axis_for(eq, br),
                   est_header = "Ratio of risk ratios (95% CI)",
                   extra_x = 1.0, extra_header = "Prediction interval",
                   plot_x = c(0.40, 0.68), est_x = 0.86,
                   xlab = "Ratio of the first component's risk ratio to the second's")
save_forest(p, "appendix_fig_component_equality", nrow(rows),
            xlab = "Ratio of the first component's risk ratio to the second's")

cat("appendix forests written\n")
