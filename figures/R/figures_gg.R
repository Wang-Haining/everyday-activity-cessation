# The three manuscript figures, in ggplot2.
#
#   Figure 1  A  adjusted standardised two-year risk of death, per cohort
#             B  the same for new limitation in activities of daily living
#             C  adjusted risk ratios across the outcomes
#   Figure 2  A  the four states a respondent could occupy for regular activity
#             B  stopped against continued, in each domain
#   Figure 3  A  retirement-linked and other work exits
#             B  each diagnosis at the next interview and one interview later
#
# Every forest is drawn in the form this journal prints: a table with a bold
# heading over each group, one line per estimate, the picture in the middle and
# the figures on the right, set in one ink. Colour is spent only in A and B of
# figure 1, where five cohorts have to be told apart along a gradient and
# position alone cannot do it.
#
# The pooled synthesis in figure 1 A and B is drawn as a line and not as an
# interval: its 95% CI is a random-effects interval on the logit scale across
# cohorts whose baseline risk differs severalfold, so it spans most of the panel
# and hides the ordering the panel exists to show.

ROOT <- Sys.getenv("MS_ROOT", unset = normalizePath("."))
DATA <- file.path(ROOT, "figures/data")
OUT  <- file.path(ROOT, "figures/lancet_r")
# Clear the output directory first, so what comes back is exactly what this run
# drew. A figure that stops being produced used to linger here, and then in the
# manuscript folder, with nothing to say it was orphaned.
unlink(OUT, recursive = TRUE)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
source(file.path(ROOT, "figures/R/lancet_theme.R"))

XLAB <- c("None", "One", "Two or more")
EST_HEAD <- "Risk ratio (95% CI)"

# ----------------------------------------------------------------- figure 1 A,B
risk_panel <- function(name, ylab, ymax, ystep) {
  d <- read_panel(name)
  cohorts <- names(COHORT_COL)[names(COHORT_COL) %in% setdiff(d$series, "Pooled")]
  d$series <- factor(d$series, levels = c(cohorts, "Pooled"))
  coh <- d[d$series != "Pooled", ]
  pool <- d[d$series == "Pooled", ]
  # A small horizontal offset per cohort keeps the intervals apart.
  coh$xo <- coh$x + (as.integer(coh$series) - (length(cohorts) + 1) / 2) * 0.075
  ends <- coh[coh$x == 2, ]

  ggplot() +
    geom_linerange(data = coh, aes(x = xo, ymin = lo, ymax = hi, colour = series),
                   linewidth = 0.28) +
    geom_line(data = coh, aes(x = xo, y = risk, colour = series), linewidth = 0.38) +
    geom_point(data = coh, aes(x = xo, y = risk, colour = series, shape = series),
               fill = "white", size = 1.1, stroke = 0.38) +
    geom_line(data = pool, aes(x = x, y = risk), colour = POOLED,
              linewidth = 0.55, linetype = "41") +
    geom_text(data = ends, aes(x = 2.16, y = risk, label = series, colour = series),
              hjust = 0, size = SMALL_SIZE / .pt, family = FONT) +
    annotate("segment", x = -0.30, xend = -0.10, y = ymax * 0.965, yend = ymax * 0.965,
             colour = POOLED, linewidth = 0.55, linetype = "41") +
    annotate("text", x = -0.04, y = ymax * 0.965, label = "Pooled", hjust = 0,
             size = SMALL_SIZE / .pt, family = FONT) +
    scale_colour_manual(values = COHORT_COL, guide = "none") +
    scale_shape_manual(values = COHORT_PCH, guide = "none") +
    scale_x_continuous(breaks = 0:2, labels = XLAB, limits = c(-0.34, 2.34),
                       expand = expansion(0)) +
    scale_y_continuous(breaks = seq(0, ymax, ystep), limits = c(0, ymax),
                       labels = function(b) fmt_break(b), expand = expansion(0)) +
    labs(x = "Activities recently stopped", y = ylab) +
    coord_cartesian(clip = "off") +
    theme_lancet() +
    theme(plot.margin = margin(4, 32, 2, 2))
}

# ------------------------------------------------------------------- figure 1 C
outcome_forest <- function() {
  d <- read_panel("fig1c_outcome_forest.csv")
  pick <- function(outcome, series)
    d[d$row_label == outcome & d$series == series, ]
  graded <- function(outcome, heading) {
    one <- pick(outcome, "One stopped"); two <- pick(outcome, "Two or more stopped")
    group_block(heading, c("One activity stopped", "Two or more stopped"),
                c(one$estimate, two$estimate), c(one$lo, two$lo),
                c(one$hi, two$hi), c(one$cohorts, two$cohorts))
  }
  rows <- forest_rows(
    graded("Death", "Death"),
    graded("New ADL limitation", "New limitation in activities of daily living"),
    graded("New IADL limitation", "New limitation in instrumental activities"),
    graded("Multimorbidity progression", "Multimorbidity progression"))

  # The lower limit was 0.84 to clear the hypertension interval, which reached
  # 0.878. With the newly reported diagnoses gone the lowest bound in the panel
  # is 1.05, and leaving the old limit would print an inch of empty axis.
  lancet_forest(rows, breaks = c(1, 1.5, 2, 3, 4), limits = c(0.95, 4.20),
                est_header = EST_HEAD, plot_x = c(0.37, 0.72), est_x = 0.925,
                extra_x = 1.0, extra_header = "Cohorts",
                favours = c("Lower risk", "Higher risk"))
}

# --------------------------------------------------------------------- figure 2
reference_forest <- function() {
  a <- read_panel("fig2a_transition_states.csv")
  a <- a[match(c("Never active", "Stopped", "Started", "Continued"), a$row_label), ]
  b <- read_panel("fig2b_stopped_vs_continued.csv")
  rows <- forest_rows(
    group_block("Against being inactive at both interviews",
                c("Inactive at both interviews", "Recently stopped",
                  "Recently started", "Continued"),
                a$estimate, a$lo, a$hi),
    group_block("Stopped against continued, in each domain",
                b$row_label, b$estimate, b$lo, b$hi))
  lancet_forest(rows, breaks = c(0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
                limits = c(0.35, 2.30), est_header = EST_HEAD,
                plot_x = c(0.33, 0.70), est_x = 0.89,
                favours = c("Lower risk of death", "Higher risk of death"))
}

# --------------------------------------------------------------------- figure 3
# One x scale for both forests. They are stacked, so a reader compares them
# down the page; with different limits the dashed null line sat at a different
# horizontal position in each and the ticks did not line up. These limits clear
# both panels' widest intervals, 0.52 and 2.96.
F3_BREAKS <- c(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
F3_LIMITS <- c(0.47, 3.15)

paired_forest <- function(name, levels, headings, breaks, limits, xlab,
                          plot_x, est_x) {
  d <- read_panel(name)
  blocks <- lapply(names(headings), function(key) {
    g <- d[d$row_label == key, ]
    g <- g[match(levels, g$series), ]
    group_block(headings[[key]], levels, g$estimate, g$lo, g$hi)
  })
  lancet_forest(do.call(forest_rows, blocks), breaks = breaks, limits = limits,
                est_header = EST_HEAD, plot_x = plot_x, est_x = est_x,
                xlab = xlab)
}

work_exit_panel <- function() {
  paired_forest("fig3a_work_exit.csv",
                c("Retirement-linked exit", "Other work exit"),
                list("HRS, death" = "HRS, death",
                     "HRS, new ADL" = "HRS, new limitation in activities of daily living",
                     "ELSA, new ADL" = "ELSA, new limitation in activities of daily living"),
                F3_BREAKS, F3_LIMITS,
                "Adjusted risk ratio versus continued work",
                plot_x = c(0.47, 0.80), est_x = 0.985)
}


# ------------------------------------------------------------------- assembly
tagged <- function(p) p & theme(plot.tag = element_text(
  size = BASE_SIZE + 1, face = "bold", family = FONT, hjust = 0, vjust = 1))

# The rule around the whole figure, and the margin that keeps the panels off it.
BOXED <- theme(
  plot.background = element_rect(fill = "white", colour = "black", linewidth = 0.45),
  plot.margin = margin(4, 4, 4, 4))

# One y scale for both panels. They carried 18 and 24 before, so the same
# vertical distance meant a different number of percentage points in each and
# the reader could not compare the two gradients by eye. 25 clears the widest
# interval in either panel, which reaches 24.6.
Y_MAX <- 25; Y_STEP <- 5
fa <- risk_panel("fig1a_death_risk.csv", "Two-year risk of death (%)", Y_MAX, Y_STEP)
fb <- risk_panel("fig1b_adl_risk.csv",
                 "Two-year risk of new\nADL limitation (%)", Y_MAX, Y_STEP)
fc <- outcome_forest()
f2 <- reference_forest()
f3a <- work_exit_panel()

# Every height is computed from the row count at one pitch, never chosen, so a
# forest that gains a row grows instead of tightening.
H_FC  <- forest_height(12, favours = TRUE)
H_F2  <- forest_height(9,  favours = TRUE)
H_F3A <- forest_height(9,  xlab = "Adjusted risk ratio versus continued work")
H_RISK <- 3.00

figure1 <- ((fa | fb) / fc) + plot_layout(heights = c(H_RISK, H_FC)) +
  plot_annotation(tag_levels = list(c("A", "B", "C")), theme = BOXED)
figure2 <- f2 + plot_annotation(theme = BOXED)
figure3 <- f3a + plot_annotation(theme = BOXED)

save_lancet(fa,  file.path(OUT, "panel_fig1_a_death_risk"),        SINGLE_COL, H_RISK)
save_lancet(fb,  file.path(OUT, "panel_fig1_b_adl_risk"),          SINGLE_COL, H_RISK)
save_lancet(fc,  file.path(OUT, "panel_fig1_c_outcome_gradient"),  DOUBLE_COL, H_FC)
save_lancet(f3a, file.path(OUT, "panel_fig3_a_work_exit_reason"),  DOUBLE_COL, H_F3A)
save_lancet(tagged(figure1), file.path(OUT, "figure1"), DOUBLE_COL, H_RISK + H_FC)
save_lancet(tagged(figure2), file.path(OUT, "figure2"), DOUBLE_COL, H_F2)
save_lancet(tagged(figure3), file.path(OUT, "figure3"), DOUBLE_COL, H_F3A)
cat(sprintf("figure heights: 1 %.2f  2 %.2f  3 %.2f\n",
            H_RISK + H_FC, H_F2, H_F3A))
