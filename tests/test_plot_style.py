import unittest

from esd_experiments.core.plot_style import (
    CATEGORICAL_COLORS,
    ERROR_CAPSIZE,
    FIGSIZE_DOUBLE,
    FIGSIZE_SINGLE,
    FONT_SCALE,
    GRID_STYLE,
    LEGEND_STYLE,
    SEMANTIC_COLORS,
    color_cycle,
    errorbar_kwargs,
    legend_kwargs,
    multipanel_figsize,
)


class PlotStyleTests(unittest.TestCase):
    def test_size_family_is_positive(self):
        for width, height in (FIGSIZE_SINGLE, FIGSIZE_DOUBLE, multipanel_figsize(2)):
            self.assertGreater(width, 0.0)
            self.assertGreater(height, 0.0)

    def test_semantic_colors_are_defined(self):
        for key in ("esd", "risk", "lower_bound", "upper_bound"):
            self.assertIn(key, SEMANTIC_COLORS)
            self.assertTrue(SEMANTIC_COLORS[key].startswith("#"))

    def test_style_helpers_are_stable(self):
        self.assertAlmostEqual(FONT_SCALE, 1.3)
        self.assertEqual(color_cycle(len(CATEGORICAL_COLORS)), CATEGORICAL_COLORS[0])
        self.assertEqual(errorbar_kwargs()["capsize"], ERROR_CAPSIZE)
        self.assertEqual(legend_kwargs()["frameon"], LEGEND_STYLE["frameon"])
        self.assertIn("linestyle", GRID_STYLE)


if __name__ == "__main__":
    unittest.main()
