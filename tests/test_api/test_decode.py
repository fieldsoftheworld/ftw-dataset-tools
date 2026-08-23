"""Tests for the DECODE derived label layers."""

import numpy as np

from ftw_dataset_tools.api.decode import boundary_from_mask, distance_from_mask


class TestBoundaryFromMask:
    """Tests for boundary_from_mask."""

    def test_marks_inner_ring_of_field(self) -> None:
        """Boundary is the one-pixel inner ring, not the field interior."""
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[2:5, 2:5] = 1

        boundary = boundary_from_mask(mask)

        # The 3x3 field has a full ring of 8 boundary pixels and a hollow centre.
        assert boundary.sum() == 8
        assert boundary[3, 3] == 0
        assert boundary[2, 2] == 1

    def test_returns_uint8(self) -> None:
        """Boundary is stored as uint8, not float32."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1

        assert boundary_from_mask(mask).dtype == np.uint8

    def test_never_marks_background(self) -> None:
        """Only field pixels can be boundary pixels."""
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:3, 1:3] = 1

        boundary = boundary_from_mask(mask)

        assert not boundary[mask == 0].any()

    def test_adjacent_fields_separated_by_line_get_own_rings(self) -> None:
        """Two fields split by a background line each get a closed boundary."""
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[1:8, 1:4] = 1
        mask[1:8, 5:8] = 1  # column 4 is the burned-in boundary line

        boundary = boundary_from_mask(mask)

        # Both fields are bounded on the side facing the separating line.
        assert boundary[4, 3] == 1
        assert boundary[4, 5] == 1
        # The separator itself is background, so it is not a boundary pixel.
        assert boundary[4, 4] == 0

    def test_field_at_chip_edge_has_no_boundary_there(self) -> None:
        """The chip border is not a field border."""
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[0:3, 0:3] = 1

        boundary = boundary_from_mask(mask)

        # Top-left corner runs off the chip, so it is interior, not boundary.
        assert boundary[0, 0] == 0
        # The sides facing background inside the chip are boundaries.
        assert boundary[2, 0] == 1
        assert boundary[0, 2] == 1

    def test_presence_only_class_treated_as_background(self) -> None:
        """Class 3 is background, so it produces a boundary like class 0 does."""
        presence_only = np.full((6, 6), 3, dtype=np.uint8)
        presence_only[1:5, 1:5] = 1

        plain = np.zeros((6, 6), dtype=np.uint8)
        plain[1:5, 1:5] = 1

        np.testing.assert_array_equal(boundary_from_mask(presence_only), boundary_from_mask(plain))

    def test_empty_mask_has_no_boundary(self) -> None:
        """A chip with no fields produces an all-zero boundary layer."""
        assert boundary_from_mask(np.zeros((8, 8), dtype=np.uint8)).sum() == 0

    def test_does_not_mutate_input(self) -> None:
        """The source mask is left untouched."""
        mask = np.full((5, 5), 3, dtype=np.uint8)
        mask[1:4, 1:4] = 1
        original = mask.copy()

        boundary_from_mask(mask)

        np.testing.assert_array_equal(mask, original)


class TestDistanceFromMask:
    """Tests for distance_from_mask."""

    def test_normalized_to_unit_range(self) -> None:
        """Values are scaled into [0, 1] with the field centre at 1.0."""
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[1:10, 1:10] = 1

        distance, max_px = distance_from_mask(mask)

        assert distance.min() == 0.0
        assert distance.max() == 1.0
        assert distance[5, 5] == 1.0
        assert max_px > 0

    def test_returns_float32(self) -> None:
        """Distance is stored as float32."""
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1

        distance, _ = distance_from_mask(mask)

        assert distance.dtype == np.float32

    def test_reported_max_undoes_normalization(self) -> None:
        """The returned max is the divisor, so scaling back gives pixel distances."""
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[1:10, 1:10] = 1

        distance, max_px = distance_from_mask(mask)

        # The centre of a 9x9 field is 5 pixels from the nearest background pixel.
        assert max_px == 5.0
        assert np.isclose(distance[5, 5] * max_px, 5.0)

    def test_background_is_zero(self) -> None:
        """Background pixels have no distance."""
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        distance, _ = distance_from_mask(mask)

        assert not distance[mask == 0].any()

    def test_empty_mask_is_all_zero(self) -> None:
        """A chip with no fields does not divide by zero."""
        distance, max_px = distance_from_mask(np.zeros((8, 8), dtype=np.uint8))

        assert max_px == 0.0
        assert distance.max() == 0.0
        assert np.isfinite(distance).all()

    def test_presence_only_class_treated_as_background(self) -> None:
        """Class 3 is background for the distance transform too."""
        presence_only = np.full((9, 9), 3, dtype=np.uint8)
        presence_only[2:7, 2:7] = 1

        plain = np.zeros((9, 9), dtype=np.uint8)
        plain[2:7, 2:7] = 1

        np.testing.assert_array_equal(
            distance_from_mask(presence_only)[0], distance_from_mask(plain)[0]
        )

    def test_normalization_is_per_chip(self) -> None:
        """The same field normalizes differently depending on the rest of the chip."""
        small_only = np.zeros((21, 21), dtype=np.uint8)
        small_only[1:4, 1:4] = 1

        with_larger_field = small_only.copy()
        with_larger_field[6:19, 6:19] = 1

        alone, _ = distance_from_mask(small_only)
        alongside, _ = distance_from_mask(with_larger_field)

        # The small field peaks at 1.0 alone but is scaled down by the big field.
        assert alone[2, 2] == 1.0
        assert alongside[2, 2] < 1.0

    def test_does_not_mutate_input(self) -> None:
        """The source mask is left untouched."""
        mask = np.full((5, 5), 3, dtype=np.uint8)
        mask[1:4, 1:4] = 1
        original = mask.copy()

        distance_from_mask(mask)

        np.testing.assert_array_equal(mask, original)

    def test_chip_that_is_entirely_field_stays_finite(self) -> None:
        """A chip with no background at all must not produce inf or nan."""
        distance, max_px = distance_from_mask(np.ones((16, 16), dtype=np.uint8))

        assert np.isfinite(distance).all()
        assert not np.isnan(distance).any()
        assert max_px > 0
        assert distance.max() == 1.0

    def test_chip_that_is_entirely_presence_only_is_all_zero(self) -> None:
        """An all-class-3 chip is all background, so there is nothing to measure."""
        distance, max_px = distance_from_mask(np.full((8, 8), 3, dtype=np.uint8))

        assert max_px == 0.0
        assert distance.max() == 0.0
        assert np.isfinite(distance).all()

    def test_single_pixel_field(self) -> None:
        """A one-pixel field is entirely boundary, at distance 1 from background."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1

        distance, max_px = distance_from_mask(mask)

        assert max_px == 1.0
        assert distance[2, 2] == 1.0
        assert boundary_from_mask(mask)[2, 2] == 1
