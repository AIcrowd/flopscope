"""Comprehensive tests for the einsum parser (upstream opt_einsum.parser)."""

import numpy
import pytest
from opt_einsum.parser import (
    alpha_canonicalize,
    convert_interleaved_input,
    convert_subscripts,
    convert_to_valid_einsum_chars,
    find_output_shape,
    find_output_str,
    gen_unused_symbols,
    get_shape,
    get_symbol,
    has_valid_einsum_chars_only,
    is_valid_einsum_char,
    parse_einsum_input,
    possibly_convert_to_numpy,
)


# ---------------------------------------------------------------------------
# is_valid_einsum_char
# ---------------------------------------------------------------------------
class TestIsValidEinsumChar:
    @pytest.mark.parametrize("ch", list("abcdefghijklmnopqrstuvwxyz"))
    def test_lowercase_valid(self, ch):
        assert is_valid_einsum_char(ch) is True

    @pytest.mark.parametrize("ch", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    def test_uppercase_valid(self, ch):
        assert is_valid_einsum_char(ch) is True

    @pytest.mark.parametrize("ch", list(",->."))
    def test_special_valid(self, ch):
        assert is_valid_einsum_char(ch) is True

    @pytest.mark.parametrize("ch", ["0", "9", " ", "\n", "\t", "@", "#", "$"])
    def test_ascii_invalid(self, ch):
        assert is_valid_einsum_char(ch) is False

    @pytest.mark.parametrize("ch", ["\u0134", "\u00d6", "\u4eac"])
    def test_unicode_invalid(self, ch):
        assert is_valid_einsum_char(ch) is False


# ---------------------------------------------------------------------------
# has_valid_einsum_chars_only
# ---------------------------------------------------------------------------
class TestHasValidEinsumCharsOnly:
    def test_simple_valid(self):
        assert has_valid_einsum_chars_only("ij,jk->ik") is True

    def test_all_letters(self):
        assert has_valid_einsum_chars_only("abcXYZ") is True

    def test_ellipsis(self):
        assert has_valid_einsum_chars_only("...ij,...jk->...ik") is True

    def test_empty_string(self):
        assert has_valid_einsum_chars_only("") is True

    def test_with_space(self):
        assert has_valid_einsum_chars_only("ij, jk") is False

    def test_with_unicode(self):
        assert has_valid_einsum_chars_only("\u00d6ver") is False

    def test_with_digit(self):
        assert has_valid_einsum_chars_only("a1b") is False


# ---------------------------------------------------------------------------
# get_symbol
# ---------------------------------------------------------------------------
class TestGetSymbol:
    def test_first_52(self):
        assert get_symbol(0) == "a"
        assert get_symbol(25) == "z"
        assert get_symbol(26) == "A"
        assert get_symbol(51) == "Z"

    def test_beyond_52(self):
        # i >= 52, i < 55296 => chr(i + 140)
        result = get_symbol(52)
        assert result == chr(52 + 140)

    def test_surrogate_skip(self):
        # i >= 55296 => chr(i + 2048)
        result = get_symbol(55296)
        assert result == chr(55296 + 2048)


# ---------------------------------------------------------------------------
# gen_unused_symbols
# ---------------------------------------------------------------------------
class TestGenUnusedSymbols:
    def test_basic(self):
        result = list(gen_unused_symbols("abd", 2))
        assert result == ["c", "e"]

    def test_empty_used(self):
        result = list(gen_unused_symbols("", 3))
        assert result == ["a", "b", "c"]

    def test_zero_needed(self):
        result = list(gen_unused_symbols("abc", 0))
        assert result == []

    def test_all_base_used(self):
        """When all 52 base symbols are used, should yield unicode symbols."""
        from opt_einsum.parser import _einsum_symbols_base

        result = list(gen_unused_symbols(_einsum_symbols_base, 2))
        # These should be the first two unicode symbols after the base set
        assert len(result) == 2
        for s in result:
            assert s not in _einsum_symbols_base

    def test_skip_used_symbols(self):
        result = list(gen_unused_symbols("ace", 3))
        assert result == ["b", "d", "f"]


# ---------------------------------------------------------------------------
# convert_to_valid_einsum_chars
# ---------------------------------------------------------------------------
class TestConvertToValidEinsumChars:
    def test_already_valid(self):
        result = convert_to_valid_einsum_chars("ij,jk->ik")
        # Symbols are re-mapped in sorted order
        assert "->" in result

    def test_unicode_chars(self):
        result = convert_to_valid_einsum_chars("\u0124\u011b\u013c\u013c\u00f6")
        assert has_valid_einsum_chars_only(result)

    def test_preserves_structure(self):
        result = convert_to_valid_einsum_chars("xy,yz->xz")
        assert "," in result
        assert "->" in result


# ---------------------------------------------------------------------------
# alpha_canonicalize
# ---------------------------------------------------------------------------
class TestAlphaCanonicalize:
    def test_reverse_order(self):
        assert alpha_canonicalize("dcba") == "abcd"

    def test_unicode(self):
        assert alpha_canonicalize("\u0124\u011b\u013c\u013c\u00f6") == "abccd"

    def test_with_arrow(self):
        result = alpha_canonicalize("zy,yx->zx")
        # z->a, y->b, x->c
        assert result == "ab,bc->ac"

    def test_preserves_special_chars(self):
        result = alpha_canonicalize("ab,ba->")
        assert result == "ab,ba->"


# ---------------------------------------------------------------------------
# find_output_str
# ---------------------------------------------------------------------------
class TestFindOutputStr:
    def test_matmul(self):
        assert find_output_str("ab,bc") == "ac"

    def test_outer_product(self):
        assert find_output_str("a,b") == "ab"

    def test_all_repeated(self):
        assert find_output_str("a,a,b,b") == ""

    def test_trace(self):
        assert find_output_str("aa") == ""

    def test_single_operand(self):
        assert find_output_str("abc") == "abc"

    def test_three_operands(self):
        # c appears twice (summed), a and b appear once each
        assert find_output_str("ac,cb,cd") == "abd"


# ---------------------------------------------------------------------------
# find_output_shape
# ---------------------------------------------------------------------------
class TestFindOutputShape:
    def test_matmul(self):
        assert find_output_shape(["ab", "bc"], [(2, 3), (3, 4)], "ac") == (2, 4)

    def test_broadcasting(self):
        assert find_output_shape(["a", "a"], [(4,), (1,)], "a") == (4,)

    def test_single_char(self):
        assert find_output_shape(["a"], [(5,)], "a") == (5,)


# ---------------------------------------------------------------------------
# get_shape
# ---------------------------------------------------------------------------
class TestGetShape:
    def test_numpy_array(self):
        a = numpy.zeros((3, 4))
        assert get_shape(a) == (3, 4)

    def test_scalar_int(self):
        assert get_shape(5) == ()

    def test_scalar_float(self):
        assert get_shape(3.14) == ()

    def test_scalar_complex(self):
        assert get_shape(1 + 2j) == ()

    def test_scalar_bool(self):
        assert get_shape(True) == ()

    def test_scalar_str(self):
        assert get_shape("hello") == ()

    def test_scalar_bytes(self):
        assert get_shape(b"hello") == ()

    def test_list_1d(self):
        assert get_shape([1, 2, 3]) == (3,)

    def test_list_2d(self):
        assert get_shape([[1, 2], [3, 4]]) == (2, 2)

    def test_nested_list(self):
        assert get_shape([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) == (2, 2, 2)

    def test_unhashable_raises(self):
        with pytest.raises(ValueError, match="Cannot determine the shape"):
            get_shape(object())

    def test_custom_shape_attr(self):
        class FakeArray:
            shape = (2, 3)

        assert get_shape(FakeArray()) == (2, 3)


# ---------------------------------------------------------------------------
# possibly_convert_to_numpy
# ---------------------------------------------------------------------------
class TestPossiblyConvertToNumpy:
    def test_scalar(self):
        result = possibly_convert_to_numpy(5)
        assert hasattr(result, "shape")
        assert result.shape == ()

    def test_list(self):
        result = possibly_convert_to_numpy([1, 2, 3])
        assert hasattr(result, "shape")
        assert result.shape == (3,)

    def test_already_array(self):
        a = numpy.array([1, 2])
        result = possibly_convert_to_numpy(a)
        assert result is a

    def test_has_shape_passthrough(self):
        class FakeArray:
            shape = (4, 5)

        fa = FakeArray()
        assert possibly_convert_to_numpy(fa) is fa


# ---------------------------------------------------------------------------
# convert_subscripts
# ---------------------------------------------------------------------------
class TestConvertSubscripts:
    def test_basic(self):
        assert convert_subscripts(["abc", "def"], {"abc": "a", "def": "b"}) == "ab"

    def test_with_ellipsis(self):
        obj = object()
        result = convert_subscripts([Ellipsis, obj], {obj: "a"})
        assert result == "...a"


# ---------------------------------------------------------------------------
# convert_interleaved_input
# ---------------------------------------------------------------------------
class TestConvertInterleavedInput:
    def test_basic_two_operands(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        subscripts, ops = convert_interleaved_input((a, [0, 1], b, [1, 2], [0, 2]))
        assert "," in subscripts
        assert "->" in subscripts
        assert len(ops) == 2
        assert ops[0] is a
        assert ops[1] is b

    def test_no_output(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        subscripts, ops = convert_interleaved_input((a, [0, 1], b, [1, 2]))
        assert "->" not in subscripts
        assert len(ops) == 2

    def test_with_ellipsis(self):
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        subscripts, ops = convert_interleaved_input(
            (a, [Ellipsis, 0, 1], b, [Ellipsis, 1, 2], [Ellipsis, 0, 2])
        )
        assert "..." in subscripts
        assert "->" in subscripts

    def test_unhashable_raises(self):
        """Unhashable objects in subscript lists should raise TypeError."""
        a = numpy.ones((3,))
        with pytest.raises(TypeError, match="hashable and comparable"):
            convert_interleaved_input((a, [[1, 2]], a, [[1, 2]]))

    def test_single_operand_with_output(self):
        a = numpy.ones((3, 4))
        subscripts, ops = convert_interleaved_input((a, [0, 1], [0, 1]))
        assert "->" in subscripts
        assert len(ops) == 1


# ---------------------------------------------------------------------------
# parse_einsum_input
# ---------------------------------------------------------------------------
class TestParseEinsumInput:
    # --- Basic string notation ---
    def test_simple_matmul(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input(("ij,jk->ik", a, b))
        assert inp == "ij,jk"
        assert out == "ik"
        assert len(ops) == 2

    def test_implicit_output(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input(("ij,jk", a, b))
        assert inp == "ij,jk"
        assert out == "ik"  # implicit: unique indices sorted

    def test_trace(self):
        a = numpy.eye(3)
        inp, out, ops = parse_einsum_input(("ii->", a))
        assert inp == "ii"
        assert out == ""

    def test_spaces_stripped(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input(("ij , jk -> ik", a, b))
        assert inp == "ij,jk"
        assert out == "ik"

    # --- Shapes mode ---
    def test_shapes_mode(self):
        inp, out, ops = parse_einsum_input(("ij,jk->ik", (3, 4), (4, 5)), shapes=True)
        assert inp == "ij,jk"
        assert out == "ik"

    def test_shapes_mode_rejects_arrays(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="looks like an array"):
            parse_einsum_input(("ij,jk->ik", a, (4, 5)), shapes=True)

    # --- Ellipsis handling ---
    def test_ellipsis_matmul(self):
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        inp, out, ops = parse_einsum_input(("...ij,...jk->...ik", a, b))
        assert "..." not in inp  # ellipsis should be expanded
        assert "..." not in out
        assert len(out) == 3  # batch + i + k

    def test_ellipsis_without_output(self):
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        inp, out, ops = parse_einsum_input(("...ij,...jk", a, b))
        assert "..." not in inp
        # Output should contain the ellipsis dims plus unique non-contracted dims
        assert len(out) > 0

    def test_ellipsis_scalar_operand(self):
        """Ellipsis with a scalar operand (shape = ()) should have 0 ellipsis dims."""
        a = numpy.array(5.0)  # scalar
        b = numpy.ones((3,))
        inp, out, ops = parse_einsum_input(("...,...i->...i", a, b))
        assert "..." not in inp

    def test_invalid_ellipsis_two_dots(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="Invalid Ellipses"):
            parse_einsum_input(("..ij,jk->ik", a, numpy.ones((4, 5))))

    def test_invalid_ellipsis_four_dots(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="Invalid Ellipses"):
            parse_einsum_input(("....ij,jk->ik", a, numpy.ones((4, 5))))

    def test_ellipsis_length_mismatch(self):
        """More subscript chars than dimensions => negative ellipsis count."""
        a = numpy.ones((3,))
        with pytest.raises(ValueError, match="Ellipses lengths do not match"):
            parse_einsum_input(("...ij", a))

    def test_ellipsis_zero_count(self):
        """Ellipsis resolves to zero extra dims when ndim == len(explicit subscripts)."""
        a = numpy.ones((3, 4))
        inp, out, ops = parse_einsum_input(("...ij->ij", a))
        assert "." not in inp

    # --- Error paths ---
    def test_no_operands(self):
        with pytest.raises(ValueError, match="No input operands"):
            parse_einsum_input(())

    def test_invalid_arrow_dash_only(self):
        a = numpy.ones((3,))
        with pytest.raises(ValueError, match="one '->'"):
            parse_einsum_input(("-i", a))

    def test_invalid_arrow_gt_only(self):
        a = numpy.ones((3,))
        with pytest.raises(ValueError, match="one '->'"):
            parse_einsum_input((">i", a))

    def test_double_arrow(self):
        a = numpy.ones((3,))
        with pytest.raises(ValueError, match="one '->'"):
            parse_einsum_input(("i->->j", a))

    def test_duplicate_output_char(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="appeared more than once"):
            parse_einsum_input(("ij->ii", a))

    def test_output_char_not_in_input(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="did not appear in the input"):
            parse_einsum_input(("ij->k", a))

    def test_operand_count_mismatch(self):
        a = numpy.ones((3, 4))
        with pytest.raises(ValueError, match="must be equal to the number of operands"):
            parse_einsum_input(("ij,jk->ik", a))

    # --- Interleaved input format ---
    def test_interleaved_basic(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input((a, [0, 1], b, [1, 2], [0, 2]))
        assert len(ops) == 2
        assert "," in inp

    def test_interleaved_no_output(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input((a, [0, 1], b, [1, 2]))
        assert len(ops) == 2

    def test_interleaved_with_ellipsis(self):
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 5))
        inp, out, ops = parse_einsum_input(
            (a, [Ellipsis, 0, 1], b, [Ellipsis, 1, 2], [Ellipsis, 0, 2])
        )
        assert len(ops) == 2

    # --- get_shape integration within parse ---
    def test_list_operand(self):
        """Lists should be converted via get_shape for dimension detection."""
        inp, out, ops = parse_einsum_input(
            ("ij,jk->ik", [[1, 2], [3, 4]], [[5, 6], [7, 8]])
        )
        assert inp == "ij,jk"
        assert out == "ik"

    # --- Edge cases ---
    def test_single_operand_identity(self):
        a = numpy.ones((3, 4))
        inp, out, ops = parse_einsum_input(("ij->ij", a))
        assert inp == "ij"
        assert out == "ij"

    def test_single_operand_transpose(self):
        a = numpy.ones((3, 4))
        inp, out, ops = parse_einsum_input(("ij->ji", a))
        assert inp == "ij"
        assert out == "ji"

    def test_empty_output(self):
        """Full contraction to scalar."""
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 3))
        inp, out, ops = parse_einsum_input(("ij,ji->", a, b))
        assert out == ""

    def test_multi_contraction(self):
        a = numpy.ones((2, 3))
        b = numpy.ones((3, 4))
        c = numpy.ones((4, 5))
        inp, out, ops = parse_einsum_input(("ij,jk,kl->il", a, b, c))
        assert inp == "ij,jk,kl"
        assert out == "il"
        assert len(ops) == 3


# ---------------------------------------------------------------------------
# Edge cases for ellipsis expansion paths
# ---------------------------------------------------------------------------
class TestEllipsisEdgeCases:
    def test_both_operands_with_ellipsis(self):
        """Both operands have ellipsis, output too."""
        a = numpy.ones((5, 3, 4))
        b = numpy.ones((5, 4, 2))
        inp, out, ops = parse_einsum_input(("...ij,...jk->...ik", a, b))
        assert "." not in inp
        assert "." not in out

    def test_only_one_operand_with_ellipsis(self):
        """One operand has ellipsis, other doesn't."""
        a = numpy.ones((5, 3, 4))
        b = numpy.ones((4, 2))
        inp, out, ops = parse_einsum_input(("...ij,jk->...ik", a, b))
        assert "." not in inp

    def test_ellipsis_high_dim(self):
        """Multiple batch dimensions via ellipsis."""
        a = numpy.ones((2, 3, 4, 5))
        b = numpy.ones((2, 3, 5, 6))
        inp, out, ops = parse_einsum_input(("...ij,...jk->...ik", a, b))
        assert "." not in inp
        # batch dims (2,3) + i + k = 4 chars output
        assert len(out) == 4

    def test_ellipsis_implicit_output_with_contraction(self):
        """Ellipsis without explicit output, with contracted indices."""
        a = numpy.ones((2, 3, 4))
        b = numpy.ones((2, 4, 3))
        inp, out, ops = parse_einsum_input(("...ij,...ji", a, b))
        # Only ellipsis dims should remain (i and j contract)
        assert "." not in inp
        assert len(out) >= 1  # at least the batch dim
