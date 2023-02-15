//! Comparison traits for `[T]`.

use crate::cmp::{self, BytewiseEq, Ordering};
use crate::ffi;
use crate::mem;

use super::from_raw_parts;
use super::memchr;

extern "C" {
    /// Calls implementation provided memcmp.
    ///
    /// Interprets the data as u8.
    ///
    /// Returns 0 for equal, < 0 for less than and > 0 for greater
    /// than.
    fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> ffi::c_int;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> PartialEq<[B]> for [A]
where
    A: PartialEq<B>,
{
    fn eq(&self, other: &[B]) -> bool {
        SlicePartialEq::equal(self, other)
    }

    fn ne(&self, other: &[B]) -> bool {
        SlicePartialEq::not_equal(self, other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for [T] {}

/// Implements comparison of vectors [lexicographically](Ord#lexicographical-comparison).
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for [T] {
    fn cmp(&self, other: &[T]) -> Ordering {
        SliceOrd::compare(self, other)
    }
}

/// Implements comparison of vectors [lexicographically](Ord#lexicographical-comparison).
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for [T] {
    fn partial_cmp(&self, other: &[T]) -> Option<Ordering> {
        SlicePartialOrd::partial_compare(self, other)
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialEq
trait SlicePartialEq<B> {
    fn equal(&self, other: &[B]) -> bool;

    fn not_equal(&self, other: &[B]) -> bool {
        !self.equal(other)
    }
}

// Generic slice equality
impl<A, B> SlicePartialEq<B> for [A]
where
    A: PartialEq<B>,
{
    default fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter().zip(other.iter()).all(|(x, y)| x == y)
    }
}

// Use memcmp for bytewise equality when the types allow
impl<A, B> SlicePartialEq<B> for [A]
where
    A: BytewiseEq<B>,
{
    fn equal(&self, other: &[B]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // SAFETY: `self` and `other` are references and are thus guaranteed to be valid.
        // The two slices have been checked to have the same size above.
        unsafe {
            let size = mem::size_of_val(self);
            memcmp(self.as_ptr() as *const u8, other.as_ptr() as *const u8, size) == 0
        }
    }
}

#[doc(hidden)]
// intermediate trait for specialization of slice's PartialOrd
trait SlicePartialOrd: Sized {
    fn partial_compare(left: &[Self], right: &[Self]) -> Option<Ordering>;
}

impl<A: PartialOrd> SlicePartialOrd for A {
    default fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
        let l = cmp::min(left.len(), right.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &left[..l];
        let rhs = &right[..l];

        for i in 0..l {
            match lhs[i].partial_cmp(&rhs[i]) {
                Some(Ordering::Equal) => (),
                non_eq => return non_eq,
            }
        }

        left.len().partial_cmp(&right.len())
    }
}

// This is the impl that we would like to have. Unfortunately it's not sound.
// See `partial_ord_slice.rs`.
/*
impl<A> SlicePartialOrd for A
where
    A: Ord,
{
    default fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(left, right))
    }
}
*/

impl<A: AlwaysApplicableOrd> SlicePartialOrd for A {
    fn partial_compare(left: &[A], right: &[A]) -> Option<Ordering> {
        Some(SliceOrd::compare(left, right))
    }
}

#[rustc_specialization_trait]
trait AlwaysApplicableOrd: SliceOrd + Ord {}

macro_rules! always_applicable_ord {
    ($([$($p:tt)*] $t:ty,)*) => {
        $(impl<$($p)*> AlwaysApplicableOrd for $t {})*
    }
}

always_applicable_ord! {
    [] u8, [] u16, [] u32, [] u64, [] u128, [] usize,
    [] i8, [] i16, [] i32, [] i64, [] i128, [] isize,
    [] bool, [] char,
    [T: ?Sized] *const T, [T: ?Sized] *mut T,
    [T: AlwaysApplicableOrd] &T,
    [T: AlwaysApplicableOrd] &mut T,
    [T: AlwaysApplicableOrd] Option<T>,
}

#[doc(hidden)]
// intermediate trait for specialization of slice's Ord
trait SliceOrd: Sized {
    fn compare(left: &[Self], right: &[Self]) -> Ordering;
}

impl<A: Ord> SliceOrd for A {
    default fn compare(left: &[Self], right: &[Self]) -> Ordering {
        let l = cmp::min(left.len(), right.len());

        // Slice to the loop iteration range to enable bound check
        // elimination in the compiler
        let lhs = &left[..l];
        let rhs = &right[..l];

        for i in 0..l {
            match lhs[i].cmp(&rhs[i]) {
                Ordering::Equal => (),
                non_eq => return non_eq,
            }
        }

        left.len().cmp(&right.len())
    }
}

// memcmp compares a sequence of unsigned bytes lexicographically.
// this matches the order we want for [u8], but no others (not even [i8]).
impl SliceOrd for u8 {
    #[inline]
    fn compare(left: &[Self], right: &[Self]) -> Ordering {
        // Since the length of a slice is always less than or equal to isize::MAX, this never underflows.
        let diff = left.len() as isize - right.len() as isize;
        // This comparison gets optimized away (on x86_64 and ARM) because the subtraction updates flags.
        let len = if left.len() < right.len() { left.len() } else { right.len() };
        // SAFETY: `left` and `right` are references and are thus guaranteed to be valid.
        // We use the minimum of both lengths which guarantees that both regions are
        // valid for reads in that interval.
        let mut order = unsafe { memcmp(left.as_ptr(), right.as_ptr(), len) as isize };
        if order == 0 {
            order = diff;
        }
        order.cmp(&0)
    }
}

pub(super) trait SliceContains: Sized {
    fn slice_contains_element(hs: &[Self], needle: &Self) -> bool;
    fn slice_contains_slice(hs: &[Self], needle: &[Self]) -> bool;
}

impl<T> SliceContains for T
where
    T: PartialEq,
{
    default fn slice_contains_element(hs: &[Self], needle: &Self) -> bool {
        hs.iter().any(|element| *element == *needle)
    }

    default fn slice_contains_slice(hs: &[Self], needle: &[Self]) -> bool {
        default_slice_contains_slice(hs, needle)
    }
}

impl SliceContains for u8 {
    #[inline]
    fn slice_contains_element(hs: &[Self], needle: &Self) -> bool {
        memchr::memchr(*needle, hs).is_some()
    }

    #[inline]
    fn slice_contains_slice(hs: &[Self], needle: &[Self]) -> bool {
        if needle.len() <= 32 {
            if let Some(result) = simd_contains(hs, needle) {
                return result;
            }
        }
        default_slice_contains_slice(hs, needle)
    }
}

unsafe fn bytes_of<T>(slice: &[T]) -> &[u8] {
    // SAFETY: caller promises that `T` and `u8` have the same memory layout,
    // thus casting `x.as_ptr()` as `*const u8` is safe.  The `x.as_ptr()` comes
    // from a reference and is thus guaranteed to be valid for reads for the
    // length of the slice `x.len()`, which cannot be larger than
    // `isize::MAX`. The returned slice is never mutated.
    unsafe { from_raw_parts(slice.as_ptr() as *const u8, slice.len()) }
}

impl SliceContains for i8 {
    #[inline]
    fn slice_contains_element(hs: &[Self], needle: &Self) -> bool {
        // SAFETY: i8 and u8 have the same memory layout
        u8::slice_contains_element(unsafe { bytes_of(hs) }, &(*needle as u8))
    }

    #[inline]
    fn slice_contains_slice(hs: &[Self], needle: &[Self]) -> bool {
        // SAFETY: i8 and u8 have the same memory layout
        unsafe { u8::slice_contains_slice(bytes_of(hs), bytes_of(needle)) }
    }
}

impl SliceContains for bool {
    #[inline]
    fn slice_contains_element(hs: &[Self], needle: &Self) -> bool {
        // SAFETY: bool and u8 have the same memory layout and all valid bool
        // bit patterns are valid u8 bit patterns.
        u8::slice_contains_element(unsafe { bytes_of(hs) }, &(*needle as u8))
    }

    #[inline]
    fn slice_contains_slice(hs: &[Self], needle: &[Self]) -> bool {
        // SAFETY: bool and u8 have the same memory layout and all valid bool
        // bit patterns are valid u8 bit patterns.
        unsafe { u8::slice_contains_slice(bytes_of(hs), bytes_of(needle)) }
    }
}

fn default_slice_contains_slice<T: PartialEq>(hs: &[T], needle: &[T]) -> bool {
    hs.windows(needle.len()).any(|slice| slice == needle)
}

/// SIMD search for short needles based on
/// Wojciech Muła's "SIMD-friendly algorithms for substring searching"[0]
///
/// It skips ahead by the vector width on each iteration (rather than the needle length as two-way
/// does) by probing the first and last byte of the needle for the whole vector width
/// and only doing full needle comparisons when the vectorized probe indicated potential matches.
///
/// Since the x86_64 baseline only offers SSE2 we only use u8x16 here.
/// If we ever ship std with for x86-64-v3 or adapt this for other platforms then wider vectors
/// should be evaluated.
///
/// For haystacks smaller than vector-size + needle length it falls back to
/// a naive O(n*m) search so this implementation should not be called on larger needles.
///
/// [0]: http://0x80.pl/articles/simd-strfind.html#sse-avx2
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn simd_contains(haystack: &[u8], needle: &[u8]) -> Option<bool> {
    debug_assert!(needle.len() > 1);

    use crate::ops::BitAnd;
    use crate::simd::mask8x16 as Mask;
    use crate::simd::u8x16 as Block;
    use crate::simd::{SimdPartialEq, ToBitMask};

    let first_probe = needle[0];
    let last_byte_offset = needle.len() - 1;

    // the offset used for the 2nd vector
    let second_probe_offset = if needle.len() == 2 {
        // never bail out on len=2 needles because the probes will fully cover them and have
        // no degenerate cases.
        1
    } else {
        // try a few bytes in case first and last byte of the needle are the same
        let Some(second_probe_offset) = (needle.len().saturating_sub(4)..needle.len()).rfind(|&idx| needle[idx] != first_probe) else {
            // fall back to other search methods if we can't find any different bytes
            // since we could otherwise hit some degenerate cases
            return None;
        };
        second_probe_offset
    };

    // do a naive search if the haystack is too small to fit
    if haystack.len() < Block::LANES + last_byte_offset {
        return Some(haystack.windows(needle.len()).any(|c| c == needle));
    }

    let first_probe: Block = Block::splat(first_probe);
    let second_probe: Block = Block::splat(needle[second_probe_offset]);
    // first byte are already checked by the outer loop. to verify a match only the
    // remainder has to be compared.
    let trimmed_needle = &needle[1..];

    // this #[cold] is load-bearing, benchmark before removing it...
    let check_mask = #[cold]
    |idx, mask: u16, skip: bool| -> bool {
        if skip {
            return false;
        }

        // and so is this. optimizations are weird.
        let mut mask = mask;

        while mask != 0 {
            let trailing = mask.trailing_zeros();
            let offset = idx + trailing as usize + 1;
            // SAFETY: mask is between 0 and 15 trailing zeroes, we skip one additional byte that was already compared
            // and then take trimmed_needle.len() bytes. This is within the bounds defined by the outer loop
            unsafe {
                let sub = haystack.get_unchecked(offset..).get_unchecked(..trimmed_needle.len());
                if small_slice_eq(sub, trimmed_needle) {
                    return true;
                }
            }
            mask &= !(1 << trailing);
        }
        return false;
    };

    let test_chunk = |idx| -> u16 {
        // SAFETY: this requires at least LANES bytes being readable at idx
        // that is ensured by the loop ranges (see comments below)
        let a: Block = unsafe { haystack.as_ptr().add(idx).cast::<Block>().read_unaligned() };
        // SAFETY: this requires LANES + block_offset bytes being readable at idx
        let b: Block = unsafe {
            haystack.as_ptr().add(idx).add(second_probe_offset).cast::<Block>().read_unaligned()
        };
        let eq_first: Mask = a.simd_eq(first_probe);
        let eq_last: Mask = b.simd_eq(second_probe);
        let both = eq_first.bitand(eq_last);
        let mask = both.to_bitmask();

        return mask;
    };

    let mut i = 0;
    let mut result = false;
    // The loop condition must ensure that there's enough headroom to read LANE bytes,
    // and not only at the current index but also at the index shifted by block_offset
    const UNROLL: usize = 4;
    while i + last_byte_offset + UNROLL * Block::LANES < haystack.len() && !result {
        let mut masks = [0u16; UNROLL];
        for j in 0..UNROLL {
            masks[j] = test_chunk(i + j * Block::LANES);
        }
        for j in 0..UNROLL {
            let mask = masks[j];
            if mask != 0 {
                result |= check_mask(i + j * Block::LANES, mask, result);
            }
        }
        i += UNROLL * Block::LANES;
    }
    while i + last_byte_offset + Block::LANES < haystack.len() && !result {
        let mask = test_chunk(i);
        if mask != 0 {
            result |= check_mask(i, mask, result);
        }
        i += Block::LANES;
    }

    // Process the tail that didn't fit into LANES-sized steps.
    // This simply repeats the same procedure but as right-aligned chunk instead
    // of a left-aligned one. The last byte must be exactly flush with the string end so
    // we don't miss a single byte or read out of bounds.
    let i = haystack.len() - last_byte_offset - Block::LANES;
    let mask = test_chunk(i);
    if mask != 0 {
        result |= check_mask(i, mask, result);
    }

    Some(result)
}

/// Compares short slices for equality.
///
/// It avoids a call to libc's memcmp which is faster on long slices
/// due to SIMD optimizations but it incurs a function call overhead.
///
/// # Safety
///
/// Both slices must have the same length.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] // only called on x86
#[inline]
unsafe fn small_slice_eq(x: &[u8], y: &[u8]) -> bool {
    debug_assert_eq!(x.len(), y.len());
    // This function is adapted from
    // https://github.com/BurntSushi/memchr/blob/8037d11b4357b0f07be2bb66dc2659d9cf28ad32/src/memmem/util.rs#L32

    // If we don't have enough bytes to do 4-byte at a time loads, then
    // fall back to the naive slow version.
    //
    // Potential alternative: We could do a copy_nonoverlapping combined with a mask instead
    // of a loop. Benchmark it.
    if x.len() < 4 {
        for (&b1, &b2) in x.iter().zip(y) {
            if b1 != b2 {
                return false;
            }
        }
        return true;
    }
    // When we have 4 or more bytes to compare, then proceed in chunks of 4 at
    // a time using unaligned loads.
    //
    // Also, why do 4 byte loads instead of, say, 8 byte loads? The reason is
    // that this particular version of memcmp is likely to be called with tiny
    // needles. That means that if we do 8 byte loads, then a higher proportion
    // of memcmp calls will use the slower variant above. With that said, this
    // is a hypothesis and is only loosely supported by benchmarks. There's
    // likely some improvement that could be made here. The main thing here
    // though is to optimize for latency, not throughput.

    // SAFETY: Via the conditional above, we know that both `px` and `py`
    // have the same length, so `px < pxend` implies that `py < pyend`.
    // Thus, derefencing both `px` and `py` in the loop below is safe.
    //
    // Moreover, we set `pxend` and `pyend` to be 4 bytes before the actual
    // end of `px` and `py`. Thus, the final dereference outside of the
    // loop is guaranteed to be valid. (The final comparison will overlap with
    // the last comparison done in the loop for lengths that aren't multiples
    // of four.)
    //
    // Finally, we needn't worry about alignment here, since we do unaligned
    // loads.
    unsafe {
        let (mut px, mut py) = (x.as_ptr(), y.as_ptr());
        let (pxend, pyend) = (px.add(x.len() - 4), py.add(y.len() - 4));
        while px < pxend {
            let vx = (px as *const u32).read_unaligned();
            let vy = (py as *const u32).read_unaligned();
            if vx != vy {
                return false;
            }
            px = px.add(4);
            py = py.add(4);
        }
        let vx = (pxend as *const u32).read_unaligned();
        let vy = (pyend as *const u32).read_unaligned();
        vx == vy
    }
}
